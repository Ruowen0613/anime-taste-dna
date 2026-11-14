from __future__ import annotations

import argparse
import copy
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anime_taste_dna.evaluation.ranking import InteractionCsr, load_interactions_csr
from anime_taste_dna.evaluation.strict_cold_start import eligible_user_indices
from anime_taste_dna.models.matrix_factorization import (
    ExplicitMatrixFactorization,
    _select_device,
    _users_for_csr,
    affine_permutation,
    fold_in_users,
)


@dataclass(frozen=True)
class StrictMfConfig:
    split_directory: Path
    output_report: Path
    embedding_dimension: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: str
    regularization_candidates: tuple[float, ...]
    observed_ratings: tuple[int, ...]
    relevance_threshold: int
    ranking_k: int
    user_batch_size: int

    @classmethod
    def from_toml(cls, path: Path) -> StrictMfConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            split_directory=resolve(raw["paths"]["split_directory"]),
            output_report=resolve(raw["paths"]["output_report"]),
            embedding_dimension=int(raw["training"]["embedding_dimension"]),
            epochs=int(raw["training"]["epochs"]),
            batch_size=int(raw["training"]["batch_size"]),
            learning_rate=float(raw["training"]["learning_rate"]),
            weight_decay=float(raw["training"]["weight_decay"]),
            seed=int(raw["training"]["seed"]),
            device=raw["training"]["device"],
            regularization_candidates=tuple(
                float(value) for value in raw["validation"]["regularization_candidates"]
            ),
            observed_ratings=tuple(
                int(value) for value in raw["validation"]["observed_ratings"]
            ),
            relevance_threshold=int(raw["validation"]["relevance_threshold"]),
            ranking_k=int(raw["validation"]["ranking_k"]),
            user_batch_size=int(raw["validation"]["user_batch_size"]),
        )


def evaluate_strict_mf(
    users: np.ndarray,
    history: InteractionCsr,
    targets: InteractionCsr,
    item_factors: np.ndarray,
    item_bias: np.ndarray,
    global_mean: float,
    supported: np.ndarray,
    observed_count: int,
    regularization: float,
    ranking_k: int,
    relevance_threshold: int,
    user_batch_size: int,
    bias_only: bool = False,
) -> dict[str, float | int]:
    discounts = 1.0 / np.log2(np.arange(2, ranking_k + 2))
    recall_sum = ndcg_sum = norm_sum = 0.0
    recommendations = []
    for start in range(0, len(users), user_batch_size):
        batch_users = users[start : start + user_batch_size]
        observed_items = np.stack(
            [history.user_items(int(user))[-observed_count:] for user in batch_users]
        )
        observed_scores = np.stack(
            [history.user_ratings(int(user))[-observed_count:] for user in batch_users]
        ).astype(np.float32)
        if bias_only:
            user_factors = np.zeros(
                (len(batch_users), item_factors.shape[1]), dtype=np.float32
            )
            user_bias = np.zeros(len(batch_users), dtype=np.float32)
        else:
            user_factors, user_bias = fold_in_users(
                item_factors,
                item_bias,
                global_mean,
                observed_items,
                observed_scores,
                regularization,
            )
        scores = user_factors @ item_factors.T
        scores += user_bias[:, None] + item_bias[None, :] + global_mean
        scores[:, ~supported] = -np.inf
        for row in range(len(batch_users)):
            scores[row, observed_items[row]] = -np.inf
        candidates = np.argpartition(scores, -ranking_k, axis=1)[:, -ranking_k:]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
        ranked = np.take_along_axis(candidates, order, axis=1)
        recommendations.append(ranked)
        norm_sum += float(np.linalg.norm(user_factors, axis=1).sum())
        for row, user_index in enumerate(batch_users):
            target_items = targets.user_items(int(user_index))
            target_ratings = targets.user_ratings(int(user_index))
            relevant = set(
                target_items[
                    (target_ratings >= relevance_threshold) & supported[target_items]
                ].tolist()
            )
            hits = np.asarray(
                [position for position, item in enumerate(ranked[row]) if item in relevant],
                dtype=np.int64,
            )
            recall_sum += len(hits) / len(relevant)
            dcg = float(discounts[hits].sum()) if len(hits) else 0.0
            idcg = float(discounts[: min(len(relevant), ranking_k)].sum())
            ndcg_sum += dcg / idcg
    recommendation_matrix = np.concatenate(recommendations)
    unique_items = len(np.unique(recommendation_matrix))
    return {
        "eligible_users": len(users),
        "recall_at_10": round(recall_sum / len(users), 8),
        "ndcg_at_10": round(ndcg_sum / len(users), 8),
        "mean_user_factor_norm": round(norm_sum / len(users), 8),
        "catalog_coverage_items": unique_items,
        "catalog_coverage_fraction": round(unique_items / int(supported.sum()), 8),
    }


def _mean_ranking_metrics(by_observed: dict[str, dict[str, float | int]]) -> tuple[float, float]:
    return (
        float(np.mean([metrics["ndcg_at_10"] for metrics in by_observed.values()])),
        float(np.mean([metrics["recall_at_10"] for metrics in by_observed.values()])),
    )


def train_and_validate(config: StrictMfConfig) -> dict[str, Any]:
    split_report = json.loads(
        (config.split_directory / "split_report.json").read_text(encoding="utf-8")
    )
    users_count = int(split_report["source"]["users"])
    items_count = int(split_report["source"]["items"])
    train = load_interactions_csr(
        config.split_directory / "train.parquet", users_count, config.batch_size
    )
    validation_history = load_interactions_csr(
        config.split_directory / "validation_history.parquet",
        users_count,
        config.batch_size,
    )
    validation_target = load_interactions_csr(
        config.split_directory / "validation_target.parquet",
        users_count,
        config.batch_size,
    )
    supported = np.bincount(train.item_indices, minlength=items_count) > 0
    validation_users = eligible_user_indices(
        validation_history,
        validation_target,
        supported,
        max(config.observed_ratings),
        config.relevance_threshold,
    )
    train_users = _users_for_csr(train)
    global_mean = float(train.ratings.astype(np.float64).mean())
    device = _select_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = ExplicitMatrixFactorization(
        users_count, items_count, config.embedding_dimension, global_mean
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    row_positions = np.arange(config.batch_size, dtype=np.int64)
    best_selection = (-math.inf, -math.inf)
    best_state: dict[str, torch.Tensor] | None = None
    history = []

    for epoch in range(1, config.epochs + 1):
        multiplier, offset = affine_permutation(len(train_users), config.seed + epoch)
        squared_error = 0.0
        model.train()
        for start in range(0, len(train_users), config.batch_size):
            length = min(config.batch_size, len(train_users) - start)
            indices = (multiplier * (row_positions[:length] + start) + offset) % len(
                train_users
            )
            user_tensor = torch.from_numpy(train_users[indices]).to(
                device=device, dtype=torch.long
            )
            item_tensor = torch.from_numpy(train.item_indices[indices]).to(
                device=device, dtype=torch.long
            )
            rating_tensor = torch.from_numpy(train.ratings[indices]).to(
                device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(user_tensor, item_tensor)
            loss = torch.nn.functional.mse_loss(predictions, rating_tensor)
            loss.backward()
            optimizer.step()
            squared_error += float(loss.detach().cpu()) * length

        item_factors = model.item_factors.weight.detach().cpu().numpy()
        item_bias = model.item_bias.weight.detach().cpu().numpy().squeeze(1)
        trials = []
        for regularization in config.regularization_candidates:
            by_observed = {
                str(observed): evaluate_strict_mf(
                    validation_users,
                    validation_history,
                    validation_target,
                    item_factors,
                    item_bias,
                    global_mean,
                    supported,
                    observed,
                    regularization,
                    config.ranking_k,
                    config.relevance_threshold,
                    config.user_batch_size,
                )
                for observed in config.observed_ratings
            }
            mean_ndcg, mean_recall = _mean_ranking_metrics(by_observed)
            trials.append(
                {
                    "regularization": regularization,
                    "mean_ndcg_at_10": round(mean_ndcg, 8),
                    "mean_recall_at_10": round(mean_recall, 8),
                    "by_observed_ratings": by_observed,
                }
            )
        selected = max(
            trials,
            key=lambda trial: (
                trial["mean_ndcg_at_10"],
                trial["mean_recall_at_10"],
                -trial["regularization"],
            ),
        )
        entry = {
            "epoch": epoch,
            "train_rmse": round(math.sqrt(squared_error / len(train_users)), 8),
            "regularization_trials": trials,
            "selected": selected,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        selection = (selected["mean_ndcg_at_10"], selected["mean_recall_at_10"])
        if selection > best_selection:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Strict MF training did not produce a checkpoint")
    best_entry = max(
        history,
        key=lambda entry: (
            entry["selected"]["mean_ndcg_at_10"],
            entry["selected"]["mean_recall_at_10"],
        ),
    )
    model.load_state_dict(best_state)
    item_factors = model.item_factors.weight.detach().cpu().numpy()
    item_bias = model.item_bias.weight.detach().cpu().numpy().squeeze(1)
    selected_regularization = float(best_entry["selected"]["regularization"])
    bias_only = {
        str(observed): evaluate_strict_mf(
            validation_users,
            validation_history,
            validation_target,
            item_factors,
            item_bias,
            global_mean,
            supported,
            observed,
            selected_regularization,
            config.ranking_k,
            config.relevance_threshold,
            config.user_batch_size,
            bias_only=True,
        )
        for observed in config.observed_ratings
    }
    report = {
        "protocol": {
            "split": "strict held-out validation users",
            "global_fit_users": split_report["user_groups"]["train"],
            "validation_users": len(validation_users),
            "candidate_exclusion": "only exposed ratings",
            "test_access": "none",
        },
        "model": {
            "name": "explicit matrix factorization",
            "embedding_dimension": config.embedding_dimension,
            "best_epoch": best_entry["epoch"],
            "selected_regularization": selected_regularization,
            "history": history,
        },
        "selected_validation": best_entry["selected"],
        "item_bias_only": bias_only,
        "reproducibility": {"seed": config.seed, "device": str(device)},
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MF under strict new-user validation")
    parser.add_argument("--config", type=Path, default=Path("configs/strict_mf.toml"))
    args = parser.parse_args()
    config = StrictMfConfig.from_toml(args.config.resolve())
    print(json.dumps(train_and_validate(config), indent=2))


if __name__ == "__main__":
    main()
