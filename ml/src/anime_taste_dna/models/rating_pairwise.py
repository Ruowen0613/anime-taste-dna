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
from torch import nn

from anime_taste_dna.evaluation.ranking import InteractionCsr, load_interactions_csr
from anime_taste_dna.evaluation.strict_cold_start import eligible_user_indices
from anime_taste_dna.models.matrix_factorization import (
    _select_device,
    _users_for_csr,
    affine_permutation,
)


@dataclass(frozen=True)
class RatingPairwiseConfig:
    split_directory: Path
    output_report: Path
    embedding_dimension: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    minimum_rating_gap: int
    seed: int
    device: str
    regularization_candidates: tuple[float, ...]
    observed_ratings: tuple[int, ...]
    relevance_threshold: int
    ranking_k: int
    user_batch_size: int
    fold_in_steps: int

    @classmethod
    def from_toml(cls, path: Path) -> RatingPairwiseConfig:
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
            minimum_rating_gap=int(raw["training"]["minimum_rating_gap"]),
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
            fold_in_steps=int(raw["validation"]["fold_in_steps"]),
        )


class RatingPairwiseMatrixFactorization(nn.Module):
    def __init__(self, users: int, items: int, dimensions: int) -> None:
        super().__init__()
        self.user_factors = nn.Embedding(users, dimensions)
        self.item_factors = nn.Embedding(items, dimensions)
        self.item_bias = nn.Embedding(items, 1)
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def pairwise_margin(
        self, users: torch.Tensor, higher_items: torch.Tensor, lower_items: torch.Tensor
    ) -> torch.Tensor:
        user_factors = self.user_factors(users)
        higher = (
            (user_factors * self.item_factors(higher_items)).sum(dim=1)
            + self.item_bias(higher_items).squeeze(1)
        )
        lower = (
            (user_factors * self.item_factors(lower_items)).sum(dim=1)
            + self.item_bias(lower_items).squeeze(1)
        )
        return higher - lower


def build_observed_rating_pairs(
    items: np.ndarray,
    ratings: np.ndarray,
    minimum_rating_gap: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate observed-only ordered pairs and preserve the full 1-10 rating gap."""
    differences = ratings[:, None].astype(np.int16) - ratings[None, :].astype(np.int16)
    higher_positions, lower_positions = np.nonzero(differences >= minimum_rating_gap)
    weights = differences[higher_positions, lower_positions].astype(np.float32) / 9.0
    return items[higher_positions], items[lower_positions], weights


def weighted_pairwise_loss(margins: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return -(torch.nn.functional.logsigmoid(margins) * weights).sum() / weights.sum()


def fold_in_rating_pairwise_users(
    item_factors: np.ndarray,
    item_bias: np.ndarray,
    observed_items: np.ndarray,
    observed_ratings: np.ndarray,
    regularization: float,
    steps: int,
    minimum_rating_gap: int = 1,
) -> np.ndarray:
    batch_size = len(observed_items)
    dimensions = item_factors.shape[1]
    user_factors = np.zeros((batch_size, dimensions), dtype=np.float32)
    identity = np.eye(dimensions, dtype=np.float32)
    for row in range(batch_size):
        higher, lower, weights = build_observed_rating_pairs(
            observed_items[row], observed_ratings[row], minimum_rating_gap
        )
        if not len(weights):
            continue
        differences = item_factors[higher] - item_factors[lower]
        offsets = item_bias[higher] - item_bias[lower]
        user = user_factors[row]
        for _ in range(steps):
            margins = differences @ user + offsets
            probabilities = 1.0 / (1.0 + np.exp(np.clip(margins, -30.0, 30.0)))
            weighted_probabilities = weights * probabilities
            gradient = -(differences * weighted_probabilities[:, None]).sum(axis=0)
            gradient += regularization * user
            curvature = weights * probabilities * (1.0 - probabilities)
            hessian = differences.T @ (differences * curvature[:, None])
            hessian += regularization * identity
            user -= np.linalg.solve(hessian, gradient).astype(np.float32)
        user_factors[row] = user
    return user_factors


def evaluate_rating_pairwise(
    users: np.ndarray,
    history: InteractionCsr,
    targets: InteractionCsr,
    item_factors: np.ndarray,
    item_bias: np.ndarray,
    supported: np.ndarray,
    observed_count: int,
    regularization: float,
    config: RatingPairwiseConfig,
    bias_only: bool = False,
) -> dict[str, float | int]:
    discounts = 1.0 / np.log2(np.arange(2, config.ranking_k + 2))
    recall_sum = ndcg_sum = norm_sum = 0.0
    recommendations: list[np.ndarray] = []
    for start in range(0, len(users), config.user_batch_size):
        batch_users = users[start : start + config.user_batch_size]
        observed_items = np.stack(
            [history.user_items(int(user))[-observed_count:] for user in batch_users]
        )
        observed_scores = np.stack(
            [history.user_ratings(int(user))[-observed_count:] for user in batch_users]
        )
        if bias_only:
            user_factors = np.zeros(
                (len(batch_users), item_factors.shape[1]), dtype=np.float32
            )
        else:
            user_factors = fold_in_rating_pairwise_users(
                item_factors,
                item_bias,
                observed_items,
                observed_scores,
                regularization,
                config.fold_in_steps,
                config.minimum_rating_gap,
            )
        scores = user_factors @ item_factors.T + item_bias[None, :]
        scores[:, ~supported] = -np.inf
        for row in range(len(batch_users)):
            scores[row, observed_items[row]] = -np.inf
        candidates = np.argpartition(scores, -config.ranking_k, axis=1)[
            :, -config.ranking_k :
        ]
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
                    (target_ratings >= config.relevance_threshold)
                    & supported[target_items]
                ].tolist()
            )
            hits = np.asarray(
                [position for position, item in enumerate(ranked[row]) if item in relevant],
                dtype=np.int64,
            )
            recall_sum += len(hits) / len(relevant)
            dcg = float(discounts[hits].sum()) if len(hits) else 0.0
            idcg = float(discounts[: min(len(relevant), config.ranking_k)].sum())
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


def _mean_metrics(by_observed: dict[str, dict[str, float | int]]) -> tuple[float, float]:
    return (
        float(np.mean([metrics["ndcg_at_10"] for metrics in by_observed.values()])),
        float(np.mean([metrics["recall_at_10"] for metrics in by_observed.values()])),
    )


def train_and_validate(config: RatingPairwiseConfig) -> dict[str, Any]:
    split_report = json.loads(
        (config.split_directory / "split_report.json").read_text(encoding="utf-8")
    )
    users_count = int(split_report["source"]["users"])
    items_count = int(split_report["source"]["items"])
    train = load_interactions_csr(
        config.split_directory / "train.parquet", users_count, config.batch_size
    )
    validation_history = load_interactions_csr(
        config.split_directory / "validation_history.parquet", users_count, config.batch_size
    )
    validation_target = load_interactions_csr(
        config.split_directory / "validation_target.parquet", users_count, config.batch_size
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
    device = _select_device(config.device)
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)
    model = RatingPairwiseMatrixFactorization(
        users_count, items_count, config.embedding_dimension
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    row_positions = np.arange(config.batch_size, dtype=np.int64)
    best_selection = (-math.inf, -math.inf)
    best_state: dict[str, torch.Tensor] | None = None
    history_entries = []

    for epoch in range(1, config.epochs + 1):
        multiplier, offset = affine_permutation(len(train_users), config.seed + epoch)
        weighted_loss_sum = weight_sum = 0.0
        valid_pairs = 0
        model.train()
        for start in range(0, len(train_users), config.batch_size):
            length = min(config.batch_size, len(train_users) - start)
            indices = (multiplier * (row_positions[:length] + start) + offset) % len(
                train_users
            )
            users = train_users[indices]
            counts = train.offsets[users + 1] - train.offsets[users]
            partner_positions = train.offsets[users] + np.floor(
                generator.random(length) * counts
            ).astype(np.int64)
            first_ratings = train.ratings[indices].astype(np.int16)
            second_ratings = train.ratings[partner_positions].astype(np.int16)
            gaps = first_ratings - second_ratings
            valid = np.abs(gaps) >= config.minimum_rating_gap
            if not valid.any():
                continue
            batch_users = users[valid]
            first_items = train.item_indices[indices[valid]]
            second_items = train.item_indices[partner_positions[valid]]
            valid_gaps = gaps[valid]
            higher_items = np.where(valid_gaps > 0, first_items, second_items)
            lower_items = np.where(valid_gaps > 0, second_items, first_items)
            weights = np.abs(valid_gaps).astype(np.float32) / 9.0

            user_tensor = torch.from_numpy(batch_users).to(device=device, dtype=torch.long)
            higher_tensor = torch.from_numpy(higher_items).to(
                device=device, dtype=torch.long
            )
            lower_tensor = torch.from_numpy(lower_items).to(
                device=device, dtype=torch.long
            )
            weight_tensor = torch.from_numpy(weights).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            margins = model.pairwise_margin(user_tensor, higher_tensor, lower_tensor)
            loss = weighted_pairwise_loss(margins, weight_tensor)
            loss.backward()
            optimizer.step()
            batch_weight = float(weights.sum())
            weighted_loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight
            valid_pairs += len(weights)

        item_factors = model.item_factors.weight.detach().cpu().numpy()
        item_bias = model.item_bias.weight.detach().cpu().numpy().squeeze(1)
        trials = []
        for regularization in config.regularization_candidates:
            by_observed = {
                str(observed): evaluate_rating_pairwise(
                    validation_users,
                    validation_history,
                    validation_target,
                    item_factors,
                    item_bias,
                    supported,
                    observed,
                    regularization,
                    config,
                )
                for observed in config.observed_ratings
            }
            mean_ndcg, mean_recall = _mean_metrics(by_observed)
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
            "train_weighted_pairwise_loss": round(weighted_loss_sum / weight_sum, 8),
            "sampled_valid_pairs": valid_pairs,
            "regularization_trials": trials,
            "selected": selected,
        }
        history_entries.append(entry)
        print(json.dumps(entry), flush=True)
        selection = (selected["mean_ndcg_at_10"], selected["mean_recall_at_10"])
        if selection > best_selection:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Rating-pairwise training did not produce a checkpoint")
    model.load_state_dict(best_state)
    best_entry = max(
        history_entries,
        key=lambda entry: (
            entry["selected"]["mean_ndcg_at_10"],
            entry["selected"]["mean_recall_at_10"],
        ),
    )
    item_factors = model.item_factors.weight.detach().cpu().numpy()
    item_bias = model.item_bias.weight.detach().cpu().numpy().squeeze(1)
    bias_only = {
        str(observed): evaluate_rating_pairwise(
            validation_users,
            validation_history,
            validation_target,
            item_factors,
            item_bias,
            supported,
            observed,
            float(best_entry["selected"]["regularization"]),
            config,
            bias_only=True,
        )
        for observed in config.observed_ratings
    }
    report = {
        "experiment": "rating-aware-pairwise-mf-validation-v1",
        "protocol": {
            "split_directory": str(config.split_directory),
            "test_split_read": False,
            "pair_definition": "same-user observed ratings only; higher rating over lower rating",
            "pair_weight": "absolute rating gap / 9",
            "unobserved_items_as_negatives": False,
            "eligible_validation_users": len(validation_users),
        },
        "config": {
            "embedding_dimension": config.embedding_dimension,
            "epochs": config.epochs,
            "seed": config.seed,
            "observed_ratings": config.observed_ratings,
            "regularization_candidates": config.regularization_candidates,
        },
        "history": history_entries,
        "selected": {"epoch": best_entry["epoch"], **best_entry["selected"]},
        "item_bias_only": bias_only,
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full-rating pairwise MF")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rating_pairwise.toml")
    )
    args = parser.parse_args()
    train_and_validate(RatingPairwiseConfig.from_toml(args.config.resolve()))


if __name__ == "__main__":
    main()
