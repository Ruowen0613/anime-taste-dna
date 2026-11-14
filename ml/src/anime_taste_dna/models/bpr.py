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

from anime_taste_dna.evaluation.fold_in_diagnostic import prepare_validation_users
from anime_taste_dna.evaluation.personalization_audit import AuditUser
from anime_taste_dna.evaluation.ranking import load_interactions_csr
from anime_taste_dna.models.matrix_factorization import _select_device, _users_for_csr


@dataclass(frozen=True)
class BprConfig:
    split_directory: Path
    output_report: Path
    embedding_dimension: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    positive_rating_threshold: int
    seed: int
    device: str
    checkpoint_users: int
    fold_in_regularization_candidates: tuple[float, ...]
    observed_ratings: int
    relevance_threshold: int
    ranking_k: int
    user_batch_size: int
    parquet_batch_size: int
    pairwise_fold_in_steps: int

    @classmethod
    def from_toml(cls, path: Path) -> BprConfig:
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
            positive_rating_threshold=int(raw["training"]["positive_rating_threshold"]),
            seed=int(raw["training"]["seed"]),
            device=raw["training"]["device"],
            checkpoint_users=int(raw["validation"]["checkpoint_users"]),
            fold_in_regularization_candidates=tuple(
                float(value) for value in raw["validation"]["fold_in_regularization_candidates"]
            ),
            observed_ratings=int(raw["validation"]["observed_ratings"]),
            relevance_threshold=int(raw["validation"]["relevance_threshold"]),
            ranking_k=int(raw["validation"]["ranking_k"]),
            user_batch_size=int(raw["validation"]["user_batch_size"]),
            parquet_batch_size=int(raw["validation"]["parquet_batch_size"]),
            pairwise_fold_in_steps=int(raw["validation"]["pairwise_fold_in_steps"]),
        )


class BprMatrixFactorization(nn.Module):
    def __init__(self, users: int, items: int, dimensions: int) -> None:
        super().__init__()
        self.user_factors = nn.Embedding(users, dimensions)
        self.item_factors = nn.Embedding(items, dimensions)
        self.item_bias = nn.Embedding(items, 1)
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def pairwise_margin(
        self, users: torch.Tensor, positive_items: torch.Tensor, negative_items: torch.Tensor
    ) -> torch.Tensor:
        user_factors = self.user_factors(users)
        positive = (
            (user_factors * self.item_factors(positive_items)).sum(dim=1)
            + self.item_bias(positive_items).squeeze(1)
        )
        negative = (
            (user_factors * self.item_factors(negative_items)).sum(dim=1)
            + self.item_bias(negative_items).squeeze(1)
        )
        return positive - negative


def sample_unobserved_items(
    users: np.ndarray,
    supported_items: np.ndarray,
    observed_keys: np.ndarray,
    item_count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    negatives = generator.choice(supported_items, size=len(users)).astype(np.int32)
    for _ in range(20):
        keys = users.astype(np.int64) * item_count + negatives
        positions = np.searchsorted(observed_keys, keys)
        clipped = np.minimum(positions, len(observed_keys) - 1)
        invalid = (positions < len(observed_keys)) & (observed_keys[clipped] == keys)
        if not invalid.any():
            return negatives
        negatives[invalid] = generator.choice(supported_items, size=int(invalid.sum()))
    raise RuntimeError("Unable to sample unobserved items after 20 attempts")


def fold_in_preference_users(
    item_factors: np.ndarray,
    observed_items: np.ndarray,
    observed_ratings: np.ndarray,
    regularization: float,
) -> np.ndarray:
    factors = item_factors[observed_items]
    targets = np.clip((observed_ratings - 7.5) / 2.5, -1.0, 1.0).astype(np.float32)
    gram = np.einsum("bni,bnj->bij", factors, factors)
    rhs = np.einsum("bni,bn->bi", factors, targets)
    penalty = np.eye(item_factors.shape[1], dtype=np.float32) * regularization
    return np.linalg.solve(gram + penalty, rhs[..., None]).squeeze(-1)


def sample_fold_in_negatives(
    observed_items: np.ndarray,
    supported_items: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    negatives = generator.choice(supported_items, size=observed_items.shape).astype(np.int32)
    for _ in range(20):
        invalid = (negatives[:, :, None] == observed_items[:, None, :]).any(axis=2)
        if not invalid.any():
            return negatives
        negatives[invalid] = generator.choice(supported_items, size=int(invalid.sum()))
    raise RuntimeError("Unable to sample fold-in negatives after 20 attempts")


def fold_in_pairwise_users(
    item_factors: np.ndarray,
    item_bias: np.ndarray,
    observed_items: np.ndarray,
    observed_ratings: np.ndarray,
    negative_items: np.ndarray,
    regularization: float,
    steps: int,
) -> np.ndarray:
    observed_factors = item_factors[observed_items]
    observed_bias = item_bias[observed_items]
    pair_differences = observed_factors[:, :, None, :] - observed_factors[:, None, :, :]
    pair_offsets = observed_bias[:, :, None] - observed_bias[:, None, :]
    rating_differences = observed_ratings[:, :, None] - observed_ratings[:, None, :]
    pair_mask = rating_differences >= 2.0
    batch_size, observations = observed_items.shape
    pair_differences = pair_differences.reshape(batch_size, observations**2, -1)
    pair_offsets = pair_offsets.reshape(batch_size, observations**2)
    pair_mask = pair_mask.reshape(batch_size, observations**2)

    sampled_differences = observed_factors - item_factors[negative_items]
    sampled_offsets = observed_bias - item_bias[negative_items]
    sampled_mask = observed_ratings >= 8.0
    differences = np.concatenate((pair_differences, sampled_differences), axis=1)
    offsets = np.concatenate((pair_offsets, sampled_offsets), axis=1)
    mask = np.concatenate((pair_mask, sampled_mask), axis=1).astype(np.float32)

    user_factors = np.zeros((batch_size, item_factors.shape[1]), dtype=np.float32)
    identity = np.eye(item_factors.shape[1], dtype=np.float32)[None, :, :]
    for _ in range(steps):
        margins = np.einsum("bpd,bd->bp", differences, user_factors) + offsets
        probabilities = 1.0 / (1.0 + np.exp(np.clip(margins, -30.0, 30.0)))
        weighted_probabilities = probabilities * mask
        gradient = -np.einsum("bpd,bp->bd", differences, weighted_probabilities)
        gradient += regularization * user_factors
        curvature = probabilities * (1.0 - probabilities) * mask
        hessian = np.einsum("bpi,bp,bpj->bij", differences, curvature, differences)
        hessian += regularization * identity
        update = np.linalg.solve(hessian, gradient[..., None]).squeeze(-1)
        user_factors -= update
    return user_factors


def evaluate_cold_start(
    users: list[AuditUser],
    item_factors: np.ndarray,
    item_bias: np.ndarray,
    supported: np.ndarray,
    regularization: float,
    config: BprConfig,
) -> dict[str, float | int]:
    discounts = 1.0 / np.log2(np.arange(2, config.ranking_k + 2))
    recall_sum = ndcg_sum = norm_sum = 0.0
    recommended: list[np.ndarray] = []
    generator = np.random.default_rng(config.seed)
    supported_indices = np.flatnonzero(supported).astype(np.int32)
    for start in range(0, len(users), config.user_batch_size):
        batch = users[start : start + config.user_batch_size]
        observed_items = np.stack(
            [user.history_items[-config.observed_ratings :] for user in batch]
        )
        observed_ratings = np.stack(
            [user.history_ratings[-config.observed_ratings :] for user in batch]
        ).astype(np.float32)
        negative_items = sample_fold_in_negatives(
            observed_items, supported_indices, generator
        )
        user_factors = fold_in_pairwise_users(
            item_factors,
            item_bias,
            observed_items,
            observed_ratings,
            negative_items,
            regularization,
            config.pairwise_fold_in_steps,
        )
        scores = user_factors @ item_factors.T + item_bias[None, :]
        scores[:, ~supported] = -np.inf
        for row, user in enumerate(batch):
            scores[row, user.history_items] = -np.inf
        candidates = np.argpartition(scores, -config.ranking_k, axis=1)[
            :, -config.ranking_k :
        ]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
        ranked = np.take_along_axis(candidates, order, axis=1)
        recommended.append(ranked)
        norm_sum += float(np.linalg.norm(user_factors, axis=1).sum())
        for row, user in enumerate(batch):
            relevant = set(user.relevant_items.tolist())
            hits = np.asarray(
                [position for position, item in enumerate(ranked[row]) if item in relevant],
                dtype=np.int64,
            )
            recall_sum += len(hits) / len(relevant)
            dcg = float(discounts[hits].sum()) if len(hits) else 0.0
            idcg = float(discounts[: min(len(relevant), config.ranking_k)].sum())
            ndcg_sum += dcg / idcg
    recommendation_matrix = np.concatenate(recommended)
    unique_items = len(np.unique(recommendation_matrix))
    return {
        "eligible_users": len(users),
        "recall_at_10": round(recall_sum / len(users), 8),
        "ndcg_at_10": round(ndcg_sum / len(users), 8),
        "mean_user_factor_norm": round(norm_sum / len(users), 8),
        "catalog_coverage_items": unique_items,
        "catalog_coverage_fraction": round(unique_items / int(supported.sum()), 8),
    }


def train_and_validate(config: BprConfig) -> dict[str, Any]:
    split_report = json.loads(
        (config.split_directory / "split_report.json").read_text(encoding="utf-8")
    )
    users_count = int(split_report["source"]["users"])
    items_count = int(split_report["source"]["items"])
    train = load_interactions_csr(
        config.split_directory / "train.parquet", users_count, config.parquet_batch_size
    )
    validation = load_interactions_csr(
        config.split_directory / "validation.parquet", users_count, config.parquet_batch_size
    )
    train_users = _users_for_csr(train)
    positive_mask = train.ratings >= config.positive_rating_threshold
    positive_users = train_users[positive_mask]
    positive_items = train.item_indices[positive_mask]
    item_counts = np.bincount(train.item_indices, minlength=items_count)
    supported = item_counts > 0
    supported_indices = np.flatnonzero(supported).astype(np.int32)
    observed_keys = np.sort(
        train_users.astype(np.int64) * items_count + train.item_indices.astype(np.int64)
    )
    validation_users = prepare_validation_users(
        train,
        validation,
        supported,
        config.observed_ratings,
        config.relevance_threshold,
    )
    checkpoint_generator = np.random.default_rng(config.seed)
    checkpoint_indices = np.sort(
        checkpoint_generator.choice(
            len(validation_users),
            size=min(config.checkpoint_users, len(validation_users)),
            replace=False,
        )
    )
    checkpoint_users = [validation_users[index] for index in checkpoint_indices]

    device = _select_device(config.device)
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)
    model = BprMatrixFactorization(
        users_count, items_count, config.embedding_dimension
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_selection: tuple[float, float] = (-math.inf, -math.inf)
    history = []

    for epoch in range(1, config.epochs + 1):
        order = generator.permutation(len(positive_users))
        loss_sum = 0.0
        model.train()
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            batch_users = positive_users[indices]
            batch_positive = positive_items[indices]
            batch_negative = sample_unobserved_items(
                batch_users,
                supported_indices,
                observed_keys,
                items_count,
                generator,
            )
            users_tensor = torch.from_numpy(batch_users).to(device=device, dtype=torch.long)
            positive_tensor = torch.from_numpy(batch_positive).to(
                device=device, dtype=torch.long
            )
            negative_tensor = torch.from_numpy(batch_negative).to(
                device=device, dtype=torch.long
            )
            optimizer.zero_grad(set_to_none=True)
            loss = -torch.nn.functional.logsigmoid(
                model.pairwise_margin(users_tensor, positive_tensor, negative_tensor)
            ).mean()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)

        item_factors = model.item_factors.weight.detach().cpu().numpy()
        item_bias = model.item_bias.weight.detach().cpu().numpy().squeeze(1)
        trials = []
        for regularization in config.fold_in_regularization_candidates:
            metrics = evaluate_cold_start(
                checkpoint_users,
                item_factors,
                item_bias,
                supported,
                regularization,
                config,
            )
            trials.append({"regularization": regularization, **metrics})
        selected = max(
            trials,
            key=lambda trial: (trial["ndcg_at_10"], trial["recall_at_10"]),
        )
        history_entry = {
            "epoch": epoch,
            "mean_bpr_loss": round(loss_sum / len(order), 8),
            "checkpoint_validation_trials": trials,
            "selected_checkpoint_metrics": selected,
        }
        history.append(history_entry)
        print(json.dumps(history_entry), flush=True)
        selection = (float(selected["ndcg_at_10"]), float(selected["recall_at_10"]))
        if selection > best_selection:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("BPR training did not produce a checkpoint")
    model.load_state_dict(best_state)
    item_factors = model.item_factors.weight.detach().cpu().numpy()
    item_bias = model.item_bias.weight.detach().cpu().numpy().squeeze(1)
    full_trials = []
    for regularization in config.fold_in_regularization_candidates:
        full_trials.append(
            {
                "regularization": regularization,
                **evaluate_cold_start(
                    validation_users,
                    item_factors,
                    item_bias,
                    supported,
                    regularization,
                    config,
                ),
            }
        )
    selected_full = max(
        full_trials,
        key=lambda trial: (trial["ndcg_at_10"], trial["recall_at_10"]),
    )
    best_epoch = max(
        history,
        key=lambda entry: (
            entry["selected_checkpoint_metrics"]["ndcg_at_10"],
            entry["selected_checkpoint_metrics"]["recall_at_10"],
        ),
    )["epoch"]
    report = {
        "protocol": {
            "split": "validation only",
            "test_access": "none",
            "positive_definition": f"train rating >= {config.positive_rating_threshold}",
            "negative_sampling": "uniform train-supported item absent from user train history",
            "fold_in": (
                "regularized pairwise logistic optimization with fixed item factors and biases"
            ),
        },
        "model": {
            "name": "bayesian personalized ranking matrix factorization",
            "embedding_dimension": config.embedding_dimension,
            "best_epoch": best_epoch,
            "history": history,
        },
        "full_validation_trials": full_trials,
        "selected_validation": selected_full,
        "reproducibility": {
            "seed": config.seed,
            "device": str(device),
            "positive_interactions": len(positive_users),
            "checkpoint_users": len(checkpoint_users),
            "full_validation_users": len(validation_users),
        },
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate BPR Model C")
    parser.add_argument("--config", type=Path, default=Path("configs/bpr.toml"))
    args = parser.parse_args()
    report = train_and_validate(BprConfig.from_toml(args.config.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
