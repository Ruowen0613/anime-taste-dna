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

from anime_taste_dna.evaluation.ranking import InteractionCsr, load_interactions_csr
from anime_taste_dna.evaluation.strict_cold_start import eligible_user_indices
from anime_taste_dna.models.matrix_factorization import _users_for_csr


@dataclass(frozen=True)
class WeightedAlsConfig:
    split_directory: Path
    output_report: Path
    dimensions: int
    iterations: int
    alpha: float
    regularization: float
    positive_rating_threshold: int
    seed: int
    fold_in_regularization_candidates: tuple[float, ...]
    popularity_blend_candidates: tuple[float, ...]
    observed_ratings: tuple[int, ...]
    relevance_threshold: int
    ranking_k: int
    user_batch_size: int
    parquet_batch_size: int

    @classmethod
    def from_toml(cls, path: Path) -> WeightedAlsConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            split_directory=resolve(raw["paths"]["split_directory"]),
            output_report=resolve(raw["paths"]["output_report"]),
            dimensions=int(raw["training"]["dimensions"]),
            iterations=int(raw["training"]["iterations"]),
            alpha=float(raw["training"]["alpha"]),
            regularization=float(raw["training"]["regularization"]),
            positive_rating_threshold=int(raw["training"]["positive_rating_threshold"]),
            seed=int(raw["training"]["seed"]),
            fold_in_regularization_candidates=tuple(
                float(value) for value in raw["validation"]["regularization_candidates"]
            ),
            popularity_blend_candidates=tuple(
                float(value) for value in raw["validation"]["popularity_blend_candidates"]
            ),
            observed_ratings=tuple(
                int(value) for value in raw["validation"]["observed_ratings"]
            ),
            relevance_threshold=int(raw["validation"]["relevance_threshold"]),
            ranking_k=int(raw["validation"]["ranking_k"]),
            user_batch_size=int(raw["validation"]["user_batch_size"]),
            parquet_batch_size=int(raw["validation"]["parquet_batch_size"]),
        )


def implicit_confidence(
    ratings: np.ndarray, threshold: int, alpha: float
) -> np.ndarray:
    strength = ratings.astype(np.float32) - threshold + 1.0
    return 1.0 + alpha * np.maximum(strength, 0.0)


def solve_implicit_factors(
    offsets: np.ndarray,
    related_indices: np.ndarray,
    confidence: np.ndarray,
    fixed_factors: np.ndarray,
    regularization: float,
) -> np.ndarray:
    dimensions = fixed_factors.shape[1]
    result = np.zeros((len(offsets) - 1, dimensions), dtype=np.float32)
    base = fixed_factors.T @ fixed_factors
    penalty = np.eye(dimensions, dtype=np.float32) * regularization
    for entity in range(len(result)):
        start, end = offsets[entity : entity + 2]
        if start == end:
            continue
        factors = fixed_factors[related_indices[start:end]]
        weights = confidence[start:end]
        system = base + (factors.T * (weights - 1.0)) @ factors + penalty
        target = factors.T @ weights
        result[entity] = np.linalg.solve(system, target)
    return result


def build_item_view(
    users: np.ndarray,
    items: np.ndarray,
    confidence: np.ndarray,
    item_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(items, kind="stable")
    sorted_items = items[order]
    counts = np.bincount(sorted_items, minlength=item_count)
    offsets = np.zeros(item_count + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets, users[order], confidence[order]


def fold_in_implicit_users(
    item_factors: np.ndarray,
    observed_items: np.ndarray,
    observed_ratings: np.ndarray,
    threshold: int,
    alpha: float,
    regularization: float,
) -> np.ndarray:
    batch, observations = observed_items.shape
    positive = observed_ratings >= threshold
    counts = positive.sum(axis=1, dtype=np.int64)
    offsets = np.zeros(batch + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    if offsets[-1] == 0:
        return np.zeros((batch, item_factors.shape[1]), dtype=np.float32)
    selected_items = observed_items[positive]
    confidence = implicit_confidence(observed_ratings[positive], threshold, alpha)
    return solve_implicit_factors(
        offsets, selected_items, confidence, item_factors, regularization
    )


def evaluate_weighted_als(
    users: np.ndarray,
    history: InteractionCsr,
    targets: InteractionCsr,
    item_factors: np.ndarray,
    item_counts: np.ndarray,
    observed_count: int,
    fold_in_regularization: float,
    popularity_blend: float,
    config: WeightedAlsConfig,
) -> dict[str, float | int]:
    supported = item_counts > 0
    discounts = 1.0 / np.log2(np.arange(2, config.ranking_k + 2))
    popularity = np.log1p(item_counts).astype(np.float32)
    popularity_supported = popularity[supported]
    popularity = (popularity - popularity_supported.mean()) / popularity_supported.std()
    recall_sum = ndcg_sum = norm_sum = 0.0
    recommendations = []
    for start in range(0, len(users), config.user_batch_size):
        batch_users = users[start : start + config.user_batch_size]
        observed_items = np.stack(
            [history.user_items(int(user))[-observed_count:] for user in batch_users]
        )
        observed_ratings = np.stack(
            [history.user_ratings(int(user))[-observed_count:] for user in batch_users]
        )
        user_factors = fold_in_implicit_users(
            item_factors,
            observed_items,
            observed_ratings,
            config.positive_rating_threshold,
            config.alpha,
            fold_in_regularization,
        )
        scores = user_factors @ item_factors.T
        supported_scores = scores[:, supported]
        score_mean = supported_scores.mean(axis=1, keepdims=True)
        score_std = supported_scores.std(axis=1, keepdims=True)
        score_std[score_std < 1e-8] = 1.0
        scores = (scores - score_mean) / score_std
        scores += popularity_blend * popularity[None, :]
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
                    (target_ratings >= config.relevance_threshold) & supported[target_items]
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


def train_and_validate(config: WeightedAlsConfig) -> dict[str, Any]:
    split_report = json.loads(
        (config.split_directory / "split_report.json").read_text(encoding="utf-8")
    )
    users_count = int(split_report["source"]["users"])
    items_count = int(split_report["source"]["items"])
    train = load_interactions_csr(
        config.split_directory / "train.parquet", users_count, config.parquet_batch_size
    )
    history = load_interactions_csr(
        config.split_directory / "validation_history.parquet",
        users_count,
        config.parquet_batch_size,
    )
    targets = load_interactions_csr(
        config.split_directory / "validation_target.parquet",
        users_count,
        config.parquet_batch_size,
    )
    all_train_users = _users_for_csr(train)
    positive = train.ratings >= config.positive_rating_threshold
    positive_users = all_train_users[positive]
    positive_items = train.item_indices[positive]
    confidence = implicit_confidence(
        train.ratings[positive], config.positive_rating_threshold, config.alpha
    )
    user_counts = np.bincount(positive_users, minlength=users_count)
    user_offsets = np.zeros(users_count + 1, dtype=np.int64)
    np.cumsum(user_counts, out=user_offsets[1:])
    item_offsets, item_users, item_confidence = build_item_view(
        positive_users, positive_items, confidence, items_count
    )
    item_counts = np.bincount(train.item_indices, minlength=items_count)
    supported = item_counts > 0
    validation_users = eligible_user_indices(
        history,
        targets,
        supported,
        max(config.observed_ratings),
        config.relevance_threshold,
    )

    generator = np.random.default_rng(config.seed)
    item_factors = generator.normal(
        0.0, 0.01, size=(items_count, config.dimensions)
    ).astype(np.float32)
    best_selection = (-math.inf, -math.inf)
    best_item_factors: np.ndarray | None = None
    history_report = []
    for iteration in range(1, config.iterations + 1):
        user_factors = solve_implicit_factors(
            user_offsets,
            positive_items,
            confidence,
            item_factors,
            config.regularization,
        )
        item_factors = solve_implicit_factors(
            item_offsets,
            item_users,
            item_confidence,
            user_factors,
            config.regularization,
        )
        trials = []
        for regularization in config.fold_in_regularization_candidates:
            for popularity_blend in config.popularity_blend_candidates:
                by_observed = {
                    str(observed): evaluate_weighted_als(
                        validation_users,
                        history,
                        targets,
                        item_factors,
                        item_counts,
                        observed,
                        regularization,
                        popularity_blend,
                        config,
                    )
                    for observed in config.observed_ratings
                }
                mean_ndcg = float(
                    np.mean([metrics["ndcg_at_10"] for metrics in by_observed.values()])
                )
                mean_recall = float(
                    np.mean([metrics["recall_at_10"] for metrics in by_observed.values()])
                )
                trials.append(
                    {
                        "regularization": regularization,
                        "popularity_blend": popularity_blend,
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
            ),
        )
        entry = {"iteration": iteration, "trials": trials, "selected": selected}
        history_report.append(entry)
        print(json.dumps(entry), flush=True)
        selection = (selected["mean_ndcg_at_10"], selected["mean_recall_at_10"])
        if selection > best_selection:
            best_selection = selection
            best_item_factors = copy.deepcopy(item_factors)

    if best_item_factors is None:
        raise RuntimeError("Weighted ALS did not produce a checkpoint")
    best_entry = max(
        history_report,
        key=lambda entry: (
            entry["selected"]["mean_ndcg_at_10"],
            entry["selected"]["mean_recall_at_10"],
        ),
    )
    report = {
        "protocol": {
            "split": "strict held-out validation users",
            "global_fit_users": split_report["user_groups"]["train"],
            "validation_users": len(validation_users),
            "candidate_exclusion": "only exposed ratings",
            "test_access": "none",
        },
        "model": {
            "name": "implicit weighted alternating least squares",
            "dimensions": config.dimensions,
            "alpha": config.alpha,
            "training_regularization": config.regularization,
            "positive_rating_threshold": config.positive_rating_threshold,
            "score_blend": "per-user standardized ALS score + weight * standardized log count",
            "positive_interactions": len(positive_items),
            "best_iteration": best_entry["iteration"],
            "history": history_report,
        },
        "selected_validation": best_entry["selected"],
        "reproducibility": {"seed": config.seed},
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Weighted ALS under strict validation")
    parser.add_argument("--config", type=Path, default=Path("configs/weighted_als.toml"))
    args = parser.parse_args()
    config = WeightedAlsConfig.from_toml(args.config.resolve())
    print(json.dumps(train_and_validate(config), indent=2))


if __name__ == "__main__":
    main()
