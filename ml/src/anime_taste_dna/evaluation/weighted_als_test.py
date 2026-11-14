from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anime_taste_dna.evaluation.ranking import load_interactions_csr
from anime_taste_dna.evaluation.strict_cold_start import (
    eligible_user_indices,
    evaluate_global_ranking_strict,
)
from anime_taste_dna.models.matrix_factorization import _users_for_csr
from anime_taste_dna.models.weighted_als import (
    WeightedAlsConfig,
    build_item_view,
    evaluate_weighted_als,
    implicit_confidence,
    solve_implicit_factors,
)


@dataclass(frozen=True)
class WeightedAlsTestConfig:
    model: WeightedAlsConfig
    processed_directory: Path
    output_report: Path
    artifact_path: Path
    fixed_iterations: int
    fixed_fold_in_regularization: float
    fixed_popularity_blend: float

    @classmethod
    def from_toml(cls, path: Path) -> WeightedAlsTestConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        model = WeightedAlsConfig(
            split_directory=resolve(raw["paths"]["split_directory"]),
            output_report=resolve(raw["paths"]["output_report"]),
            dimensions=int(raw["training"]["dimensions"]),
            iterations=int(raw["fixed_selection"]["iterations"]),
            alpha=float(raw["training"]["alpha"]),
            regularization=float(raw["training"]["regularization"]),
            positive_rating_threshold=int(raw["training"]["positive_rating_threshold"]),
            seed=int(raw["training"]["seed"]),
            fold_in_regularization_candidates=(
                float(raw["fixed_selection"]["fold_in_regularization"]),
            ),
            popularity_blend_candidates=(
                float(raw["fixed_selection"]["popularity_blend"]),
            ),
            observed_ratings=tuple(
                int(value) for value in raw["evaluation"]["observed_ratings"]
            ),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
            user_batch_size=int(raw["evaluation"]["user_batch_size"]),
            parquet_batch_size=int(raw["evaluation"]["parquet_batch_size"]),
        )
        return cls(
            model=model,
            processed_directory=resolve(raw["paths"]["processed_directory"]),
            output_report=resolve(raw["paths"]["output_report"]),
            artifact_path=resolve(raw["paths"]["artifact_path"]),
            fixed_iterations=int(raw["fixed_selection"]["iterations"]),
            fixed_fold_in_regularization=float(
                raw["fixed_selection"]["fold_in_regularization"]
            ),
            fixed_popularity_blend=float(raw["fixed_selection"]["popularity_blend"]),
        )


def _mean_metrics(by_observed: dict[str, dict[str, float | int]]) -> tuple[float, float]:
    recall = float(np.mean([metrics["recall_at_10"] for metrics in by_observed.values()]))
    ndcg = float(np.mean([metrics["ndcg_at_10"] for metrics in by_observed.values()]))
    return recall, ndcg


def run_final_test(config: WeightedAlsTestConfig) -> dict[str, Any]:
    model_config = config.model
    split_report = json.loads(
        (model_config.split_directory / "split_report.json").read_text(encoding="utf-8")
    )
    users_count = int(split_report["source"]["users"])
    items_count = int(split_report["source"]["items"])
    train = load_interactions_csr(
        model_config.split_directory / "train.parquet",
        users_count,
        model_config.parquet_batch_size,
    )
    history = load_interactions_csr(
        model_config.split_directory / "test_history.parquet",
        users_count,
        model_config.parquet_batch_size,
    )
    targets = load_interactions_csr(
        model_config.split_directory / "test_target.parquet",
        users_count,
        model_config.parquet_batch_size,
    )
    all_train_users = _users_for_csr(train)
    positive = train.ratings >= model_config.positive_rating_threshold
    positive_users = all_train_users[positive]
    positive_items = train.item_indices[positive]
    confidence = implicit_confidence(
        train.ratings[positive],
        model_config.positive_rating_threshold,
        model_config.alpha,
    )
    user_counts = np.bincount(positive_users, minlength=users_count)
    user_offsets = np.zeros(users_count + 1, dtype=np.int64)
    np.cumsum(user_counts, out=user_offsets[1:])
    item_offsets, item_users, item_confidence = build_item_view(
        positive_users, positive_items, confidence, items_count
    )
    item_counts = np.bincount(train.item_indices, minlength=items_count)
    supported = item_counts > 0
    test_users = eligible_user_indices(
        history,
        targets,
        supported,
        max(model_config.observed_ratings),
        model_config.relevance_threshold,
    )

    generator = np.random.default_rng(model_config.seed)
    item_factors = generator.normal(
        0.0, 0.01, size=(items_count, model_config.dimensions)
    ).astype(np.float32)
    for iteration in range(1, config.fixed_iterations + 1):
        user_factors = solve_implicit_factors(
            user_offsets,
            positive_items,
            confidence,
            item_factors,
            model_config.regularization,
        )
        item_factors = solve_implicit_factors(
            item_offsets,
            item_users,
            item_confidence,
            user_factors,
            model_config.regularization,
        )
        print(json.dumps({"completed_iteration": iteration}), flush=True)

    hybrid_by_observed = {
        str(observed): evaluate_weighted_als(
            test_users,
            history,
            targets,
            item_factors,
            item_counts,
            observed,
            config.fixed_fold_in_regularization,
            config.fixed_popularity_blend,
            model_config,
        )
        for observed in model_config.observed_ratings
    }
    count_ranking = np.flatnonzero(supported)[
        np.argsort(-item_counts[supported], kind="stable")
    ]
    baseline_by_observed = {
        str(observed): evaluate_global_ranking_strict(
            count_ranking,
            history,
            targets,
            test_users,
            observed,
            model_config.relevance_threshold,
            model_config.ranking_k,
        )
        for observed in model_config.observed_ratings
    }
    hybrid_recall, hybrid_ndcg = _mean_metrics(hybrid_by_observed)
    baseline_recall, baseline_ndcg = _mean_metrics(baseline_by_observed)
    passed = hybrid_recall > baseline_recall and hybrid_ndcg > baseline_ndcg

    artifact: dict[str, Any] | None = None
    if passed:
        items = pd.read_parquet(config.processed_directory / "items.parquet")
        config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.artifact_path,
            item_factors=item_factors,
            item_counts=item_counts.astype(np.int64),
            item_index=items["item_index"].to_numpy(dtype=np.int32),
            mal_id=items["mal_id"].to_numpy(dtype=np.int64),
            title=items["title"].fillna("").to_numpy(dtype=np.str_),
            supported=supported,
            alpha=np.asarray([model_config.alpha], dtype=np.float32),
            training_regularization=np.asarray(
                [model_config.regularization], dtype=np.float32
            ),
            fold_in_regularization=np.asarray(
                [config.fixed_fold_in_regularization], dtype=np.float32
            ),
            popularity_blend=np.asarray([config.fixed_popularity_blend], dtype=np.float32),
            positive_rating_threshold=np.asarray(
                [model_config.positive_rating_threshold], dtype=np.int16
            ),
            model_version=np.asarray(["weighted-als-hybrid-v1"], dtype=np.str_),
        )
        artifact = {"path": str(config.artifact_path), "exported": True}

    report = {
        "protocol": {
            "split": "strict untouched test users",
            "test_users": len(test_users),
            "parameter_search": "none; all values frozen from validation",
            "candidate_exclusion": "only exposed ratings",
        },
        "fixed_selection": {
            "iterations": config.fixed_iterations,
            "fold_in_regularization": config.fixed_fold_in_regularization,
            "popularity_blend": config.fixed_popularity_blend,
        },
        "interaction_count_baseline": {
            "mean_recall_at_10": round(baseline_recall, 8),
            "mean_ndcg_at_10": round(baseline_ndcg, 8),
            "by_observed_ratings": baseline_by_observed,
        },
        "weighted_als_hybrid": {
            "mean_recall_at_10": round(hybrid_recall, 8),
            "mean_ndcg_at_10": round(hybrid_ndcg, 8),
            "by_observed_ratings": hybrid_by_observed,
        },
        "passed_both_metrics": passed,
        "artifact": artifact,
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Weighted ALS final test once")
    parser.add_argument("--config", type=Path, default=Path("configs/weighted_als_test.toml"))
    args = parser.parse_args()
    config = WeightedAlsTestConfig.from_toml(args.config.resolve())
    print(json.dumps(run_final_test(config), indent=2))


if __name__ == "__main__":
    main()
