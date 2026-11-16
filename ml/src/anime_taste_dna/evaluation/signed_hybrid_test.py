from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anime_taste_dna.evaluation.ranking import load_interactions_csr
from anime_taste_dna.evaluation.signed_hybrid_validation import (
    FixedArtifactParameters,
    SignedHybridValidationConfig,
    evaluate_candidate,
)
from anime_taste_dna.evaluation.strict_cold_start import (
    eligible_user_indices,
    evaluate_global_ranking_strict,
)
from anime_taste_dna.models.signed_profile import normalize_item_factors


@dataclass(frozen=True)
class SignedHybridTestConfig:
    validation: SignedHybridValidationConfig
    output_report: Path
    signed_regularization: float
    signed_weight: float

    @classmethod
    def from_toml(cls, path: Path) -> SignedHybridTestConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()
        validation = SignedHybridValidationConfig(
            split_directory=(root / raw["paths"]["split_directory"]).resolve(),
            artifact_path=(root / raw["paths"]["artifact_path"]).resolve(),
            output_report=(root / raw["paths"]["output_report"]).resolve(),
            signed_regularization_candidates=(
                float(raw["fixed_selection"]["signed_regularization"]),
            ),
            signed_weight_candidates=(float(raw["fixed_selection"]["signed_weight"]),),
            selection_users=0,
            observed_ratings=tuple(
                int(value) for value in raw["evaluation"]["observed_ratings"]
            ),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
            user_batch_size=int(raw["evaluation"]["user_batch_size"]),
            parquet_batch_size=int(raw["evaluation"]["parquet_batch_size"]),
            max_counterfactual_overlap=float(
                raw["behavior_gates"]["max_counterfactual_top_10_overlap"]
            ),
            max_popularity_overlap=float(
                raw["behavior_gates"]["max_popularity_top_10_overlap"]
            ),
            minimum_catalog_coverage_fraction=float(
                raw["behavior_gates"]["minimum_catalog_coverage_fraction"]
            ),
            seed=int(raw["evaluation"]["seed"]),
        )
        return cls(
            validation=validation,
            output_report=validation.output_report,
            signed_regularization=float(raw["fixed_selection"]["signed_regularization"]),
            signed_weight=float(raw["fixed_selection"]["signed_weight"]),
        )


def _mean_metrics(by_observed: dict[str, dict[str, float | int]]) -> tuple[float, float]:
    return (
        float(np.mean([metrics["recall_at_10"] for metrics in by_observed.values()])),
        float(np.mean([metrics["ndcg_at_10"] for metrics in by_observed.values()])),
    )


def run_final_test(config: SignedHybridTestConfig) -> dict[str, Any]:
    evaluation = config.validation
    split_report = json.loads(
        (evaluation.split_directory / "split_report.json").read_text(encoding="utf-8")
    )
    users_count = int(split_report["source"]["users"])
    with np.load(evaluation.artifact_path, allow_pickle=False) as artifact:
        raw_item_factors = artifact["item_factors"].astype(np.float32)
        item_counts = artifact["item_counts"].astype(np.int64)
        source_model_version = str(artifact["model_version"][0])
        artifact_parameters = FixedArtifactParameters(
            alpha=float(artifact["alpha"][0]),
            fold_in_regularization=float(artifact["fold_in_regularization"][0]),
            positive_rating_threshold=int(artifact["positive_rating_threshold"][0]),
            popularity_blend=float(artifact["popularity_blend"][0]),
        )
    supported = item_counts > 0
    normalized_item_factors = normalize_item_factors(raw_item_factors)
    popularity = np.log1p(item_counts).astype(np.float32)
    popularity = (popularity - popularity[supported].mean()) / popularity[supported].std()
    history = load_interactions_csr(
        evaluation.split_directory / "test_history.parquet",
        users_count,
        evaluation.parquet_batch_size,
    )
    targets = load_interactions_csr(
        evaluation.split_directory / "test_target.parquet",
        users_count,
        evaluation.parquet_batch_size,
    )
    test_users = eligible_user_indices(
        history,
        targets,
        supported,
        max(evaluation.observed_ratings),
        evaluation.relevance_threshold,
    )

    candidate_by_observed = {
        str(observed): evaluate_candidate(
            test_users,
            history,
            targets,
            raw_item_factors,
            normalized_item_factors,
            popularity,
            supported,
            observed,
            config.signed_regularization,
            config.signed_weight,
            artifact_parameters,
            evaluation,
            include_counterfactual=observed == 10,
        )
        for observed in evaluation.observed_ratings
    }
    current_by_observed = {
        str(observed): evaluate_candidate(
            test_users,
            history,
            targets,
            raw_item_factors,
            normalized_item_factors,
            popularity,
            supported,
            observed,
            config.signed_regularization,
            0.0,
            artifact_parameters,
            evaluation,
            include_counterfactual=observed == 10,
        )
        for observed in evaluation.observed_ratings
    }
    count_ranking = np.flatnonzero(supported)[
        np.argsort(-item_counts[supported], kind="stable")
    ]
    popularity_by_observed = {
        str(observed): evaluate_global_ranking_strict(
            count_ranking,
            history,
            targets,
            test_users,
            observed,
            evaluation.relevance_threshold,
            evaluation.ranking_k,
        )
        for observed in evaluation.observed_ratings
    }
    candidate_recall, candidate_ndcg = _mean_metrics(candidate_by_observed)
    current_recall, current_ndcg = _mean_metrics(current_by_observed)
    popularity_recall, popularity_ndcg = _mean_metrics(popularity_by_observed)
    ten_rating_metrics = candidate_by_observed["10"]
    passed_behavior = (
        float(ten_rating_metrics["high_to_low_counterfactual_top_10_overlap"])
        <= evaluation.max_counterfactual_overlap
        and float(ten_rating_metrics["popularity_top_10_overlap"])
        <= evaluation.max_popularity_overlap
        and float(ten_rating_metrics["catalog_coverage_fraction"])
        >= evaluation.minimum_catalog_coverage_fraction
    )
    passed_relevance_baseline = (
        candidate_recall > popularity_recall and candidate_ndcg > popularity_ndcg
    )
    report = {
        "protocol": {
            "split": "strict fixed test users",
            "parameter_search": "none; values frozen from validation",
            "source_item_model": source_model_version,
            "test_users": len(test_users),
        },
        "fixed_selection": {
            "signed_regularization": config.signed_regularization,
            "signed_weight": config.signed_weight,
        },
        "interaction_count_baseline": {
            "mean_recall_at_10": round(popularity_recall, 8),
            "mean_ndcg_at_10": round(popularity_ndcg, 8),
            "by_observed_ratings": popularity_by_observed,
        },
        "current_production_reference": {
            "mean_recall_at_10": round(current_recall, 8),
            "mean_ndcg_at_10": round(current_ndcg, 8),
            "by_observed_ratings": current_by_observed,
        },
        "signed_hybrid_candidate": {
            "mean_recall_at_10": round(candidate_recall, 8),
            "mean_ndcg_at_10": round(candidate_ndcg, 8),
            "by_observed_ratings": candidate_by_observed,
        },
        "passed_relevance_baseline": passed_relevance_baseline,
        "passed_behavior_gates": passed_behavior,
        "accepted_for_production": passed_relevance_baseline and passed_behavior,
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen signed-hybrid final test once")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/signed_hybrid_test.toml")
    )
    args = parser.parse_args()
    config = SignedHybridTestConfig.from_toml(args.config.resolve())
    print(json.dumps(run_final_test(config), indent=2))


if __name__ == "__main__":
    main()
