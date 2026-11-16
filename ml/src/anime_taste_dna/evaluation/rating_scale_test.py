from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anime_taste_dna.evaluation.ranking import load_interactions_csr
from anime_taste_dna.evaluation.rating_scale_validation import popularity_blend_for_count
from anime_taste_dna.evaluation.signed_hybrid_validation import (
    FixedArtifactParameters,
    SignedHybridValidationConfig,
    evaluate_candidate,
)
from anime_taste_dna.evaluation.strict_cold_start import eligible_user_indices
from anime_taste_dna.models.signed_profile import normalize_item_factors


@dataclass(frozen=True)
class RatingScaleTestConfig:
    evaluation: SignedHybridValidationConfig
    validation_report: Path
    output_report: Path
    signed_regularization: float
    signed_weight: float
    popularity_decay_pivot: int

    @classmethod
    def from_toml(cls, path: Path) -> RatingScaleTestConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()
        evaluation = SignedHybridValidationConfig(
            split_directory=(root / raw["paths"]["split_directory"]).resolve(),
            artifact_path=(root / raw["paths"]["artifact_path"]).resolve(),
            output_report=(root / raw["paths"]["output_report"]).resolve(),
            signed_regularization_candidates=(),
            signed_weight_candidates=(),
            selection_users=0,
            observed_ratings=tuple(
                int(value) for value in raw["evaluation"]["observed_ratings"]
            ),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
            user_batch_size=int(raw["evaluation"]["user_batch_size"]),
            parquet_batch_size=int(raw["evaluation"]["parquet_batch_size"]),
            max_counterfactual_overlap=1.0,
            max_popularity_overlap=1.0,
            minimum_catalog_coverage_fraction=0.0,
            seed=int(raw["evaluation"]["seed"]),
        )
        return cls(
            evaluation=evaluation,
            validation_report=(root / raw["paths"]["validation_report"]).resolve(),
            output_report=evaluation.output_report,
            signed_regularization=float(raw["fixed_model"]["signed_regularization"]),
            signed_weight=float(raw["fixed_model"]["signed_weight"]),
            popularity_decay_pivot=int(raw["fixed_model"]["popularity_decay_pivot"]),
        )


def _mean_metric(by_observed: dict[str, dict[str, float | int]], name: str) -> float:
    return float(np.mean([float(metrics[name]) for metrics in by_observed.values()]))


def run_test(config: RatingScaleTestConfig) -> dict[str, Any]:
    validation_report = json.loads(config.validation_report.read_text(encoding="utf-8"))
    selected_exponent = float(
        validation_report["selection"]["selected_popularity_decay_exponent"]
    )
    evaluation = config.evaluation
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

    results: dict[str, Any] = {}
    for name, exponent in (("fixed_popularity", 0.0), ("decayed_popularity", selected_exponent)):
        by_observed: dict[str, dict[str, float | int]] = {}
        for observed_count in evaluation.observed_ratings:
            blend = popularity_blend_for_count(
                artifact_parameters.popularity_blend,
                observed_count,
                config.popularity_decay_pivot,
                exponent,
            )
            metrics = evaluate_candidate(
                test_users,
                history,
                targets,
                raw_item_factors,
                normalized_item_factors,
                popularity,
                supported,
                observed_count,
                config.signed_regularization,
                config.signed_weight,
                artifact_parameters,
                evaluation,
                include_counterfactual=observed_count == config.popularity_decay_pivot,
                popularity_blend_override=blend,
            )
            by_observed[str(observed_count)] = {
                "effective_popularity_blend": round(blend, 8),
                **metrics,
            }
            print(
                json.dumps({"model": name, "observed_ratings": observed_count, **metrics}),
                flush=True,
            )
        results[name] = {
            "popularity_decay_exponent": exponent,
            "mean_recall_at_10": round(_mean_metric(by_observed, "recall_at_10"), 8),
            "mean_ndcg_at_10": round(_mean_metric(by_observed, "ndcg_at_10"), 8),
            "mean_popularity_top_10_overlap": round(
                _mean_metric(by_observed, "popularity_top_10_overlap"), 8
            ),
            "by_observed_ratings": by_observed,
        }

    fixed = results["fixed_popularity"]
    candidate = results["decayed_popularity"]
    accepted = (
        float(candidate["mean_recall_at_10"]) >= float(fixed["mean_recall_at_10"])
        and float(candidate["mean_ndcg_at_10"]) >= float(fixed["mean_ndcg_at_10"])
        and float(candidate["mean_popularity_top_10_overlap"])
        < float(fixed["mean_popularity_top_10_overlap"])
    )
    report = {
        "protocol": {
            "split": "strict fixed test users",
            "parameter_search": "none; decay exponent frozen from validation report",
            "test_users": len(test_users),
            "source_item_model": source_model_version,
        },
        "frozen_selection": {
            "popularity_decay_exponent": selected_exponent,
            "popularity_decay_pivot": config.popularity_decay_pivot,
        },
        "results": results,
        "accepted_for_production": accepted,
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen rating-scale popularity decay test")
    parser.add_argument("--config", type=Path, default=Path("configs/rating_scale_test.toml"))
    args = parser.parse_args()
    print(json.dumps(run_test(RatingScaleTestConfig.from_toml(args.config)), indent=2))


if __name__ == "__main__":
    main()
