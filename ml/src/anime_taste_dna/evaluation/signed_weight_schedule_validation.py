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
class SignedWeightScheduleConfig:
    evaluation: SignedHybridValidationConfig
    output_report: Path
    signed_regularization: float
    baseline_signed_weight: float
    signed_weight_candidates: tuple[float, ...]
    popularity_decay_exponent: float
    popularity_decay_pivot: int
    maximum_recall_loss_fraction: float
    maximum_popularity_overlap_increase: float
    minimum_coverage_retention_fraction: float

    @classmethod
    def from_toml(cls, path: Path) -> SignedWeightScheduleConfig:
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
            output_report=evaluation.output_report,
            signed_regularization=float(raw["fixed_model"]["signed_regularization"]),
            baseline_signed_weight=float(raw["fixed_model"]["signed_weight"]),
            signed_weight_candidates=tuple(
                float(value) for value in raw["selection"]["signed_weight_candidates"]
            ),
            popularity_decay_exponent=float(
                raw["fixed_model"]["popularity_decay_exponent"]
            ),
            popularity_decay_pivot=int(raw["fixed_model"]["popularity_decay_pivot"]),
            maximum_recall_loss_fraction=float(
                raw["selection"]["maximum_recall_loss_fraction"]
            ),
            maximum_popularity_overlap_increase=float(
                raw["selection"]["maximum_popularity_overlap_increase"]
            ),
            minimum_coverage_retention_fraction=float(
                raw["selection"]["minimum_coverage_retention_fraction"]
            ),
        )


def run_validation(config: SignedWeightScheduleConfig) -> dict[str, Any]:
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
        evaluation.split_directory / "validation_history.parquet",
        users_count,
        evaluation.parquet_batch_size,
    )
    targets = load_interactions_csr(
        evaluation.split_directory / "validation_target.parquet",
        users_count,
        evaluation.parquet_batch_size,
    )
    validation_users = eligible_user_indices(
        history,
        targets,
        supported,
        max(evaluation.observed_ratings),
        evaluation.relevance_threshold,
    )

    by_observed: dict[str, dict[str, Any]] = {}
    selected_weights: dict[str, float] = {}
    for observed_count in evaluation.observed_ratings:
        popularity_blend = popularity_blend_for_count(
            artifact_parameters.popularity_blend,
            observed_count,
            config.popularity_decay_pivot,
            config.popularity_decay_exponent,
        )
        trials = []
        for signed_weight in config.signed_weight_candidates:
            metrics = evaluate_candidate(
                validation_users,
                history,
                targets,
                raw_item_factors,
                normalized_item_factors,
                popularity,
                supported,
                observed_count,
                config.signed_regularization,
                signed_weight,
                artifact_parameters,
                evaluation,
                include_counterfactual=False,
                popularity_blend_override=popularity_blend,
            )
            trial = {"signed_weight": signed_weight, **metrics}
            trials.append(trial)
            print(json.dumps({"observed_ratings": observed_count, **trial}), flush=True)

        baseline = next(
            trial
            for trial in trials
            if float(trial["signed_weight"]) == config.baseline_signed_weight
        )
        eligible = [
            trial
            for trial in trials
            if float(trial["recall_at_10"])
            >= float(baseline["recall_at_10"])
            * (1.0 - config.maximum_recall_loss_fraction)
            and float(trial["popularity_top_10_overlap"])
            <= float(baseline["popularity_top_10_overlap"])
            + config.maximum_popularity_overlap_increase
            and int(trial["catalog_coverage_items"])
            >= int(baseline["catalog_coverage_items"])
            * config.minimum_coverage_retention_fraction
        ]
        selected = max(
            eligible,
            key=lambda trial: (
                float(trial["ndcg_at_10"]),
                float(trial["recall_at_10"]),
                -float(trial["popularity_top_10_overlap"]),
            ),
        )
        selected_weights[str(observed_count)] = float(selected["signed_weight"])
        by_observed[str(observed_count)] = {
            "effective_popularity_blend": round(popularity_blend, 8),
            "baseline": baseline,
            "selected": selected,
            "trials": trials,
        }

    report = {
        "experiment": "rating-count-aware-signed-weight-validation-v1",
        "protocol": {
            "split": "strict held-out validation users",
            "test_access": "none",
            "common_users": len(validation_users),
            "source_item_model": source_model_version,
            "selection_metric": "NDCG@10 after recall, popularity-overlap, and coverage gates",
        },
        "fixed_model": {
            "signed_regularization": config.signed_regularization,
            "baseline_signed_weight": config.baseline_signed_weight,
            "popularity_decay_exponent": config.popularity_decay_exponent,
            "popularity_decay_pivot": config.popularity_decay_pivot,
        },
        "selected_weights": selected_weights,
        "by_observed_ratings": by_observed,
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select signed hybrid weights by observed-rating count on validation users"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/signed_weight_schedule_validation.toml"),
    )
    args = parser.parse_args()
    config = SignedWeightScheduleConfig.from_toml(args.config.resolve())
    print(json.dumps(run_validation(config), indent=2))


if __name__ == "__main__":
    main()
