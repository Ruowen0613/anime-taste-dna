from __future__ import annotations

import argparse
import hashlib
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
from anime_taste_dna.evaluation.strict_cold_start import eligible_user_indices
from anime_taste_dna.models.signed_profile import normalize_item_factors


@dataclass(frozen=True)
class RatingScaleValidationConfig:
    evaluation: SignedHybridValidationConfig
    output_report: Path
    signed_regularization: float
    signed_weight: float
    popularity_decay_exponents: tuple[float, ...]
    popularity_decay_pivot: int
    maximum_mean_ndcg_loss_fraction: float
    maximum_mean_recall_loss_fraction: float

    @classmethod
    def from_toml(cls, path: Path) -> RatingScaleValidationConfig:
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
            signed_weight=float(raw["fixed_model"]["signed_weight"]),
            popularity_decay_exponents=tuple(
                float(value) for value in raw["selection"]["popularity_decay_exponents"]
            ),
            popularity_decay_pivot=int(raw["selection"]["popularity_decay_pivot"]),
            maximum_mean_ndcg_loss_fraction=float(
                raw["selection"]["maximum_mean_ndcg_loss_fraction"]
            ),
            maximum_mean_recall_loss_fraction=float(
                raw["selection"]["maximum_mean_recall_loss_fraction"]
            ),
        )


def popularity_blend_for_count(
    base_blend: float, observed_count: int, pivot: int, decay_exponent: float
) -> float:
    if observed_count <= 0 or pivot <= 0:
        raise ValueError("observed_count and pivot must be positive")
    return base_blend * min(1.0, (pivot / observed_count) ** decay_exponent)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_metric(by_observed: dict[str, dict[str, float | int]], name: str) -> float:
    return float(np.mean([float(metrics[name]) for metrics in by_observed.values()]))


def run_validation(config: RatingScaleValidationConfig) -> dict[str, Any]:
    evaluation = config.evaluation
    split_report_path = evaluation.split_directory / "split_report.json"
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
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

    trials: list[dict[str, Any]] = []
    for exponent in config.popularity_decay_exponents:
        by_observed: dict[str, dict[str, float | int]] = {}
        for observed_count in evaluation.observed_ratings:
            blend = popularity_blend_for_count(
                artifact_parameters.popularity_blend,
                observed_count,
                config.popularity_decay_pivot,
                exponent,
            )
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
                json.dumps(
                    {
                        "popularity_decay_exponent": exponent,
                        "observed_ratings": observed_count,
                        **by_observed[str(observed_count)],
                    }
                ),
                flush=True,
            )
        trials.append(
            {
                "popularity_decay_exponent": exponent,
                "mean_recall_at_10": round(_mean_metric(by_observed, "recall_at_10"), 8),
                "mean_ndcg_at_10": round(_mean_metric(by_observed, "ndcg_at_10"), 8),
                "mean_popularity_top_10_overlap": round(
                    _mean_metric(by_observed, "popularity_top_10_overlap"), 8
                ),
                "by_observed_ratings": by_observed,
            }
        )

    fixed = next(
        trial for trial in trials if float(trial["popularity_decay_exponent"]) == 0.0
    )
    minimum_ndcg = float(fixed["mean_ndcg_at_10"]) * (
        1.0 - config.maximum_mean_ndcg_loss_fraction
    )
    minimum_recall = float(fixed["mean_recall_at_10"]) * (
        1.0 - config.maximum_mean_recall_loss_fraction
    )
    eligible = [
        trial
        for trial in trials
        if float(trial["mean_ndcg_at_10"]) >= minimum_ndcg
        and float(trial["mean_recall_at_10"]) >= minimum_recall
    ]
    selected = min(
        eligible,
        key=lambda trial: (
            float(trial["mean_popularity_top_10_overlap"]),
            -float(trial["mean_ndcg_at_10"]),
        ),
    )
    report = {
        "protocol": {
            "split": "strict held-out validation users",
            "test_access": "none",
            "common_users_eligible_for_max_observations": len(validation_users),
            "source_item_model": source_model_version,
            "rating_counts": list(evaluation.observed_ratings),
            "ranking_formula": (
                "standardized ALS plus decayed popularity, then standardized, plus fixed signed "
                "explicit-rating residual"
            ),
        },
        "fixed_model": {
            "base_popularity_blend": artifact_parameters.popularity_blend,
            "signed_regularization": config.signed_regularization,
            "signed_weight": config.signed_weight,
        },
        "selection": {
            "popularity_decay_pivot": config.popularity_decay_pivot,
            "maximum_mean_ndcg_loss_fraction": config.maximum_mean_ndcg_loss_fraction,
            "maximum_mean_recall_loss_fraction": config.maximum_mean_recall_loss_fraction,
            "selected_popularity_decay_exponent": selected["popularity_decay_exponent"],
        },
        "trials": trials,
        "source": {
            "artifact_sha256": _sha256(evaluation.artifact_path),
            "split_report_sha256": _sha256(split_report_path),
        },
        "reproducibility": {"seed": evaluation.seed},
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate rating-count-aware popularity decay on strict validation users"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rating_scale_validation.toml")
    )
    args = parser.parse_args()
    print(json.dumps(run_validation(RatingScaleValidationConfig.from_toml(args.config)), indent=2))


if __name__ == "__main__":
    main()
