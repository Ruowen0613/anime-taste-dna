from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anime_taste_dna.evaluation.ranking import load_interactions_csr
from anime_taste_dna.evaluation.strict_cold_start import eligible_user_indices
from anime_taste_dna.models.weighted_als import WeightedAlsConfig, evaluate_weighted_als


@dataclass(frozen=True)
class NegativeFeedbackConfig:
    split_directory: Path
    artifact_path: Path
    output_report: Path
    negative_thresholds: tuple[int, ...]
    negative_alphas: tuple[float, ...]
    model: WeightedAlsConfig

    @classmethod
    def from_toml(cls, path: Path) -> NegativeFeedbackConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        split_directory = resolve(raw["paths"]["split_directory"])
        output_report = resolve(raw["paths"]["output_report"])
        model = WeightedAlsConfig(
            split_directory=split_directory,
            output_report=output_report,
            dimensions=int(raw["model"]["dimensions"]),
            iterations=int(raw["model"]["iterations"]),
            alpha=float(raw["model"]["positive_alpha"]),
            regularization=float(raw["model"]["training_regularization"]),
            positive_rating_threshold=int(raw["model"]["positive_rating_threshold"]),
            seed=int(raw["model"]["seed"]),
            fold_in_regularization_candidates=(
                float(raw["model"]["fold_in_regularization"]),
            ),
            popularity_blend_candidates=(float(raw["model"]["popularity_blend"]),),
            observed_ratings=tuple(
                int(value) for value in raw["evaluation"]["observed_ratings"]
            ),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
            user_batch_size=int(raw["evaluation"]["user_batch_size"]),
            parquet_batch_size=int(raw["evaluation"]["parquet_batch_size"]),
        )
        return cls(
            split_directory=split_directory,
            artifact_path=resolve(raw["paths"]["artifact_path"]),
            output_report=output_report,
            negative_thresholds=tuple(
                int(value) for value in raw["search"]["negative_thresholds"]
            ),
            negative_alphas=tuple(
                float(value) for value in raw["search"]["negative_alphas"]
            ),
            model=model,
        )


def _summarize(by_observed: dict[str, dict[str, float | int]]) -> tuple[float, float]:
    recall = float(np.mean([value["recall_at_10"] for value in by_observed.values()]))
    ndcg = float(np.mean([value["ndcg_at_10"] for value in by_observed.values()]))
    return round(recall, 8), round(ndcg, 8)


def run_evaluation(config: NegativeFeedbackConfig) -> dict[str, Any]:
    split_report = json.loads(
        (config.split_directory / "split_report.json").read_text(encoding="utf-8")
    )
    users_count = int(split_report["source"]["users"])
    history = load_interactions_csr(
        config.split_directory / "validation_history.parquet",
        users_count,
        config.model.parquet_batch_size,
    )
    targets = load_interactions_csr(
        config.split_directory / "validation_target.parquet",
        users_count,
        config.model.parquet_batch_size,
    )
    with np.load(config.artifact_path, allow_pickle=False) as artifact:
        item_factors = artifact["item_factors"].astype(np.float32)
        item_counts = artifact["item_counts"].astype(np.int64)
        model_version = str(artifact["model_version"][0])
    supported = item_counts > 0
    users = eligible_user_indices(
        history,
        targets,
        supported,
        max(config.model.observed_ratings),
        config.model.relevance_threshold,
    )

    def evaluate(threshold: int | None, alpha: float) -> dict[str, Any]:
        by_observed = {
            str(observed): evaluate_weighted_als(
                users,
                history,
                targets,
                item_factors,
                item_counts,
                observed,
                config.model.fold_in_regularization_candidates[0],
                config.model.popularity_blend_candidates[0],
                config.model,
                negative_rating_threshold=threshold,
                negative_alpha=alpha,
            )
            for observed in config.model.observed_ratings
        }
        recall, ndcg = _summarize(by_observed)
        return {
            "negative_rating_threshold": threshold,
            "negative_alpha": alpha,
            "mean_recall_at_10": recall,
            "mean_ndcg_at_10": ndcg,
            "by_observed_ratings": by_observed,
        }

    baseline = evaluate(None, 0.0)
    trials = [
        evaluate(threshold, alpha)
        for threshold in config.negative_thresholds
        for alpha in config.negative_alphas
    ]
    selected = max(
        trials,
        key=lambda trial: (trial["mean_ndcg_at_10"], trial["mean_recall_at_10"]),
    )
    report = {
        "protocol": {
            "split": "strict held-out validation users",
            "test_access": "none",
            "model_version": model_version,
            "fixed_global_model": True,
            "negative_preference": "confidence-weighted implicit target p=0",
        },
        "positive_only_baseline": baseline,
        "negative_feedback_trials": trials,
        "selected_validation": selected,
        "improvement": {
            "recall_at_10": round(
                selected["mean_recall_at_10"] - baseline["mean_recall_at_10"], 8
            ),
            "ndcg_at_10": round(
                selected["mean_ndcg_at_10"] - baseline["mean_ndcg_at_10"], 8
            ),
        },
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate explicit negative rating fold-in")
    parser.add_argument("--config", type=Path, default=Path("configs/negative_feedback.toml"))
    args = parser.parse_args()
    config = NegativeFeedbackConfig.from_toml(args.config.resolve())
    print(json.dumps(run_evaluation(config), indent=2))


if __name__ == "__main__":
    main()
