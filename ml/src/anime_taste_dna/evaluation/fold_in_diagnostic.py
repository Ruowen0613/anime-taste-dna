from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anime_taste_dna.evaluation.personalization_audit import (
    AuditUser,
    evaluate_mode,
    mean_top_k_overlap,
)
from anime_taste_dna.evaluation.ranking import InteractionCsr, load_interactions_csr


@dataclass(frozen=True)
class FoldInDiagnosticConfig:
    split_directory: Path
    artifact_path: Path
    training_report_path: Path
    output_report: Path
    regularization_candidates: tuple[float, ...]
    observed_ratings: tuple[int, ...]
    relevance_threshold: int
    ranking_k: int
    user_batch_size: int
    parquet_batch_size: int
    pair_samples: int
    seed: int

    @classmethod
    def from_toml(cls, path: Path) -> FoldInDiagnosticConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            split_directory=resolve(raw["paths"]["split_directory"]),
            artifact_path=resolve(raw["paths"]["artifact_path"]),
            training_report_path=resolve(raw["paths"]["training_report_path"]),
            output_report=resolve(raw["paths"]["output_report"]),
            regularization_candidates=tuple(
                float(value) for value in raw["evaluation"]["regularization_candidates"]
            ),
            observed_ratings=tuple(
                int(value) for value in raw["evaluation"]["observed_ratings"]
            ),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
            user_batch_size=int(raw["evaluation"]["user_batch_size"]),
            parquet_batch_size=int(raw["evaluation"]["parquet_batch_size"]),
            pair_samples=int(raw["evaluation"]["pair_samples"]),
            seed=int(raw["evaluation"]["seed"]),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def prepare_validation_users(
    train: InteractionCsr,
    validation: InteractionCsr,
    supported: np.ndarray,
    maximum_observations: int,
    relevance_threshold: int,
) -> list[AuditUser]:
    users = []
    for user_index in range(train.users):
        history_items = train.user_items(user_index)
        if len(history_items) < maximum_observations:
            continue
        target_items = validation.user_items(user_index)
        target_ratings = validation.user_ratings(user_index)
        relevant = target_items[
            (target_ratings >= relevance_threshold) & supported[target_items]
        ]
        if len(relevant) == 0:
            continue
        users.append(
            AuditUser(
                user_index=user_index,
                history_items=history_items,
                history_ratings=train.user_ratings(user_index),
                relevant_items=np.unique(relevant),
            )
        )
    return users


def run_diagnostic(config: FoldInDiagnosticConfig) -> dict[str, Any]:
    split_report_path = config.split_directory / "split_report.json"
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    users_count = int(split_report["source"]["users"])
    training_report = json.loads(config.training_report_path.read_text(encoding="utf-8"))
    with np.load(config.artifact_path, allow_pickle=False) as artifact:
        item_factors = artifact["item_factors"].astype(np.float32)
        item_bias = artifact["item_bias"].astype(np.float32)
        global_mean = float(artifact["global_mean"][0])
        supported = artifact["supported"].astype(np.bool_)
        selected_regularization = float(artifact["fold_in_regularization"][0])
        model_version = str(artifact["model_version"][0])

    train = load_interactions_csr(
        config.split_directory / "train.parquet", users_count, config.parquet_batch_size
    )
    validation = load_interactions_csr(
        config.split_directory / "validation.parquet", users_count, config.parquet_batch_size
    )
    users = prepare_validation_users(
        train,
        validation,
        supported,
        max(config.observed_ratings),
        config.relevance_threshold,
    )

    bias_metrics, bias_recommendations = evaluate_mode(
        users,
        item_factors,
        item_bias,
        global_mean,
        supported,
        selected_regularization,
        10,
        "item_bias_only",
        config,
    )
    trials = []
    for regularization in config.regularization_candidates:
        by_observed: dict[str, dict[str, float | int]] = {}
        recommendations_at_10: np.ndarray | None = None
        for observed_count in config.observed_ratings:
            metrics, recommendations = evaluate_mode(
                users,
                item_factors,
                item_bias,
                global_mean,
                supported,
                regularization,
                observed_count,
                "full",
                config,
            )
            by_observed[str(observed_count)] = metrics
            if observed_count == 10:
                recommendations_at_10 = recommendations
        if recommendations_at_10 is None:
            raise ValueError("observed_ratings must include 10")
        metrics_at_10 = by_observed["10"]
        trial = {
            "regularization": regularization,
            "by_observed_ratings": by_observed,
            "full_vs_item_bias_top_10_overlap": round(
                mean_top_k_overlap(recommendations_at_10, bias_recommendations), 8
            ),
            "recall_at_10_delta_vs_item_bias": round(
                float(metrics_at_10["recall_at_10"]) - float(bias_metrics["recall_at_10"]), 8
            ),
            "ndcg_at_10_delta_vs_item_bias": round(
                float(metrics_at_10["ndcg_at_10"]) - float(bias_metrics["ndcg_at_10"]), 8
            ),
        }
        trials.append(trial)
        print(json.dumps(trial), flush=True)

    report = {
        "protocol": {
            "model_version": model_version,
            "split": "validation only",
            "common_cohort": (
                "validation users eligible for the maximum observation count "
                "and one relevant target"
            ),
            "candidate_set": "all train-supported items excluding train history",
            "test_access": "none",
        },
        "selected_artifact_regularization": selected_regularization,
        "item_bias_only_at_10_ratings": bias_metrics,
        "regularization_trials": trials,
        "training_curve": {
            "best_epoch": training_report["model"]["best_epoch"],
            "history": training_report["model"]["training_history"],
        },
        "source": {
            "artifact_sha256": _sha256(config.artifact_path),
            "split_report_sha256": _sha256(split_report_path),
            "training_report_sha256": _sha256(config.training_report_path),
        },
        "reproducibility": {"seed": config.seed, "eligible_users": len(users)},
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose fold-in personalization on validation")
    parser.add_argument("--config", type=Path, default=Path("configs/fold_in_diagnostic.toml"))
    args = parser.parse_args()
    report = run_diagnostic(FoldInDiagnosticConfig.from_toml(args.config.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
