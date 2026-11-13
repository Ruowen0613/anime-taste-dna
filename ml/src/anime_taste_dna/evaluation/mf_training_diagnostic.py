from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from anime_taste_dna.evaluation.fold_in_diagnostic import prepare_validation_users
from anime_taste_dna.evaluation.personalization_audit import evaluate_mode, mean_top_k_overlap
from anime_taste_dna.evaluation.ranking import load_interactions_csr
from anime_taste_dna.models.matrix_factorization import (
    MatrixFactorizationConfig,
    _select_device,
    train_model,
)


@dataclass(frozen=True)
class TrainingVariant:
    name: str
    learning_rate: float
    weight_decay: float
    epochs: int
    early_stopping_patience: int


@dataclass(frozen=True)
class MfTrainingDiagnosticConfig:
    base_config_path: Path
    output_report: Path
    variants: tuple[TrainingVariant, ...]
    regularization_candidates: tuple[float, ...]
    observed_ratings: tuple[int, ...]
    relevance_threshold: int
    ranking_k: int
    user_batch_size: int
    parquet_batch_size: int
    pair_samples: int
    seed: int

    @classmethod
    def from_toml(cls, path: Path) -> MfTrainingDiagnosticConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            base_config_path=resolve(raw["paths"]["base_config_path"]),
            output_report=resolve(raw["paths"]["output_report"]),
            variants=tuple(
                TrainingVariant(
                    name=entry["name"],
                    learning_rate=float(entry["learning_rate"]),
                    weight_decay=float(entry["weight_decay"]),
                    epochs=int(entry["epochs"]),
                    early_stopping_patience=int(entry["early_stopping_patience"]),
                )
                for entry in raw["variants"]
            ),
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


def select_trial(trials: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        trials,
        key=lambda trial: (
            trial["full_metrics"]["ndcg_at_10"],
            trial["full_metrics"]["recall_at_10"],
            -trial["regularization"],
        ),
    )


def run_diagnostic(config: MfTrainingDiagnosticConfig) -> dict[str, Any]:
    base = MatrixFactorizationConfig.from_toml(config.base_config_path)
    split_report = json.loads(
        (base.split_directory / "split_report.json").read_text(encoding="utf-8")
    )
    users_count = int(split_report["source"]["users"])
    items_count = int(split_report["source"]["items"])
    train = load_interactions_csr(
        base.split_directory / "train.parquet", users_count, config.parquet_batch_size
    )
    validation = load_interactions_csr(
        base.split_directory / "validation.parquet", users_count, config.parquet_batch_size
    )
    supported = np.bincount(train.item_indices, minlength=items_count) > 0
    users = prepare_validation_users(
        train,
        validation,
        supported,
        max(config.observed_ratings),
        config.relevance_threshold,
    )
    device = _select_device(base.device)
    variant_results = []

    for variant in config.variants:
        print(json.dumps({"training_variant": variant.name}), flush=True)
        training_config = replace(
            base,
            learning_rate=variant.learning_rate,
            weight_decay=variant.weight_decay,
            epochs=variant.epochs,
            early_stopping_patience=variant.early_stopping_patience,
        )
        model, history, best_epoch = train_model(
            train, validation, items_count, training_config, device
        )
        item_factors = model.item_factors.weight.detach().cpu().numpy()
        item_bias = model.item_bias.weight.detach().cpu().numpy().squeeze(1)
        global_mean = float(model.global_mean.cpu())
        bias_metrics, bias_recommendations = evaluate_mode(
            users,
            item_factors,
            item_bias,
            global_mean,
            supported,
            1.0,
            10,
            "item_bias_only",
            config,
        )
        fold_in_trials = []
        for regularization in config.regularization_candidates:
            full_metrics, full_recommendations = evaluate_mode(
                users,
                item_factors,
                item_bias,
                global_mean,
                supported,
                regularization,
                10,
                "full",
                config,
            )
            fold_in_trials.append(
                {
                    "regularization": regularization,
                    "full_metrics": full_metrics,
                    "full_vs_item_bias_top_10_overlap": round(
                        mean_top_k_overlap(full_recommendations, bias_recommendations), 8
                    ),
                    "recall_delta_vs_item_bias": round(
                        float(full_metrics["recall_at_10"])
                        - float(bias_metrics["recall_at_10"]),
                        8,
                    ),
                    "ndcg_delta_vs_item_bias": round(
                        float(full_metrics["ndcg_at_10"])
                        - float(bias_metrics["ndcg_at_10"]),
                        8,
                    ),
                }
            )
        selected = select_trial(fold_in_trials)
        result = {
            "name": variant.name,
            "training": {
                "learning_rate": variant.learning_rate,
                "weight_decay": variant.weight_decay,
                "best_epoch_by_rmse": best_epoch,
                "history": history,
            },
            "item_bias_only": bias_metrics,
            "fold_in_trials": fold_in_trials,
            "selected_by_validation_ndcg": selected,
        }
        variant_results.append(result)
        print(
            json.dumps(
                {
                    "completed_variant": variant.name,
                    "best_epoch": best_epoch,
                    "selected": selected,
                }
            ),
            flush=True,
        )

    report = {
        "protocol": {
            "split": "validation only",
            "test_access": "none",
            "selection": "highest NDCG@10, then Recall@10, then lower regularization",
            "common_cohort": (
                "validation users eligible for 10 observations and one relevant target"
            ),
        },
        "variants": variant_results,
        "reproducibility": {
            "seed": config.seed,
            "device": str(device),
            "eligible_users": len(users),
        },
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare explicit-MF training variants")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mf_training_diagnostic.toml")
    )
    args = parser.parse_args()
    report = run_diagnostic(MfTrainingDiagnosticConfig.from_toml(args.config.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
