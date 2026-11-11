from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from anime_taste_dna.evaluation.ranking import (
    InteractionCsr,
    evaluate_global_ranking,
    load_interactions_csr,
)


@dataclass(frozen=True)
class PopularityConfig:
    processed_directory: Path
    split_directory: Path
    output_directory: Path
    shrinkage_candidates: tuple[float, ...]
    observed_ratings: tuple[int, ...]
    relevance_threshold: int
    ranking_k: int
    batch_size: int

    @classmethod
    def from_toml(cls, config_path: Path) -> PopularityConfig:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
        root = config_path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            processed_directory=resolve(raw["paths"]["processed_directory"]),
            split_directory=resolve(raw["paths"]["split_directory"]),
            output_directory=resolve(raw["paths"]["output_directory"]),
            shrinkage_candidates=tuple(
                float(value) for value in raw["model"]["shrinkage_candidates"]
            ),
            observed_ratings=tuple(
                int(value) for value in raw["evaluation"]["observed_ratings"]
            ),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
            batch_size=int(raw["evaluation"]["batch_size"]),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_config(config: PopularityConfig) -> dict[str, Any]:
    if not config.shrinkage_candidates or min(config.shrinkage_candidates) < 0:
        raise ValueError("Shrinkage candidates must be non-empty and non-negative")
    if not config.observed_ratings or min(config.observed_ratings) <= 0:
        raise ValueError("Observed rating counts must be positive")
    if not 1 <= config.relevance_threshold <= 10:
        raise ValueError("Relevance threshold must be between 1 and 10")
    if config.ranking_k != 10:
        raise ValueError("The MVP evaluation contract requires ranking_k = 10")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(f"Popularity output already exists: {config.output_directory}")

    report_path = config.split_directory / "split_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("Split report does not exist")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for split_name in ("train", "validation", "test"):
        split_path = config.split_directory / f"{split_name}.parquet"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        expected = report["output"]["files"][split_name]["sha256"]
        actual = _sha256(split_path)
        if actual != expected:
            raise ValueError(f"Fingerprint mismatch for {split_name}: {actual}")
    return report


def fit_popularity(
    train_path: Path, items: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray, float]:
    counts = np.zeros(items, dtype=np.int64)
    sums = np.zeros(items, dtype=np.float64)
    for batch in pq.ParquetFile(train_path).iter_batches(
        columns=["item_index", "rating"], batch_size=batch_size
    ):
        item_indices = batch.column(0).to_numpy()
        ratings = batch.column(1).to_numpy()
        counts += np.bincount(item_indices, minlength=items)
        sums += np.bincount(item_indices, weights=ratings, minlength=items)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("Training split is empty")
    return counts, sums, float(sums.sum() / total)


def popularity_ranking(
    counts: np.ndarray, sums: np.ndarray, global_mean: float, shrinkage: float
) -> tuple[np.ndarray, np.ndarray]:
    scores = (sums + shrinkage * global_mean) / (counts + shrinkage)
    supported = np.flatnonzero(counts > 0)
    order = np.lexsort((supported, -counts[supported], -scores[supported]))
    return supported[order].astype(np.int32), scores


def _evaluate_observation_levels(
    ranking: np.ndarray,
    train: InteractionCsr,
    targets: InteractionCsr,
    additional_exclusions: InteractionCsr | None,
    config: PopularityConfig,
) -> dict[str, dict[str, float | int]]:
    return {
        str(observed): evaluate_global_ranking(
            ranking=ranking,
            observed=train,
            targets=targets,
            additional_exclusions=additional_exclusions,
            observed_count=observed,
            relevance_threshold=config.relevance_threshold,
            ranking_k=config.ranking_k,
        )
        for observed in config.observed_ratings
    }


def train_and_evaluate(config: PopularityConfig) -> dict[str, Any]:
    split_report = _validate_config(config)
    users = int(split_report["source"]["users"])
    items = int(split_report["source"]["items"])
    train_path = config.split_directory / "train.parquet"
    validation_path = config.split_directory / "validation.parquet"
    test_path = config.split_directory / "test.parquet"

    counts, sums, global_mean = fit_popularity(train_path, items, config.batch_size)
    train = load_interactions_csr(train_path, users, config.batch_size)
    validation = load_interactions_csr(validation_path, users, config.batch_size)
    validation_trials = []
    for shrinkage in config.shrinkage_candidates:
        ranking, _ = popularity_ranking(counts, sums, global_mean, shrinkage)
        metrics = _evaluate_observation_levels(ranking, train, validation, None, config)
        mean_ndcg = float(np.mean([entry["ndcg_at_10"] for entry in metrics.values()]))
        mean_recall = float(np.mean([entry["recall_at_10"] for entry in metrics.values()]))
        validation_trials.append(
            {
                "shrinkage": shrinkage,
                "mean_recall_at_10": round(mean_recall, 8),
                "mean_ndcg_at_10": round(mean_ndcg, 8),
                "by_observed_ratings": metrics,
            }
        )

    best_trial = max(
        validation_trials,
        key=lambda trial: (
            trial["mean_ndcg_at_10"],
            trial["mean_recall_at_10"],
            -trial["shrinkage"],
        ),
    )
    selected_shrinkage = float(best_trial["shrinkage"])
    ranking, scores = popularity_ranking(counts, sums, global_mean, selected_shrinkage)
    test = load_interactions_csr(test_path, users, config.batch_size)
    test_metrics = _evaluate_observation_levels(ranking, train, test, validation, config)

    config.output_directory.mkdir(parents=True)
    items_frame = pd.read_parquet(config.processed_directory / "items.parquet")
    artifact = items_frame[["item_index", "mal_id", "title"]].copy()
    artifact["train_count"] = counts[artifact["item_index"]]
    artifact["train_mean"] = np.divide(
        sums[artifact["item_index"]],
        artifact["train_count"],
        out=np.zeros(len(artifact), dtype=np.float64),
        where=artifact["train_count"] > 0,
    )
    artifact["popularity_score"] = scores[artifact["item_index"]]
    artifact["rank"] = np.nan
    artifact.loc[ranking, "rank"] = np.arange(1, len(ranking) + 1)
    artifact["rank"] = artifact["rank"].astype("Int32")
    artifact_path = config.output_directory / "popularity.parquet"
    artifact.to_parquet(artifact_path, index=False, compression="zstd")

    report = {
        "model": {
            "name": "bayesian popularity baseline",
            "formula": (
                "(train_rating_sum + shrinkage * train_global_mean) / "
                "(train_count + shrinkage)"
            ),
            "fit_data": "train only",
            "global_train_mean": round(global_mean, 8),
            "supported_items": int((counts > 0).sum()),
            "selected_shrinkage": selected_shrinkage,
        },
        "evaluation": {
            "relevance_threshold": config.relevance_threshold,
            "ranking_k": config.ranking_k,
            "selection_metric": (
                "mean validation NDCG@10; Recall@10 then lower shrinkage break ties"
            ),
            "candidate_set": (
                "all train-supported items excluding the user history available before "
                "the target split"
            ),
            "validation_trials": validation_trials,
            "test_by_observed_ratings": test_metrics,
        },
        "source": {
            "split_report_sha256": _sha256(config.split_directory / "split_report.json"),
            "train_sha256": split_report["output"]["files"]["train"]["sha256"],
            "validation_sha256": split_report["output"]["files"]["validation"]["sha256"],
            "test_sha256": split_report["output"]["files"]["test"]["sha256"],
        },
        "artifact": {
            "path": "popularity.parquet",
            "sha256": _sha256(artifact_path),
        },
    }
    report_path = config.output_directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate the popularity baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/popularity.toml"))
    args = parser.parse_args()
    report = train_and_evaluate(PopularityConfig.from_toml(args.config.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
