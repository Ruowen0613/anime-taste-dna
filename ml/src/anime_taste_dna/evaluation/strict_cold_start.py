from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anime_taste_dna.evaluation.ranking import InteractionCsr, load_interactions_csr
from anime_taste_dna.models.popularity import fit_popularity, popularity_ranking


@dataclass(frozen=True)
class StrictColdStartConfig:
    split_directory: Path
    output_report: Path
    shrinkage_candidates: tuple[float, ...]
    observed_ratings: tuple[int, ...]
    relevance_threshold: int
    ranking_k: int
    batch_size: int

    @classmethod
    def from_toml(cls, path: Path) -> StrictColdStartConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            split_directory=resolve(raw["paths"]["split_directory"]),
            output_report=resolve(raw["paths"]["output_report"]),
            shrinkage_candidates=tuple(
                float(value) for value in raw["popularity"]["shrinkage_candidates"]
            ),
            observed_ratings=tuple(
                int(value) for value in raw["evaluation"]["observed_ratings"]
            ),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
            batch_size=int(raw["evaluation"]["batch_size"]),
        )


def eligible_user_indices(
    history: InteractionCsr,
    targets: InteractionCsr,
    supported: np.ndarray,
    maximum_observations: int,
    relevance_threshold: int,
) -> np.ndarray:
    eligible = []
    for user_index in range(history.users):
        if len(history.user_items(user_index)) < maximum_observations:
            continue
        target_items = targets.user_items(user_index)
        target_ratings = targets.user_ratings(user_index)
        relevant = target_items[
            (target_ratings >= relevance_threshold) & supported[target_items]
        ]
        if len(relevant):
            eligible.append(user_index)
    return np.asarray(eligible, dtype=np.int32)


def evaluate_global_ranking_strict(
    ranking: np.ndarray,
    history: InteractionCsr,
    targets: InteractionCsr,
    users: np.ndarray,
    observed_count: int,
    relevance_threshold: int,
    ranking_k: int,
) -> dict[str, float | int]:
    discounts = 1.0 / np.log2(np.arange(2, ranking_k + 2))
    recall_sum = ndcg_sum = 0.0
    recommendations = []
    for user_index in users:
        observed = set(history.user_items(int(user_index))[-observed_count:].tolist())
        recommended = np.asarray(
            [item for item in ranking if int(item) not in observed][:ranking_k],
            dtype=np.int32,
        )
        recommendations.append(recommended)
        target_items = targets.user_items(int(user_index))
        target_ratings = targets.user_ratings(int(user_index))
        relevant = set(target_items[target_ratings >= relevance_threshold].tolist())
        hits = np.asarray(
            [position for position, item in enumerate(recommended) if item in relevant],
            dtype=np.int64,
        )
        recall_sum += len(hits) / len(relevant)
        dcg = float(discounts[hits].sum()) if len(hits) else 0.0
        idcg = float(discounts[: min(len(relevant), ranking_k)].sum())
        ndcg_sum += dcg / idcg
    recommendation_matrix = np.stack(recommendations)
    unique_items = len(np.unique(recommendation_matrix))
    return {
        "eligible_users": len(users),
        "recall_at_10": round(recall_sum / len(users), 8),
        "ndcg_at_10": round(ndcg_sum / len(users), 8),
        "catalog_coverage_items": unique_items,
        "candidate_exclusion": f"only the {observed_count} exposed ratings",
    }


def run_validation(config: StrictColdStartConfig) -> dict[str, Any]:
    split_report = json.loads(
        (config.split_directory / "split_report.json").read_text(encoding="utf-8")
    )
    users_count = int(split_report["source"]["users"])
    items_count = int(split_report["source"]["items"])
    train_path = config.split_directory / "train.parquet"
    counts, sums, global_mean = fit_popularity(train_path, items_count, config.batch_size)
    supported = counts > 0
    history = load_interactions_csr(
        config.split_directory / "validation_history.parquet",
        users_count,
        config.batch_size,
    )
    targets = load_interactions_csr(
        config.split_directory / "validation_target.parquet",
        users_count,
        config.batch_size,
    )
    users = eligible_user_indices(
        history,
        targets,
        supported,
        max(config.observed_ratings),
        config.relevance_threshold,
    )
    trials = []
    for shrinkage in config.shrinkage_candidates:
        ranking, _ = popularity_ranking(counts, sums, global_mean, shrinkage)
        by_observed = {
            str(observed): evaluate_global_ranking_strict(
                ranking,
                history,
                targets,
                users,
                observed,
                config.relevance_threshold,
                config.ranking_k,
            )
            for observed in config.observed_ratings
        }
        trials.append(
            {
                "shrinkage": shrinkage,
                "mean_ndcg_at_10": round(
                    float(np.mean([value["ndcg_at_10"] for value in by_observed.values()])),
                    8,
                ),
                "mean_recall_at_10": round(
                    float(np.mean([value["recall_at_10"] for value in by_observed.values()])),
                    8,
                ),
                "by_observed_ratings": by_observed,
            }
        )
    selected = max(
        trials,
        key=lambda trial: (
            trial["mean_ndcg_at_10"],
            trial["mean_recall_at_10"],
            -trial["shrinkage"],
        ),
    )
    count_ranking = np.flatnonzero(supported)[
        np.argsort(-counts[supported], kind="stable")
    ]
    count_by_observed = {
        str(observed): evaluate_global_ranking_strict(
            count_ranking,
            history,
            targets,
            users,
            observed,
            config.relevance_threshold,
            config.ranking_k,
        )
        for observed in config.observed_ratings
    }
    report = {
        "protocol": {
            "split": "strict held-out validation users",
            "global_fit_users": split_report["user_groups"]["train"],
            "validation_users": len(users),
            "candidate_set": "all train-supported items",
            "candidate_exclusion": "only exposed ratings; hidden history is not excluded",
            "relevance": f"target rating >= {config.relevance_threshold}",
            "test_access": "none",
        },
        "model": {
            "name": "bayesian popularity baseline",
            "supported_items": int(supported.sum()),
            "global_train_mean": round(global_mean, 8),
        },
        "validation_trials": trials,
        "selected_validation": selected,
        "interaction_count_baseline": {"by_observed_ratings": count_by_observed},
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strict cold-start popularity")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/strict_cold_start.toml")
    )
    args = parser.parse_args()
    config = StrictColdStartConfig.from_toml(args.config.resolve())
    print(json.dumps(run_validation(config), indent=2))


if __name__ == "__main__":
    main()
