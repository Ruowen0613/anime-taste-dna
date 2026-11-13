from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from anime_taste_dna.evaluation.ranking import InteractionCsr, load_interactions_csr
from anime_taste_dna.models.matrix_factorization import fold_in_users

AuditMode = Literal["full", "item_bias_only", "interaction_only"]


@dataclass(frozen=True)
class PersonalizationAuditConfig:
    split_directory: Path
    artifact_path: Path
    output_report: Path
    observed_ratings: tuple[int, ...]
    relevance_threshold: int
    ranking_k: int
    user_batch_size: int
    parquet_batch_size: int
    pair_samples: int
    seed: int

    @classmethod
    def from_toml(cls, path: Path) -> PersonalizationAuditConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            split_directory=resolve(raw["paths"]["split_directory"]),
            artifact_path=resolve(raw["paths"]["artifact_path"]),
            output_report=resolve(raw["paths"]["output_report"]),
            observed_ratings=tuple(int(value) for value in raw["evaluation"]["observed_ratings"]),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
            user_batch_size=int(raw["evaluation"]["user_batch_size"]),
            parquet_batch_size=int(raw["evaluation"]["parquet_batch_size"]),
            pair_samples=int(raw["evaluation"]["pair_samples"]),
            seed=int(raw["evaluation"]["seed"]),
        )


@dataclass(frozen=True)
class AuditUser:
    user_index: int
    history_items: np.ndarray
    history_ratings: np.ndarray
    relevant_items: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def mean_top_k_overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Recommendation matrices must have the same two-dimensional shape")
    overlaps = [
        len(set(left[row].tolist()) & set(right[row].tolist())) / left.shape[1]
        for row in range(len(left))
    ]
    return float(np.mean(overlaps))


def mean_pairwise_jaccard(
    recommendations: np.ndarray, samples: int, seed: int
) -> float:
    if len(recommendations) < 2:
        return 1.0
    generator = np.random.default_rng(seed)
    similarities = []
    for _ in range(samples):
        left, right = generator.choice(len(recommendations), size=2, replace=False)
        left_items = set(recommendations[left].tolist())
        right_items = set(recommendations[right].tolist())
        similarities.append(len(left_items & right_items) / len(left_items | right_items))
    return float(np.mean(similarities))


def prepare_users(
    train: InteractionCsr,
    validation: InteractionCsr,
    test: InteractionCsr,
    supported: np.ndarray,
    maximum_observations: int,
    relevance_threshold: int,
) -> list[AuditUser]:
    users = []
    for user_index in range(train.users):
        history_items = np.concatenate(
            (train.user_items(user_index), validation.user_items(user_index))
        )
        if len(history_items) < maximum_observations:
            continue
        history_ratings = np.concatenate(
            (train.user_ratings(user_index), validation.user_ratings(user_index))
        )
        target_items = test.user_items(user_index)
        target_ratings = test.user_ratings(user_index)
        relevant = target_items[
            (target_ratings >= relevance_threshold) & supported[target_items]
        ]
        if len(relevant) == 0:
            continue
        users.append(
            AuditUser(
                user_index=user_index,
                history_items=history_items,
                history_ratings=history_ratings,
                relevant_items=np.unique(relevant),
            )
        )
    return users


def evaluate_mode(
    users: list[AuditUser],
    item_factors: np.ndarray,
    item_bias: np.ndarray,
    global_mean: float,
    supported: np.ndarray,
    regularization: float,
    observed_count: int,
    mode: AuditMode,
    config: PersonalizationAuditConfig,
) -> tuple[dict[str, float | int], np.ndarray]:
    recommendations = np.empty((len(users), config.ranking_k), dtype=np.int32)
    discounts = 1.0 / np.log2(np.arange(2, config.ranking_k + 2))
    recall_sum = ndcg_sum = norm_sum = interaction_std_sum = 0.0
    zero_bias = np.zeros_like(item_bias)

    for start in range(0, len(users), config.user_batch_size):
        batch = users[start : start + config.user_batch_size]
        observed_items = np.stack([user.history_items[-observed_count:] for user in batch])
        observed_ratings = np.stack(
            [user.history_ratings[-observed_count:] for user in batch]
        ).astype(np.float32)
        if mode == "item_bias_only":
            user_factors = np.zeros((len(batch), item_factors.shape[1]), dtype=np.float32)
            user_bias = np.zeros(len(batch), dtype=np.float32)
        else:
            fold_in_bias = item_bias if mode == "full" else zero_bias
            user_factors, user_bias = fold_in_users(
                item_factors,
                fold_in_bias,
                global_mean,
                observed_items,
                observed_ratings,
                regularization,
            )
        scoring_bias = item_bias if mode in {"full", "item_bias_only"} else zero_bias
        scores = user_factors @ item_factors.T
        interaction_std_sum += float(scores[:, supported].std(axis=1).sum())
        scores += user_bias[:, None] + scoring_bias[None, :] + global_mean
        scores[:, ~supported] = -np.inf
        for row, user in enumerate(batch):
            scores[row, user.history_items] = -np.inf
        top_indices = np.argpartition(scores, -config.ranking_k, axis=1)[
            :, -config.ranking_k :
        ]
        top_scores = np.take_along_axis(scores, top_indices, axis=1)
        order = np.argsort(-top_scores, axis=1, kind="stable")
        ranked = np.take_along_axis(top_indices, order, axis=1)
        recommendations[start : start + len(batch)] = ranked
        norm_sum += float(np.linalg.norm(user_factors, axis=1).sum())

        for row, user in enumerate(batch):
            relevant = set(user.relevant_items.tolist())
            hits = np.asarray(
                [index for index, item in enumerate(ranked[row]) if item in relevant],
                dtype=np.int64,
            )
            recall_sum += len(hits) / len(relevant)
            dcg = float(discounts[hits].sum()) if len(hits) else 0.0
            idcg = float(discounts[: min(len(relevant), config.ranking_k)].sum())
            ndcg_sum += dcg / idcg

    unique_items, counts = np.unique(recommendations, return_counts=True)
    most_common_slots = int(np.sort(counts)[-10:].sum())
    total_slots = recommendations.size
    metrics: dict[str, float | int] = {
        "eligible_users": len(users),
        "recall_at_10": round(recall_sum / len(users), 8),
        "ndcg_at_10": round(ndcg_sum / len(users), 8),
        "mean_user_factor_norm": round(norm_sum / len(users), 8),
        "mean_interaction_score_std": round(interaction_std_sum / len(users), 8),
        "catalog_coverage_items": len(unique_items),
        "catalog_coverage_fraction": round(len(unique_items) / int(supported.sum()), 8),
        "top_10_items_share_of_slots": round(most_common_slots / total_slots, 8),
        "sampled_user_pair_jaccard": round(
            mean_pairwise_jaccard(recommendations, config.pair_samples, config.seed), 8
        ),
    }
    return metrics, recommendations


def run_audit(config: PersonalizationAuditConfig) -> dict[str, Any]:
    split_report_path = config.split_directory / "split_report.json"
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    users_count = int(split_report["source"]["users"])
    with np.load(config.artifact_path, allow_pickle=False) as artifact:
        item_factors = artifact["item_factors"].astype(np.float32)
        item_bias = artifact["item_bias"].astype(np.float32)
        global_mean = float(artifact["global_mean"][0])
        supported = artifact["supported"].astype(np.bool_)
        regularization = float(artifact["fold_in_regularization"][0])
        model_version = str(artifact["model_version"][0])

    train = load_interactions_csr(
        config.split_directory / "train.parquet", users_count, config.parquet_batch_size
    )
    validation = load_interactions_csr(
        config.split_directory / "validation.parquet", users_count, config.parquet_batch_size
    )
    test = load_interactions_csr(
        config.split_directory / "test.parquet", users_count, config.parquet_batch_size
    )
    users = prepare_users(
        train,
        validation,
        test,
        supported,
        max(config.observed_ratings),
        config.relevance_threshold,
    )

    full_results: dict[str, dict[str, float | int]] = {}
    full_recommendations: dict[int, np.ndarray] = {}
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
        full_results[str(observed_count)] = metrics
        full_recommendations[observed_count] = recommendations
        print(json.dumps({"mode": "full", "observed": observed_count, **metrics}), flush=True)

    audit_count = 10
    item_bias_metrics, item_bias_recommendations = evaluate_mode(
        users,
        item_factors,
        item_bias,
        global_mean,
        supported,
        regularization,
        audit_count,
        "item_bias_only",
        config,
    )
    interaction_metrics, interaction_recommendations = evaluate_mode(
        users,
        item_factors,
        item_bias,
        global_mean,
        supported,
        regularization,
        audit_count,
        "interaction_only",
        config,
    )
    comparisons = {
        "full_vs_item_bias_top_10_overlap": round(
            mean_top_k_overlap(full_recommendations[audit_count], item_bias_recommendations), 8
        ),
        "full_vs_interaction_only_top_10_overlap": round(
            mean_top_k_overlap(full_recommendations[audit_count], interaction_recommendations), 8
        ),
        "full_5_vs_10_top_10_overlap": round(
            mean_top_k_overlap(full_recommendations[5], full_recommendations[10]), 8
        ),
        "full_10_vs_20_top_10_overlap": round(
            mean_top_k_overlap(full_recommendations[10], full_recommendations[20]), 8
        ),
    }
    report = {
        "protocol": {
            "model_version": model_version,
            "common_cohort": "test users eligible for 20 observations and one relevant target",
            "candidate_set": "all train-supported items excluding train and validation history",
            "inference_ablation_note": (
                "item-bias and interaction terms are removed at inference; "
                "they are not retrained models"
            ),
        },
        "full_model_by_observed_ratings": full_results,
        "ablations_at_10_ratings": {
            "item_bias_only": item_bias_metrics,
            "interaction_only_without_item_bias": interaction_metrics,
        },
        "comparisons": comparisons,
        "source": {
            "artifact_sha256": _sha256(config.artifact_path),
            "split_report_sha256": _sha256(split_report_path),
        },
        "reproducibility": {"seed": config.seed, "sampled_user_pairs": config.pair_samples},
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit matrix-factorization personalization")
    parser.add_argument("--config", type=Path, default=Path("configs/personalization_audit.toml"))
    args = parser.parse_args()
    report = run_audit(PersonalizationAuditConfig.from_toml(args.config.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
