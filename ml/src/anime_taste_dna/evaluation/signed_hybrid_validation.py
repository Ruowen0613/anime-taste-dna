from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anime_taste_dna.evaluation.ranking import InteractionCsr, load_interactions_csr
from anime_taste_dna.evaluation.strict_cold_start import eligible_user_indices
from anime_taste_dna.models.signed_profile import (
    fold_in_signed_profiles,
    normalize_item_factors,
)
from anime_taste_dna.models.weighted_als import fold_in_implicit_users


@dataclass(frozen=True)
class SignedHybridValidationConfig:
    split_directory: Path
    artifact_path: Path
    output_report: Path
    signed_regularization_candidates: tuple[float, ...]
    signed_weight_candidates: tuple[float, ...]
    selection_users: int
    observed_ratings: tuple[int, ...]
    relevance_threshold: int
    ranking_k: int
    user_batch_size: int
    parquet_batch_size: int
    max_counterfactual_overlap: float
    max_popularity_overlap: float
    minimum_catalog_coverage_fraction: float
    seed: int

    @classmethod
    def from_toml(cls, path: Path) -> SignedHybridValidationConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            split_directory=resolve(raw["paths"]["split_directory"]),
            artifact_path=resolve(raw["paths"]["artifact_path"]),
            output_report=resolve(raw["paths"]["output_report"]),
            signed_regularization_candidates=tuple(
                float(value)
                for value in raw["selection"]["signed_regularization_candidates"]
            ),
            signed_weight_candidates=tuple(
                float(value) for value in raw["selection"]["signed_weight_candidates"]
            ),
            selection_users=int(raw["selection"]["users"]),
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


@dataclass(frozen=True)
class FixedArtifactParameters:
    alpha: float
    fold_in_regularization: float
    positive_rating_threshold: int
    popularity_blend: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _top_k_overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Recommendation matrices must have matching two-dimensional shapes")
    return float(
        np.mean(
            [
                len(set(left[row].tolist()) & set(right[row].tolist())) / left.shape[1]
                for row in range(len(left))
            ]
        )
    )


def _rank_scores(scores: np.ndarray, ranking_k: int) -> np.ndarray:
    candidates = np.argpartition(scores, -ranking_k, axis=1)[:, -ranking_k:]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1)


def _standardize_rows(scores: np.ndarray, supported: np.ndarray) -> np.ndarray:
    supported_scores = scores[:, supported]
    means = supported_scores.mean(axis=1, keepdims=True)
    standard_deviations = supported_scores.std(axis=1, keepdims=True)
    standard_deviations[standard_deviations < 1e-8] = 1.0
    return (scores - means) / standard_deviations


def _ranking_metrics(
    users: np.ndarray,
    targets: InteractionCsr,
    recommendations: np.ndarray,
    supported: np.ndarray,
    relevance_threshold: int,
) -> tuple[float, float]:
    ranking_k = recommendations.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, ranking_k + 2))
    recall_sum = ndcg_sum = 0.0
    for row, user_index in enumerate(users):
        target_items = targets.user_items(int(user_index))
        target_ratings = targets.user_ratings(int(user_index))
        relevant = set(
            target_items[
                (target_ratings >= relevance_threshold) & supported[target_items]
            ].tolist()
        )
        hits = np.asarray(
            [position for position, item in enumerate(recommendations[row]) if item in relevant],
            dtype=np.int64,
        )
        recall_sum += len(hits) / len(relevant)
        dcg = float(discounts[hits].sum()) if len(hits) else 0.0
        idcg = float(discounts[: min(len(relevant), ranking_k)].sum())
        ndcg_sum += dcg / idcg
    return recall_sum / len(users), ndcg_sum / len(users)


def recommend_signed_hybrid(
    users: np.ndarray,
    history: InteractionCsr,
    raw_item_factors: np.ndarray,
    normalized_item_factors: np.ndarray,
    popularity: np.ndarray,
    supported: np.ndarray,
    observed_count: int,
    signed_regularization: float,
    signed_weight: float,
    artifact_parameters: FixedArtifactParameters,
    ranking_k: int,
    user_batch_size: int,
    flip_high_ratings: bool = False,
    popularity_blend_override: float | None = None,
) -> tuple[np.ndarray, float]:
    recommendations = []
    signed_norm_sum = 0.0
    for start in range(0, len(users), user_batch_size):
        batch_users = users[start : start + user_batch_size]
        observed_items = np.stack(
            [history.user_items(int(user))[-observed_count:] for user in batch_users]
        )
        observed_ratings = np.stack(
            [history.user_ratings(int(user))[-observed_count:] for user in batch_users]
        ).astype(np.float32)
        if flip_high_ratings:
            observed_ratings = observed_ratings.copy()
            observed_ratings[observed_ratings >= 8] = 1

        implicit_profiles = fold_in_implicit_users(
            raw_item_factors,
            observed_items,
            observed_ratings,
            artifact_parameters.positive_rating_threshold,
            artifact_parameters.alpha,
            artifact_parameters.fold_in_regularization,
        )
        production_scores = _standardize_rows(
            implicit_profiles @ raw_item_factors.T, supported
        )
        popularity_blend = (
            artifact_parameters.popularity_blend
            if popularity_blend_override is None
            else popularity_blend_override
        )
        production_scores += popularity_blend * popularity[None, :]
        production_scores = _standardize_rows(production_scores, supported)

        signed_profiles = fold_in_signed_profiles(
            normalized_item_factors,
            observed_items,
            observed_ratings,
            signed_regularization,
        )
        signed_norm_sum += float(np.linalg.norm(signed_profiles, axis=1).sum())
        signed_scores = _standardize_rows(
            signed_profiles @ normalized_item_factors.T, supported
        )
        scores = production_scores + signed_weight * signed_scores
        scores[:, ~supported] = -np.inf
        for row in range(len(batch_users)):
            scores[row, observed_items[row]] = -np.inf
        recommendations.append(_rank_scores(scores, ranking_k))
    return np.concatenate(recommendations), signed_norm_sum / len(users)


def evaluate_candidate(
    users: np.ndarray,
    history: InteractionCsr,
    targets: InteractionCsr,
    raw_item_factors: np.ndarray,
    normalized_item_factors: np.ndarray,
    popularity: np.ndarray,
    supported: np.ndarray,
    observed_count: int,
    signed_regularization: float,
    signed_weight: float,
    artifact_parameters: FixedArtifactParameters,
    config: SignedHybridValidationConfig,
    include_counterfactual: bool,
    popularity_blend_override: float | None = None,
) -> dict[str, float | int]:
    recommendations, mean_norm = recommend_signed_hybrid(
        users,
        history,
        raw_item_factors,
        normalized_item_factors,
        popularity,
        supported,
        observed_count,
        signed_regularization,
        signed_weight,
        artifact_parameters,
        config.ranking_k,
        config.user_batch_size,
        popularity_blend_override=popularity_blend_override,
    )
    recall, ndcg = _ranking_metrics(
        users,
        targets,
        recommendations,
        supported,
        config.relevance_threshold,
    )
    popularity_scores = np.broadcast_to(popularity, (len(users), len(popularity))).copy()
    popularity_scores[:, ~supported] = -np.inf
    for row, user_index in enumerate(users):
        popularity_scores[
            row, history.user_items(int(user_index))[-observed_count:]
        ] = -np.inf
    popularity_recommendations = _rank_scores(popularity_scores, config.ranking_k)
    unique_items = len(np.unique(recommendations))
    metrics: dict[str, float | int] = {
        "eligible_users": len(users),
        "recall_at_10": round(recall, 8),
        "ndcg_at_10": round(ndcg, 8),
        "mean_signed_user_factor_norm": round(mean_norm, 8),
        "catalog_coverage_items": unique_items,
        "catalog_coverage_fraction": round(unique_items / int(supported.sum()), 8),
        "popularity_top_10_overlap": round(
            _top_k_overlap(recommendations, popularity_recommendations), 8
        ),
    }
    if include_counterfactual:
        flipped, _ = recommend_signed_hybrid(
            users,
            history,
            raw_item_factors,
            normalized_item_factors,
            popularity,
            supported,
            observed_count,
            signed_regularization,
            signed_weight,
            artifact_parameters,
            config.ranking_k,
            config.user_batch_size,
            flip_high_ratings=True,
            popularity_blend_override=popularity_blend_override,
        )
        metrics["high_to_low_counterfactual_top_10_overlap"] = round(
            _top_k_overlap(recommendations, flipped), 8
        )
    return metrics


def _passes_behavior_gates(
    metrics: dict[str, float | int], config: SignedHybridValidationConfig
) -> bool:
    return (
        float(metrics["high_to_low_counterfactual_top_10_overlap"])
        <= config.max_counterfactual_overlap
        and float(metrics["popularity_top_10_overlap"])
        <= config.max_popularity_overlap
        and float(metrics["catalog_coverage_fraction"])
        >= config.minimum_catalog_coverage_fraction
    )


def run_validation(config: SignedHybridValidationConfig) -> dict[str, Any]:
    split_report_path = config.split_directory / "split_report.json"
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    users_count = int(split_report["source"]["users"])
    with np.load(config.artifact_path, allow_pickle=False) as artifact:
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
        config.split_directory / "validation_history.parquet",
        users_count,
        config.parquet_batch_size,
    )
    targets = load_interactions_csr(
        config.split_directory / "validation_target.parquet",
        users_count,
        config.parquet_batch_size,
    )
    validation_users = eligible_user_indices(
        history,
        targets,
        supported,
        max(config.observed_ratings),
        config.relevance_threshold,
    )
    generator = np.random.default_rng(config.seed)
    selection_users = np.sort(
        generator.choice(
            validation_users,
            size=min(config.selection_users, len(validation_users)),
            replace=False,
        )
    )

    trials = []
    for regularization in config.signed_regularization_candidates:
        for signed_weight in config.signed_weight_candidates:
            metrics = evaluate_candidate(
                selection_users,
                history,
                targets,
                raw_item_factors,
                normalized_item_factors,
                popularity,
                supported,
                10,
                regularization,
                signed_weight,
                artifact_parameters,
                config,
                include_counterfactual=True,
            )
            trial = {
                "signed_regularization": regularization,
                "signed_weight": signed_weight,
                "passes_behavior_gates": _passes_behavior_gates(metrics, config),
                **metrics,
            }
            trials.append(trial)
            print(json.dumps(trial), flush=True)
    eligible = [trial for trial in trials if trial["passes_behavior_gates"]]
    selected = (
        max(eligible, key=lambda trial: (trial["ndcg_at_10"], trial["recall_at_10"]))
        if eligible
        else None
    )

    full_validation = None
    if selected is not None:
        full_validation = {}
        for observed_count in config.observed_ratings:
            metrics = evaluate_candidate(
                validation_users,
                history,
                targets,
                raw_item_factors,
                normalized_item_factors,
                popularity,
                supported,
                observed_count,
                float(selected["signed_regularization"]),
                float(selected["signed_weight"]),
                artifact_parameters,
                config,
                include_counterfactual=observed_count == 10,
            )
            full_validation[str(observed_count)] = metrics
            print(json.dumps({"observed_ratings": observed_count, **metrics}), flush=True)

    report = {
        "protocol": {
            "split": "strict held-out validation users",
            "source_item_model": source_model_version,
            "ranking_formula": (
                "standardized production hybrid score + signed_weight * "
                "standardized all-rating profile score"
            ),
            "rating_signal": "all 1-10 ratings mapped linearly around neutral midpoint 5.5",
            "test_access": "none",
            "selection_cohort": len(selection_users),
        },
        "behavior_gates": {
            "max_counterfactual_top_10_overlap": config.max_counterfactual_overlap,
            "max_popularity_top_10_overlap": config.max_popularity_overlap,
            "minimum_catalog_coverage_fraction": config.minimum_catalog_coverage_fraction,
        },
        "selection_trials_at_10_ratings": trials,
        "selected": selected,
        "full_validation_by_observed_ratings": full_validation,
        "source": {
            "artifact_sha256": _sha256(config.artifact_path),
            "split_report_sha256": _sha256(split_report_path),
        },
        "reproducibility": {"seed": config.seed},
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a production-score plus signed explicit-rating hybrid"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/signed_hybrid_validation.toml")
    )
    args = parser.parse_args()
    config = SignedHybridValidationConfig.from_toml(args.config.resolve())
    print(json.dumps(run_validation(config), indent=2))


if __name__ == "__main__":
    main()
