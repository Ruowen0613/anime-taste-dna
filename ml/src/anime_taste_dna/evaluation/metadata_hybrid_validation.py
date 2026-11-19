from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anime_taste_dna.evaluation.ranking import InteractionCsr, load_interactions_csr
from anime_taste_dna.evaluation.rating_scale_validation import popularity_blend_for_count
from anime_taste_dna.evaluation.signed_hybrid_validation import (
    FixedArtifactParameters,
    SignedHybridValidationConfig,
    _rank_scores,
    _ranking_metrics,
    _standardize_rows,
    _top_k_overlap,
)
from anime_taste_dna.evaluation.strict_cold_start import eligible_user_indices
from anime_taste_dna.models.signed_profile import (
    fold_in_signed_profiles,
    normalize_item_factors,
)
from anime_taste_dna.models.weighted_als import fold_in_implicit_users


@dataclass(frozen=True)
class MetadataHybridConfig:
    evaluation: SignedHybridValidationConfig
    items_path: Path
    output_report: Path
    signed_regularization: float
    signed_weight: float
    popularity_decay_exponent: float
    popularity_decay_pivot: int
    metadata_weight_candidates: tuple[float, ...]
    minimum_genre_frequency: int
    maximum_recall_loss_fraction: float
    maximum_popularity_overlap_increase: float
    minimum_coverage_retention_fraction: float

    @classmethod
    def from_toml(cls, path: Path) -> MetadataHybridConfig:
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
            items_path=(root / raw["paths"]["items_path"]).resolve(),
            output_report=evaluation.output_report,
            signed_regularization=float(raw["fixed_model"]["signed_regularization"]),
            signed_weight=float(raw["fixed_model"]["signed_weight"]),
            popularity_decay_exponent=float(
                raw["fixed_model"]["popularity_decay_exponent"]
            ),
            popularity_decay_pivot=int(raw["fixed_model"]["popularity_decay_pivot"]),
            metadata_weight_candidates=tuple(
                float(value) for value in raw["selection"]["metadata_weight_candidates"]
            ),
            minimum_genre_frequency=int(raw["selection"]["minimum_genre_frequency"]),
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


def _genre_tokens(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    return tuple(token.strip() for token in value.split(",") if token.strip())


def build_genre_features(
    items_path: Path,
    item_count: int,
    supported: np.ndarray,
    minimum_frequency: int,
) -> tuple[np.ndarray, tuple[str, ...], float]:
    items = pd.read_parquet(items_path, columns=["item_index", "genres"])
    token_rows: dict[int, tuple[str, ...]] = {}
    counts: Counter[str] = Counter()
    for row in items.itertuples(index=False):
        item_index = int(row.item_index)
        if not 0 <= item_index < item_count:
            continue
        tokens = _genre_tokens(row.genres)
        token_rows[item_index] = tokens
        if supported[item_index]:
            counts.update(set(tokens))
    vocabulary = tuple(
        sorted(token for token, count in counts.items() if count >= minimum_frequency)
    )
    if not vocabulary:
        raise ValueError("No genres satisfy the configured minimum frequency")
    feature_index = {token: index for index, token in enumerate(vocabulary)}
    features = np.zeros((item_count, len(vocabulary)), dtype=np.float32)
    supported_count = int(supported.sum())
    for item_index, tokens in token_rows.items():
        for token in set(tokens):
            column = feature_index.get(token)
            if column is not None:
                document_frequency = counts[token]
                features[item_index, column] = np.log(
                    (supported_count + 1) / (document_frequency + 1)
                ) + 1.0
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features /= np.where(norms >= 1e-8, norms, 1.0)
    coverage = float(np.mean(np.linalg.norm(features[supported], axis=1) > 0))
    return features, vocabulary, coverage


def fold_in_genre_profiles(
    genre_features: np.ndarray,
    observed_items: np.ndarray,
    observed_ratings: np.ndarray,
) -> np.ndarray:
    centered = np.clip((observed_ratings.astype(np.float32) - 5.5) / 4.5, -1.0, 1.0)
    profiles = np.einsum("bo,bof->bf", centered, genre_features[observed_items])
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    return profiles / np.where(norms >= 1e-8, norms, 1.0)


def recommend_metadata_hybrid(
    users: np.ndarray,
    history: InteractionCsr,
    raw_item_factors: np.ndarray,
    normalized_item_factors: np.ndarray,
    genre_features: np.ndarray,
    popularity: np.ndarray,
    supported: np.ndarray,
    observed_count: int,
    metadata_weight: float,
    popularity_blend: float,
    artifact_parameters: FixedArtifactParameters,
    config: MetadataHybridConfig,
) -> np.ndarray:
    recommendations = []
    for start in range(0, len(users), config.evaluation.user_batch_size):
        batch_users = users[start : start + config.evaluation.user_batch_size]
        observed_items = np.stack(
            [history.user_items(int(user))[-observed_count:] for user in batch_users]
        )
        observed_ratings = np.stack(
            [history.user_ratings(int(user))[-observed_count:] for user in batch_users]
        ).astype(np.float32)
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
        production_scores += popularity_blend * popularity[None, :]
        production_scores = _standardize_rows(production_scores, supported)
        signed_profiles = fold_in_signed_profiles(
            normalized_item_factors,
            observed_items,
            observed_ratings,
            config.signed_regularization,
        )
        signed_scores = _standardize_rows(
            signed_profiles @ normalized_item_factors.T, supported
        )
        genre_profiles = fold_in_genre_profiles(
            genre_features, observed_items, observed_ratings
        )
        genre_scores = _standardize_rows(genre_profiles @ genre_features.T, supported)
        scores = (
            production_scores
            + config.signed_weight * signed_scores
            + metadata_weight * genre_scores
        )
        scores[:, ~supported] = -np.inf
        for row in range(len(batch_users)):
            scores[row, observed_items[row]] = -np.inf
        recommendations.append(_rank_scores(scores, config.evaluation.ranking_k))
    return np.concatenate(recommendations)


def evaluate_metadata_candidate(
    users: np.ndarray,
    history: InteractionCsr,
    targets: InteractionCsr,
    raw_item_factors: np.ndarray,
    normalized_item_factors: np.ndarray,
    genre_features: np.ndarray,
    popularity: np.ndarray,
    supported: np.ndarray,
    observed_count: int,
    metadata_weight: float,
    popularity_blend: float,
    artifact_parameters: FixedArtifactParameters,
    config: MetadataHybridConfig,
) -> dict[str, float | int]:
    recommendations = recommend_metadata_hybrid(
        users,
        history,
        raw_item_factors,
        normalized_item_factors,
        genre_features,
        popularity,
        supported,
        observed_count,
        metadata_weight,
        popularity_blend,
        artifact_parameters,
        config,
    )
    recall, ndcg = _ranking_metrics(
        users,
        targets,
        recommendations,
        supported,
        config.evaluation.relevance_threshold,
    )
    popularity_scores = np.broadcast_to(popularity, (len(users), len(popularity))).copy()
    popularity_scores[:, ~supported] = -np.inf
    for row, user_index in enumerate(users):
        popularity_scores[
            row, history.user_items(int(user_index))[-observed_count:]
        ] = -np.inf
    popularity_recommendations = _rank_scores(
        popularity_scores, config.evaluation.ranking_k
    )
    unique_items = len(np.unique(recommendations))
    return {
        "eligible_users": len(users),
        "recall_at_10": round(recall, 8),
        "ndcg_at_10": round(ndcg, 8),
        "catalog_coverage_items": unique_items,
        "catalog_coverage_fraction": round(unique_items / int(supported.sum()), 8),
        "popularity_top_10_overlap": round(
            _top_k_overlap(recommendations, popularity_recommendations), 8
        ),
    }


def _mean(by_observed: dict[str, dict[str, float | int]], metric: str) -> float:
    return float(np.mean([float(metrics[metric]) for metrics in by_observed.values()]))


def run_validation(config: MetadataHybridConfig) -> dict[str, Any]:
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
    genre_features, genre_vocabulary, metadata_coverage = build_genre_features(
        config.items_path,
        len(item_counts),
        supported,
        config.minimum_genre_frequency,
    )
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

    trials = []
    for metadata_weight in config.metadata_weight_candidates:
        by_observed: dict[str, dict[str, float | int]] = {}
        for observed_count in evaluation.observed_ratings:
            popularity_blend = popularity_blend_for_count(
                artifact_parameters.popularity_blend,
                observed_count,
                config.popularity_decay_pivot,
                config.popularity_decay_exponent,
            )
            metrics = evaluate_metadata_candidate(
                validation_users,
                history,
                targets,
                raw_item_factors,
                normalized_item_factors,
                genre_features,
                popularity,
                supported,
                observed_count,
                metadata_weight,
                popularity_blend,
                artifact_parameters,
                config,
            )
            by_observed[str(observed_count)] = metrics
            print(
                json.dumps(
                    {
                        "metadata_weight": metadata_weight,
                        "observed_ratings": observed_count,
                        **metrics,
                    }
                ),
                flush=True,
            )
        trials.append(
            {
                "metadata_weight": metadata_weight,
                "mean_recall_at_10": round(_mean(by_observed, "recall_at_10"), 8),
                "mean_ndcg_at_10": round(_mean(by_observed, "ndcg_at_10"), 8),
                "mean_popularity_top_10_overlap": round(
                    _mean(by_observed, "popularity_top_10_overlap"), 8
                ),
                "mean_catalog_coverage_items": round(
                    _mean(by_observed, "catalog_coverage_items"), 1
                ),
                "by_observed_ratings": by_observed,
            }
        )
    baseline = next(trial for trial in trials if float(trial["metadata_weight"]) == 0.0)
    eligible = [
        trial
        for trial in trials
        if float(trial["mean_recall_at_10"])
        >= float(baseline["mean_recall_at_10"])
        * (1.0 - config.maximum_recall_loss_fraction)
        and float(trial["mean_popularity_top_10_overlap"])
        <= float(baseline["mean_popularity_top_10_overlap"])
        + config.maximum_popularity_overlap_increase
        and float(trial["mean_catalog_coverage_items"])
        >= float(baseline["mean_catalog_coverage_items"])
        * config.minimum_coverage_retention_fraction
    ]
    selected = max(
        eligible,
        key=lambda trial: (
            float(trial["mean_ndcg_at_10"]),
            float(trial["mean_recall_at_10"]),
            -float(trial["mean_popularity_top_10_overlap"]),
        ),
    )
    report = {
        "experiment": "genre-metadata-hybrid-validation-v1",
        "protocol": {
            "split": "strict held-out validation users",
            "test_access": "none",
            "common_users": len(validation_users),
            "source_model": source_model_version,
            "metadata_fit": "static item genres; IDF fit on train-supported items only",
        },
        "metadata": {
            "feature_count": len(genre_vocabulary),
            "features": list(genre_vocabulary),
            "supported_item_coverage": round(metadata_coverage, 8),
        },
        "baseline": baseline,
        "selected": selected,
        "trials": trials,
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a genre-metadata reranker over the signed hybrid"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/metadata_hybrid_validation.toml"),
    )
    args = parser.parse_args()
    config = MetadataHybridConfig.from_toml(args.config.resolve())
    print(json.dumps(run_validation(config), indent=2))


if __name__ == "__main__":
    main()
