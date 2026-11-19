from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from anime_taste_dna.evaluation.ranking import InteractionCsr, load_interactions_csr
from anime_taste_dna.evaluation.rating_scale_validation import popularity_blend_for_count
from anime_taste_dna.evaluation.signed_hybrid_validation import (
    FixedArtifactParameters,
    _standardize_rows,
)
from anime_taste_dna.evaluation.strict_cold_start import eligible_user_indices
from anime_taste_dna.models.signed_profile import (
    fold_in_signed_profiles,
    normalize_item_factors,
)
from anime_taste_dna.models.weighted_als import fold_in_implicit_users


@dataclass(frozen=True)
class CandidateRerankerConfig:
    split_directory: Path
    artifact_path: Path
    output_report: Path
    train_users: int
    pseudo_target_count: int
    observed_ratings: tuple[int, ...]
    candidates: int
    negatives_per_positive: int
    hidden_units: int
    residual_weight_candidates: tuple[float, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    signed_regularization: float
    signed_weight: float
    popularity_decay_exponent: float
    popularity_decay_pivot: int
    relevance_threshold: int
    ranking_k: int
    user_batch_size: int
    parquet_batch_size: int
    seed: int

    @classmethod
    def from_toml(cls, path: Path) -> CandidateRerankerConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()

        def resolve(value: str) -> Path:
            return (root / value).resolve()

        return cls(
            split_directory=resolve(raw["paths"]["split_directory"]),
            artifact_path=resolve(raw["paths"]["artifact_path"]),
            output_report=resolve(raw["paths"]["output_report"]),
            train_users=int(raw["training"]["users"]),
            pseudo_target_count=int(raw["training"]["pseudo_target_count"]),
            observed_ratings=tuple(
                int(value) for value in raw["evaluation"]["observed_ratings"]
            ),
            candidates=int(raw["model"]["candidates"]),
            negatives_per_positive=int(raw["training"]["negatives_per_positive"]),
            hidden_units=int(raw["model"]["hidden_units"]),
            residual_weight_candidates=tuple(
                float(value) for value in raw["model"]["residual_weight_candidates"]
            ),
            epochs=int(raw["training"]["epochs"]),
            batch_size=int(raw["training"]["batch_size"]),
            learning_rate=float(raw["training"]["learning_rate"]),
            weight_decay=float(raw["training"]["weight_decay"]),
            signed_regularization=float(raw["fixed_model"]["signed_regularization"]),
            signed_weight=float(raw["fixed_model"]["signed_weight"]),
            popularity_decay_exponent=float(
                raw["fixed_model"]["popularity_decay_exponent"]
            ),
            popularity_decay_pivot=int(raw["fixed_model"]["popularity_decay_pivot"]),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
            user_batch_size=int(raw["evaluation"]["user_batch_size"]),
            parquet_batch_size=int(raw["evaluation"]["parquet_batch_size"]),
            seed=int(raw["training"]["seed"]),
        )


class PairwiseReranker(nn.Module):
    def __init__(self, features: int, hidden_units: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


@dataclass(frozen=True)
class CandidateBatch:
    users: np.ndarray
    items: np.ndarray
    features: np.ndarray
    observed_items: np.ndarray


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms >= 1e-8, norms, 1.0)


def build_candidate_batch(
    users: np.ndarray,
    history: InteractionCsr,
    observed_count: int,
    holdout_count: int,
    raw_item_factors: np.ndarray,
    normalized_item_factors: np.ndarray,
    item_counts: np.ndarray,
    popularity: np.ndarray,
    supported: np.ndarray,
    artifact_parameters: FixedArtifactParameters,
    config: CandidateRerankerConfig,
) -> CandidateBatch:
    observed_item_rows = []
    observed_rating_rows = []
    for user in users:
        items = history.user_items(int(user))
        ratings = history.user_ratings(int(user))
        end = len(items) - holdout_count if holdout_count else len(items)
        start = end - observed_count
        observed_item_rows.append(items[start:end])
        observed_rating_rows.append(ratings[start:end])
    observed_items = np.stack(observed_item_rows).astype(np.int32)
    observed_ratings = np.stack(observed_rating_rows).astype(np.float32)

    implicit_profiles = fold_in_implicit_users(
        raw_item_factors,
        observed_items,
        observed_ratings,
        artifact_parameters.positive_rating_threshold,
        artifact_parameters.alpha,
        artifact_parameters.fold_in_regularization,
    )
    implicit_scores = _standardize_rows(
        implicit_profiles @ raw_item_factors.T, supported
    )
    popularity_blend = popularity_blend_for_count(
        artifact_parameters.popularity_blend,
        observed_count,
        config.popularity_decay_pivot,
        config.popularity_decay_exponent,
    )
    production_scores = _standardize_rows(
        implicit_scores + popularity_blend * popularity[None, :], supported
    )
    signed_profiles = fold_in_signed_profiles(
        normalized_item_factors,
        observed_items,
        observed_ratings,
        config.signed_regularization,
    )
    signed_scores = _standardize_rows(
        signed_profiles @ normalized_item_factors.T, supported
    )
    base_scores = production_scores + config.signed_weight * signed_scores
    base_scores[:, ~supported] = -np.inf
    for row in range(len(users)):
        base_scores[row, observed_items[row]] = -np.inf
    candidates = np.argpartition(base_scores, -config.candidates, axis=1)[
        :, -config.candidates :
    ]
    candidate_base_scores = np.take_along_axis(base_scores, candidates, axis=1)
    order = np.argsort(-candidate_base_scores, axis=1, kind="stable")
    candidates = np.take_along_axis(candidates, order, axis=1)

    positive_weights = np.maximum(observed_ratings - 7.0, 0.0)
    positive_profiles = np.einsum(
        "bo,bod->bd", positive_weights, normalized_item_factors[observed_items]
    )
    positive_profiles = _normalize_rows(positive_profiles)
    centered_weights = np.clip((observed_ratings - 5.5) / 4.5, -1.0, 1.0)
    centered_profiles = np.einsum(
        "bo,bod->bd", centered_weights, normalized_item_factors[observed_items]
    )
    centered_profiles = _normalize_rows(centered_profiles)
    candidate_factors = normalized_item_factors[candidates]
    positive_similarity = np.einsum("bcd,bd->bc", candidate_factors, positive_profiles)
    centered_similarity = np.einsum("bcd,bd->bc", candidate_factors, centered_profiles)
    observed_feature = np.full_like(
        positive_similarity, np.log1p(observed_count) / np.log(81.0)
    )
    factor_norm = np.linalg.norm(raw_item_factors[candidates], axis=2)
    features = np.stack(
        [
            np.take_along_axis(base_scores, candidates, axis=1),
            np.take_along_axis(production_scores, candidates, axis=1),
            np.take_along_axis(signed_scores, candidates, axis=1),
            popularity[candidates],
            positive_similarity,
            centered_similarity,
            factor_norm,
            observed_feature,
        ],
        axis=2,
    ).astype(np.float32)
    return CandidateBatch(
        users=users,
        items=candidates.astype(np.int32),
        features=features,
        observed_items=observed_items,
    )


def build_training_pairs(
    batches: list[CandidateBatch],
    history: InteractionCsr,
    pseudo_target_count: int,
    relevance_threshold: int,
    negatives_per_positive: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positive_rows = []
    negative_rows = []
    users_with_candidate_positive = 0
    candidate_positives = 0
    for batch in batches:
        for row, user in enumerate(batch.users):
            target_items = history.user_items(int(user))[-pseudo_target_count:]
            target_ratings = history.user_ratings(int(user))[-pseudo_target_count:]
            relevant = set(target_items[target_ratings >= relevance_threshold].tolist())
            positive_positions = [
                index for index, item in enumerate(batch.items[row]) if item in relevant
            ]
            if not positive_positions:
                continue
            users_with_candidate_positive += 1
            candidate_positives += len(positive_positions)
            negative_positions = [
                index for index, item in enumerate(batch.items[row]) if item not in relevant
            ]
            for positive_position in positive_positions:
                chosen_negatives = negative_positions[:negatives_per_positive]
                positive_rows.extend(
                    [batch.features[row, positive_position]] * len(chosen_negatives)
                )
                negative_rows.extend(batch.features[row, chosen_negatives])
    if not positive_rows:
        raise RuntimeError("Candidate generator retrieved no pseudo-target positives")
    return (
        np.asarray(positive_rows, dtype=np.float32),
        np.asarray(negative_rows, dtype=np.float32),
        {
            "users_with_candidate_positive": users_with_candidate_positive,
            "candidate_positives": candidate_positives,
            "training_pairs": len(positive_rows),
        },
    )


def _relevant_items(
    users: np.ndarray,
    targets: InteractionCsr,
    supported: np.ndarray,
    relevance_threshold: int,
) -> list[set[int]]:
    result = []
    for user in users:
        items = targets.user_items(int(user))
        ratings = targets.user_ratings(int(user))
        result.append(
            set(items[(ratings >= relevance_threshold) & supported[items]].tolist())
        )
    return result


def evaluate_reranker(
    model: PairwiseReranker,
    batches: list[CandidateBatch],
    relevant_by_batch: list[list[set[int]]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    popularity_ranking: np.ndarray,
    ranking_k: int,
    residual_weight: float,
) -> dict[str, float | int]:
    recall_sum = ndcg_sum = popularity_overlap_sum = 0.0
    recommendations = []
    eligible_users = 0
    discounts = 1.0 / np.log2(np.arange(2, ranking_k + 2))
    model.eval()
    with torch.no_grad():
        for batch, relevant_rows in zip(batches, relevant_by_batch, strict=True):
            normalized = (batch.features - feature_mean) / feature_std
            residual_scores = model(
                torch.from_numpy(normalized.reshape(-1, normalized.shape[-1]))
            )
            residual_scores = residual_scores.numpy().reshape(len(batch.users), -1)
            residual_mean = residual_scores.mean(axis=1, keepdims=True)
            residual_std = residual_scores.std(axis=1, keepdims=True)
            residual_std[residual_std < 1e-8] = 1.0
            residual_scores = (residual_scores - residual_mean) / residual_std
            scores = batch.features[:, :, 0] + residual_weight * residual_scores
            top_positions = np.argpartition(scores, -ranking_k, axis=1)[:, -ranking_k:]
            top_scores = np.take_along_axis(scores, top_positions, axis=1)
            order = np.argsort(-top_scores, axis=1, kind="stable")
            top_positions = np.take_along_axis(top_positions, order, axis=1)
            ranked = np.take_along_axis(batch.items, top_positions, axis=1)
            recommendations.append(ranked)
            for row, relevant in enumerate(relevant_rows):
                if not relevant:
                    continue
                hits = np.asarray(
                    [index for index, item in enumerate(ranked[row]) if item in relevant],
                    dtype=np.int64,
                )
                recall_sum += len(hits) / len(relevant)
                dcg = float(discounts[hits].sum()) if len(hits) else 0.0
                idcg = float(discounts[: min(len(relevant), ranking_k)].sum())
                ndcg_sum += dcg / idcg
                excluded = set(batch.observed_items[row].tolist())
                popular = [
                    int(item) for item in popularity_ranking if item not in excluded
                ][:ranking_k]
                popularity_overlap_sum += len(set(ranked[row].tolist()) & set(popular)) / ranking_k
                eligible_users += 1
    matrix = np.concatenate(recommendations)
    return {
        "eligible_users": eligible_users,
        "recall_at_10": round(recall_sum / eligible_users, 8),
        "ndcg_at_10": round(ndcg_sum / eligible_users, 8),
        "catalog_coverage_items": len(np.unique(matrix)),
        "popularity_top_10_overlap": round(popularity_overlap_sum / eligible_users, 8),
    }


def evaluate_base_candidates(
    batches: list[CandidateBatch],
    relevant_by_batch: list[list[set[int]]],
    popularity_ranking: np.ndarray,
    ranking_k: int,
) -> dict[str, float | int]:
    class BaseScore(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return features[:, 0]

    feature_count = batches[0].features.shape[2]
    return evaluate_reranker(
        BaseScore(),
        batches,
        relevant_by_batch,
        np.zeros(feature_count, dtype=np.float32),
        np.ones(feature_count, dtype=np.float32),
        popularity_ranking,
        ranking_k,
        0.0,
    )


def run_validation(config: CandidateRerankerConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)
    split_report = json.loads(
        (config.split_directory / "split_report.json").read_text(encoding="utf-8")
    )
    users_count = int(split_report["source"]["users"])
    train = load_interactions_csr(
        config.split_directory / "train.parquet", users_count, config.parquet_batch_size
    )
    validation_history = load_interactions_csr(
        config.split_directory / "validation_history.parquet",
        users_count,
        config.parquet_batch_size,
    )
    validation_targets = load_interactions_csr(
        config.split_directory / "validation_target.parquet",
        users_count,
        config.parquet_batch_size,
    )
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
    popularity_ranking = np.flatnonzero(supported)[
        np.argsort(-popularity[supported], kind="stable")
    ]

    lengths = np.diff(train.offsets)
    required = max(config.observed_ratings) + config.pseudo_target_count
    eligible_train_users = np.flatnonzero(lengths >= required)
    selected_train_users = np.sort(
        generator.choice(
            eligible_train_users,
            size=min(config.train_users, len(eligible_train_users)),
            replace=False,
        )
    )
    train_batches = []
    for observed_count in config.observed_ratings:
        count_users = selected_train_users[
            np.arange(len(selected_train_users)) % len(config.observed_ratings)
            == config.observed_ratings.index(observed_count)
        ]
        for start in range(0, len(count_users), config.user_batch_size):
            train_batches.append(
                build_candidate_batch(
                    count_users[start : start + config.user_batch_size],
                    train,
                    observed_count,
                    config.pseudo_target_count,
                    raw_item_factors,
                    normalized_item_factors,
                    item_counts,
                    popularity,
                    supported,
                    artifact_parameters,
                    config,
                )
            )
    positive_features, negative_features, training_stats = build_training_pairs(
        train_batches,
        train,
        config.pseudo_target_count,
        config.relevance_threshold,
        config.negatives_per_positive,
    )
    feature_values = np.concatenate((positive_features, negative_features), axis=0)
    feature_mean = feature_values.mean(axis=0)
    feature_std = feature_values.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    positive_tensor = torch.from_numpy((positive_features - feature_mean) / feature_std)
    negative_tensor = torch.from_numpy((negative_features - feature_mean) / feature_std)

    validation_users = eligible_user_indices(
        validation_history,
        validation_targets,
        supported,
        max(config.observed_ratings),
        config.relevance_threshold,
    )
    validation_batches: dict[str, list[CandidateBatch]] = {}
    validation_relevant: dict[str, list[list[set[int]]]] = {}
    baseline_by_observed: dict[str, dict[str, float | int]] = {}
    for observed_count in config.observed_ratings:
        batches = []
        relevant_batches = []
        for start in range(0, len(validation_users), config.user_batch_size):
            users = validation_users[start : start + config.user_batch_size]
            batches.append(
                build_candidate_batch(
                    users,
                    validation_history,
                    observed_count,
                    0,
                    raw_item_factors,
                    normalized_item_factors,
                    item_counts,
                    popularity,
                    supported,
                    artifact_parameters,
                    config,
                )
            )
            relevant_batches.append(
                _relevant_items(
                    users,
                    validation_targets,
                    supported,
                    config.relevance_threshold,
                )
            )
        validation_batches[str(observed_count)] = batches
        validation_relevant[str(observed_count)] = relevant_batches
        baseline_by_observed[str(observed_count)] = evaluate_base_candidates(
            batches, relevant_batches, popularity_ranking, config.ranking_k
        )

    model = PairwiseReranker(positive_features.shape[1], config.hidden_units)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history_report = []
    best_selection = (-np.inf, -np.inf)
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(len(positive_tensor))
        loss_sum = 0.0
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            margin = model(positive_tensor[indices]) - model(negative_tensor[indices])
            loss = torch.nn.functional.softplus(-margin).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(indices)
        residual_trials = []
        for residual_weight in config.residual_weight_candidates:
            by_observed = {
                str(observed_count): evaluate_reranker(
                    model,
                    validation_batches[str(observed_count)],
                    validation_relevant[str(observed_count)],
                    feature_mean,
                    feature_std,
                    popularity_ranking,
                    config.ranking_k,
                    residual_weight,
                )
                for observed_count in config.observed_ratings
            }
            mean_ndcg = float(
                np.mean([metrics["ndcg_at_10"] for metrics in by_observed.values()])
            )
            mean_recall = float(
                np.mean([metrics["recall_at_10"] for metrics in by_observed.values()])
            )
            residual_trials.append(
                {
                    "residual_weight": residual_weight,
                    "mean_recall_at_10": round(mean_recall, 8),
                    "mean_ndcg_at_10": round(mean_ndcg, 8),
                    "by_observed_ratings": by_observed,
                }
            )
        selected_residual = max(
            residual_trials,
            key=lambda trial: (
                trial["mean_ndcg_at_10"], trial["mean_recall_at_10"]
            ),
        )
        entry = {
            "epoch": epoch,
            "train_pairwise_loss": round(loss_sum / len(positive_tensor), 8),
            **selected_residual,
            "residual_trials": residual_trials,
        }
        history_report.append(entry)
        print(json.dumps(entry), flush=True)
        selection = (
            float(selected_residual["mean_ndcg_at_10"]),
            float(selected_residual["mean_recall_at_10"]),
        )
        if selection > best_selection:
            best_selection = selection
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Reranker training produced no checkpoint")
    model.load_state_dict(best_state)
    selected = max(
        history_report,
        key=lambda entry: (entry["mean_ndcg_at_10"], entry["mean_recall_at_10"]),
    )
    baseline_mean_recall = float(
        np.mean([metrics["recall_at_10"] for metrics in baseline_by_observed.values()])
    )
    baseline_mean_ndcg = float(
        np.mean([metrics["ndcg_at_10"] for metrics in baseline_by_observed.values()])
    )
    report = {
        "experiment": "two-stage-candidate-reranker-validation-v1",
        "protocol": {
            "split": "reranker fit on global-training users; strict held-out validation users",
            "test_access": "none",
            "source_model": source_model_version,
            "candidate_count": config.candidates,
            "validation_users": len(validation_users),
        },
        "training": {
            "selected_train_users": len(selected_train_users),
            **training_stats,
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
        },
        "baseline": {
            "mean_recall_at_10": round(baseline_mean_recall, 8),
            "mean_ndcg_at_10": round(baseline_mean_ndcg, 8),
            "by_observed_ratings": baseline_by_observed,
        },
        "selected_validation": selected,
        "history": history_report,
        "reproducibility": {"seed": config.seed},
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and validate a two-stage candidate reranker"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/candidate_reranker_validation.toml"),
    )
    args = parser.parse_args()
    config = CandidateRerankerConfig.from_toml(args.config.resolve())
    print(json.dumps(run_validation(config), indent=2))


if __name__ == "__main__":
    main()
