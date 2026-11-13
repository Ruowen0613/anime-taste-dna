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


@dataclass(frozen=True)
class PairwiseEvaluationConfig:
    split_directory: Path
    artifact_path: Path
    output_report: Path
    observed_ratings: int
    quiz_rounds: tuple[int, ...]
    preference_weight_candidates: tuple[float, ...]
    minimum_rating_gap: int
    relevance_threshold: int
    ranking_k: int
    user_batch_size: int
    parquet_batch_size: int
    seed: int

    @classmethod
    def from_toml(cls, path: Path) -> PairwiseEvaluationConfig:
        with path.open("rb") as file:
            raw = tomllib.load(file)
        root = path.parent.parent.resolve()
        resolve = lambda value: (root / value).resolve()  # noqa: E731
        return cls(
            split_directory=resolve(raw["paths"]["split_directory"]),
            artifact_path=resolve(raw["paths"]["artifact_path"]),
            output_report=resolve(raw["paths"]["output_report"]),
            observed_ratings=int(raw["evaluation"]["observed_ratings"]),
            quiz_rounds=tuple(int(value) for value in raw["evaluation"]["quiz_rounds"]),
            preference_weight_candidates=tuple(
                float(value) for value in raw["validation"]["preference_weight_candidates"]
            ),
            minimum_rating_gap=int(raw["evaluation"]["minimum_rating_gap"]),
            relevance_threshold=int(raw["evaluation"]["relevance_threshold"]),
            ranking_k=int(raw["evaluation"]["ranking_k"]),
            user_batch_size=int(raw["evaluation"]["user_batch_size"]),
            parquet_batch_size=int(raw["evaluation"]["parquet_batch_size"]),
            seed=int(raw["evaluation"]["seed"]),
        )


@dataclass(frozen=True)
class PreparedUser:
    user_index: int
    observed_items: np.ndarray
    observed_ratings: np.ndarray
    preference_pairs: np.ndarray
    excluded_items: np.ndarray
    relevant_items: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_preference_pairs(
    items: np.ndarray,
    ratings: np.ndarray,
    rounds: int,
    minimum_rating_gap: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(items))
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for left in order:
        for right in order:
            difference = int(ratings[left]) - int(ratings[right])
            if difference < minimum_rating_gap:
                continue
            pair = (int(items[left]), int(items[right]))
            if pair in seen:
                continue
            pairs.append(pair)
            seen.add(pair)
            if len(pairs) == rounds:
                return np.asarray(pairs, dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def fold_in_with_preferences(
    item_factors: np.ndarray,
    item_bias: np.ndarray,
    global_mean: float,
    observed_items: np.ndarray,
    observed_ratings: np.ndarray,
    preference_pairs: np.ndarray,
    regularization: float,
    preference_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    factors = item_factors[observed_items]
    design = np.concatenate(
        (factors, np.ones((*factors.shape[:2], 1), dtype=np.float32)), axis=2
    )
    targets = observed_ratings - global_mean - item_bias[observed_items]
    if preference_pairs.shape[1] > 0 and preference_weight > 0:
        most = preference_pairs[:, :, 0]
        least = preference_pairs[:, :, 1]
        pair_design = np.concatenate(
            (
                item_factors[most] - item_factors[least],
                np.zeros((*most.shape, 1), dtype=np.float32),
            ),
            axis=2,
        )
        pair_targets = 1.0 - item_bias[most] + item_bias[least]
        scale = np.sqrt(preference_weight)
        design = np.concatenate((design, pair_design * scale), axis=1)
        targets = np.concatenate((targets, pair_targets * scale), axis=1)
    gram = np.einsum("bni,bnj->bij", design, design)
    rhs = np.einsum("bni,bn->bi", design, targets)
    penalty = np.eye(design.shape[2], dtype=np.float32) * regularization
    parameters = np.linalg.solve(gram + penalty, rhs[..., None]).squeeze(-1)
    return parameters[:, :-1], parameters[:, -1]


def _combined_history(
    primary: InteractionCsr, additional: InteractionCsr | None, user_index: int
) -> tuple[np.ndarray, np.ndarray]:
    items = primary.user_items(user_index)
    ratings = primary.user_ratings(user_index)
    if additional is not None:
        items = np.concatenate((items, additional.user_items(user_index)))
        ratings = np.concatenate((ratings, additional.user_ratings(user_index)))
    return items, ratings


def prepare_users(
    primary: InteractionCsr,
    additional: InteractionCsr | None,
    targets: InteractionCsr,
    supported: np.ndarray,
    config: PairwiseEvaluationConfig,
) -> list[PreparedUser]:
    maximum_rounds = max(config.quiz_rounds)
    users: list[PreparedUser] = []
    for user_index in range(primary.users):
        history_items, history_ratings = _combined_history(primary, additional, user_index)
        if len(history_items) <= config.observed_ratings:
            continue
        target_items = targets.user_items(user_index)
        target_ratings = targets.user_ratings(user_index)
        relevant = target_items[
            (target_ratings >= config.relevance_threshold) & supported[target_items]
        ]
        if len(relevant) == 0:
            continue
        preference_pairs = build_preference_pairs(
            history_items[: -config.observed_ratings],
            history_ratings[: -config.observed_ratings],
            rounds=maximum_rounds,
            minimum_rating_gap=config.minimum_rating_gap,
            seed=config.seed + user_index,
        )
        if len(preference_pairs) < maximum_rounds:
            continue
        users.append(
            PreparedUser(
                user_index=user_index,
                observed_items=history_items[-config.observed_ratings :].astype(np.int32),
                observed_ratings=history_ratings[-config.observed_ratings :].astype(np.float32),
                preference_pairs=preference_pairs,
                excluded_items=history_items,
                relevant_items=np.unique(relevant),
            )
        )
    return users


def evaluate_condition(
    users: list[PreparedUser],
    item_factors: np.ndarray,
    item_bias: np.ndarray,
    global_mean: float,
    supported: np.ndarray,
    regularization: float,
    preference_weight: float,
    rounds: int,
    config: PairwiseEvaluationConfig,
    baseline: dict[int, tuple[np.ndarray, float, float]] | None = None,
) -> tuple[dict[str, float | int], dict[int, tuple[np.ndarray, float, float]]]:
    discounts = 1.0 / np.log2(np.arange(2, config.ranking_k + 2))
    recall_sum = ndcg_sum = satisfaction_sum = replaced_sum = 0.0
    improved = unchanged = worsened = 0
    per_user: dict[int, tuple[np.ndarray, float, float]] = {}
    candidate_mask = supported.astype(np.bool_, copy=False)

    for start in range(0, len(users), config.user_batch_size):
        batch = users[start : start + config.user_batch_size]
        observed_items = np.stack([user.observed_items for user in batch])
        observed_ratings = np.stack([user.observed_ratings for user in batch])
        if rounds:
            preferences = np.stack([user.preference_pairs[:rounds] for user in batch])
        else:
            preferences = np.empty((len(batch), 0, 2), dtype=np.int32)
        user_factors, user_bias = fold_in_with_preferences(
            item_factors,
            item_bias,
            global_mean,
            observed_items,
            observed_ratings,
            preferences,
            regularization,
            preference_weight,
        )
        scores = user_factors @ item_factors.T
        scores += user_bias[:, None] + item_bias[None, :] + global_mean
        scores[:, ~candidate_mask] = -np.inf
        for row, user in enumerate(batch):
            scores[row, user.excluded_items] = -np.inf
        top_indices = np.argpartition(scores, -config.ranking_k, axis=1)[
            :, -config.ranking_k :
        ]
        top_scores = np.take_along_axis(scores, top_indices, axis=1)
        order = np.argsort(-top_scores, axis=1, kind="stable")
        recommendations = np.take_along_axis(top_indices, order, axis=1)

        for row, user in enumerate(batch):
            relevant = set(user.relevant_items.tolist())
            hits = np.asarray(
                [index for index, item in enumerate(recommendations[row]) if item in relevant],
                dtype=np.int64,
            )
            recall = len(hits) / len(relevant)
            dcg = float(discounts[hits].sum()) if len(hits) else 0.0
            idcg = float(discounts[: min(len(relevant), config.ranking_k)].sum())
            ndcg = dcg / idcg
            recall_sum += recall
            ndcg_sum += ndcg
            if rounds:
                chosen = preferences[row]
                most_scores = (
                    item_factors[chosen[:, 0]] @ user_factors[row]
                    + item_bias[chosen[:, 0]]
                )
                least_scores = (
                    item_factors[chosen[:, 1]] @ user_factors[row]
                    + item_bias[chosen[:, 1]]
                )
                satisfaction_sum += float(np.mean(most_scores > least_scores))
            if baseline is not None:
                baseline_items, _, baseline_ndcg = baseline[user.user_index]
                replaced_sum += len(
                    set(baseline_items.tolist()) - set(recommendations[row].tolist())
                )
                if ndcg > baseline_ndcg + 1e-12:
                    improved += 1
                elif ndcg < baseline_ndcg - 1e-12:
                    worsened += 1
                else:
                    unchanged += 1
            per_user[user.user_index] = (recommendations[row], recall, ndcg)

    count = len(users)
    metrics: dict[str, float | int] = {
        "eligible_users": count,
        "recall_at_10": round(recall_sum / count, 8),
        "ndcg_at_10": round(ndcg_sum / count, 8),
        "preference_satisfaction": round(satisfaction_sum / count, 8) if rounds else 0.0,
    }
    if baseline is not None:
        metrics.update(
            {
                "mean_top_10_replacements": round(replaced_sum / count, 8),
                "users_improved": improved,
                "users_unchanged": unchanged,
                "users_worsened": worsened,
            }
        )
    return metrics, per_user


def run_evaluation(config: PairwiseEvaluationConfig) -> dict[str, Any]:
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
    validation_users = prepare_users(train, None, validation, supported, config)
    max_rounds = max(config.quiz_rounds)
    weight_trials = []
    for weight in config.preference_weight_candidates:
        metrics, _ = evaluate_condition(
            validation_users,
            item_factors,
            item_bias,
            global_mean,
            supported,
            regularization,
            weight,
            max_rounds,
            config,
        )
        weight_trials.append({"preference_weight": weight, **metrics})
        print(json.dumps(weight_trials[-1]), flush=True)
    selected_weight = max(
        weight_trials,
        key=lambda trial: (trial["ndcg_at_10"], trial["recall_at_10"], -trial["preference_weight"]),
    )["preference_weight"]

    def evaluate_rounds(
        prepared: list[PreparedUser],
    ) -> dict[str, dict[str, float | int]]:
        baseline_metrics, baseline = evaluate_condition(
            prepared,
            item_factors,
            item_bias,
            global_mean,
            supported,
            regularization,
            float(selected_weight),
            0,
            config,
        )
        results = {"0": baseline_metrics}
        for rounds in config.quiz_rounds:
            if rounds == 0:
                continue
            metrics, _ = evaluate_condition(
                prepared,
                item_factors,
                item_bias,
                global_mean,
                supported,
                regularization,
                float(selected_weight),
                rounds,
                config,
                baseline,
            )
            results[str(rounds)] = metrics
        return results

    validation_results = evaluate_rounds(validation_users)
    test = load_interactions_csr(
        config.split_directory / "test.parquet", users_count, config.parquet_batch_size
    )
    test_users = prepare_users(train, validation, test, supported, config)
    test_results = evaluate_rounds(test_users)
    report = {
        "protocol": {
            "model_version": model_version,
            "observed_ratings": config.observed_ratings,
            "quiz_rounds": list(config.quiz_rounds),
            "minimum_rating_gap": config.minimum_rating_gap,
            "pair_source": "chronological history before the observed-rating window",
            "common_cohort": "users eligible for the maximum configured quiz rounds",
            "candidate_set": "all train-supported items excluding all available user history",
            "proxy_limitation": "historical rating differences proxy pre-watch interest choices",
        },
        "selection": {
            "metric": "validation NDCG@10 at maximum quiz rounds, then Recall@10",
            "preference_weight_trials": weight_trials,
            "selected_preference_weight": selected_weight,
        },
        "validation": validation_results,
        "test": test_results,
        "reproducibility": {"seed": config.seed},
        "source": {
            "artifact_sha256": _sha256(config.artifact_path),
            "split_report_sha256": _sha256(split_report_path),
        },
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pairwise Taste Quiz refinement")
    parser.add_argument("--config", type=Path, default=Path("configs/pairwise_quiz.toml"))
    args = parser.parse_args()
    report = run_evaluation(PairwiseEvaluationConfig.from_toml(args.config.resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
