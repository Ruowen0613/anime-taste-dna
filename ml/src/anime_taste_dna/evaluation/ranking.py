from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class InteractionCsr:
    offsets: np.ndarray
    item_indices: np.ndarray
    ratings: np.ndarray

    @property
    def users(self) -> int:
        return len(self.offsets) - 1

    def user_items(self, user_index: int) -> np.ndarray:
        start, end = self.offsets[user_index : user_index + 2]
        return self.item_indices[start:end]

    def user_ratings(self, user_index: int) -> np.ndarray:
        start, end = self.offsets[user_index : user_index + 2]
        return self.ratings[start:end]


def load_interactions_csr(path: Path, users: int, batch_size: int) -> InteractionCsr:
    parquet = pq.ParquetFile(path)
    counts = np.zeros(users, dtype=np.int64)
    total_rows = 0
    for batch in parquet.iter_batches(columns=["user_index"], batch_size=batch_size):
        user_indices = batch.column(0).to_numpy()
        counts += np.bincount(user_indices, minlength=users)
        total_rows += len(user_indices)

    offsets = np.zeros(users + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    item_indices = np.empty(total_rows, dtype=np.int32)
    ratings = np.empty(total_rows, dtype=np.uint8)
    cursors = offsets[:-1].copy()

    for batch in parquet.iter_batches(
        columns=["user_index", "item_index", "rating"], batch_size=batch_size
    ):
        user_indices = batch.column(0).to_numpy()
        batch_items = batch.column(1).to_numpy()
        batch_ratings = batch.column(2).to_numpy()
        boundaries = np.flatnonzero(user_indices[1:] != user_indices[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(user_indices)]))
        for start, end in zip(starts, ends, strict=True):
            user_index = int(user_indices[start])
            destination = int(cursors[user_index])
            length = int(end - start)
            item_indices[destination : destination + length] = batch_items[start:end]
            ratings[destination : destination + length] = batch_ratings[start:end]
            cursors[user_index] += length

    if not np.array_equal(cursors, offsets[1:]):
        raise RuntimeError(f"Failed to construct interaction index for {path}")
    return InteractionCsr(offsets=offsets, item_indices=item_indices, ratings=ratings)


def evaluate_global_ranking(
    ranking: np.ndarray,
    observed: InteractionCsr,
    targets: InteractionCsr,
    additional_exclusions: InteractionCsr | None,
    observed_count: int,
    relevance_threshold: int,
    ranking_k: int,
) -> dict[str, float | int]:
    if observed.users != targets.users:
        raise ValueError("Observed and target interactions must have the same user count")
    if additional_exclusions is not None and additional_exclusions.users != observed.users:
        raise ValueError("Additional exclusions must have the same user count")

    maximum_item = max(
        int(ranking.max()),
        int(observed.item_indices.max(initial=0)),
        int(targets.item_indices.max(initial=0)),
        int(
            additional_exclusions.item_indices.max(initial=0)
            if additional_exclusions is not None
            else 0
        ),
    )
    candidate_mask = np.zeros(maximum_item + 1, dtype=np.bool_)
    candidate_mask[ranking] = True
    excluded_mask = np.zeros_like(candidate_mask)
    discounts = 1.0 / np.log2(np.arange(2, ranking_k + 2))
    recall_sum = 0.0
    ndcg_sum = 0.0
    eligible_users = 0

    for user_index in range(observed.users):
        observed_items = observed.user_items(user_index)
        if len(observed_items) < observed_count:
            continue
        target_items = targets.user_items(user_index)
        target_ratings = targets.user_ratings(user_index)
        relevant = target_items[
            (target_ratings >= relevance_threshold) & candidate_mask[target_items]
        ]
        if len(relevant) == 0:
            continue

        excluded_mask[observed_items] = True
        if additional_exclusions is not None:
            extra_items = additional_exclusions.user_items(user_index)
            excluded_mask[extra_items] = True
        recommendations: list[int] = []
        for item_index in ranking:
            if not excluded_mask[item_index]:
                recommendations.append(int(item_index))
                if len(recommendations) == ranking_k:
                    break

        relevant_set = set(relevant.tolist())
        hit_positions = np.asarray(
            [position for position, item in enumerate(recommendations) if item in relevant_set],
            dtype=np.int64,
        )
        hits = len(hit_positions)
        recall_sum += hits / len(relevant_set)
        dcg = float(discounts[hit_positions].sum()) if hits else 0.0
        idcg = float(discounts[: min(len(relevant_set), ranking_k)].sum())
        ndcg_sum += dcg / idcg
        eligible_users += 1

        excluded_mask[observed_items] = False
        if additional_exclusions is not None:
            excluded_mask[extra_items] = False

    if eligible_users == 0:
        return {"eligible_users": 0, "recall_at_10": 0.0, "ndcg_at_10": 0.0}
    return {
        "eligible_users": eligible_users,
        "recall_at_10": round(recall_sum / eligible_users, 8),
        "ndcg_at_10": round(ndcg_sum / eligible_users, 8),
    }
