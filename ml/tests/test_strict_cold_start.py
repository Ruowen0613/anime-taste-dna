import numpy as np

from anime_taste_dna.evaluation.ranking import InteractionCsr
from anime_taste_dna.evaluation.strict_cold_start import evaluate_global_ranking_strict


def _csr(items: list[int], ratings: list[int]) -> InteractionCsr:
    return InteractionCsr(
        offsets=np.asarray([0, len(items)], dtype=np.int64),
        item_indices=np.asarray(items, dtype=np.int32),
        ratings=np.asarray(ratings, dtype=np.uint8),
    )


def test_strict_ranking_excludes_only_exposed_observations() -> None:
    metrics = evaluate_global_ranking_strict(
        ranking=np.asarray([0, 1, 2, 3], dtype=np.int32),
        history=_csr([0, 1], [8, 8]),
        targets=_csr([2], [10]),
        users=np.asarray([0], dtype=np.int32),
        observed_count=1,
        relevance_threshold=8,
        ranking_k=2,
    )

    # Item 0 is hidden history and remains a candidate; only exposed item 1 is excluded.
    assert metrics["recall_at_10"] == 1.0
    assert metrics["ndcg_at_10"] == 0.63092975
