import numpy as np

from anime_taste_dna.evaluation.ranking import InteractionCsr
from anime_taste_dna.models.strict_matrix_factorization import evaluate_strict_mf


def _csr(items: list[int], ratings: list[int]) -> InteractionCsr:
    return InteractionCsr(
        offsets=np.asarray([0, len(items)], dtype=np.int64),
        item_indices=np.asarray(items, dtype=np.int32),
        ratings=np.asarray(ratings, dtype=np.uint8),
    )


def test_strict_mf_excludes_only_the_exposed_suffix() -> None:
    metrics = evaluate_strict_mf(
        users=np.asarray([0], dtype=np.int32),
        history=_csr([0, 1], [10, 10]),
        targets=_csr([2], [10]),
        item_factors=np.zeros((3, 1), dtype=np.float32),
        item_bias=np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
        global_mean=0.0,
        supported=np.ones(3, dtype=np.bool_),
        observed_count=1,
        regularization=1.0,
        ranking_k=2,
        relevance_threshold=8,
        user_batch_size=1,
        bias_only=True,
    )

    # Hidden-history item 0 stays ranked first; exposed item 1 alone is excluded.
    assert metrics["recall_at_10"] == 1.0
    assert metrics["ndcg_at_10"] == 0.63092975
