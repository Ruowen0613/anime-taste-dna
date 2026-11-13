import numpy as np
import pytest

from anime_taste_dna.evaluation.personalization_audit import (
    mean_pairwise_jaccard,
    mean_top_k_overlap,
)


def test_top_k_overlap_uses_per_user_item_sets() -> None:
    left = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    right = np.asarray([[1, 2, 7], [4, 8, 9]], dtype=np.int32)

    assert mean_top_k_overlap(left, right) == pytest.approx(0.5)


def test_pairwise_jaccard_is_deterministic() -> None:
    recommendations = np.asarray(
        [[1, 2, 3], [1, 2, 4], [8, 9, 10]], dtype=np.int32
    )

    first = mean_pairwise_jaccard(recommendations, samples=20, seed=42)
    second = mean_pairwise_jaccard(recommendations, samples=20, seed=42)

    assert first == second
    assert 0 <= first <= 1
