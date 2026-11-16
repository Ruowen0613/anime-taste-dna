import pytest

from anime_taste_dna.evaluation.rating_scale_validation import popularity_blend_for_count


def test_popularity_blend_stays_fixed_through_pivot_then_decays() -> None:
    assert popularity_blend_for_count(32.0, 5, 10, 0.5) == 32.0
    assert popularity_blend_for_count(32.0, 10, 10, 0.5) == 32.0
    assert popularity_blend_for_count(32.0, 40, 10, 0.5) == 16.0


def test_zero_decay_keeps_original_blend() -> None:
    assert popularity_blend_for_count(32.0, 80, 10, 0.0) == 32.0


@pytest.mark.parametrize(("observed", "pivot"), [(0, 10), (10, 0)])
def test_popularity_blend_rejects_non_positive_counts(observed: int, pivot: int) -> None:
    with pytest.raises(ValueError):
        popularity_blend_for_count(32.0, observed, pivot, 0.5)
