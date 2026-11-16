import numpy as np
import pytest

from anime_taste_dna.models.signed_profile import (
    fold_in_signed_profiles,
    normalize_item_factors,
    signed_rating_targets,
)


def test_signed_rating_targets_use_all_explicit_ratings() -> None:
    ratings = np.asarray([1, 5, 6, 10], dtype=np.float32)

    targets = signed_rating_targets(ratings)

    assert targets.tolist() == pytest.approx([-1.0, -1 / 9, 1 / 9, 1.0])


def test_low_rating_reverses_profile_direction() -> None:
    factors = normalize_item_factors(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    items = np.asarray([[0, 1]], dtype=np.int32)
    liked = fold_in_signed_profiles(
        factors,
        items,
        np.asarray([[10, 10]], dtype=np.float32),
        regularization=1.0,
    )
    disliked = fold_in_signed_profiles(
        factors,
        items,
        np.asarray([[1, 1]], dtype=np.float32),
        regularization=1.0,
    )

    assert np.all(liked > 0)
    assert np.all(disliked < 0)
    assert disliked == pytest.approx(-liked)


def test_item_factor_normalization_preserves_zero_rows() -> None:
    factors = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = normalize_item_factors(factors)

    assert normalized[0].tolist() == pytest.approx([0.6, 0.8])
    assert normalized[1].tolist() == [0.0, 0.0]
