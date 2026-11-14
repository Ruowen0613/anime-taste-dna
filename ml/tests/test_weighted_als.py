import numpy as np

from anime_taste_dna.models.weighted_als import (
    fold_in_implicit_users,
    implicit_confidence,
    solve_implicit_factors,
)


def test_implicit_confidence_increases_with_rating() -> None:
    confidence = implicit_confidence(
        np.asarray([8, 9, 10], dtype=np.uint8), threshold=8, alpha=2.0
    )

    assert confidence.tolist() == [3.0, 5.0, 7.0]


def test_implicit_factor_solver_points_toward_observed_factor() -> None:
    factors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    solved = solve_implicit_factors(
        offsets=np.asarray([0, 1], dtype=np.int64),
        related_indices=np.asarray([0], dtype=np.int32),
        confidence=np.asarray([5.0], dtype=np.float32),
        fixed_factors=factors,
        regularization=1.0,
    )

    assert solved[0, 0] > solved[0, 1]


def test_fold_in_ignores_below_threshold_ratings() -> None:
    factors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    user = fold_in_implicit_users(
        factors,
        observed_items=np.asarray([[0, 1]], dtype=np.int32),
        observed_ratings=np.asarray([[10, 5]], dtype=np.uint8),
        threshold=8,
        alpha=10.0,
        regularization=1.0,
    )

    assert user[0, 0] > user[0, 1]
