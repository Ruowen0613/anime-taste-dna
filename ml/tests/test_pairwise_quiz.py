import numpy as np

from anime_taste_dna.evaluation.pairwise_quiz import (
    build_preference_pairs,
    fold_in_with_preferences,
)


def test_preference_pairs_are_deterministic_and_respect_rating_gap() -> None:
    items = np.asarray([10, 11, 12, 13], dtype=np.int32)
    ratings = np.asarray([10, 8, 6, 4], dtype=np.float32)

    first = build_preference_pairs(items, ratings, rounds=3, minimum_rating_gap=2, seed=42)
    second = build_preference_pairs(items, ratings, rounds=3, minimum_rating_gap=2, seed=42)

    assert np.array_equal(first, second)
    rating_by_item = dict(zip(items.tolist(), ratings.tolist(), strict=True))
    assert all(rating_by_item[most] - rating_by_item[least] >= 2 for most, least in first)


def test_pairwise_fold_in_moves_score_toward_preferred_item() -> None:
    item_factors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    item_bias = np.zeros(3, dtype=np.float32)
    observed_items = np.asarray([[2]], dtype=np.int32)
    observed_ratings = np.asarray([[5.0]], dtype=np.float32)
    no_pairs = np.empty((1, 0, 2), dtype=np.int32)
    pair = np.asarray([[[0, 1]]], dtype=np.int32)

    baseline_factors, _ = fold_in_with_preferences(
        item_factors, item_bias, 5.0, observed_items, observed_ratings, no_pairs, 1.0, 1.0
    )
    refined_factors, _ = fold_in_with_preferences(
        item_factors, item_bias, 5.0, observed_items, observed_ratings, pair, 1.0, 1.0
    )

    baseline_difference = baseline_factors[0] @ (item_factors[0] - item_factors[1])
    refined_difference = refined_factors[0] @ (item_factors[0] - item_factors[1])
    assert refined_difference > baseline_difference
