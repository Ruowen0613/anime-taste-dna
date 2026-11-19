import numpy as np
import pytest
import torch

from anime_taste_dna.models.rating_pairwise import (
    build_observed_rating_pairs,
    fold_in_rating_pairwise_users,
    weighted_pairwise_loss,
)


def test_pairs_use_all_rating_levels_and_normalized_gaps() -> None:
    items = np.asarray([10, 20, 30, 40, 50], dtype=np.int32)
    ratings = np.asarray([1, 5, 7, 10, 10], dtype=np.uint8)

    higher, lower, weights = build_observed_rating_pairs(items, ratings)

    pairs = {(int(high), int(low)): float(weight) for high, low, weight in zip(
        higher, lower, weights, strict=True
    )}
    assert (20, 10) in pairs
    assert (30, 20) in pairs
    assert (40, 50) not in pairs
    assert pairs[(40, 10)] == pytest.approx(1.0)
    assert pairs[(30, 20)] == pytest.approx(2 / 9)


def test_pair_builder_never_invents_unobserved_items() -> None:
    items = np.asarray([2, 7, 11], dtype=np.int32)
    higher, lower, _ = build_observed_rating_pairs(
        items, np.asarray([1, 10, 5], dtype=np.uint8)
    )
    assert set(higher.tolist() + lower.tolist()) <= set(items.tolist())


def test_weighted_pairwise_loss_matches_hand_calculation() -> None:
    margins = torch.tensor([0.0, 1.0])
    weights = torch.tensor([1.0, 0.5])
    expected = (
        torch.nn.functional.softplus(-margins) * weights
    ).sum() / weights.sum()
    assert weighted_pairwise_loss(margins, weights).item() == pytest.approx(
        expected.item()
    )


def test_fold_in_orders_high_rating_above_low_rating() -> None:
    item_factors = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    item_bias = np.zeros(2, dtype=np.float32)
    items = np.asarray([[0, 1]], dtype=np.int32)

    user = fold_in_rating_pairwise_users(
        item_factors,
        item_bias,
        items,
        np.asarray([[10, 1]], dtype=np.uint8),
        regularization=1.0,
        steps=8,
    )
    reversed_user = fold_in_rating_pairwise_users(
        item_factors,
        item_bias,
        items,
        np.asarray([[1, 10]], dtype=np.uint8),
        regularization=1.0,
        steps=8,
    )
    assert float(user[0] @ item_factors[0]) > float(user[0] @ item_factors[1])
    assert float(reversed_user[0] @ item_factors[0]) < float(
        reversed_user[0] @ item_factors[1]
    )


def test_tied_or_single_ratings_return_finite_zero_profile() -> None:
    factors = np.eye(3, dtype=np.float32)
    biases = np.zeros(3, dtype=np.float32)
    users = fold_in_rating_pairwise_users(
        factors,
        biases,
        np.asarray([[0, 1], [2, 2]], dtype=np.int32),
        np.asarray([[5, 5], [8, 8]], dtype=np.uint8),
        regularization=1.0,
        steps=4,
    )
    assert np.isfinite(users).all()
    np.testing.assert_array_equal(users, np.zeros_like(users))


def test_stronger_regularization_shrinks_user_profile() -> None:
    factors = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    biases = np.zeros(2, dtype=np.float32)
    items = np.asarray([[0, 1]], dtype=np.int32)
    ratings = np.asarray([[10, 1]], dtype=np.uint8)
    weak = fold_in_rating_pairwise_users(factors, biases, items, ratings, 0.1, 8)
    strong = fold_in_rating_pairwise_users(factors, biases, items, ratings, 100.0, 8)
    assert np.linalg.norm(weak) > np.linalg.norm(strong)
