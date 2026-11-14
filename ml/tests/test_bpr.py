import numpy as np
import torch

from anime_taste_dna.models.bpr import (
    BprMatrixFactorization,
    fold_in_pairwise_users,
    fold_in_preference_users,
    sample_unobserved_items,
)


def test_bpr_margin_compares_positive_and_negative_scores() -> None:
    model = BprMatrixFactorization(users=1, items=2, dimensions=1)
    with torch.no_grad():
        model.user_factors.weight[:] = torch.tensor([[2.0]])
        model.item_factors.weight[:] = torch.tensor([[3.0], [1.0]])
        model.item_bias.weight.zero_()

    margin = model.pairwise_margin(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))

    assert margin.item() == 4.0


def test_negative_sampler_excludes_observed_user_items() -> None:
    users = np.asarray([0, 1], dtype=np.int32)
    supported = np.asarray([0, 1, 2], dtype=np.int32)
    observed_keys = np.asarray([0, 1, 4], dtype=np.int64)

    negatives = sample_unobserved_items(
        users, supported, observed_keys, item_count=3, generator=np.random.default_rng(42)
    )

    assert negatives[0] == 2
    assert negatives[1] != 1


def test_fold_in_preference_users_points_toward_higher_rated_item() -> None:
    factors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    user = fold_in_preference_users(
        factors,
        observed_items=np.asarray([[0, 1]], dtype=np.int32),
        observed_ratings=np.asarray([[10.0, 5.0]], dtype=np.float32),
        regularization=0.1,
    )

    scores = user @ factors.T
    assert scores[0, 0] > scores[0, 1]


def test_pairwise_fold_in_ranks_high_rating_above_low_rating() -> None:
    factors = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    user = fold_in_pairwise_users(
        factors,
        item_bias=np.zeros(3, dtype=np.float32),
        observed_items=np.asarray([[0, 1]], dtype=np.int32),
        observed_ratings=np.asarray([[10.0, 5.0]], dtype=np.float32),
        negative_items=np.asarray([[2, 2]], dtype=np.int32),
        regularization=1.0,
        steps=8,
    )

    scores = user @ factors.T
    assert scores[0, 0] > scores[0, 1]
