import numpy as np
import torch

from anime_taste_dna.models.matrix_factorization import (
    ExplicitMatrixFactorization,
    affine_permutation,
    fold_in_users,
)


def test_affine_permutation_visits_every_position_once() -> None:
    length = 101
    multiplier, offset = affine_permutation(length, seed=42)
    positions = (multiplier * np.arange(length) + offset) % length

    assert sorted(positions.tolist()) == list(range(length))


def test_matrix_factorization_forward_includes_biases_and_interaction() -> None:
    model = ExplicitMatrixFactorization(users=1, items=1, dimensions=2, global_mean=5.0)
    with torch.no_grad():
        model.user_factors.weight[:] = torch.tensor([[1.0, 2.0]])
        model.item_factors.weight[:] = torch.tensor([[3.0, 4.0]])
        model.user_bias.weight[:] = torch.tensor([[0.5]])
        model.item_bias.weight[:] = torch.tensor([[-0.25]])

    prediction = model(torch.tensor([0]), torch.tensor([0]))

    assert prediction.item() == 16.25


def test_fold_in_solves_regularized_user_parameters() -> None:
    item_factors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    item_bias = np.zeros(2, dtype=np.float32)
    observed_items = np.asarray([[0, 1]], dtype=np.int32)
    observed_ratings = np.asarray([[7.0, 9.0]], dtype=np.float32)

    factors, bias = fold_in_users(
        item_factors,
        item_bias,
        global_mean=5.0,
        observed_items=observed_items,
        observed_ratings=observed_ratings,
        regularization=0.1,
    )

    predictions = 5.0 + bias[:, None] + factors @ item_factors.T
    assert predictions.shape == (1, 2)
    assert predictions[0, 1] > predictions[0, 0]
