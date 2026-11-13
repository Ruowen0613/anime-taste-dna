import numpy as np

from anime_taste_dna.evaluation.fold_in_diagnostic import prepare_validation_users
from anime_taste_dna.evaluation.ranking import InteractionCsr


def _csr(items: list[list[int]], ratings: list[list[float]]) -> InteractionCsr:
    lengths = np.asarray([len(row) for row in items], dtype=np.int64)
    return InteractionCsr(
        offsets=np.concatenate(([0], np.cumsum(lengths))),
        item_indices=np.asarray([item for row in items for item in row], dtype=np.int32),
        ratings=np.asarray([rating for row in ratings for rating in row], dtype=np.float32),
    )


def test_prepare_validation_users_uses_train_history_and_relevant_validation_targets() -> None:
    train = _csr([[0, 1], [1]], [[6, 9], [10]])
    validation = _csr([[2], [2]], [[8], [7]])

    users = prepare_validation_users(
        train,
        validation,
        supported=np.asarray([True, True, True]),
        maximum_observations=2,
        relevance_threshold=8,
    )

    assert len(users) == 1
    assert users[0].history_items.tolist() == [0, 1]
    assert users[0].relevant_items.tolist() == [2]
