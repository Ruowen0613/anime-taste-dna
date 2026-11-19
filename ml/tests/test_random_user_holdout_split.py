import numpy as np

from anime_taste_dna.data.random_user_holdout_split import random_target_positions


def test_random_target_positions_are_deterministic_and_disjoint() -> None:
    history, target = random_target_positions(100, 0.2, 10, 42)
    repeated_history, repeated_target = random_target_positions(100, 0.2, 10, 42)

    assert len(history) == 80
    assert len(target) == 20
    assert not set(history) & set(target)
    assert set(history) | set(target) == set(range(100))
    assert np.array_equal(history, repeated_history)
    assert np.array_equal(target, repeated_target)
