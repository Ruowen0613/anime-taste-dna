import numpy as np

from anime_taste_dna.data.prepare_user_animelist_2025_pilot import (
    deduplicate_sorted_interactions,
)


def test_deduplicate_keeps_exact_pair_and_drops_conflicting_pair() -> None:
    users = np.asarray([2, 1, 1, 1, 1], dtype=np.int32)
    items = np.asarray([3, 2, 2, 4, 4], dtype=np.int32)
    ratings = np.asarray([8, 7, 7, 6, 9], dtype=np.uint8)

    keep, removed, conflicts = deduplicate_sorted_interactions(users, items, ratings)

    order = np.lexsort((ratings, items, users))
    assert list(zip(users[order][keep], items[order][keep], ratings[order][keep], strict=True)) == [
        (1, 2, 7),
        (2, 3, 8),
    ]
    assert removed == 2
    assert conflicts == 1
