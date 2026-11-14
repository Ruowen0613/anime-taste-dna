import numpy as np
import pandas as pd

from anime_taste_dna.data.user_holdout_split import assign_user_groups, label_user_rows


def test_assign_user_groups_is_deterministic_and_disjoint() -> None:
    groups = assign_user_groups(100, validation_fraction=0.1, test_fraction=0.2, seed=42)

    assert np.array_equal(groups, assign_user_groups(100, 0.1, 0.2, seed=42))
    assert np.count_nonzero(groups == 0) == 70
    assert np.count_nonzero(groups == 1) == 10
    assert np.count_nonzero(groups == 2) == 20


def test_label_user_rows_keeps_train_users_whole_and_splits_held_out_by_time() -> None:
    frame = pd.DataFrame(
        {
            "user_index": [0] * 5 + [1] * 5,
            "item_index": list(range(10)),
            "rating": [8] * 10,
            "updated_at": pd.date_range("2020-01-01", periods=10, freq="D")[::-1],
        }
    )
    groups = np.asarray([0, 1], dtype=np.int8)

    labelled = label_user_rows(
        frame,
        groups,
        target_fraction=0.4,
        minimum_target=1,
        minimum_history=3,
    )

    assert set(labelled.loc[labelled.user_index == 0, "split"]) == {"train"}
    validation = labelled.loc[labelled.user_index == 1]
    assert validation["split"].tolist() == [
        "validation_history",
        "validation_history",
        "validation_history",
        "validation_target",
        "validation_target",
    ]
    history_time = validation.loc[
        validation["split"] == "validation_history", "updated_at"
    ].max()
    target_time = validation.loc[
        validation["split"] == "validation_target", "updated_at"
    ].min()
    assert history_time < target_time
