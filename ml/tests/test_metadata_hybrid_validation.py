from pathlib import Path

import numpy as np
import pandas as pd

from anime_taste_dna.evaluation.metadata_hybrid_validation import (
    build_genre_features,
    fold_in_genre_profiles,
)


def test_genre_features_and_signed_profile_point_toward_liked_genre(
    tmp_path: Path,
) -> None:
    items_path = tmp_path / "items.parquet"
    pd.DataFrame(
        {
            "item_index": [0, 1, 2],
            "genres": ["Action, Sci-Fi", "Romance", "Action"],
        }
    ).to_parquet(items_path, index=False)
    features, vocabulary, coverage = build_genre_features(
        items_path,
        item_count=3,
        supported=np.asarray([True, True, True]),
        minimum_frequency=1,
    )
    profile = fold_in_genre_profiles(
        features,
        observed_items=np.asarray([[0, 1]], dtype=np.int32),
        observed_ratings=np.asarray([[10, 1]], dtype=np.float32),
    )
    scores = profile @ features.T

    assert vocabulary == ("Action", "Romance", "Sci-Fi")
    assert coverage == 1.0
    assert scores[0, 2] > scores[0, 1]
