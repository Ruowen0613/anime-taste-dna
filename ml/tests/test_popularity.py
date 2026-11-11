from pathlib import Path

import numpy as np
import pandas as pd

from anime_taste_dna.evaluation.ranking import (
    evaluate_global_ranking,
    load_interactions_csr,
)
from anime_taste_dna.models.popularity import fit_popularity, popularity_ranking


def test_popularity_fit_and_ranking_use_training_ratings_only(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "user_index": [0, 0, 1, 1],
            "item_index": [0, 1, 0, 2],
            "rating": [10, 2, 8, 7],
        }
    ).to_parquet(train_path, index=False)

    counts, sums, global_mean = fit_popularity(train_path, items=3, batch_size=2)
    ranking, scores = popularity_ranking(counts, sums, global_mean, shrinkage=2.0)

    assert counts.tolist() == [2, 1, 1]
    assert sums.tolist() == [18.0, 2.0, 7.0]
    assert global_mean == 6.75
    assert ranking.tolist() == [0, 2, 1]
    assert scores[0] == (18 + 2 * 6.75) / 4


def test_ranking_metrics_exclude_history_and_require_relevant_targets(tmp_path: Path) -> None:
    observed_path = tmp_path / "observed.parquet"
    target_path = tmp_path / "target.parquet"
    pd.DataFrame(
        {
            "user_index": [0, 0, 1, 1],
            "item_index": [0, 1, 0, 2],
            "rating": [10, 9, 8, 7],
        }
    ).to_parquet(observed_path, index=False)
    pd.DataFrame(
        {
            "user_index": [0, 0, 1],
            "item_index": [2, 3, 4],
            "rating": [9, 4, 9],
        }
    ).to_parquet(target_path, index=False)
    observed = load_interactions_csr(observed_path, users=2, batch_size=2)
    targets = load_interactions_csr(target_path, users=2, batch_size=2)

    metrics = evaluate_global_ranking(
        ranking=np.asarray([0, 1, 2, 3], dtype=np.int32),
        observed=observed,
        targets=targets,
        additional_exclusions=None,
        observed_count=2,
        relevance_threshold=8,
        ranking_k=10,
    )

    assert metrics["eligible_users"] == 1
    assert metrics["recall_at_10"] == 1.0
    assert metrics["ndcg_at_10"] == 1.0
