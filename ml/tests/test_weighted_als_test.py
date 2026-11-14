import pytest

from anime_taste_dna.evaluation.weighted_als_test import _mean_metrics


def test_mean_metrics_macro_averages_observation_levels() -> None:
    recall, ndcg = _mean_metrics(
        {
            "5": {"recall_at_10": 0.1, "ndcg_at_10": 0.2},
            "10": {"recall_at_10": 0.3, "ndcg_at_10": 0.4},
        }
    )

    assert recall == pytest.approx(0.2)
    assert ndcg == pytest.approx(0.3)
