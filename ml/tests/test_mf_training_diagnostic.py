from anime_taste_dna.evaluation.mf_training_diagnostic import select_trial


def test_select_trial_prefers_validation_ndcg_then_recall() -> None:
    trials = [
        {
            "regularization": 10.0,
            "full_metrics": {"ndcg_at_10": 0.2, "recall_at_10": 0.4},
        },
        {
            "regularization": 100.0,
            "full_metrics": {"ndcg_at_10": 0.3, "recall_at_10": 0.2},
        },
    ]

    assert select_trial(trials)["regularization"] == 100.0
