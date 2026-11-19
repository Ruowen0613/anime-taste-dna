from anime_taste_dna.data.resample_fixed_user_holdout import fixed_sample_positions


def test_fixed_sample_positions_require_and_partition_exact_count() -> None:
    assert fixed_sample_positions(99, 80, 20, 42) is None
    sampled = fixed_sample_positions(150, 80, 20, 42)
    assert sampled is not None
    history, target = sampled
    assert len(history) == 80
    assert len(target) == 20
    assert not set(history) & set(target)
