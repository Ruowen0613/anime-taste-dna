import numpy as np

from anime_taste_dna.evaluation.candidate_reranker_validation import _normalize_rows


def test_normalize_rows_keeps_zero_profile_finite() -> None:
    values = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = _normalize_rows(values)

    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [0.0, 0.0])
