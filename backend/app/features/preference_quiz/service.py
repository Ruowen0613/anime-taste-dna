from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.features.recommendations.artifact import (
    WeightedAlsArtifact,
    get_weighted_als_artifact,
)


class PreferenceQuizService:
    def __init__(self, artifact: WeightedAlsArtifact) -> None:
        self.artifact = artifact

    def candidate_mal_ids(self, excluded_mal_ids: list[int], pool_size: int = 24) -> list[int]:
        excluded = set(excluded_mal_ids)
        supported = np.flatnonzero(self.artifact.supported)
        supported = supported[
            np.argsort(self.artifact.item_counts[supported], kind="stable")[::-1]
        ][: min(1000, len(supported))]
        available = np.asarray(
            [index for index in supported if int(self.artifact.mal_ids[index]) not in excluded],
            dtype=np.int64,
        )
        if not len(available):
            return []
        rng = np.random.default_rng()
        selected = rng.choice(available, size=min(pool_size, len(available)), replace=False)
        return [int(self.artifact.mal_ids[index]) for index in selected]


@lru_cache
def get_preference_quiz_service() -> PreferenceQuizService:
    return PreferenceQuizService(get_weighted_als_artifact())
