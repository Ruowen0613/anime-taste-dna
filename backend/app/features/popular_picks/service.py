from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.features.popular_picks.schemas import PopularPick
from app.features.recommendations.artifact import (
    SignedHybridArtifact,
    get_signed_hybrid_artifact,
)


class PopularPicksService:
    def __init__(self, artifact: SignedHybridArtifact) -> None:
        self.artifact = artifact

    def top_items(self, limit: int = 100) -> list[PopularPick]:
        supported = np.flatnonzero(self.artifact.supported)
        order = np.lexsort(
            (self.artifact.mal_ids[supported], -self.artifact.item_counts[supported])
        )
        selected = supported[order[: min(limit, len(supported))]]
        return [
            PopularPick(
                mal_id=int(self.artifact.mal_ids[index]),
                title=str(self.artifact.titles[index]),
                model_interactions=int(self.artifact.item_counts[index]),
            )
            for index in selected
        ]


@lru_cache
def get_popular_picks_service() -> PopularPicksService:
    return PopularPicksService(get_signed_hybrid_artifact())
