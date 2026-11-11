from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.features.recommendations.artifact import (
    RecommendationArtifact,
    get_recommendation_artifact,
)
from app.features.recommendations.schemas import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


class InsufficientMappedRatingsError(ValueError):
    """Raised when too few request ratings belong to the model item universe."""


class RecommendationService:
    def __init__(self, artifact: RecommendationArtifact, minimum_mapped_ratings: int) -> None:
        self.artifact = artifact
        self.minimum_mapped_ratings = minimum_mapped_ratings
        self._item_by_mal_id = {
            int(mal_id): item_index for item_index, mal_id in enumerate(artifact.mal_ids)
        }

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        mapped: list[tuple[int, int]] = []
        unmapped_mal_ids: list[int] = []
        for user_rating in request.ratings:
            item_index = self._item_by_mal_id.get(user_rating.mal_id)
            if item_index is None or not self.artifact.supported[item_index]:
                unmapped_mal_ids.append(user_rating.mal_id)
            else:
                mapped.append((item_index, user_rating.rating))
        if len(mapped) < self.minimum_mapped_ratings:
            raise InsufficientMappedRatingsError(
                f"At least {self.minimum_mapped_ratings} ratings must map to "
                "model-supported anime; "
                f"received {len(mapped)}"
            )

        observed_items = np.asarray([item for item, _ in mapped], dtype=np.int32)
        observed_ratings = np.asarray([rating for _, rating in mapped], dtype=np.float32)
        user_factor, user_bias = self._fold_in(observed_items, observed_ratings)
        scores = self.artifact.item_factors @ user_factor
        scores += self.artifact.item_bias + self.artifact.global_mean + user_bias
        scores[~self.artifact.supported] = -np.inf
        scores[observed_items] = -np.inf
        candidates = np.flatnonzero(np.isfinite(scores))
        result_limit = min(request.limit, len(candidates))
        if result_limit:
            candidate_scores = scores[candidates]
            selected = np.argpartition(candidate_scores, -result_limit)[-result_limit:]
            selected_items = candidates[selected]
            selected_items = selected_items[
                np.lexsort((self.artifact.mal_ids[selected_items], -scores[selected_items]))
            ]
        else:
            selected_items = np.asarray([], dtype=np.int64)

        items = [
            RecommendationItem(
                mal_id=int(self.artifact.mal_ids[item_index]),
                title=str(self.artifact.titles[item_index]),
                rank=rank,
                predicted_rating=round(float(np.clip(scores[item_index], 1.0, 10.0)), 4),
            )
            for rank, item_index in enumerate(selected_items, start=1)
        ]
        return RecommendationResponse(
            model_version=self.artifact.model_version,
            mapped_ratings=len(mapped),
            unmapped_mal_ids=unmapped_mal_ids,
            items=items,
        )

    def _fold_in(
        self, observed_items: np.ndarray, observed_ratings: np.ndarray
    ) -> tuple[np.ndarray, float]:
        factors = self.artifact.item_factors[observed_items]
        design = np.concatenate(
            (factors, np.ones((len(factors), 1), dtype=np.float32)), axis=1
        )
        targets = (
            observed_ratings
            - self.artifact.global_mean
            - self.artifact.item_bias[observed_items]
        )
        penalty = np.eye(design.shape[1], dtype=np.float32)
        penalty *= self.artifact.fold_in_regularization
        parameters = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        return parameters[:-1], float(parameters[-1])


@lru_cache
def get_recommendation_service() -> RecommendationService:
    return RecommendationService(
        artifact=get_recommendation_artifact(),
        minimum_mapped_ratings=get_settings().minimum_mapped_ratings,
    )
