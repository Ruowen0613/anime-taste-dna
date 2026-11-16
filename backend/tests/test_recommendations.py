from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.features.anime.client import AniListError, get_anilist_client
from app.features.anime.schemas import AnimeSearchResult
from app.features.recommendations.artifact import RecommendationArtifact, SignedHybridArtifact
from app.features.recommendations.router import (
    deduplicate_seasons,
    expand_season_family,
    get_available_recommendation_service,
)
from app.features.recommendations.schemas import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from app.features.recommendations.service import (
    RecommendationService,
    SignedHybridRecommendationService,
)
from app.main import app


def _service(preference_weight: float = 0.0) -> RecommendationService:
    artifact = RecommendationArtifact(
        item_factors=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32),
        item_bias=np.zeros(4, dtype=np.float32),
        global_mean=5.0,
        mal_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        titles=np.asarray(["One", "Two", "Three", "Four"]),
        supported=np.ones(4, dtype=np.bool_),
        fold_in_regularization=1.0,
        model_version="fixture-v1",
    )
    return RecommendationService(
        artifact=artifact,
        minimum_mapped_ratings=2,
        preference_weight=preference_weight,
    )


def _signed_hybrid_service() -> SignedHybridRecommendationService:
    artifact = SignedHybridArtifact(
        item_factors=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32
        ),
        item_counts=np.ones(4, dtype=np.int64),
        mal_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        titles=np.asarray(["One", "Two", "Three", "Four"]),
        supported=np.ones(4, dtype=np.bool_),
        alpha=10.0,
        fold_in_regularization=1.0,
        popularity_blend=0.0,
        positive_rating_threshold=8,
        signed_fold_in_regularization=1.0,
        signed_score_weight=0.5,
        popularity_decay_pivot=10,
        popularity_decay_exponent=1.0,
        model_version="signed-hybrid-fixture-v1",
    )
    return SignedHybridRecommendationService(artifact, minimum_mapped_ratings=2)


def test_signed_hybrid_uses_low_ratings_and_does_not_fake_predicted_ratings() -> None:
    ratings = [
        {"mal_id": mal_id, "rating": 10 if mal_id == 1 else 1}
        for mal_id in [1, 2, 100, 101, 102, 103, 104, 105, 106, 107]
    ]

    result = _signed_hybrid_service().recommend(
        RecommendationRequest(ratings=ratings, limit=2)
    )

    assert result.model_version == "signed-hybrid-fixture-v1"
    assert [item.mal_id for item in result.items] == [3, 4]
    assert all(item.predicted_rating is None for item in result.items)


def test_changing_high_rating_to_low_changes_signed_hybrid_ranking() -> None:
    service = _signed_hybrid_service()
    unmapped = [{"mal_id": mal_id, "rating": 5} for mal_id in range(100, 108)]
    high_first = service.recommend(
        RecommendationRequest(
            ratings=[
                {"mal_id": 1, "rating": 10},
                {"mal_id": 2, "rating": 1},
                *unmapped,
            ],
            limit=2,
        )
    )
    low_first = service.recommend(
        RecommendationRequest(
            ratings=[
                {"mal_id": 1, "rating": 1},
                {"mal_id": 2, "rating": 10},
                *unmapped,
            ],
            limit=2,
        )
    )

    assert [item.mal_id for item in high_first.items] == [3, 4]
    assert [item.mal_id for item in low_first.items] == [4, 3]


def test_popularity_blend_decays_after_rating_pivot() -> None:
    fixture = _signed_hybrid_service()
    service = SignedHybridRecommendationService(
        replace(fixture.artifact, popularity_blend=32.0), minimum_mapped_ratings=2
    )

    assert service._effective_popularity_blend(5) == 32.0
    assert service._effective_popularity_blend(10) == 32.0
    assert service._effective_popularity_blend(20) == 16.0
    assert service._effective_popularity_blend(80) == 4.0


def test_request_accepts_up_to_100_unique_ratings() -> None:
    request = RecommendationRequest(
        ratings=[{"mal_id": mal_id, "rating": 8} for mal_id in range(1, 101)]
    )

    assert len(request.ratings) == 100


def test_request_rejects_more_than_100_ratings() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(
            ratings=[{"mal_id": mal_id, "rating": 8} for mal_id in range(1, 102)]
        )


class FakeRecommendationMetadataClient:
    async def get_season_relations(self, mal_ids: list[int]) -> dict[int, set[int]]:
        return {mal_id: set() for mal_id in mal_ids}

    async def get_by_mal_ids(self, mal_ids: list[int]) -> dict[int, AnimeSearchResult]:
        assert mal_ids == [3, 4]
        return {
            mal_id: AnimeSearchResult(
                mal_id=mal_id,
                anilist_id=mal_id + 1000,
                title=f"Enriched {mal_id}",
                cover_image_url=f"https://example.com/{mal_id}.jpg",
                year=2020 + mal_id,
                format="TV",
                genres=["Drama"],
                average_score=80 + mal_id,
            )
            for mal_id in mal_ids
        }


class FailingRecommendationMetadataClient:
    async def get_season_relations(self, mal_ids: list[int]) -> dict[int, set[int]]:
        raise AniListError("AniList is unavailable")

    async def get_by_mal_ids(self, mal_ids: list[int]) -> dict[int, AnimeSearchResult]:
        raise AniListError("AniList is unavailable")


def test_deduplicate_seasons_keeps_highest_ranked_title_per_family() -> None:
    response = RecommendationResponse(
        model_version="fixture",
        mapped_ratings=10,
        unmapped_mal_ids=[],
        items=[
            RecommendationItem(mal_id=1, title="Season 1", rank=1),
            RecommendationItem(mal_id=2, title="Season 2", rank=2),
            RecommendationItem(mal_id=3, title="Another show", rank=3),
        ],
    )

    deduplicated = deduplicate_seasons(
        response,
        {1: {2}, 2: {1}, 3: set()},
        limit=10,
    )

    assert [item.mal_id for item in deduplicated.items] == [1, 3]
    assert [item.rank for item in deduplicated.items] == [1, 2]


class FakeSeasonGraphClient:
    async def get_season_relations(self, mal_ids: list[int]) -> dict[int, set[int]]:
        graph = {1: {2}, 2: {1, 3}, 3: {2}}
        return {mal_id: graph.get(mal_id, set()) for mal_id in mal_ids}


def test_not_interested_expands_across_entire_season_chain() -> None:
    import asyncio

    expanded = asyncio.run(expand_season_family(FakeSeasonGraphClient(), [1]))

    assert expanded == {1, 2, 3}


def test_recommendations_fold_in_user_and_exclude_rated_items() -> None:
    service = _service()
    app.dependency_overrides[get_available_recommendation_service] = lambda: service
    app.dependency_overrides[get_anilist_client] = FakeRecommendationMetadataClient
    ratings = [
        {"mal_id": mal_id, "rating": 10 if mal_id == 1 else 1}
        for mal_id in [1, 2, 100, 101, 102, 103, 104, 105, 106, 107]
    ]
    try:
        response = TestClient(app).post(
            "/api/v1/recommendations", json={"ratings": ratings, "limit": 2}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "fixture-v1"
    assert payload["mapped_ratings"] == 2
    assert payload["metadata_status"] == "complete"
    assert payload["unmapped_mal_ids"] == [100, 101, 102, 103, 104, 105, 106, 107]
    assert [item["mal_id"] for item in payload["items"]] == [3, 4]
    assert payload["items"][0]["title"] == "Enriched 3"
    assert payload["items"][0]["cover_image_url"] == "https://example.com/3.jpg"
    assert payload["items"][0]["genres"] == ["Drama"]
    assert all(item["mal_id"] not in {1, 2} for item in payload["items"])


def test_recommendations_reject_duplicate_mal_ids() -> None:
    app.dependency_overrides[get_available_recommendation_service] = _service
    app.dependency_overrides[get_anilist_client] = FakeRecommendationMetadataClient
    ratings = [{"mal_id": 1, "rating": 8} for _ in range(10)]
    try:
        response = TestClient(app).post(
            "/api/v1/recommendations", json={"ratings": ratings}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_recommendations_exclude_archived_mal_ids_without_using_them_for_fold_in() -> None:
    ratings = [
        {"mal_id": mal_id, "rating": 10 if mal_id == 1 else 1}
        for mal_id in [1, 2, 100, 101, 102, 103, 104, 105, 106, 107]
    ]

    result = _service().recommend(
        RecommendationRequest(ratings=ratings, excluded_mal_ids=[3], limit=2)
    )

    assert result.mapped_ratings == 2
    assert [item.mal_id for item in result.items] == [4]


def test_pairwise_preference_refines_user_vector_without_becoming_a_rating() -> None:
    ratings = [
        {"mal_id": mal_id, "rating": 10 if mal_id == 1 else 1}
        for mal_id in [1, 2, 100, 101, 102, 103, 104, 105, 106, 107]
    ]
    baseline = _service().recommend(RecommendationRequest(ratings=ratings, limit=2))
    refined = _service(preference_weight=1.0).recommend(
        RecommendationRequest(
            ratings=ratings,
            preferences=[{"most_mal_id": 4, "least_mal_id": 3}],
            limit=2,
        )
    )

    baseline_scores = {item.mal_id: item.predicted_rating for item in baseline.items}
    refined_scores = {item.mal_id: item.predicted_rating for item in refined.items}
    assert refined.mapped_ratings == baseline.mapped_ratings == 2
    assert refined.mapped_preferences == 1
    assert refined.pairwise_refinement_active is True
    assert refined_scores[4] - refined_scores[3] > baseline_scores[4] - baseline_scores[3]


def test_recommendations_require_enough_model_mappings() -> None:
    service = _service()
    app.dependency_overrides[get_available_recommendation_service] = lambda: service
    app.dependency_overrides[get_anilist_client] = FakeRecommendationMetadataClient
    ratings = [{"mal_id": mal_id, "rating": 8} for mal_id in range(100, 110)]
    try:
        response = TestClient(app).post(
            "/api/v1/recommendations", json={"ratings": ratings}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert "At least 2 ratings" in response.json()["detail"]


def test_recommendations_survive_metadata_outage() -> None:
    service = _service()
    app.dependency_overrides[get_available_recommendation_service] = lambda: service
    app.dependency_overrides[get_anilist_client] = FailingRecommendationMetadataClient
    ratings = [
        {"mal_id": mal_id, "rating": 10 if mal_id == 1 else 1}
        for mal_id in [1, 2, 100, 101, 102, 103, 104, 105, 106, 107]
    ]
    try:
        response = TestClient(app).post(
            "/api/v1/recommendations", json={"ratings": ratings, "limit": 2}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata_status"] == "unavailable"
    assert [item["title"] for item in payload["items"]] == ["Three", "Four"]
