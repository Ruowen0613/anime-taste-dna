import numpy as np
from fastapi.testclient import TestClient

from app.features.anime.client import AniListError, get_anilist_client
from app.features.anime.schemas import AnimeSearchResult
from app.features.recommendations.artifact import RecommendationArtifact
from app.features.recommendations.router import get_available_recommendation_service
from app.features.recommendations.service import RecommendationService
from app.main import app


def _service() -> RecommendationService:
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
    return RecommendationService(artifact=artifact, minimum_mapped_ratings=2)


class FakeRecommendationMetadataClient:
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
    async def get_by_mal_ids(self, mal_ids: list[int]) -> dict[int, AnimeSearchResult]:
        raise AniListError("AniList is unavailable")


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
