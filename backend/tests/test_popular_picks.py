import numpy as np
from fastapi.testclient import TestClient

from app.features.anime.client import get_anilist_client
from app.features.anime.schemas import AnimeSearchResult
from app.features.popular_picks.router import get_available_popular_picks_service
from app.features.popular_picks.service import PopularPicksService
from app.features.recommendations.artifact import SignedHybridArtifact
from app.main import app


def artifact() -> SignedHybridArtifact:
    return SignedHybridArtifact(
        item_factors=np.ones((4, 2), dtype=np.float32),
        item_counts=np.asarray([5, 20, 20, 100]),
        mal_ids=np.asarray([40, 30, 20, 10]),
        titles=np.asarray(["Forty", "Thirty", "Twenty", "Ten"]),
        supported=np.asarray([True, True, True, False]),
        alpha=20.0,
        fold_in_regularization=1.0,
        popularity_blend=2.0,
        positive_rating_threshold=8,
        signed_fold_in_regularization=10.0,
        signed_score_weight=1.0,
        popularity_decay_pivot=10,
        popularity_decay_exponent=1.0,
        model_version="test-model",
    )


def test_service_returns_supported_items_by_interactions_then_mal_id() -> None:
    items = PopularPicksService(artifact()).top_items(limit=3)

    assert [item.mal_id for item in items] == [20, 30, 40]
    assert [item.model_interactions for item in items] == [20, 20, 5]


class FakeMetadataClient:
    calls: list[list[int]] = []

    async def get_by_mal_ids(self, mal_ids: list[int]) -> dict[int, AnimeSearchResult]:
        self.calls.append(mal_ids)
        return {
            mal_id: AnimeSearchResult(
                mal_id=mal_id,
                anilist_id=1000 + mal_id,
                title=f"Anime {mal_id}",
                cover_image_url=f"https://example.com/{mal_id}.jpg",
                year=2020,
                format="TV",
                genres=["Action"],
                popularity=1000,
            )
            for mal_id in mal_ids
        }


def test_endpoint_enriches_model_supported_popular_items() -> None:
    client = FakeMetadataClient()
    app.dependency_overrides[get_available_popular_picks_service] = (
        lambda: PopularPicksService(artifact())
    )
    app.dependency_overrides[get_anilist_client] = lambda: client
    try:
        response = TestClient(app).get("/api/v1/popular-picks")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "test-model"
    assert payload["metadata_status"] == "complete"
    assert [item["mal_id"] for item in payload["items"]] == [20, 30, 40]
    assert payload["items"][0]["anilist_id"] == 1020
