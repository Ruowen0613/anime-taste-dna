from fastapi.testclient import TestClient

from app.features.anime.client import AniListError, get_anilist_client
from app.features.anime.schemas import AnimeSearchResult
from app.main import app


class FakeAniListClient:
    async def search(self, query: str, limit: int) -> list[AnimeSearchResult]:
        assert query == "Fullmetal"
        assert limit == 5
        return [
            AnimeSearchResult(
                mal_id=5114,
                anilist_id=5114,
                title="Fullmetal Alchemist: Brotherhood",
                cover_image_url="https://example.com/cover.jpg",
                year=2009,
                format="TV",
                genres=["Action", "Adventure"],
            )
        ]


def test_search_returns_normalized_anime() -> None:
    app.dependency_overrides[get_anilist_client] = lambda: FakeAniListClient()
    try:
        response = TestClient(app).get("/api/v1/anime/search?q=Fullmetal&limit=5")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["items"][0]["mal_id"] == 5114
    assert response.json()["items"][0]["genres"] == ["Action", "Adventure"]


def test_search_rejects_blank_query() -> None:
    response = TestClient(app).get("/api/v1/anime/search?q=%20%20")
    assert response.status_code == 422


class FailingAniListClient:
    async def search(self, query: str, limit: int) -> list[AnimeSearchResult]:
        raise AniListError("AniList search is temporarily unavailable")


def test_search_converts_upstream_failure_to_bad_gateway() -> None:
    app.dependency_overrides[get_anilist_client] = lambda: FailingAniListClient()
    try:
        response = TestClient(app).get("/api/v1/anime/search?q=Naruto")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {"detail": "AniList search is temporarily unavailable"}
