from fastapi.testclient import TestClient

from app.features.anime.client import AniListError, get_anilist_client
from app.features.anime.schemas import AnimeDetails, AnimeReview, AnimeSearchResult
from app.main import app


class FakeAniListClient:
    async def search(self, query: str, limit: int, sort: str) -> list[AnimeSearchResult]:
        assert query == "Fullmetal"
        assert limit == 5
        assert sort == "POPULARITY_DESC"
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


def test_search_rejects_unknown_sort() -> None:
    response = TestClient(app).get("/api/v1/anime/search?q=Naruto&sort=trending")
    assert response.status_code == 422


class FailingAniListClient:
    async def search(self, query: str, limit: int, sort: str) -> list[AnimeSearchResult]:
        raise AniListError("AniList search is temporarily unavailable")


def test_search_converts_upstream_failure_to_bad_gateway() -> None:
    app.dependency_overrides[get_anilist_client] = lambda: FailingAniListClient()
    try:
        response = TestClient(app).get("/api/v1/anime/search?q=Naruto")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {"detail": "AniList search is temporarily unavailable"}


class FakeDetailsClient:
    async def get_details(self, anilist_id: int) -> AnimeDetails | None:
        assert anilist_id == 16498
        return AnimeDetails(
            mal_id=16498,
            anilist_id=16498,
            title="Attack on Titan",
            cover_image_url="https://example.com/cover.jpg",
            year=2013,
            format="TV",
            genres=["Action", "Drama"],
            description="Humanity fights for survival.",
            average_score=84,
            episodes=25,
            duration_minutes=24,
            status="FINISHED",
            source="MANGA",
            banner_image_url="https://example.com/banner.jpg",
            anilist_url="https://anilist.co/anime/16498",
            reviews=[
                AnimeReview(
                    id=1,
                    summary="A tense and memorable adaptation.",
                    helpful_count=18,
                    rating_count=20,
                    user_name="reviewer",
                )
            ],
        )


def test_details_returns_score_description_and_reviews() -> None:
    app.dependency_overrides[get_anilist_client] = lambda: FakeDetailsClient()
    try:
        response = TestClient(app).get("/api/v1/anime/16498")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["average_score"] == 84
    assert payload["description"] == "Humanity fights for survival."
    assert payload["reviews"][0]["user_name"] == "reviewer"


class MissingDetailsClient:
    async def get_details(self, anilist_id: int) -> None:
        return None


def test_details_returns_not_found() -> None:
    app.dependency_overrides[get_anilist_client] = lambda: MissingDetailsClient()
    try:
        response = TestClient(app).get("/api/v1/anime/999999999")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
