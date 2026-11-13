from fastapi.testclient import TestClient

from app.features.anime.client import get_anilist_client
from app.features.anime.schemas import AnimeSearchResult
from app.features.preference_quiz.service import get_preference_quiz_service
from app.main import app


class FakeQuizService:
    def candidate_mal_ids(self, excluded_mal_ids: list[int], pool_size: int = 24) -> list[int]:
        assert excluded_mal_ids == [99]
        assert pool_size == 24
        return [1, 2, 3, 4]


class FakeQuizMetadataClient:
    async def get_by_mal_ids(self, mal_ids: list[int]) -> dict[int, AnimeSearchResult]:
        return {
            mal_id: AnimeSearchResult(
                mal_id=mal_id,
                anilist_id=1000 + mal_id,
                title=f"Option {mal_id}",
                cover_image_url=f"https://example.com/{mal_id}.jpg",
                year=2020 + mal_id,
                format="TV",
                genres=[["Action"], ["Romance"], ["Comedy"], ["Mystery"]][mal_id - 1],
                popularity=100 - mal_id,
                summary=f"Short summary {mal_id}",
            )
            for mal_id in mal_ids
        }


def test_quiz_returns_four_model_supported_metadata_options() -> None:
    app.dependency_overrides[get_preference_quiz_service] = FakeQuizService
    app.dependency_overrides[get_anilist_client] = FakeQuizMetadataClient
    try:
        response = TestClient(app).post(
            "/api/v1/preference-quiz/options",
            json={"excluded_mal_ids": [99]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 4
    assert {item["genres"][0] for item in payload["items"]} == {
        "Action",
        "Romance",
        "Comedy",
        "Mystery",
    }
    assert payload["items"][0]["summary"].startswith("Short summary")
