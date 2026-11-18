from fastapi.testclient import TestClient

from app.features.anime.client import get_anilist_client
from app.features.anime.schemas import AnimeSearchResult, AnimeTag
from app.features.taste_dna.schemas import TasteRating
from app.features.taste_dna.service import build_taste_dna
from app.main import app


def tag(name: str, category: str, rank: int = 100) -> AnimeTag:
    return AnimeTag(name=name, category=category, rank=rank)


def test_service_builds_fixed_genres_and_ranked_tag_groups() -> None:
    result = build_taste_dna(
        [
            TasteRating(
                mal_id=1,
                title="Action Ten",
                rating=10,
                genres=["Action"],
                tags=[tag("Survival", "Theme-Other", 90)],
            ),
            TasteRating(
                mal_id=2,
                title="Action Nine",
                rating=9,
                genres=["Action"],
                tags=[tag("Post-Apocalyptic", "Setting-Universe", 80)],
            ),
            TasteRating(
                mal_id=3,
                title="Drama Ten",
                rating=10,
                genres=["Drama"],
                tags=[tag("Ensemble Cast", "Cast-Main Cast", 90)],
            ),
            TasteRating(mal_id=4, title="Drama Two", rating=2, genres=["Drama"]),
            TasteRating(mal_id=5, title="Unknown", rating=8, genres=[]),
        ]
    )

    assert result.method_version == "metadata-affinity-v2"
    assert result.mapped_ratings == 4
    assert result.metadata_status == "partial"
    assert result.unmapped_mal_ids == [5]
    assert [dimension.label for dimension in result.genre_dimensions] == [
        "Action",
        "Comedy",
        "Drama",
        "Romance",
        "Fantasy",
        "Sci-Fi",
        "Slice of Life",
        "Psychological",
    ]
    assert result.genre_dimensions[0].score == 72.2
    assert result.genre_dimensions[1].score == 50.0
    assert result.genre_dimensions[1].support == 0
    assert result.genre_dimensions[1].average_rating is None
    assert [item.title for item in result.genre_dimensions[0].contributing_titles] == [
        "Action Ten",
        "Action Nine",
    ]
    assert result.story_themes[0].label == "Survival"
    assert result.worlds_and_settings[0].label == "Post-Apocalyptic"
    assert result.characters[0].label == "Ensemble Cast"


class FakeMetadataClient:
    async def get_by_mal_ids(self, mal_ids: list[int]) -> dict[int, AnimeSearchResult]:
        assert mal_ids == [1, 2, 3, 4, 5]
        return {
            5: AnimeSearchResult(
                mal_id=5,
                anilist_id=105,
                title="Enriched Five",
                genres=["Fantasy"],
                tags=[tag("Isekai", "Theme-Fantasy", 90)],
            )
        }


def test_endpoint_enriches_ratings_missing_local_genres() -> None:
    app.dependency_overrides[get_anilist_client] = lambda: FakeMetadataClient()
    try:
        response = TestClient(app).post(
            "/api/v1/taste-dna",
            json={
                "ratings": [
                    {"mal_id": 1, "title": "One", "rating": 10, "genres": ["Action"]},
                    {"mal_id": 2, "title": "Two", "rating": 8, "genres": ["Drama"]},
                    {"mal_id": 3, "title": "Three", "rating": 7, "genres": ["Comedy"]},
                    {"mal_id": 4, "title": "Four", "rating": 6, "genres": ["Romance"]},
                    {"mal_id": 5, "title": "Five", "rating": 9, "genres": []},
                ]
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata_status"] == "complete"
    assert payload["mapped_ratings"] == 5
    assert any(
        dimension["label"] == "Fantasy"
        for dimension in payload["genre_dimensions"]
    )
    assert payload["story_themes"][0]["label"] == "Isekai"


def test_endpoint_requires_five_ratings() -> None:
    response = TestClient(app).post(
        "/api/v1/taste-dna",
        json={"ratings": [{"mal_id": 1, "title": "One", "rating": 10, "genres": []}]},
    )
    assert response.status_code == 422


def test_endpoint_rejects_duplicate_ratings() -> None:
    ratings = [
        {"mal_id": mal_id, "title": str(mal_id), "rating": 8, "genres": ["Action"]}
        for mal_id in [1, 2, 3, 4, 4]
    ]
    response = TestClient(app).post("/api/v1/taste-dna", json={"ratings": ratings})
    assert response.status_code == 422
