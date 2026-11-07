from pydantic import BaseModel


class AnimeSearchResult(BaseModel):
    mal_id: int | None
    anilist_id: int
    title: str
    cover_image_url: str | None = None
    year: int | None = None
    format: str | None = None
    genres: list[str]
    average_score: int | None = None
    popularity: int = 0


class AnimeSearchResponse(BaseModel):
    items: list[AnimeSearchResult]


class AnimeReview(BaseModel):
    id: int
    summary: str
    helpful_count: int
    rating_count: int
    user_name: str
    user_avatar_url: str | None = None


class AnimeDetails(AnimeSearchResult):
    description: str | None = None
    episodes: int | None = None
    duration_minutes: int | None = None
    status: str | None = None
    source: str | None = None
    banner_image_url: str | None = None
    anilist_url: str
    reviews: list[AnimeReview]
