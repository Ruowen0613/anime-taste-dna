from pydantic import BaseModel, Field, PositiveInt


class AnimeTag(BaseModel):
    name: str
    rank: int = Field(ge=0, le=100)
    category: str
    is_adult: bool = False
    is_general_spoiler: bool = False
    is_media_spoiler: bool = False


class AnimeSearchResult(BaseModel):
    mal_id: int | None
    anilist_id: int
    title: str
    cover_image_url: str | None = None
    year: int | None = None
    format: str | None = None
    genres: list[str]
    tags: list[AnimeTag] = Field(default_factory=list)
    average_score: int | None = None
    popularity: int = 0
    summary: str | None = None


class AnimeSearchResponse(BaseModel):
    items: list[AnimeSearchResult]


class AnimeMetadataRequest(BaseModel):
    mal_ids: list[PositiveInt] = Field(min_length=1, max_length=100)


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
