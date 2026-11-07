from pydantic import BaseModel


class AnimeSearchResult(BaseModel):
    mal_id: int | None
    anilist_id: int
    title: str
    cover_image_url: str | None = None
    year: int | None = None
    format: str | None = None
    genres: list[str]


class AnimeSearchResponse(BaseModel):
    items: list[AnimeSearchResult]
