from typing import Literal

from pydantic import BaseModel, Field


class PopularPick(BaseModel):
    mal_id: int
    anilist_id: int | None = None
    title: str
    cover_image_url: str | None = None
    year: int | None = None
    format: str | None = None
    genres: list[str] = Field(default_factory=list)
    average_score: int | None = None
    popularity: int = 0
    model_interactions: int


class PopularPicksResponse(BaseModel):
    model_version: str
    metadata_status: Literal["complete", "partial", "unavailable"]
    items: list[PopularPick]
