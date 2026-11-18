from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.features.anime.schemas import AnimeTag


class TasteRating(BaseModel):
    mal_id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=300)
    rating: int = Field(ge=1, le=10)
    genres: list[str] = Field(default_factory=list, max_length=20)
    tags: list[AnimeTag] = Field(default_factory=list, max_length=100)


class TasteDnaRequest(BaseModel):
    ratings: list[TasteRating] = Field(min_length=5, max_length=100)

    @model_validator(mode="after")
    def ratings_must_have_unique_mal_ids(self) -> TasteDnaRequest:
        mal_ids = [rating.mal_id for rating in self.ratings]
        if len(mal_ids) != len(set(mal_ids)):
            raise ValueError("ratings must contain unique mal_id values")
        return self

class TasteContribution(BaseModel):
    mal_id: int
    title: str
    rating: int


class TasteDimension(BaseModel):
    label: str
    score: float = Field(ge=0, le=100)
    support: int = Field(ge=0)
    average_rating: float | None = Field(default=None, ge=1, le=10)
    contributing_titles: list[TasteContribution]


class TasteDnaResponse(BaseModel):
    method_version: str
    mapped_ratings: int
    total_ratings: int
    metadata_status: Literal["complete", "partial", "unavailable"]
    unmapped_mal_ids: list[int]
    genre_dimensions: list[TasteDimension]
    story_themes: list[TasteDimension]
    worlds_and_settings: list[TasteDimension]
    characters: list[TasteDimension]
