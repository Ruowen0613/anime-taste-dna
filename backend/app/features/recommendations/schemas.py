from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class UserRating(BaseModel):
    mal_id: int = Field(gt=0)
    rating: int = Field(ge=1, le=10)


class RecommendationRequest(BaseModel):
    ratings: list[UserRating] = Field(min_length=10, max_length=20)
    limit: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def ratings_must_have_unique_mal_ids(self) -> RecommendationRequest:
        mal_ids = [rating.mal_id for rating in self.ratings]
        if len(mal_ids) != len(set(mal_ids)):
            raise ValueError("ratings must contain unique mal_id values")
        return self


class RecommendationItem(BaseModel):
    mal_id: int
    title: str
    rank: int
    predicted_rating: float
    anilist_id: int | None = None
    cover_image_url: str | None = None
    year: int | None = None
    format: str | None = None
    genres: list[str] = Field(default_factory=list)
    average_score: int | None = None


class RecommendationResponse(BaseModel):
    model_version: str
    mapped_ratings: int
    unmapped_mal_ids: list[int]
    metadata_status: Literal["complete", "partial", "unavailable"] = "unavailable"
    items: list[RecommendationItem]
