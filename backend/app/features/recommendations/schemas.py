from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class UserRating(BaseModel):
    mal_id: int = Field(gt=0)
    rating: int = Field(ge=1, le=10)


class PairwisePreference(BaseModel):
    most_mal_id: int = Field(gt=0)
    least_mal_id: int = Field(gt=0)

    @model_validator(mode="after")
    def choices_must_differ(self) -> PairwisePreference:
        if self.most_mal_id == self.least_mal_id:
            raise ValueError("most_mal_id and least_mal_id must differ")
        return self


class RecommendationRequest(BaseModel):
    ratings: list[UserRating] = Field(min_length=10, max_length=100)
    preferences: list[PairwisePreference] = Field(default_factory=list, max_length=20)
    excluded_mal_ids: list[int] = Field(default_factory=list, max_length=500)
    not_interested_mal_ids: list[int] = Field(default_factory=list, max_length=100)
    limit: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def ratings_must_have_unique_mal_ids(self) -> RecommendationRequest:
        mal_ids = [rating.mal_id for rating in self.ratings]
        if len(mal_ids) != len(set(mal_ids)):
            raise ValueError("ratings must contain unique mal_id values")
        if any(mal_id <= 0 for mal_id in self.excluded_mal_ids):
            raise ValueError("excluded_mal_ids must contain positive values")
        if len(self.excluded_mal_ids) != len(set(self.excluded_mal_ids)):
            raise ValueError("excluded_mal_ids must contain unique values")
        if any(mal_id <= 0 for mal_id in self.not_interested_mal_ids):
            raise ValueError("not_interested_mal_ids must contain positive values")
        if len(self.not_interested_mal_ids) != len(set(self.not_interested_mal_ids)):
            raise ValueError("not_interested_mal_ids must contain unique values")
        return self


class RecommendationItem(BaseModel):
    mal_id: int
    title: str
    rank: int
    predicted_rating: float | None = None
    anilist_id: int | None = None
    cover_image_url: str | None = None
    year: int | None = None
    format: str | None = None
    genres: list[str] = Field(default_factory=list)
    average_score: int | None = None


class RecommendationResponse(BaseModel):
    model_version: str
    mapped_ratings: int
    mapped_preferences: int = 0
    pairwise_refinement_active: bool = False
    unmapped_mal_ids: list[int]
    metadata_status: Literal["complete", "partial", "unavailable"] = "unavailable"
    items: list[RecommendationItem]
