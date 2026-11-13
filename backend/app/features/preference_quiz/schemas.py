from pydantic import BaseModel, Field


class QuizOptionsRequest(BaseModel):
    excluded_mal_ids: list[int] = Field(default_factory=list, max_length=1000)


class QuizOption(BaseModel):
    mal_id: int
    anilist_id: int
    title: str
    cover_image_url: str | None = None
    year: int | None = None
    format: str | None = None
    genres: list[str] = Field(default_factory=list)
    summary: str | None = None


class QuizOptionsResponse(BaseModel):
    items: list[QuizOption]
