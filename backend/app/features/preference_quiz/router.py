from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.features.anime.client import AniListClient, AniListError, get_anilist_client
from app.features.preference_quiz.schemas import (
    QuizOption,
    QuizOptionsRequest,
    QuizOptionsResponse,
)
from app.features.preference_quiz.service import (
    PreferenceQuizService,
    get_preference_quiz_service,
)

router = APIRouter(prefix="/api/v1/preference-quiz", tags=["preference-quiz"])


def short_summary(summary: str | None, limit: int = 320) -> str | None:
    if summary is None or len(summary) <= limit:
        return summary
    shortened = summary[: limit - 1].rsplit(" ", 1)[0]
    return f"{shortened}…"


@router.post("/options", response_model=QuizOptionsResponse)
async def get_quiz_options(
    request: QuizOptionsRequest,
    service: Annotated[PreferenceQuizService, Depends(get_preference_quiz_service)],
    client: Annotated[AniListClient, Depends(get_anilist_client)],
) -> QuizOptionsResponse:
    candidate_ids = service.candidate_mal_ids(request.excluded_mal_ids)
    try:
        metadata = await client.get_by_mal_ids(candidate_ids)
    except AniListError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    candidates = list(metadata.values())
    selected = []
    covered_genres: set[str] = set()
    while candidates and len(selected) < 4:
        best = max(
            candidates,
            key=lambda anime: (len(set(anime.genres) - covered_genres), anime.popularity),
        )
        selected.append(best)
        covered_genres.update(best.genres)
        candidates.remove(best)
    if len(selected) < 4:
        raise HTTPException(status_code=503, detail="Not enough quiz options are available")
    return QuizOptionsResponse(
        items=[
            QuizOption(
                mal_id=anime.mal_id,
                anilist_id=anime.anilist_id,
                title=anime.title,
                cover_image_url=anime.cover_image_url,
                year=anime.year,
                format=anime.format,
                genres=anime.genres,
                summary=short_summary(anime.summary),
            )
            for anime in selected
            if anime.mal_id is not None
        ]
    )
