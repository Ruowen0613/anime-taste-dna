from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.features.anime.client import AniListClient, AniListError, get_anilist_client
from app.features.anime.schemas import AnimeDetails, AnimeSearchResponse

router = APIRouter(prefix="/api/v1/anime", tags=["anime"])


@router.get("/search", response_model=AnimeSearchResponse)
async def search_anime(
    q: Annotated[str, Query(min_length=1, max_length=100)],
    client: Annotated[AniListClient, Depends(get_anilist_client)],
    limit: Annotated[int, Query(ge=1, le=20)] = 10,
) -> AnimeSearchResponse:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Search query cannot be blank")
    try:
        items = await client.search(query, limit)
    except AniListError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AnimeSearchResponse(items=items)


@router.get("/{anilist_id}", response_model=AnimeDetails)
async def get_anime_details(
    anilist_id: int,
    client: Annotated[AniListClient, Depends(get_anilist_client)],
) -> AnimeDetails:
    try:
        details = await client.get_details(anilist_id)
    except AniListError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if details is None:
        raise HTTPException(status_code=404, detail="Anime not found")
    return details
