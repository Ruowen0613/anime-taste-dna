from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.features.anime.client import AniListClient, AniListError, get_anilist_client
from app.features.anime.schemas import AnimeDetails, AnimeMetadataRequest, AnimeSearchResponse

router = APIRouter(prefix="/api/v1/anime", tags=["anime"])

SearchSort = Literal["popularity", "score", "relevance", "newest"]
ANILIST_SORTS: dict[SearchSort, str] = {
    "popularity": "POPULARITY_DESC",
    "score": "SCORE_DESC",
    "relevance": "SEARCH_MATCH",
    "newest": "START_DATE_DESC",
}


@router.get("/search", response_model=AnimeSearchResponse)
async def search_anime(
    q: Annotated[str, Query(min_length=1, max_length=100)],
    client: Annotated[AniListClient, Depends(get_anilist_client)],
    limit: Annotated[int, Query(ge=1, le=20)] = 10,
    sort: SearchSort = "popularity",
) -> AnimeSearchResponse:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Search query cannot be blank")
    try:
        items = await client.search(query, limit, ANILIST_SORTS[sort])
    except AniListError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AnimeSearchResponse(items=items)


@router.post("/metadata", response_model=AnimeSearchResponse)
async def get_anime_metadata(
    request: AnimeMetadataRequest,
    client: Annotated[AniListClient, Depends(get_anilist_client)],
) -> AnimeSearchResponse:
    mal_ids = list(dict.fromkeys(request.mal_ids))
    try:
        metadata = await client.get_by_mal_ids(mal_ids)
    except AniListError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AnimeSearchResponse(items=[metadata[mal_id] for mal_id in mal_ids if mal_id in metadata])


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
