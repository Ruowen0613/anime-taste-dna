from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.anilist import AniListClient, AniListError, get_anilist_client
from app.config import get_settings
from app.schemas import AnimeSearchResponse, HealthResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Online inference and metadata API for Anime Taste DNA.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(service="anime-taste-dna-api")


@app.get(
    "/api/v1/anime/search",
    response_model=AnimeSearchResponse,
    tags=["anime"],
)
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
