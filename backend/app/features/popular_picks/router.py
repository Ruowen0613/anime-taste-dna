from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.features.anime.client import AniListClient, AniListError, get_anilist_client
from app.features.popular_picks.schemas import PopularPicksResponse
from app.features.popular_picks.service import PopularPicksService, get_popular_picks_service
from app.features.recommendations.artifact import ArtifactError

router = APIRouter(prefix="/api/v1/popular-picks", tags=["popular-picks"])


def get_available_popular_picks_service() -> PopularPicksService:
    try:
        return get_popular_picks_service()
    except ArtifactError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is unavailable",
        ) from error


@router.get("", response_model=PopularPicksResponse)
async def get_popular_picks(
    service: Annotated[PopularPicksService, Depends(get_available_popular_picks_service)],
    client: Annotated[AniListClient, Depends(get_anilist_client)],
) -> PopularPicksResponse:
    items = service.top_items()
    metadata = {}
    try:
        for start in range(0, len(items), 50):
            batch = items[start : start + 50]
            metadata.update(await client.get_by_mal_ids([item.mal_id for item in batch]))
    except AniListError:
        return PopularPicksResponse(
            model_version=service.artifact.model_version,
            metadata_status="unavailable",
            items=items,
        )

    enriched = []
    for item in items:
        anime = metadata.get(item.mal_id)
        if anime is None:
            enriched.append(item)
            continue
        enriched.append(
            item.model_copy(
                update={
                    "anilist_id": anime.anilist_id,
                    "title": anime.title,
                    "cover_image_url": anime.cover_image_url,
                    "year": anime.year,
                    "format": anime.format,
                    "genres": anime.genres,
                    "average_score": anime.average_score,
                    "popularity": anime.popularity,
                }
            )
        )
    status_value = "complete" if len(metadata) == len(items) else "partial"
    return PopularPicksResponse(
        model_version=service.artifact.model_version,
        metadata_status=status_value,
        items=enriched,
    )
