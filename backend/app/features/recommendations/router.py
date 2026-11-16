from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.features.anime.client import AniListClient, AniListError, get_anilist_client
from app.features.recommendations.artifact import ArtifactError
from app.features.recommendations.schemas import RecommendationRequest, RecommendationResponse
from app.features.recommendations.service import (
    InsufficientMappedRatingsError,
    RecommendationService,
    SignedHybridRecommendationService,
    get_signed_hybrid_recommendation_service,
)

router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])


def get_available_recommendation_service() -> SignedHybridRecommendationService:
    try:
        return get_signed_hybrid_recommendation_service()
    except ArtifactError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is unavailable",
        ) from error


@router.post("", response_model=RecommendationResponse)
async def recommend(
    request: RecommendationRequest,
    service: Annotated[
        RecommendationService | SignedHybridRecommendationService,
        Depends(get_available_recommendation_service),
    ],
    anilist_client: Annotated[AniListClient, Depends(get_anilist_client)],
) -> RecommendationResponse:
    try:
        response = service.recommend(request)
    except InsufficientMappedRatingsError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    try:
        metadata = await anilist_client.get_by_mal_ids(
            [item.mal_id for item in response.items]
        )
    except AniListError:
        return response.model_copy(update={"metadata_status": "unavailable"})

    enriched_items = []
    for item in response.items:
        anime = metadata.get(item.mal_id)
        if anime is None:
            enriched_items.append(item)
            continue
        enriched_items.append(
            item.model_copy(
                update={
                    "title": anime.title,
                    "anilist_id": anime.anilist_id,
                    "cover_image_url": anime.cover_image_url,
                    "year": anime.year,
                    "format": anime.format,
                    "genres": anime.genres,
                    "average_score": anime.average_score,
                }
            )
        )
    metadata_status = "complete" if len(metadata) == len(response.items) else "partial"
    return response.model_copy(
        update={"metadata_status": metadata_status, "items": enriched_items}
    )
