from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.features.recommendations.artifact import ArtifactError
from app.features.recommendations.schemas import RecommendationRequest, RecommendationResponse
from app.features.recommendations.service import (
    InsufficientMappedRatingsError,
    RecommendationService,
    get_recommendation_service,
)

router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])


def get_available_recommendation_service() -> RecommendationService:
    try:
        return get_recommendation_service()
    except ArtifactError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is unavailable",
        ) from error


@router.post("", response_model=RecommendationResponse)
def recommend(
    request: RecommendationRequest,
    service: Annotated[RecommendationService, Depends(get_available_recommendation_service)],
) -> RecommendationResponse:
    try:
        return service.recommend(request)
    except InsufficientMappedRatingsError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
