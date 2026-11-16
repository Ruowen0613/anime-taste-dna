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


async def expand_season_family(
    client: AniListClient, seed_mal_ids: list[int]
) -> set[int]:
    discovered = set(seed_mal_ids)
    frontier = set(seed_mal_ids)
    while frontier and len(discovered) < 500:
        relations = await client.get_season_relations(sorted(frontier))
        neighbors = {related for values in relations.values() for related in values}
        frontier = neighbors - discovered
        discovered.update(frontier)
    return discovered


def deduplicate_seasons(
    response: RecommendationResponse,
    relations: dict[int, set[int]],
    limit: int,
) -> RecommendationResponse:
    parents: dict[int, int] = {}

    def find(item: int) -> int:
        parents.setdefault(item, item)
        if parents[item] != item:
            parents[item] = find(parents[item])
        return parents[item]

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for mal_id, neighbors in relations.items():
        for neighbor in neighbors:
            union(mal_id, neighbor)

    selected = []
    seen_families: set[int] = set()
    for item in response.items:
        family = find(item.mal_id)
        if family in seen_families:
            continue
        seen_families.add(family)
        selected.append(item.model_copy(update={"rank": len(selected) + 1}))
        if len(selected) == limit:
            break
    return response.model_copy(update={"items": selected})


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
    requested_limit = request.limit
    expanded_exclusions = set(request.excluded_mal_ids)
    if request.not_interested_mal_ids:
        try:
            expanded_exclusions.update(
                await expand_season_family(
                    anilist_client, request.not_interested_mal_ids
                )
            )
        except AniListError:
            expanded_exclusions.update(request.not_interested_mal_ids)
    candidate_request = request.model_copy(
        update={
            "excluded_mal_ids": sorted(expanded_exclusions),
            "limit": min(max(requested_limit * 5, requested_limit), 50),
        }
    )
    try:
        response = service.recommend(candidate_request)
    except InsufficientMappedRatingsError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    try:
        relations = await anilist_client.get_season_relations(
            [item.mal_id for item in response.items]
        )
        response = deduplicate_seasons(response, relations, requested_limit)
        metadata = await anilist_client.get_by_mal_ids(
            [item.mal_id for item in response.items]
        )
    except AniListError:
        response = response.model_copy(update={"items": response.items[:requested_limit]})
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
