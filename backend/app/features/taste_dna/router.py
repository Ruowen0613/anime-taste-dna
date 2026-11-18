from typing import Annotated

from fastapi import APIRouter, Depends

from app.features.anime.client import AniListClient, AniListError, get_anilist_client
from app.features.taste_dna.schemas import TasteDnaRequest, TasteDnaResponse
from app.features.taste_dna.service import build_taste_dna

router = APIRouter(prefix="/api/v1/taste-dna", tags=["taste-dna"])


@router.post("", response_model=TasteDnaResponse)
async def get_taste_dna(
    request: TasteDnaRequest,
    client: Annotated[AniListClient, Depends(get_anilist_client)],
) -> TasteDnaResponse:
    metadata_ids = [rating.mal_id for rating in request.ratings]
    metadata = {}
    if metadata_ids:
        try:
            metadata = await client.get_by_mal_ids(metadata_ids)
        except AniListError:
            metadata = {}

    enriched_ratings = []
    for rating in request.ratings:
        anime = metadata.get(rating.mal_id)
        if anime is None:
            enriched_ratings.append(rating)
        else:
            enriched_ratings.append(
                rating.model_copy(
                    update={
                        "title": anime.title,
                        "genres": anime.genres,
                        "tags": anime.tags,
                    }
                )
            )
    return build_taste_dna(enriched_ratings)
