import html
import re
from functools import lru_cache
from typing import Any

import httpx

from app.features.anime.cache import TtlCache
from app.features.anime.schemas import AnimeDetails, AnimeReview, AnimeSearchResult, AnimeTag

ANILIST_API_URL = "https://graphql.anilist.co"
SEARCH_QUERY = """
query SearchAnime($search: String!, $perPage: Int!, $sort: [MediaSort!]!) {
  Page(page: 1, perPage: $perPage) {
    media(type: ANIME, search: $search, sort: $sort) {
      id
      idMal
      title { romaji english native }
      description(asHtml: false)
      coverImage { large }
      seasonYear
      format
      genres
      tags { name rank category isAdult isGeneralSpoiler isMediaSpoiler }
      averageScore
      popularity
    }
  }
}
"""
DETAIL_QUERY = """
query AnimeDetails($id: Int!) {
  Media(id: $id, type: ANIME) {
    id
    idMal
    title { romaji english native }
    description(asHtml: false)
    averageScore
    episodes
    duration
    status
    source
    bannerImage
    coverImage { extraLarge large }
    seasonYear
    format
    genres
    tags { name rank category isAdult isGeneralSpoiler isMediaSpoiler }
    siteUrl
    reviews(page: 1, perPage: 3, sort: RATING_DESC) {
      nodes {
        id
        summary
        rating
        ratingAmount
        user { name avatar { large } }
      }
    }
  }
}
"""
MAL_IDS_QUERY = """
query AnimeByMalIds($malIds: [Int], $perPage: Int!) {
  Page(page: 1, perPage: $perPage) {
    media(type: ANIME, idMal_in: $malIds) {
      id
      idMal
      title { romaji english native }
      description(asHtml: false)
      coverImage { large }
      seasonYear
      format
      genres
      tags { name rank category isAdult isGeneralSpoiler isMediaSpoiler }
      averageScore
      popularity
    }
  }
}
"""
SEASON_RELATIONS_QUERY = """
query AnimeSeasonRelations($malIds: [Int], $perPage: Int!) {
  Page(page: 1, perPage: $perPage) {
    media(type: ANIME, idMal_in: $malIds) {
      idMal
      relations {
        edges {
          relationType
          node { type idMal }
        }
      }
    }
  }
}
"""

METADATA_CACHE_TTL_SECONDS = 6 * 60 * 60
RELATION_CACHE_TTL_SECONDS = 24 * 60 * 60
ANILIST_CACHE_MAX_ENTRIES = 5_000


class AniListError(RuntimeError):
    """Raised when AniList cannot provide a valid search response."""


class AniListClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._metadata_cache = TtlCache[int, AnimeSearchResult](
            ttl_seconds=METADATA_CACHE_TTL_SECONDS,
            max_entries=ANILIST_CACHE_MAX_ENTRIES,
        )
        self._season_relation_cache = TtlCache[int, set[int]](
            ttl_seconds=RELATION_CACHE_TTL_SECONDS,
            max_entries=ANILIST_CACHE_MAX_ENTRIES,
        )

    async def search(self, query: str, limit: int, sort: str) -> list[AnimeSearchResult]:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=8.0)
        try:
            response = await client.post(
                ANILIST_API_URL,
                json={
                    "query": SEARCH_QUERY,
                    "variables": {"search": query, "perPage": limit, "sort": [sort]},
                },
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AniListError("AniList search is temporarily unavailable") from exc
        finally:
            if owns_client:
                await client.aclose()

        if payload.get("errors"):
            raise AniListError("AniList rejected the search request")

        try:
            media = payload["data"]["Page"]["media"]
            return [self._parse_item(item) for item in media]
        except (KeyError, TypeError, AttributeError) as exc:
            raise AniListError("AniList returned an invalid response") from exc

    async def get_details(self, anilist_id: int) -> AnimeDetails | None:
        payload = await self._request(DETAIL_QUERY, {"id": anilist_id})
        try:
            item = payload["data"]["Media"]
            if item is None:
                return None
            base = self._parse_item(item)
            reviews = item.get("reviews", {}).get("nodes") or []
            return AnimeDetails(
                **base.model_dump(),
                description=self._clean_text(item.get("description")),
                episodes=item.get("episodes"),
                duration_minutes=item.get("duration"),
                status=item.get("status"),
                source=item.get("source"),
                banner_image_url=item.get("bannerImage"),
                anilist_url=item["siteUrl"],
                reviews=[self._parse_review(review) for review in reviews],
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise AniListError("AniList returned an invalid response") from exc

    async def get_by_mal_ids(self, mal_ids: list[int]) -> dict[int, AnimeSearchResult]:
        unique_ids = list(dict.fromkeys(mal_ids))
        result: dict[int, AnimeSearchResult] = {}
        missing = []
        for mal_id in unique_ids:
            cached = self._metadata_cache.get(mal_id)
            if cached is None:
                missing.append(mal_id)
            else:
                result[mal_id] = cached
        for start in range(0, len(missing), 50):
            batch = missing[start : start + 50]
            payload = await self._request(
                MAL_IDS_QUERY,
                {"malIds": batch, "perPage": len(batch)},
            )
            try:
                media = payload["data"]["Page"]["media"]
                for item in media:
                    parsed = self._parse_item(item)
                    if parsed.mal_id is None:
                        continue
                    result[parsed.mal_id] = parsed
                    self._metadata_cache.set(parsed.mal_id, parsed)
            except (KeyError, TypeError, AttributeError) as exc:
                raise AniListError("AniList returned invalid recommendation metadata") from exc
        return result

    async def get_season_relations(self, mal_ids: list[int]) -> dict[int, set[int]]:
        unique_ids = list(dict.fromkeys(mal_ids))
        result: dict[int, set[int]] = {}
        missing = []
        for mal_id in unique_ids:
            cached = self._season_relation_cache.get(mal_id)
            if cached is None:
                missing.append(mal_id)
            else:
                result[mal_id] = set(cached)
        for start in range(0, len(missing), 50):
            batch = missing[start : start + 50]
            payload = await self._request(
                SEASON_RELATIONS_QUERY,
                {"malIds": batch, "perPage": len(batch)},
            )
            try:
                media = payload["data"]["Page"]["media"]
                batch_relations = {mal_id: set() for mal_id in batch}
                for item in media:
                    mal_id = item.get("idMal")
                    if mal_id is None:
                        continue
                    neighbors = batch_relations.setdefault(mal_id, set())
                    for edge in (item.get("relations") or {}).get("edges") or []:
                        node = edge.get("node") or {}
                        related_mal_id = node.get("idMal")
                        if (
                            edge.get("relationType") in {"PREQUEL", "SEQUEL"}
                            and node.get("type") == "ANIME"
                            and related_mal_id is not None
                        ):
                            neighbors.add(int(related_mal_id))
                for mal_id, neighbors in batch_relations.items():
                    self._season_relation_cache.set(mal_id, set(neighbors))
                    result[mal_id] = neighbors
            except (KeyError, TypeError, AttributeError) as exc:
                raise AniListError("AniList returned invalid season relations") from exc
        return {mal_id: set(result.get(mal_id, set())) for mal_id in unique_ids}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _request(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=8.0)
        try:
            response = await client.post(
                ANILIST_API_URL,
                json={"query": query, "variables": variables},
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AniListError("AniList is temporarily unavailable") from exc
        finally:
            if owns_client:
                await client.aclose()
        if not isinstance(payload, dict) or payload.get("errors"):
            raise AniListError("AniList rejected the request")
        return payload

    @staticmethod
    def _parse_item(item: dict[str, Any]) -> AnimeSearchResult:
        titles = item.get("title") or {}
        title = titles.get("english") or titles.get("romaji") or titles.get("native")
        if not title:
            raise AniListError("AniList returned an item without a title")
        cover = item.get("coverImage") or {}
        return AnimeSearchResult(
            mal_id=item.get("idMal"),
            anilist_id=item["id"],
            title=title,
            cover_image_url=cover.get("extraLarge") or cover.get("large"),
            year=item.get("seasonYear"),
            format=item.get("format"),
            genres=item.get("genres") or [],
            tags=[
                AnimeTag(
                    name=tag["name"],
                    rank=tag.get("rank") or 0,
                    category=tag.get("category") or "",
                    is_adult=tag.get("isAdult") or False,
                    is_general_spoiler=tag.get("isGeneralSpoiler") or False,
                    is_media_spoiler=tag.get("isMediaSpoiler") or False,
                )
                for tag in item.get("tags") or []
            ],
            average_score=item.get("averageScore"),
            popularity=item.get("popularity") or 0,
            summary=AniListClient._clean_text(item.get("description")),
        )

    @staticmethod
    def _parse_review(review: dict[str, Any]) -> AnimeReview:
        user = review.get("user") or {}
        avatar = user.get("avatar") or {}
        return AnimeReview(
            id=review["id"],
            summary=review.get("summary") or "",
            helpful_count=review.get("rating") or 0,
            rating_count=review.get("ratingAmount") or 0,
            user_name=user.get("name") or "AniList user",
            user_avatar_url=avatar.get("large"),
        )

    @staticmethod
    def _clean_text(value: str | None) -> str | None:
        if not value:
            return None
        with_line_breaks = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
        without_tags = re.sub(r"<[^>]+>", "", with_line_breaks)
        return html.unescape(without_tags).strip()


@lru_cache
def get_anilist_client() -> AniListClient:
    http_client = httpx.AsyncClient(
        timeout=8.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    return AniListClient(http_client)


async def close_anilist_client() -> None:
    if get_anilist_client.cache_info().currsize:
        await get_anilist_client().aclose()
        get_anilist_client.cache_clear()
