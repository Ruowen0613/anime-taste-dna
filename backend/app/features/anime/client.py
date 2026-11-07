import html
import re
from typing import Any

import httpx

from app.features.anime.schemas import AnimeDetails, AnimeReview, AnimeSearchResult

ANILIST_API_URL = "https://graphql.anilist.co"
SEARCH_QUERY = """
query SearchAnime($search: String!, $perPage: Int!) {
  Page(page: 1, perPage: $perPage) {
    media(type: ANIME, search: $search, sort: SEARCH_MATCH) {
      id
      idMal
      title { romaji english native }
      coverImage { large }
      seasonYear
      format
      genres
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


class AniListError(RuntimeError):
    """Raised when AniList cannot provide a valid search response."""


class AniListClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def search(self, query: str, limit: int) -> list[AnimeSearchResult]:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=8.0)
        try:
            response = await client.post(
                ANILIST_API_URL,
                json={"query": SEARCH_QUERY, "variables": {"search": query, "perPage": limit}},
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
                average_score=item.get("averageScore"),
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


def get_anilist_client() -> AniListClient:
    return AniListClient()
