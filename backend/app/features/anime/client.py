from typing import Any

import httpx

from app.features.anime.schemas import AnimeSearchResult

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
            cover_image_url=cover.get("large"),
            year=item.get("seasonYear"),
            format=item.get("format"),
            genres=item.get("genres") or [],
        )


def get_anilist_client() -> AniListClient:
    return AniListClient()
