import asyncio
import json

import httpx

from app.features.anime.cache import TtlCache
from app.features.anime.client import AniListClient


def test_ttl_cache_expires_and_evicts_oldest_entry() -> None:
    now = [0.0]
    cache = TtlCache[int, str](ttl_seconds=10, max_entries=2, clock=lambda: now[0])
    cache.set(1, "one")
    cache.set(2, "two")
    cache.set(3, "three")

    assert cache.get(1) is None
    assert cache.get(2) == "two"
    now[0] = 11.0
    assert cache.get(2) is None


def test_anilist_metadata_and_relations_are_cached() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        variables = payload["variables"]
        if "AnimeByMalIds" in payload["query"]:
            calls.append("metadata")
            media = [
                {
                    "id": 1_000 + mal_id,
                    "idMal": mal_id,
                    "title": {"english": f"Anime {mal_id}"},
                    "description": None,
                    "coverImage": {"large": f"https://example.com/{mal_id}.jpg"},
                    "seasonYear": 2020,
                    "format": "TV",
                    "genres": ["Drama"],
                    "averageScore": 80,
                    "popularity": 1_000,
                }
                for mal_id in variables["malIds"]
            ]
        else:
            calls.append("relations")
            media = [
                {"idMal": mal_id, "relations": {"edges": []}}
                for mal_id in variables["malIds"]
            ]
        return httpx.Response(200, json={"data": {"Page": {"media": media}}})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = AniListClient(http_client)
            await client.get_by_mal_ids([1, 2])
            await client.get_by_mal_ids([1, 2])
            await client.get_season_relations([1, 2])
            await client.get_season_relations([1, 2])

    asyncio.run(scenario())

    assert calls == ["metadata", "relations"]
