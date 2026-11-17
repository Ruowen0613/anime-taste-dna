import { API_BASE_URL } from '../../api/health'
import type { AnimeSearchResult } from '../discover/types'

type AnimeMetadataResponse = { items: AnimeSearchResult[] }

export async function getAnimeMetadata(malIds: number[], signal?: AbortSignal): Promise<AnimeSearchResult[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/anime/metadata`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mal_ids: malIds }),
    signal,
  })
  if (!response.ok) throw new Error('Could not refresh anime metadata.')
  return ((await response.json()) as AnimeMetadataResponse).items
}
