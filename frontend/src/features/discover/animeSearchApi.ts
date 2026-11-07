import { API_BASE_URL } from '../../api/health'
import type { AnimeSearchResult, AnimeSearchSort } from './types'

type AnimeSearchResponse = { items: AnimeSearchResult[] }

export async function searchAnime(query: string, sort: AnimeSearchSort, signal?: AbortSignal): Promise<AnimeSearchResult[]> {
  const params = new URLSearchParams({ q: query, limit: '10', sort })
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 10_000)
  const abortRequest = () => controller.abort()
  signal?.addEventListener('abort', abortRequest, { once: true })

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/anime/search?${params}`, { signal: controller.signal })
    if (!response.ok) {
      throw new Error(response.status === 502 ? 'Anime search is temporarily unavailable.' : 'Search failed. Please try again.')
    }
    return ((await response.json()) as AnimeSearchResponse).items
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      if (signal?.aborted) throw error
      throw new Error('Search timed out. Check that the API is running on port 8000.')
    }
    if (error instanceof TypeError) {
      throw new Error('Cannot reach the API. Start FastAPI on port 8000 and try again.')
    }
    throw error
  } finally {
    window.clearTimeout(timeout)
    signal?.removeEventListener('abort', abortRequest)
  }
}
