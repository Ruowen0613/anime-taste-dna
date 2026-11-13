import { API_BASE_URL } from '../../api/health'
import type { RatedAnime } from '../discover/types'
import type { RecommendationResponse } from './types'

export async function getRecommendations(ratings: RatedAnime[], signal?: AbortSignal): Promise<RecommendationResponse> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 15_000)
  const abortRequest = () => controller.abort()
  signal?.addEventListener('abort', abortRequest, { once: true })

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/recommendations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ratings: ratings.slice(-20).map(({ mal_id, rating }) => ({ mal_id, rating })),
        limit: 20,
      }),
      signal: controller.signal,
    })
    if (!response.ok) {
      const payload = await response.json().catch(() => null) as { detail?: string } | null
      throw new Error(payload?.detail ?? 'Could not generate recommendations.')
    }
    const result = await response.json() as RecommendationResponse
    const ratedMalIds = new Set(ratings.map(({ mal_id }) => mal_id))
    return {
      ...result,
      items: result.items
        .filter(({ mal_id }) => !ratedMalIds.has(mal_id))
        .slice(0, 10)
        .map((item, index) => ({ ...item, rank: index + 1 })),
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      if (signal?.aborted) throw error
      throw new Error('Recommendations timed out. Please try again.')
    }
    if (error instanceof TypeError) throw new Error('Cannot reach the recommendation API.')
    throw error
  } finally {
    window.clearTimeout(timeout)
    signal?.removeEventListener('abort', abortRequest)
  }
}
