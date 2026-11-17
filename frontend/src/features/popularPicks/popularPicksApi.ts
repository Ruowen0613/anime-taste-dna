import { API_BASE_URL } from '../../api/health'
import type { PopularPicksResponse } from './types'

export async function getPopularPicks(signal?: AbortSignal): Promise<PopularPicksResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/popular-picks`, { signal })
  if (!response.ok) {
    throw new Error(
      response.status === 503
        ? 'The recommendation model is unavailable.'
        : 'Could not load Popular Picks.',
    )
  }
  return response.json() as Promise<PopularPicksResponse>
}
