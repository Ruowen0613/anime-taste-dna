import { API_BASE_URL } from '../../api/health'
import type { RatedAnime } from '../discover/types'
import type { TasteDnaResponse } from './types'

export async function getTasteDna(ratings: RatedAnime[], signal?: AbortSignal): Promise<TasteDnaResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/taste-dna`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ratings: ratings.slice(-100).map(({ mal_id, title, rating, genres }) => ({ mal_id, title, rating, genres })),
    }),
    signal,
  })
  if (!response.ok) {
    throw new Error(response.status === 422 ? 'Rate at least 5 anime to build your Taste DNA.' : 'Could not build your Taste DNA. Please try again.')
  }
  return response.json() as Promise<TasteDnaResponse>
}
