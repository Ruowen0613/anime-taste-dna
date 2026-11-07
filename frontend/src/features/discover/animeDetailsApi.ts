import { API_BASE_URL } from '../../api/health'
import type { AnimeDetails } from './types'

export async function getAnimeDetails(anilistId: number, signal?: AbortSignal): Promise<AnimeDetails> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 10_000)
  const abortRequest = () => controller.abort()
  signal?.addEventListener('abort', abortRequest, { once: true })

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/anime/${anilistId}`, { signal: controller.signal })
    if (!response.ok) throw new Error('Could not load anime details.')
    return (await response.json()) as AnimeDetails
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      if (signal?.aborted) throw error
      throw new Error('Anime details timed out. Please try again.')
    }
    if (error instanceof TypeError) throw new Error('Cannot reach the API.')
    throw error
  } finally {
    window.clearTimeout(timeout)
    signal?.removeEventListener('abort', abortRequest)
  }
}
