import { useRef, useState } from 'react'
import { getAnimeDetails } from './animeDetailsApi'
import type { AnimeDetails, AnimeSearchResult } from './types'

export function useAnimeDetails() {
  const [selected, setSelected] = useState<AnimeSearchResult | null>(null)
  const [details, setDetails] = useState<AnimeDetails | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)
  const cache = useRef(new Map<number, AnimeDetails>())

  async function open(anime: AnimeSearchResult) {
    controller.current?.abort()
    setSelected(anime)
    setError('')
    const cached = cache.current.get(anime.anilist_id)
    if (cached) {
      setDetails(cached)
      setLoading(false)
      return
    }

    const nextController = new AbortController()
    controller.current = nextController
    setDetails(null)
    setLoading(true)
    try {
      const loaded = await getAnimeDetails(anime.anilist_id, nextController.signal)
      cache.current.set(anime.anilist_id, loaded)
      setDetails(loaded)
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      setError(caught instanceof Error ? caught.message : 'Could not load anime details.')
    } finally {
      if (controller.current === nextController) setLoading(false)
    }
  }

  function close() {
    controller.current?.abort()
    controller.current = null
    setSelected(null)
    setDetails(null)
    setError('')
    setLoading(false)
  }

  return { selected, details, loading, error, open, close }
}
