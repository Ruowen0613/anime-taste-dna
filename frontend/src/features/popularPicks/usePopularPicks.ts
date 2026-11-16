import { useEffect, useState } from 'react'
import { getPopularPicks } from './popularPicksApi'
import type { PopularPicksResponse } from './types'

export function usePopularPicks() {
  const [data, setData] = useState<PopularPicksResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    getPopularPicks(controller.signal)
      .then(setData)
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === 'AbortError') return
        setError(caught instanceof Error ? caught.message : 'Could not load Popular Picks.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [])

  return { data, loading, error }
}
