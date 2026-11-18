import { useCallback, useEffect, useRef, useState } from 'react'
import type { RatedAnime } from '../discover/types'
import { getTasteDna } from './tasteDnaApi'
import type { TasteDnaResponse } from './types'

export function useTasteDna(ratings: RatedAnime[]) {
  const [data, setData] = useState<TasteDnaResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)

  const generate = useCallback(async () => {
    if (ratings.length < 5) return
    controller.current?.abort()
    const nextController = new AbortController()
    controller.current = nextController
    setLoading(true)
    setError('')
    try {
      setData(await getTasteDna(ratings, nextController.signal))
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      setError(caught instanceof Error ? caught.message : 'Could not build your Taste DNA.')
    } finally {
      if (controller.current === nextController) setLoading(false)
    }
  }, [ratings])

  useEffect(() => {
    void generate()
    return () => controller.current?.abort()
  }, [generate])

  return { data, loading, error, generate }
}
