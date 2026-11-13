import { useCallback, useEffect, useRef, useState } from 'react'
import type { RatedAnime } from '../discover/types'
import { getRecommendations } from './recommendationsApi'
import type { RecommendationResponse } from './types'

export function useRecommendations(ratings: RatedAnime[]) {
  const [result, setResult] = useState<RecommendationResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)

  const generate = useCallback(async () => {
    controller.current?.abort()
    const nextController = new AbortController()
    controller.current = nextController
    setLoading(true)
    setError('')
    try {
      const response = await getRecommendations(ratings, nextController.signal)
      setResult(response)
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      setError(caught instanceof Error ? caught.message : 'Could not generate recommendations.')
    } finally {
      if (controller.current === nextController) setLoading(false)
    }
  }, [ratings])

  useEffect(() => {
    void generate()
    return () => controller.current?.abort()
  }, [generate])

  return { result, loading, error, generate }
}
