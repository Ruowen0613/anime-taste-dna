import { useCallback, useEffect, useRef, useState } from 'react'
import type { RatedAnime } from '../discover/types'
import type { PairwisePreference } from '../preferenceQuiz/types'
import { getRecommendations } from './recommendationsApi'
import type { RecommendationResponse } from './types'

export function useRecommendations(ratings: RatedAnime[], archivedMalIds: number[], preferences: PairwisePreference[]) {
  const [result, setResult] = useState<RecommendationResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)
  const previousItems = useRef<number[] | null>(null)
  const previousPreferenceCount = useRef(preferences.length)
  const [refinementChangedCount, setRefinementChangedCount] = useState<number | null>(null)

  const generate = useCallback(async () => {
    controller.current?.abort()
    const nextController = new AbortController()
    controller.current = nextController
    setLoading(true)
    setError('')
    try {
      const response = await getRecommendations(ratings, archivedMalIds, preferences, nextController.signal)
      if (preferences.length > previousPreferenceCount.current && previousItems.current) {
        const nextIds = new Set(response.items.map(({ mal_id }) => mal_id))
        setRefinementChangedCount(previousItems.current.filter((malId) => !nextIds.has(malId)).length)
      }
      previousItems.current = response.items.map(({ mal_id }) => mal_id)
      previousPreferenceCount.current = preferences.length
      setResult(response)
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      setError(caught instanceof Error ? caught.message : 'Could not generate recommendations.')
    } finally {
      if (controller.current === nextController) setLoading(false)
    }
  }, [ratings, archivedMalIds, preferences])

  useEffect(() => {
    void generate()
    return () => controller.current?.abort()
  }, [generate])

  return { result, loading, error, generate, refinementChangedCount }
}
