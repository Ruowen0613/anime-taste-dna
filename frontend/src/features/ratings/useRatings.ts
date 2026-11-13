import { useEffect, useState } from 'react'
import type { AnimeSearchResult, RatedAnime } from '../discover/types'

const STORAGE_KEY = 'anime-taste-dna:ratings:v1'

function readRatings(): RatedAnime[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (item): item is RatedAnime =>
        typeof item === 'object' && item !== null &&
        typeof (item as RatedAnime).mal_id === 'number' &&
        Number.isInteger((item as RatedAnime).rating) &&
        (item as RatedAnime).rating >= 1 && (item as RatedAnime).rating <= 10,
    )
  } catch {
    return []
  }
}

export function useRatings() {
  const [ratings, setRatings] = useState<RatedAnime[]>(readRatings)

  useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(ratings)), [ratings])

  function setRating(anime: AnimeSearchResult, rating: number) {
    if (anime.mal_id === null || !Number.isInteger(rating) || rating < 1 || rating > 10) return
    const ratedAnime: RatedAnime = { ...anime, mal_id: anime.mal_id, rating }
    setRatings((current) => [...current.filter((item) => item.mal_id !== anime.mal_id), ratedAnime])
  }

  function removeRating(malId: number) {
    setRatings((current) => current.filter((item) => item.mal_id !== malId))
  }

  return { ratings, setRating, removeRating }
}
