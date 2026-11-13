import { useEffect, useState } from 'react'
import type { PairwisePreference } from './types'

const STORAGE_KEY = 'anime-taste-dna:preference-quiz:v1'

type StoredQuiz = {
  preferences: PairwisePreference[]
  seenMalIds: number[]
}

function readQuiz(): StoredQuiz {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') as Partial<StoredQuiz>
    const preferences = Array.isArray(parsed.preferences) ? parsed.preferences.filter(
      (item): item is PairwisePreference =>
        typeof item?.most_mal_id === 'number' && typeof item?.least_mal_id === 'number' &&
        item.most_mal_id !== item.least_mal_id,
    ) : []
    const seenMalIds = Array.isArray(parsed.seenMalIds)
      ? parsed.seenMalIds.filter((value): value is number => Number.isInteger(value) && value > 0)
      : []
    return { preferences: preferences.slice(-20), seenMalIds: [...new Set(seenMalIds)] }
  } catch {
    return { preferences: [], seenMalIds: [] }
  }
}

export function usePreferenceQuiz() {
  const [quiz, setQuiz] = useState<StoredQuiz>(readQuiz)

  useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(quiz)), [quiz])

  function recordPreference(preference: PairwisePreference, shownMalIds: number[]) {
    setQuiz((current) => ({
      preferences: [...current.preferences, preference].slice(-20),
      seenMalIds: [...new Set([...current.seenMalIds, ...shownMalIds])],
    }))
  }

  function recordSeen(shownMalIds: number[]) {
    setQuiz((current) => ({
      ...current,
      seenMalIds: [...new Set([...current.seenMalIds, ...shownMalIds])],
    }))
  }

  return { ...quiz, recordPreference, recordSeen }
}
