import { useEffect, useState } from 'react'

const STORAGE_KEY = 'anime-taste-dna:archived-recommendations:v1'

function readArchivedMalIds(): number[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
    if (!Array.isArray(parsed)) return []
    return [...new Set(parsed.filter((value): value is number => Number.isInteger(value) && value > 0))]
  } catch {
    return []
  }
}

export function useArchivedRecommendations() {
  const [archivedMalIds, setArchivedMalIds] = useState<number[]>(readArchivedMalIds)

  useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(archivedMalIds)), [archivedMalIds])

  function archive(malId: number) {
    setArchivedMalIds((current) => current.includes(malId) ? current : [...current, malId])
  }

  function restoreAll() {
    setArchivedMalIds([])
  }

  return { archivedMalIds, archive, restoreAll }
}
