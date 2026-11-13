import { useEffect, useMemo, useState } from 'react'
import type { ArchivedRecommendation } from './types'

const STORAGE_KEY = 'anime-taste-dna:archived-recommendations:v1'

function readArchivedRecommendations(): ArchivedRecommendation[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
    if (!Array.isArray(parsed)) return []
    const archived = new Map<number, ArchivedRecommendation>()
    for (const value of parsed) {
      if (Number.isInteger(value) && Number(value) > 0) {
        const malId = Number(value)
        archived.set(malId, { mal_id: malId, title: `MAL #${malId}` })
      } else if (
        typeof value === 'object' && value !== null &&
        Number.isInteger((value as ArchivedRecommendation).mal_id) &&
        (value as ArchivedRecommendation).mal_id > 0 &&
        typeof (value as ArchivedRecommendation).title === 'string'
      ) {
        const item = value as ArchivedRecommendation
        archived.set(item.mal_id, { mal_id: item.mal_id, title: item.title })
      }
    }
    return [...archived.values()]
  } catch {
    return []
  }
}

export function useArchivedRecommendations() {
  const [archived, setArchived] = useState<ArchivedRecommendation[]>(readArchivedRecommendations)
  const archivedMalIds = useMemo(() => archived.map(({ mal_id }) => mal_id), [archived])

  useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(archived)), [archived])

  function archive(item: ArchivedRecommendation) {
    setArchived((current) => current.some(({ mal_id }) => mal_id === item.mal_id) ? current : [...current, item])
  }

  function restore(malIds: number[]) {
    const restored = new Set(malIds)
    setArchived((current) => current.filter(({ mal_id }) => !restored.has(mal_id)))
  }

  return { archived, archivedMalIds, archive, restore }
}
