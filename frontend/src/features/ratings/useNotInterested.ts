import { useEffect, useMemo, useState } from 'react'
import type { AnimeSearchResult } from '../discover/types'

const STORAGE_KEY = 'anime-taste-dna:archived-recommendations:v1'

export type NotInterestedAnime = AnimeSearchResult & { mal_id: number }

function emptyAnime(malId: number, title: string): NotInterestedAnime {
  return {
    mal_id: malId,
    anilist_id: null,
    title,
    cover_image_url: null,
    year: null,
    format: null,
    genres: [],
    average_score: null,
    popularity: 0,
  }
}

function normalizeAnime(value: unknown): NotInterestedAnime | null {
  if (Number.isInteger(value) && Number(value) > 0) {
    const malId = Number(value)
    return emptyAnime(malId, `MAL #${malId}`)
  }
  if (typeof value !== 'object' || value === null) return null
  const candidate = value as Partial<NotInterestedAnime>
  if (!Number.isInteger(candidate.mal_id) || Number(candidate.mal_id) <= 0 || typeof candidate.title !== 'string') return null
  const malId = Number(candidate.mal_id)
  return {
    ...emptyAnime(malId, candidate.title),
    anilist_id: Number.isInteger(candidate.anilist_id) ? Number(candidate.anilist_id) : null,
    cover_image_url: typeof candidate.cover_image_url === 'string' ? candidate.cover_image_url : null,
    year: Number.isInteger(candidate.year) ? Number(candidate.year) : null,
    format: typeof candidate.format === 'string' ? candidate.format : null,
    genres: Array.isArray(candidate.genres) ? candidate.genres.filter((genre): genre is string => typeof genre === 'string') : [],
    average_score: Number.isFinite(candidate.average_score) ? Number(candidate.average_score) : null,
    popularity: Number.isFinite(candidate.popularity) ? Number(candidate.popularity) : 0,
  }
}

function readNotInterested(): NotInterestedAnime[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
    if (!Array.isArray(parsed)) return []
    const items = new Map<number, NotInterestedAnime>()
    for (const value of parsed) {
      const item = normalizeAnime(value)
      if (item) items.set(item.mal_id, item)
    }
    return [...items.values()]
  } catch {
    return []
  }
}

export function useNotInterested() {
  const [items, setItems] = useState<NotInterestedAnime[]>(readNotInterested)
  const malIds = useMemo(() => items.map(({ mal_id }) => mal_id), [items])

  useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(items)), [items])

  function add(item: NotInterestedAnime) {
    setItems((current) => {
      const existing = current.find(({ mal_id }) => mal_id === item.mal_id)
      if (!existing) return [...current, item]
      return current.map((candidate) => candidate.mal_id === item.mal_id ? { ...candidate, ...item } : candidate)
    })
  }

  function restore(malIdsToRestore: number[]) {
    const restored = new Set(malIdsToRestore)
    setItems((current) => current.filter(({ mal_id }) => !restored.has(mal_id)))
  }

  return { items, malIds, add, restore }
}
