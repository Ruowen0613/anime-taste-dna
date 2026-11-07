import { type FormEvent, useCallback, useEffect, useRef, useState } from 'react'
import { searchAnime } from './animeSearchApi'
import type { AnimeSearchResult, AnimeSearchSort } from './types'

export function useAnimeSearch() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<AnimeSearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState('')
  const [sortBy, setSortBy] = useState<AnimeSearchSort>('popularity')
  const requestId = useRef(0)

  const runSearch = useCallback(async (normalizedQuery: string, signal?: AbortSignal) => {
    const currentRequest = ++requestId.current
    setSearching(true)
    setError('')
    try {
      const items = await searchAnime(normalizedQuery, sortBy, signal)
      if (currentRequest === requestId.current) setResults(items)
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      if (currentRequest === requestId.current) {
        setError(caught instanceof Error ? caught.message : 'Search failed. Please try again.')
      }
    } finally {
      if (currentRequest === requestId.current) setSearching(false)
    }
  }, [sortBy])

  useEffect(() => {
    const normalizedQuery = query.trim()
    if (normalizedQuery.length < 3) {
      requestId.current += 1
      setResults([])
      setSearching(false)
      setError('')
      return
    }
    const controller = new AbortController()
    const timer = window.setTimeout(() => void runSearch(normalizedQuery, controller.signal), 350)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [query, runSearch])

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const normalizedQuery = query.trim()
    if (normalizedQuery) void runSearch(normalizedQuery)
  }

  return { query, setQuery, results, searching, error, sortBy, setSortBy, submit }
}
