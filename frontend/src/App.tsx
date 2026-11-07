import { type FormEvent, useEffect, useRef, useState } from 'react'
import { searchAnime, type AnimeSearchResult } from './api/anime'
import { getHealth } from './api/health'
import { useRatings } from './ratings'

type ApiState = 'checking' | 'online' | 'offline'

function App() {
  const [apiState, setApiState] = useState<ApiState>('checking')
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<AnimeSearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const requestId = useRef(0)
  const { ratings, setRating, removeRating } = useRatings()

  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then(() => setApiState('online'))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        setApiState('offline')
      })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const normalizedQuery = query.trim()
    if (normalizedQuery.length < 3) {
      requestId.current += 1
      setResults([])
      setSearching(false)
      setSearchError('')
      return
    }

    const controller = new AbortController()
    const timer = window.setTimeout(async () => {
      const currentRequest = ++requestId.current
      setSearching(true)
      setSearchError('')
      try {
        const items = await searchAnime(normalizedQuery, controller.signal)
        if (currentRequest === requestId.current) setResults(items)
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') return
        if (currentRequest === requestId.current) {
          setSearchError(error instanceof Error ? error.message : 'Search failed. Please try again.')
        }
      } finally {
        if (currentRequest === requestId.current) setSearching(false)
      }
    }, 350)

    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [query])

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const normalizedQuery = query.trim()
    if (!normalizedQuery) return
    const currentRequest = ++requestId.current
    setSearching(true)
    setSearchError('')
    try {
      const items = await searchAnime(normalizedQuery)
      if (currentRequest === requestId.current) setResults(items)
    } catch (error) {
      if (currentRequest === requestId.current) {
        setSearchError(error instanceof Error ? error.message : 'Search failed. Please try again.')
      }
    } finally {
      if (currentRequest === requestId.current) setSearching(false)
    }
  }

  const statusText = {
    checking: 'Checking API…',
    online: 'API connected',
    offline: 'API offline — start FastAPI on port 8000',
  }[apiState]

  return (
    <main className="min-h-screen bg-ink text-white">
      <div className="mx-auto max-w-6xl px-6 py-8 sm:px-10">
        <nav className="flex flex-wrap items-center justify-between gap-4" aria-label="Primary navigation">
          <a className="text-lg font-semibold tracking-tight" href="/">
            Anime Taste <span className="text-sakura">DNA</span>
          </a>
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300" aria-live="polite">
            <span className={`mr-2 inline-block h-2 w-2 rounded-full ${apiState === 'online' ? 'bg-emerald-400' : 'bg-amber-400'}`} />
            {statusText}
          </span>
        </nav>

        <header className="py-16 sm:py-24">
          <p className="mb-4 text-sm font-semibold uppercase tracking-[0.25em] text-violet">Discover your taste</p>
          <h1 className="max-w-4xl text-4xl font-bold tracking-tight sm:text-6xl">Rate anime you have watched.</h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-300">
            Add 10–20 ratings to unlock recommendations. Your list stays in this browser.
          </p>
          <form className="mt-9 flex max-w-2xl gap-3" onSubmit={handleSearch}>
            <label className="sr-only" htmlFor="anime-search">Search anime</label>
            <input
              id="anime-search"
              className="min-w-0 flex-1 rounded-full border border-white/15 bg-white/5 px-5 py-3 outline-none transition placeholder:text-slate-500 focus:border-violet"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Type at least 3 characters…"
              maxLength={100}
            />
            <button className="rounded-full bg-white px-6 py-3 font-semibold text-ink disabled:opacity-50" disabled={searching || !query.trim()} type="submit">
              {searching ? 'Searching…' : 'Search'}
            </button>
          </form>
          {searchError && <p className="mt-3 text-sm text-rose-300" role="alert">{searchError}</p>}
        </header>

        {results.length > 0 && (
          <section aria-labelledby="results-heading" className="pb-16">
            <h2 id="results-heading" className="mb-5 text-2xl font-semibold">Search results</h2>
            <div className="grid gap-4 sm:grid-cols-2">
              {results.map((anime) => {
                const currentRating = anime.mal_id === null ? undefined : ratings.find((item) => item.mal_id === anime.mal_id)?.rating
                return (
                  <article className="flex gap-4 rounded-2xl border border-white/10 bg-white/5 p-4" key={anime.anilist_id}>
                    {anime.cover_image_url ? <img className="h-32 w-24 rounded-lg object-cover" src={anime.cover_image_url} alt="" /> : <div className="h-32 w-24 rounded-lg bg-white/10" />}
                    <div className="min-w-0 flex-1">
                      <h3 className="font-semibold">{anime.title}</h3>
                      <p className="mt-1 text-sm text-slate-400">{[anime.year, anime.format].filter(Boolean).join(' · ') || 'Details unavailable'}</p>
                      <p className="mt-2 line-clamp-2 text-xs text-slate-400">{anime.genres.join(' · ')}</p>
                      {anime.mal_id === null ? (
                        <p className="mt-4 text-xs text-amber-300">No MyAnimeList mapping — rating unavailable</p>
                      ) : (
                        <label className="mt-4 flex items-center gap-2 text-sm">
                          Your rating
                          <select
                            className="rounded-lg border border-white/15 bg-ink px-2 py-1"
                            value={currentRating ?? ''}
                            onChange={(event) => setRating(anime, Number(event.target.value))}
                          >
                            <option value="" disabled>1–10</option>
                            {Array.from({ length: 10 }, (_, index) => index + 1).map((rating) => <option value={rating} key={rating}>{rating}</option>)}
                          </select>
                        </label>
                      )}
                    </div>
                  </article>
                )
              })}
            </div>
          </section>
        )}

        <section className="border-t border-white/10 py-10" aria-labelledby="ratings-heading">
          <div className="flex items-end justify-between gap-4">
            <div>
              <h2 id="ratings-heading" className="text-2xl font-semibold">Your ratings</h2>
              <p className="mt-1 text-sm text-slate-400">{ratings.length}/10 minimum ratings</p>
            </div>
            <div className="h-2 w-32 overflow-hidden rounded-full bg-white/10" aria-hidden="true">
              <div className="h-full bg-gradient-to-r from-sakura to-violet" style={{ width: `${Math.min(ratings.length * 10, 100)}%` }} />
            </div>
          </div>
          {ratings.length === 0 ? (
            <p className="mt-8 rounded-2xl border border-dashed border-white/15 p-8 text-center text-slate-400">Search above and rate your first anime.</p>
          ) : (
            <ul className="mt-6 divide-y divide-white/10">
              {ratings.map((anime) => (
                <li className="flex items-center justify-between gap-4 py-4" key={anime.mal_id}>
                  <span className="min-w-0 truncate font-medium">{anime.title}</span>
                  <div className="flex items-center gap-3">
                    <span className="rounded-full bg-violet/20 px-3 py-1 text-sm text-violet">{anime.rating}/10</span>
                    <button className="text-sm text-slate-400 hover:text-white" type="button" onClick={() => removeRating(anime.mal_id)}>Remove</button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </main>
  )
}

export default App
