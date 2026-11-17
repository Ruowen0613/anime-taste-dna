import { useEffect, useMemo, useState } from 'react'
import type { AnimeSearchResult } from '../../discover/types'
import { getAnimeMetadata } from '../notInterestedApi'
import type { NotInterestedAnime } from '../useNotInterested'

type NotInterestedPanelProps = {
  items: NotInterestedAnime[]
  onOpen: (anime: AnimeSearchResult) => void
  onRestore: (malIds: number[]) => void
}

export function NotInterestedPanel({ items, onOpen, onRestore }: NotInterestedPanelProps) {
  const [expanded, setExpanded] = useState(false)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [metadata, setMetadata] = useState<Map<number, NotInterestedAnime>>(new Map())
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const visibleItems = useMemo(() => items.map((item) => metadata.get(item.mal_id) ?? item), [items, metadata])
  const allSelected = items.length > 0 && selected.size === items.length

  useEffect(() => {
    setSelected((current) => new Set([...current].filter((malId) => items.some((item) => item.mal_id === malId))))
  }, [items])

  useEffect(() => {
    if (!expanded || items.length === 0) return
    const controller = new AbortController()
    setLoading(true)
    setError('')
    void getAnimeMetadata(items.map(({ mal_id }) => mal_id), controller.signal)
      .then((loaded) => {
        const next = new Map<number, NotInterestedAnime>()
        for (const item of loaded) {
          if (item.mal_id !== null) next.set(item.mal_id, { ...item, mal_id: item.mal_id })
        }
        setMetadata(next)
      })
      .catch((caught) => {
        if (caught instanceof DOMException && caught.name === 'AbortError') return
        setError(caught instanceof Error ? caught.message : 'Could not refresh anime metadata.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [expanded, items])

  function toggle(malId: number) {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(malId)) next.delete(malId)
      else next.add(malId)
      return next
    })
  }

  if (items.length === 0) return null

  return (
    <section className="border-t border-white/10 py-10" aria-labelledby="not-interested-heading">
      <button
        className="flex w-full items-center justify-between gap-4 text-left"
        type="button"
        aria-expanded={expanded}
        aria-controls="not-interested-content"
        onClick={() => setExpanded((current) => !current)}
      >
        <div>
          <h2 id="not-interested-heading" className="text-2xl font-semibold">Not interested</h2>
          <p className="mt-1 text-sm text-slate-400">{items.length} hidden title{items.length === 1 ? '' : 's'} excluded from recommendations.</p>
        </div>
        <span className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300">{expanded ? 'Collapse ↑' : 'Manage ↓'}</span>
      </button>

      {expanded && (
        <div id="not-interested-content" className="mt-7">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <label className="flex cursor-pointer items-center gap-3 text-sm text-slate-300">
              <input type="checkbox" checked={allSelected} onChange={() => setSelected(allSelected ? new Set() : new Set(items.map(({ mal_id }) => mal_id)))} />
              Select all
            </label>
            <button
              className="rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-ink transition hover:bg-violet hover:text-white disabled:opacity-40"
              type="button"
              disabled={selected.size === 0}
              onClick={() => onRestore([...selected])}
            >
              Restore selected ({selected.size})
            </button>
          </div>
          {loading && <p className="mt-4 text-sm text-slate-400">Refreshing anime details…</p>}
          {error && <p className="mt-4 text-sm text-amber-300">{error} Stored titles are still available below.</p>}
          <ul className="mt-5 grid gap-4 md:grid-cols-2">
            {visibleItems.map((anime) => (
              <li className="group flex min-h-44 overflow-hidden rounded-2xl border border-white/10 bg-white/5 transition hover:border-white/20 hover:bg-white/[0.07]" key={anime.mal_id}>
                {anime.cover_image_url ? (
                  <img className="w-28 shrink-0 object-cover sm:w-32" src={anime.cover_image_url} alt={`${anime.title} cover`} loading="lazy" />
                ) : (
                  <div className="flex w-28 shrink-0 items-center justify-center bg-white/5 px-3 text-center text-xs text-slate-500 sm:w-32">No image</div>
                )}
                <div className="flex min-w-0 flex-1 flex-col p-4">
                  <div className="flex items-start justify-between gap-3">
                    <h3 className="line-clamp-2 font-semibold leading-snug">{anime.title}</h3>
                    <input aria-label={`Select ${anime.title}`} type="checkbox" checked={selected.has(anime.mal_id)} onChange={() => toggle(anime.mal_id)} />
                  </div>
                  <div className="mt-3 flex flex-wrap gap-1.5 text-xs text-slate-300">
                    {anime.year && <span className="rounded-full bg-white/10 px-2.5 py-1">{anime.year}</span>}
                    {anime.format && <span className="rounded-full bg-white/10 px-2.5 py-1">{anime.format}</span>}
                    {anime.genres.slice(0, 3).map((genre) => <span className="rounded-full bg-white/10 px-2.5 py-1" key={genre}>{genre}</span>)}
                  </div>
                  {anime.average_score !== null && <p className="mt-3 text-sm text-slate-400">AniList community score <span className="text-slate-200">{anime.average_score}/100</span></p>}
                  <div className="mt-auto flex items-end justify-between gap-3 pt-4">
                    <button className="text-sm text-violet transition hover:text-white disabled:text-slate-600" type="button" disabled={anime.anilist_id === null} onClick={() => onOpen(anime)}>{anime.anilist_id === null ? 'Details unavailable' : 'View details'}</button>
                    <button className="rounded-lg bg-white/10 px-3 py-1.5 text-sm text-slate-200 transition hover:bg-white/15 hover:text-white" type="button" onClick={() => onRestore([anime.mal_id])}>Restore</button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}
