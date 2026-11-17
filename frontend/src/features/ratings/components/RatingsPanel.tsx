import { useMemo, useState } from 'react'
import type { AnimeSearchResult, RatedAnime } from '../../discover/types'

type SortOption = 'score-desc' | 'score-asc' | 'title' | 'year-desc'

type RatingsPanelProps = {
  ratings: RatedAnime[]
  onRate: (anime: AnimeSearchResult, rating: number) => void
  onRemove: (malId: number) => void
  onOpen: (anime: AnimeSearchResult) => void
  onRecommend: () => void
  hideVisualTitle?: boolean
}

export function RatingsPanel({ ratings, onRate, onRemove, onOpen, onRecommend, hideVisualTitle = false }: RatingsPanelProps) {
  const [editingMalId, setEditingMalId] = useState<number | null>(null)
  const [sortBy, setSortBy] = useState<SortOption>('score-desc')
  const sortedRatings = useMemo(() => [...ratings].sort((left, right) => {
    if (sortBy === 'score-desc') return right.rating - left.rating || left.title.localeCompare(right.title)
    if (sortBy === 'score-asc') return left.rating - right.rating || left.title.localeCompare(right.title)
    if (sortBy === 'year-desc') return (right.year ?? 0) - (left.year ?? 0) || left.title.localeCompare(right.title)
    return left.title.localeCompare(right.title)
  }), [ratings, sortBy])

  return (
    <section className="border-t border-white/10 py-10" aria-labelledby="ratings-heading">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h2 id="ratings-heading" className={hideVisualTitle ? 'sr-only' : 'text-2xl font-semibold'}>Your ratings</h2>
          <p className="mt-1 text-sm text-slate-400">{ratings.length}/10 minimum ratings</p>
        </div>
        <div className="flex flex-col items-end gap-3">
          {ratings.length > 1 && (
            <label className="flex items-center gap-2 text-sm text-slate-400">
              Sort
              <select className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white outline-none focus:border-violet" value={sortBy} onChange={(event) => setSortBy(event.target.value as SortOption)}>
                <option value="score-desc">Highest score</option>
                <option value="score-asc">Lowest score</option>
                <option value="title">Title A–Z</option>
                <option value="year-desc">Newest release</option>
              </select>
            </label>
          )}
          <div className="h-2 w-32 overflow-hidden rounded-full bg-white/10" aria-hidden="true">
            <div className="h-full bg-gradient-to-r from-sakura to-violet" style={{ width: `${Math.min(ratings.length * 10, 100)}%` }} />
          </div>
        </div>
      </div>
      {ratings.length >= 10 && (
        <div className="mt-8 flex flex-col items-start justify-between gap-4 rounded-2xl border border-violet/30 bg-gradient-to-r from-violet/15 to-sakura/10 p-5 sm:flex-row sm:items-center">
          <div>
            <h3 className="font-semibold">Your taste profile is ready</h3>
            <p className="mt-1 text-sm text-slate-300">Generate a personalized Top 10 from your {ratings.length} ratings.</p>
          </div>
          <button className="rounded-xl bg-white px-5 py-3 font-semibold text-ink transition hover:bg-violet hover:text-white" type="button" onClick={onRecommend}>Get recommendations</button>
        </div>
      )}
      {ratings.length === 0 ? (
        <p className="mt-8 rounded-2xl border border-dashed border-white/15 p-8 text-center text-slate-400">Go to Popular Picks to rate your first anime.</p>
      ) : (
        <ul className="mt-8 grid gap-4 md:grid-cols-2">
          {sortedRatings.map((anime) => (
            <li className="group flex min-h-44 cursor-pointer overflow-hidden rounded-2xl border border-white/10 bg-white/5 transition hover:border-white/20 hover:bg-white/[0.07]" key={anime.mal_id} onClick={() => onOpen(anime)}>
              {anime.cover_image_url ? (
                <img className="w-28 shrink-0 object-cover sm:w-32" src={anime.cover_image_url} alt={`${anime.title} cover`} />
              ) : (
                <div className="flex w-28 shrink-0 items-center justify-center bg-white/5 px-3 text-center text-xs text-slate-500 sm:w-32">No image</div>
              )}
              <div className="flex min-w-0 flex-1 flex-col p-4">
                <div className="flex items-start justify-between gap-3">
                  <h3 className="line-clamp-2 font-semibold leading-snug">{anime.title}</h3>
                  <span className="shrink-0 rounded-full bg-violet/20 px-3 py-1 text-sm font-semibold text-violet">{anime.rating}/10</span>
                </div>
                <div className="mt-3 flex flex-wrap gap-1.5 text-xs text-slate-300">
                  {anime.year && <span className="rounded-full bg-white/10 px-2.5 py-1">{anime.year}</span>}
                  {anime.format && <span className="rounded-full bg-white/10 px-2.5 py-1">{anime.format}</span>}
                  {anime.genres.slice(0, 3).map((genre) => (
                    <span className="rounded-full bg-white/10 px-2.5 py-1" key={genre}>{genre}</span>
                  ))}
                </div>
                <div className="mt-auto flex items-end justify-between gap-3 pt-4">
                  <button className="text-sm text-violet hover:text-white" type="button" onClick={(event) => { event.stopPropagation(); onOpen(anime) }}>View details</button>
                  <div className="flex items-center gap-2" onClick={(event) => event.stopPropagation()}>
                    {editingMalId === anime.mal_id ? (
                      <select
                        className="rounded-lg border border-violet/40 bg-ink px-2.5 py-1.5 text-sm text-white outline-none focus:border-violet"
                        aria-label={`Edit score for ${anime.title}`}
                        value={anime.rating}
                        onChange={(event) => {
                          onRate(anime, Number(event.target.value))
                          setEditingMalId(null)
                        }}
                      >
                        {Array.from({ length: 10 }, (_, index) => index + 1).map((rating) => <option value={rating} key={rating}>{rating}/10</option>)}
                      </select>
                    ) : (
                      <button className="rounded-lg bg-white/10 px-3 py-1.5 text-sm text-slate-200 transition hover:bg-white/15 hover:text-white" type="button" onClick={() => setEditingMalId(anime.mal_id)}>Edit score</button>
                    )}
                    <button className="rounded-lg bg-black/40 px-3 py-1.5 text-sm text-slate-300 transition hover:bg-rose-950/70 hover:text-rose-200" type="button" onClick={() => onRemove(anime.mal_id)}>Remove</button>
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
