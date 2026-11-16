import type { AnimeSearchResult, AnimeSearchSort, RatedAnime } from '../types'

const compactNumber = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 })

type SearchResultsProps = {
  results: AnimeSearchResult[]
  ratings: RatedAnime[]
  onRate: (anime: AnimeSearchResult, rating: number) => void
  onOpen: (anime: AnimeSearchResult) => void
  sortBy: AnimeSearchSort
  onSortChange: (sort: AnimeSearchSort) => void
  searching: boolean
  archivedMalIds: Set<number>
  onRemove: (malId: number) => void
  onNotInterested: (anime: AnimeSearchResult & { mal_id: number }) => void
  onRestoreInterest: (malId: number) => void
}

export function SearchResults({ results, ratings, onRate, onOpen, sortBy, onSortChange, searching, archivedMalIds, onRemove, onNotInterested, onRestoreInterest }: SearchResultsProps) {
  if (results.length === 0) return null

  return (
    <section aria-labelledby="results-heading" className="pb-16">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <h2 id="results-heading" className="text-2xl font-semibold">Search results</h2>
        <label className="flex items-center gap-2 text-sm text-slate-400">
          Sort
          <select className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white outline-none focus:border-violet" value={sortBy} disabled={searching} onChange={(event) => onSortChange(event.target.value as AnimeSearchSort)}>
            <option value="popularity">Most popular</option>
            <option value="score">Highest rated</option>
            <option value="relevance">Best match</option>
            <option value="newest">Newest release</option>
          </select>
        </label>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {results.map((anime) => {
          const currentRating = anime.mal_id === null ? undefined : ratings.find((item) => item.mal_id === anime.mal_id)?.rating
          const notInterested = anime.mal_id !== null && archivedMalIds.has(anime.mal_id)
          return (
            <article className="flex cursor-pointer gap-4 rounded-2xl border border-white/10 bg-white/5 p-4 transition hover:border-white/20 hover:bg-white/[0.07]" key={anime.anilist_id} onClick={() => onOpen(anime)}>
              {anime.cover_image_url ? <img className="h-32 w-24 rounded-lg object-cover" src={anime.cover_image_url} alt="" /> : <div className="h-32 w-24 rounded-lg bg-white/10" />}
              <div className="min-w-0 flex-1">
                <h3 className="font-semibold">{anime.title}</h3>
                <p className="mt-1 text-sm text-slate-400">{[anime.year, anime.format].filter(Boolean).join(' · ') || 'Details unavailable'}</p>
                <p className="mt-2 line-clamp-2 text-xs text-slate-400">{anime.genres.join(' · ')}</p>
                <div className="mt-2 flex gap-3 text-xs text-slate-400">
                  {anime.average_score != null && <span className="text-sakura">★ {anime.average_score}/100</span>}
                  <span>{compactNumber.format(anime.popularity)} users</span>
                </div>
                {anime.mal_id === null ? (
                  <p className="mt-4 text-xs text-amber-300">No MyAnimeList mapping — rating unavailable</p>
                ) : (
                  <label className="mt-4 flex items-center gap-2 text-sm">
                    Your rating
                    <select
                      className="rounded-lg border border-white/15 bg-ink px-2 py-1"
                      value={currentRating ?? (notInterested ? 'not-interested' : '')}
                      onClick={(event) => event.stopPropagation()}
                      onChange={(event) => {
                        const value = event.target.value
                        if (anime.mal_id === null) return
                        if (!value) {
                          onRemove(anime.mal_id)
                          onRestoreInterest(anime.mal_id)
                        } else if (value === 'not-interested') {
                          onRemove(anime.mal_id)
                          onNotInterested({ ...anime, mal_id: anime.mal_id })
                        } else {
                          onRestoreInterest(anime.mal_id)
                          onRate(anime, Number(value))
                        }
                      }}
                    >
                      <option value="">Not watched</option>
                      <option value="not-interested">Not interested</option>
                      {Array.from({ length: 10 }, (_, index) => index + 1).map((rating) => <option value={rating} key={rating}>{rating}</option>)}
                    </select>
                  </label>
                )}
                <button className="mt-3 text-sm text-violet hover:text-white" type="button" onClick={(event) => { event.stopPropagation(); onOpen(anime) }}>View details</button>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}
