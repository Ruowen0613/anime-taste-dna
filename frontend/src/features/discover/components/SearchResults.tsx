import type { AnimeSearchResult, RatedAnime } from '../types'

type SearchResultsProps = {
  results: AnimeSearchResult[]
  ratings: RatedAnime[]
  onRate: (anime: AnimeSearchResult, rating: number) => void
  onOpen: (anime: AnimeSearchResult) => void
}

export function SearchResults({ results, ratings, onRate, onOpen }: SearchResultsProps) {
  if (results.length === 0) return null

  return (
    <section aria-labelledby="results-heading" className="pb-16">
      <h2 id="results-heading" className="mb-5 text-2xl font-semibold">Search results</h2>
      <div className="grid gap-4 sm:grid-cols-2">
        {results.map((anime) => {
          const currentRating = anime.mal_id === null ? undefined : ratings.find((item) => item.mal_id === anime.mal_id)?.rating
          return (
            <article className="flex cursor-pointer gap-4 rounded-2xl border border-white/10 bg-white/5 p-4 transition hover:border-white/20 hover:bg-white/[0.07]" key={anime.anilist_id} onClick={() => onOpen(anime)}>
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
                    <select className="rounded-lg border border-white/15 bg-ink px-2 py-1" value={currentRating ?? ''} onClick={(event) => event.stopPropagation()} onChange={(event) => onRate(anime, Number(event.target.value))}>
                      <option value="" disabled>1–10</option>
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
