import type { AnimeSearchResult, RatedAnime } from '../types'

type RatingsPanelProps = {
  ratings: RatedAnime[]
  onRemove: (malId: number) => void
  onOpen: (anime: AnimeSearchResult) => void
}

export function RatingsPanel({ ratings, onRemove, onOpen }: RatingsPanelProps) {
  return (
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
        <ul className="mt-8 grid gap-4 md:grid-cols-2">
          {ratings.map((anime) => (
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
                  <button className="text-sm text-slate-400 transition hover:text-rose-300" type="button" onClick={(event) => { event.stopPropagation(); onRemove(anime.mal_id) }}>Remove</button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
