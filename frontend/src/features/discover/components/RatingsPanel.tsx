import type { RatedAnime } from '../types'

type RatingsPanelProps = {
  ratings: RatedAnime[]
  onRemove: (malId: number) => void
}

export function RatingsPanel({ ratings, onRemove }: RatingsPanelProps) {
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
        <ul className="mt-6 divide-y divide-white/10">
          {ratings.map((anime) => (
            <li className="flex items-center justify-between gap-4 py-4" key={anime.mal_id}>
              <span className="min-w-0 truncate font-medium">{anime.title}</span>
              <div className="flex items-center gap-3">
                <span className="rounded-full bg-violet/20 px-3 py-1 text-sm text-violet">{anime.rating}/10</span>
                <button className="text-sm text-slate-400 hover:text-white" type="button" onClick={() => onRemove(anime.mal_id)}>Remove</button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
