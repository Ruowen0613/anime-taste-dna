import type { RatedAnime } from '../../discover/types'
import type { PopularPick } from '../types'

type PopularPickCardProps = {
  anime: PopularPick
  rating: RatedAnime | undefined
  notInterested: boolean
  onRate: (anime: PopularPick, rating: number) => void
  onRemove: (malId: number) => void
  onNotInterested: (anime: PopularPick) => void
  onRestoreInterest: (malId: number) => void
  onOpen: (anime: PopularPick) => void
}

export function PopularPickCard({ anime, rating, notInterested, onRate, onRemove, onNotInterested, onRestoreInterest, onOpen }: PopularPickCardProps) {
  function updateRating(value: string) {
    if (!value) {
      onRemove(anime.mal_id)
      onRestoreInterest(anime.mal_id)
      return
    }
    if (value === 'not-interested') {
      onRemove(anime.mal_id)
      onNotInterested(anime)
      return
    }
    onRestoreInterest(anime.mal_id)
    onRate(anime, Number(value))
  }

  return (
    <article className="group min-w-0 overflow-hidden rounded-xl border border-white/10 bg-white/[0.04] transition hover:-translate-y-0.5 hover:border-violet/50 hover:bg-white/[0.07]">
      <button className="block w-full text-left" type="button" onClick={() => onOpen(anime)} aria-label={`View details for ${anime.title}`}>
        <div className="aspect-[2/3] overflow-hidden bg-gradient-to-br from-violet/20 to-sakura/10">
          {anime.cover_image_url ? (
            <img className="h-full w-full object-cover transition duration-300 group-hover:scale-105" src={anime.cover_image_url} alt="" loading="lazy" />
          ) : (
            <div className="flex h-full items-center justify-center px-2 text-center text-xs text-slate-500">Cover unavailable</div>
          )}
        </div>
        <h2 className="h-14 overflow-hidden px-2 pt-2 text-xs font-semibold leading-4 text-white" title={anime.title}>{anime.title}</h2>
      </button>
      <div className="px-2 pb-2">
        <select
          className="w-full rounded-lg border border-white/10 bg-ink px-2 py-1.5 text-xs text-slate-200 outline-none transition focus:border-violet"
          aria-label={`Rating for ${anime.title}`}
          value={rating?.rating ?? (notInterested ? 'not-interested' : '')}
          onChange={(event) => updateRating(event.target.value)}
        >
          <option value="">Not watched</option>
          <option value="not-interested">Not interested</option>
          {Array.from({ length: 10 }, (_, index) => index + 1).map((score) => (
            <option value={score} key={score}>{score}/10</option>
          ))}
        </select>
      </div>
    </article>
  )
}
