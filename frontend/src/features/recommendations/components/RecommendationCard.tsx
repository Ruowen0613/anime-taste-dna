import type { AnimeSearchResult } from '../../discover/types'
import type { RecommendationItem } from '../types'

type RecommendationCardProps = {
  item: RecommendationItem
  onOpen: (anime: AnimeSearchResult) => void
}

export function RecommendationCard({ item, onOpen }: RecommendationCardProps) {
  const canOpen = item.anilist_id !== null
  const open = () => {
    if (!canOpen) return
    onOpen({
      mal_id: item.mal_id,
      anilist_id: item.anilist_id!,
      title: item.title,
      cover_image_url: item.cover_image_url,
      year: item.year,
      format: item.format,
      genres: item.genres,
      average_score: item.average_score,
      popularity: 0,
    })
  }

  return (
    <li className="group relative flex min-h-52 overflow-hidden rounded-2xl border border-white/10 bg-white/5 transition hover:border-violet/40 hover:bg-white/[0.07]">
      <span className="absolute left-3 top-3 z-10 rounded-full bg-ink/85 px-3 py-1 text-sm font-bold text-sakura backdrop-blur">#{item.rank}</span>
      {item.cover_image_url ? (
        <img className="w-32 shrink-0 object-cover sm:w-40" src={item.cover_image_url} alt={`${item.title} cover`} />
      ) : (
        <div className="flex w-32 shrink-0 items-center justify-center bg-white/5 px-4 text-center text-xs text-slate-500 sm:w-40">Metadata unavailable</div>
      )}
      <div className="flex min-w-0 flex-1 flex-col p-5">
        <div className="flex items-start justify-between gap-3">
          <h2 className="line-clamp-2 text-lg font-semibold leading-snug">{item.title}</h2>
          <div className="shrink-0 text-right"><strong className="block text-lg text-violet">{item.predicted_rating.toFixed(1)}</strong><span className="text-[11px] text-slate-500">predicted / 10</span></div>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5 text-xs text-slate-300">
          {item.year && <span className="rounded-full bg-white/10 px-2.5 py-1">{item.year}</span>}
          {item.format && <span className="rounded-full bg-white/10 px-2.5 py-1">{item.format}</span>}
          {item.genres.slice(0, 3).map((genre) => <span className="rounded-full bg-white/10 px-2.5 py-1" key={genre}>{genre}</span>)}
        </div>
        {item.average_score !== null && <p className="mt-3 text-sm text-slate-400">AniList community score <span className="text-slate-200">{item.average_score}/100</span></p>}
        <button className="mt-auto self-start pt-5 text-sm text-violet transition hover:text-white disabled:text-slate-600" type="button" disabled={!canOpen} onClick={open}>{canOpen ? 'View details' : 'Details unavailable'}</button>
      </div>
    </li>
  )
}
