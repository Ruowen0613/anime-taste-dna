import { useState } from 'react'
import type { AnimeSearchResult } from '../../discover/types'
import type { ArchivedRecommendation, RecommendationItem } from '../types'

type RecommendationCardProps = {
  item: RecommendationItem
  onOpen: (anime: AnimeSearchResult) => void
  onRate: (anime: AnimeSearchResult, rating: number) => void
  onArchive: (item: ArchivedRecommendation) => void
}

export function RecommendationCard({ item, onOpen, onRate, onArchive }: RecommendationCardProps) {
  const [ratingOpen, setRatingOpen] = useState(false)
  const canOpen = item.anilist_id !== null
  const anime = item.anilist_id !== null ? {
      mal_id: item.mal_id,
      anilist_id: item.anilist_id,
      title: item.title,
      cover_image_url: item.cover_image_url,
      year: item.year,
      format: item.format,
      genres: item.genres,
      average_score: item.average_score,
      popularity: 0,
    } satisfies AnimeSearchResult : null
  const open = () => {
    if (anime) onOpen(anime)
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
          <div className="shrink-0 text-right">
            {item.predicted_rating === null ? (
              <><strong className="block text-lg text-violet">#{item.rank}</strong><span className="text-[11px] text-slate-500">model pick</span></>
            ) : (
              <><strong className="block text-lg text-violet">{item.predicted_rating.toFixed(1)}</strong><span className="text-[11px] text-slate-500">predicted / 10</span></>
            )}
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5 text-xs text-slate-300">
          {item.year && <span className="rounded-full bg-white/10 px-2.5 py-1">{item.year}</span>}
          {item.format && <span className="rounded-full bg-white/10 px-2.5 py-1">{item.format}</span>}
          {item.genres.slice(0, 3).map((genre) => <span className="rounded-full bg-white/10 px-2.5 py-1" key={genre}>{genre}</span>)}
        </div>
        {item.average_score !== null && <p className="mt-3 text-sm text-slate-400">AniList community score <span className="text-slate-200">{item.average_score}/100</span></p>}
        <div className="mt-auto flex flex-wrap items-end gap-x-4 gap-y-2 pt-5">
          <button className="text-sm text-violet transition hover:text-white disabled:text-slate-600" type="button" disabled={!canOpen} onClick={open}>{canOpen ? 'View details' : 'Details unavailable'}</button>
          {ratingOpen ? (
            <label className="flex items-center gap-2 text-sm text-slate-300">
              Your score
              <select
                className="rounded-lg border border-violet/40 bg-ink px-2.5 py-1.5 text-sm text-white outline-none focus:border-violet"
                aria-label={`Rate ${item.title}`}
                defaultValue=""
                onChange={(event) => {
                  if (!anime) return
                  onRate(anime, Number(event.target.value))
                }}
              >
                <option value="" disabled>1–10</option>
                {Array.from({ length: 10 }, (_, index) => index + 1).map((rating) => <option value={rating} key={rating}>{rating}/10</option>)}
              </select>
            </label>
          ) : (
            <button className="text-sm text-sakura transition hover:text-white disabled:text-slate-600" type="button" disabled={!canOpen} onClick={() => setRatingOpen(true)}>{canOpen ? 'Already watched? Rate it' : 'Rating unavailable'}</button>
          )}
          <button className="rounded-lg bg-black/30 px-2.5 py-1 text-sm text-slate-400 transition hover:bg-black/50 hover:text-white" type="button" onClick={() => onArchive({ mal_id: item.mal_id, title: item.title })}>Not interested</button>
        </div>
      </div>
    </li>
  )
}
