import { useEffect } from 'react'
import type { AnimeDetails, AnimeSearchResult } from '../types'

type AnimeDetailsModalProps = {
  anime: AnimeSearchResult | null
  details: AnimeDetails | null
  loading: boolean
  error: string
  onClose: () => void
}

function formatLabel(value: string | null) {
  return value?.toLowerCase().replaceAll('_', ' ').replace(/^./, (letter) => letter.toUpperCase())
}

export function AnimeDetailsModal({ anime, details, loading, error, onClose }: AnimeDetailsModalProps) {
  useEffect(() => {
    if (!anime) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [anime, onClose])

  if (!anime) return null
  const displayed = details ?? anime

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/75 p-0 backdrop-blur-sm sm:items-center sm:p-6" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="relative max-h-[92vh] w-full max-w-4xl overflow-y-auto rounded-t-3xl border border-white/10 bg-ink shadow-2xl sm:rounded-3xl" role="dialog" aria-modal="true" aria-labelledby="anime-details-title">
        <button className="absolute right-4 top-4 z-10 rounded-full bg-black/60 px-3 py-2 text-sm text-white transition hover:bg-black/80" type="button" onClick={onClose} aria-label="Close anime details">Close</button>
        <div className="relative h-44 overflow-hidden bg-gradient-to-br from-violet/30 to-sakura/20 sm:h-56">
          {(details?.banner_image_url || displayed.cover_image_url) && <img className="h-full w-full object-cover opacity-60" src={details?.banner_image_url ?? displayed.cover_image_url ?? ''} alt="" />}
          <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/20 to-transparent" />
        </div>
        <div className="relative -mt-16 px-5 pb-8 sm:px-8">
          <div className="flex items-end gap-5">
            {displayed.cover_image_url && <img className="hidden h-48 w-32 rounded-xl object-cover shadow-xl sm:block" src={displayed.cover_image_url} alt={`${displayed.title} cover`} />}
            <div className="min-w-0 pb-2">
              <h2 id="anime-details-title" className="text-2xl font-bold sm:text-4xl">{displayed.title}</h2>
              <div className="mt-3 flex flex-wrap gap-2 text-sm text-slate-300">
                {displayed.year && <span>{displayed.year}</span>}
                {displayed.format && <span>· {displayed.format}</span>}
                {details?.episodes && <span>· {details.episodes} episodes</span>}
                {details?.duration_minutes && <span>· {details.duration_minutes} min</span>}
              </div>
            </div>
            {details?.average_score != null && (
              <div className="ml-auto shrink-0 rounded-2xl bg-white/10 px-4 py-3 text-center">
                <strong className="block text-2xl text-sakura">{details.average_score}</strong>
                <span className="text-xs text-slate-400">AniList / 100</span>
              </div>
            )}
          </div>

          {loading && <p className="py-12 text-center text-slate-400">Loading details…</p>}
          {error && <p className="mt-8 rounded-xl bg-rose-400/10 p-4 text-rose-200" role="alert">{error}</p>}
          {details && (
            <div className="mt-8 space-y-8">
              <div className="flex flex-wrap gap-2">
                {details.genres.map((genre) => <span className="rounded-full bg-white/10 px-3 py-1 text-sm text-slate-200" key={genre}>{genre}</span>)}
                {details.status && <span className="rounded-full bg-white/10 px-3 py-1 text-sm text-slate-200">{formatLabel(details.status)}</span>}
                {details.source && <span className="rounded-full bg-white/10 px-3 py-1 text-sm text-slate-200">From {formatLabel(details.source)}</span>}
              </div>
              <div>
                <h3 className="text-lg font-semibold">About</h3>
                <p className="mt-3 whitespace-pre-line leading-7 text-slate-300">{details.description || 'No description is available from AniList.'}</p>
              </div>
              <div>
                <div className="flex items-center justify-between gap-4">
                  <h3 className="text-lg font-semibold">Community reviews</h3>
                  <a className="text-sm text-violet hover:text-white" href={details.anilist_url} target="_blank" rel="noreferrer">View on AniList ↗</a>
                </div>
                {details.reviews.length === 0 ? (
                  <p className="mt-3 text-slate-400">No public reviews are available for this title.</p>
                ) : (
                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    {details.reviews.map((review) => (
                      <article className="rounded-2xl border border-white/10 bg-white/5 p-4" key={review.id}>
                        <div className="flex items-center gap-3 text-sm text-slate-400">
                          {review.user_avatar_url && <img className="h-8 w-8 rounded-full object-cover" src={review.user_avatar_url} alt="" />}
                          <span>{review.user_name}</span>
                          <span className="ml-auto">{review.helpful_count}/{review.rating_count} helpful</span>
                        </div>
                        <p className="mt-3 line-clamp-4 text-sm leading-6 text-slate-300">{review.summary}</p>
                      </article>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
