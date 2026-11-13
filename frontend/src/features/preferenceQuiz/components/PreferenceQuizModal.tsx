import { useCallback, useEffect, useRef, useState } from 'react'
import { getQuizOptions } from '../preferenceQuizApi'
import type { PairwisePreference, QuizOption } from '../types'

type PreferenceQuizModalProps = {
  excludedMalIds: number[]
  completedRounds: number
  onSubmit: (preference: PairwisePreference, shownMalIds: number[]) => void
  onSkip: (shownMalIds: number[]) => void
  onClose: () => void
}

export function PreferenceQuizModal({ excludedMalIds, completedRounds, onSubmit, onSkip, onClose }: PreferenceQuizModalProps) {
  const [items, setItems] = useState<QuizOption[]>([])
  const [most, setMost] = useState<number | null>(null)
  const [least, setLeast] = useState<number | null>(null)
  const [expanded, setExpanded] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)

  const load = useCallback(async (extraExcluded: number[] = []) => {
    controller.current?.abort()
    const nextController = new AbortController()
    controller.current = nextController
    setLoading(true)
    setError('')
    setMost(null)
    setLeast(null)
    setExpanded(null)
    try {
      setItems(await getQuizOptions([...new Set([...excludedMalIds, ...extraExcluded])], nextController.signal))
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      setError(caught instanceof Error ? caught.message : 'Could not load taste quiz options.')
    } finally {
      if (controller.current === nextController) setLoading(false)
    }
  }, [excludedMalIds])

  useEffect(() => {
    void load()
    return () => controller.current?.abort()
  }, [load])

  const shownMalIds = items.map(({ mal_id }) => mal_id)
  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-ink/95 p-4 backdrop-blur-sm sm:p-8" role="dialog" aria-modal="true" aria-labelledby="preference-quiz-title">
      <div className="mx-auto max-w-6xl rounded-3xl border border-white/10 bg-[#111625] p-5 shadow-2xl sm:p-8">
        <div className="flex items-start justify-between gap-6">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.25em] text-violet">Experimental taste refinement</p>
            <h2 id="preference-quiz-title" className="mt-2 text-3xl font-bold sm:text-4xl">Pick your instinct</h2>
            <p className="mt-2 max-w-2xl text-slate-300">Choose one you are most interested in and one you are least interested in. You do not need to have watched them.</p>
            <p className="mt-2 text-sm text-slate-500">{completedRounds} preference round{completedRounds === 1 ? '' : 's'} saved</p>
          </div>
          <button className="rounded-full bg-white/5 px-4 py-2 text-slate-300 transition hover:bg-white/10 hover:text-white" type="button" onClick={onClose} aria-label="Close taste quiz">Close</button>
        </div>

        {loading && <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4" aria-label="Loading taste quiz">{Array.from({ length: 4 }, (_, index) => <div className="h-[28rem] animate-pulse rounded-2xl bg-white/5" key={index} />)}</div>}
        {error && <div className="mt-8 rounded-2xl border border-rose-400/20 bg-rose-400/10 p-5 text-rose-200" role="alert">{error}<button className="ml-3 underline" type="button" onClick={() => void load()}>Try again</button></div>}
        {!loading && !error && (
          <>
            <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {items.map((item) => (
                <article className={`overflow-hidden rounded-2xl border bg-white/5 transition ${most === item.mal_id ? 'border-emerald-400/70' : least === item.mal_id ? 'border-rose-400/70' : 'border-white/10'}`} key={item.mal_id}>
                  {item.cover_image_url ? <img className="h-64 w-full object-cover" src={item.cover_image_url} alt={`${item.title} cover`} /> : <div className="flex h-64 items-center justify-center bg-white/5 text-slate-500">No image</div>}
                  <div className="p-4">
                    <h3 className="line-clamp-2 min-h-12 font-semibold">{item.title}</h3>
                    <p className="mt-1 text-xs text-slate-400">{[item.year, item.format].filter(Boolean).join(' · ')}</p>
                    <div className="mt-3 flex flex-wrap gap-1">{item.genres.slice(0, 3).map((genre) => <span className="rounded-full bg-white/10 px-2 py-1 text-[11px] text-slate-300" key={genre}>{genre}</span>)}</div>
                    <button className="mt-3 text-sm text-violet hover:text-white" type="button" onClick={() => setExpanded(expanded === item.mal_id ? null : item.mal_id)}>{expanded === item.mal_id ? 'Hide summary' : 'Quick summary'}</button>
                    {expanded === item.mal_id && <p className="mt-2 text-sm leading-relaxed text-slate-300">{item.summary ?? 'No short summary is available.'}</p>}
                    <div className="mt-4 grid grid-cols-2 gap-2">
                      <button className={`rounded-lg px-2 py-2 text-xs font-semibold transition ${most === item.mal_id ? 'bg-emerald-400 text-ink' : 'bg-emerald-400/10 text-emerald-300 hover:bg-emerald-400/20'}`} type="button" onClick={() => { setMost(item.mal_id); if (least === item.mal_id) setLeast(null) }}>Most interested</button>
                      <button className={`rounded-lg px-2 py-2 text-xs font-semibold transition ${least === item.mal_id ? 'bg-rose-400 text-ink' : 'bg-rose-400/10 text-rose-300 hover:bg-rose-400/20'}`} type="button" onClick={() => { setLeast(item.mal_id); if (most === item.mal_id) setMost(null) }}>Least interested</button>
                    </div>
                  </div>
                </article>
              ))}
            </div>
            <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
              <button className="text-sm text-slate-400 underline decoration-white/20 underline-offset-4 hover:text-white" type="button" onClick={() => onSkip(shownMalIds)}>Skip this set</button>
              <button className="rounded-xl bg-white px-5 py-3 font-semibold text-ink transition hover:bg-violet hover:text-white disabled:opacity-40" type="button" disabled={most === null || least === null} onClick={() => { if (most === null || least === null) return; onSubmit({ most_mal_id: most, least_mal_id: least }, shownMalIds) }}>Refine my taste</button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
