import { useState } from 'react'
import type { TasteDimension } from '../types'

type TasteTagGroupProps = {
  title: string
  description: string
  dimensions: TasteDimension[]
  defaultExpanded?: boolean
}

export function TasteTagGroup({ title, description, dimensions, defaultExpanded = false }: TasteTagGroupProps) {
  const [expanded, setExpanded] = useState(defaultExpanded)

  return (
    <section className="rounded-3xl border border-white/10 bg-white/[0.04] p-5 sm:p-6">
      <button
        className="flex w-full items-start justify-between gap-4 text-left lg:pointer-events-none"
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((current) => !current)}
      >
        <span>
          <span className="block text-xl font-semibold">{title}</span>
          <span className="mt-1 block text-sm font-normal text-slate-400">{description}</span>
        </span>
        <span className="mt-1 text-slate-400 lg:hidden" aria-hidden="true">{expanded ? '−' : '+'}</span>
      </button>
      <div className={`${expanded ? 'block' : 'hidden'} lg:block`}>
        {dimensions.length > 0 ? (
          <ol className="mt-5 space-y-4">
          {dimensions.map((dimension) => (
            <li key={dimension.label}>
              <div className="flex min-w-0 items-end justify-between gap-4">
                <div className="min-w-0">
                  <h3 className="break-words font-medium text-slate-100">{dimension.label}</h3>
                  <details className="group relative mt-0.5 w-fit max-w-full">
                    <summary className="cursor-pointer list-none text-xs text-slate-500 underline decoration-dotted underline-offset-4 transition hover:text-slate-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-violet">
                      {dimension.support} supporting title{dimension.support === 1 ? '' : 's'}
                    </summary>
                    <div className="relative z-10 mt-2 hidden w-[min(17rem,calc(100vw-5rem))] rounded-xl border border-white/10 bg-[#151a2b] p-3 text-xs leading-5 text-slate-300 shadow-2xl group-hover:block group-focus-within:block group-open:block sm:absolute sm:left-0 sm:top-full">
                      <p className="font-semibold text-white">Supporting titles</p>
                      <ul className="mt-1 max-h-52 space-y-1 overflow-y-auto pr-1">
                        {dimension.contributing_titles.map((contribution) => (
                          <li key={contribution.mal_id}>{contribution.title} <span className="text-slate-500">{contribution.rating}/10</span></li>
                        ))}
                      </ul>
                    </div>
                  </details>
                </div>
                <strong className="shrink-0 text-sm text-violet">{dimension.score.toFixed(1)}</strong>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-sakura to-violet"
                  style={{ width: `${dimension.score}%` }}
                />
              </div>
            </li>
          ))}
          </ol>
        ) : (
          <p className="mt-5 text-sm text-slate-500">Not enough AniList metadata yet.</p>
        )}
      </div>
    </section>
  )
}
