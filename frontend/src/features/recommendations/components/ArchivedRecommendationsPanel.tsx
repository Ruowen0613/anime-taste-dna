import { useState } from 'react'
import type { ArchivedRecommendation } from '../types'

type ArchivedRecommendationsPanelProps = {
  items: ArchivedRecommendation[]
  onRestore: (malIds: number[]) => void
  onClose: () => void
}

export function ArchivedRecommendationsPanel({ items, onRestore, onClose }: ArchivedRecommendationsPanelProps) {
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const allSelected = selected.size === items.length

  function toggle(malId: number) {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(malId)) next.delete(malId)
      else next.add(malId)
      return next
    })
  }

  return (
    <section className="mt-5 rounded-2xl border border-white/10 bg-white/5 p-5" aria-labelledby="archived-recommendations-heading">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 id="archived-recommendations-heading" className="font-semibold">Hidden recommendations</h2>
          <p className="mt-1 text-sm text-slate-400">Select only the titles you want to put back into the recommendation pool.</p>
        </div>
        <button className="text-sm text-slate-400 transition hover:text-white" type="button" onClick={onClose}>Close</button>
      </div>
      <label className="mt-5 flex cursor-pointer items-center gap-3 border-b border-white/10 pb-3 text-sm text-slate-300">
        <input type="checkbox" checked={allSelected} onChange={() => setSelected(allSelected ? new Set() : new Set(items.map(({ mal_id }) => mal_id)))} />
        Select all
      </label>
      <div className="grid gap-2 py-3 sm:grid-cols-2">
        {items.map((item) => (
          <label className="flex cursor-pointer items-center gap-3 rounded-lg px-2 py-2 text-sm text-slate-200 transition hover:bg-white/5" key={item.mal_id}>
            <input type="checkbox" checked={selected.has(item.mal_id)} onChange={() => toggle(item.mal_id)} />
            <span className="line-clamp-2">{item.title}</span>
          </label>
        ))}
      </div>
      <button
        className="rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-ink transition hover:bg-violet hover:text-white disabled:opacity-40"
        type="button"
        disabled={selected.size === 0}
        onClick={() => onRestore([...selected])}
      >
        Restore selected ({selected.size})
      </button>
    </section>
  )
}
