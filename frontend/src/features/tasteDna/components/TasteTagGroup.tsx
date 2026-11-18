import type { TasteDimension } from '../types'

type TasteTagGroupProps = {
  title: string
  description: string
  dimensions: TasteDimension[]
}

export function TasteTagGroup({ title, description, dimensions }: TasteTagGroupProps) {
  return (
    <section className="rounded-3xl border border-white/10 bg-white/[0.04] p-5 sm:p-6">
      <h2 className="text-xl font-semibold">{title}</h2>
      <p className="mt-1 text-sm text-slate-400">{description}</p>
      {dimensions.length > 0 ? (
        <ol className="mt-5 space-y-4">
          {dimensions.map((dimension) => (
            <li key={dimension.label}>
              <div className="flex items-end justify-between gap-4">
                <div>
                  <h3 className="font-medium text-slate-100">{dimension.label}</h3>
                  <p className="mt-0.5 text-xs text-slate-500">
                    {dimension.support} supporting title{dimension.support === 1 ? '' : 's'}
                  </p>
                </div>
                <strong className="text-sm text-violet">{dimension.score.toFixed(1)}</strong>
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
    </section>
  )
}
