import type { RatedAnime } from '../discover/types'
import { TasteRadar } from './components/TasteRadar'
import { TasteTagGroup } from './components/TasteTagGroup'
import { useTasteDna } from './useTasteDna'

type TasteDnaPageProps = {
  ratings: RatedAnime[]
  onPopularPicks: () => void
}

export function TasteDnaPage({ ratings, onPopularPicks }: TasteDnaPageProps) {
  const tasteDna = useTasteDna(ratings)

  return (
    <section className="pb-20 pt-16" aria-labelledby="taste-dna-heading">
      <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.28em] text-violet">Your ratings, decoded</p>
          <h1 id="taste-dna-heading" className="mt-3 text-4xl font-bold sm:text-6xl">Taste DNA</h1>
          <p className="mt-4 max-w-3xl text-slate-300">A transparent genre profile derived from your scores—not names assigned to hidden model factors. More ratings make each signal better supported.</p>
        </div>
        {ratings.length >= 5 && <button className="self-start rounded-xl border border-white/15 bg-white/5 px-4 py-2.5 text-sm transition hover:bg-white/10 disabled:opacity-50" type="button" disabled={tasteDna.loading} onClick={() => void tasteDna.generate()}>Refresh profile</button>}
      </div>

      {ratings.length < 5 ? (
        <div className="mt-12 rounded-3xl border border-dashed border-white/15 bg-white/[0.03] p-10 text-center">
          <p className="text-2xl font-semibold">Your profile needs a little more signal</p>
          <p className="mx-auto mt-3 max-w-xl text-slate-400">Rate at least 5 anime to reveal your first genre affinities. Ten or more gives the profile a stronger foundation.</p>
          <button className="mt-6 rounded-xl bg-white px-5 py-3 font-semibold text-ink transition hover:bg-violet hover:text-white" type="button" onClick={onPopularPicks}>Rate Popular Picks</button>
        </div>
      ) : tasteDna.loading && !tasteDna.data ? (
        <div className="mt-12 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          <div className="h-[420px] animate-pulse rounded-3xl bg-white/5" />
          <div className="grid gap-4 sm:grid-cols-2">{Array.from({ length: 6 }, (_, index) => <div className="h-48 animate-pulse rounded-2xl bg-white/5" key={index} />)}</div>
        </div>
      ) : tasteDna.error ? (
        <div className="mt-10 rounded-2xl border border-rose-400/20 bg-rose-400/10 p-5 text-rose-200" role="alert">
          <p>{tasteDna.error}</p>
          <button className="mt-3 text-sm underline" type="button" onClick={() => void tasteDna.generate()}>Try again</button>
        </div>
      ) : tasteDna.data ? (
        <>
          <div className="mt-8 flex flex-wrap items-center gap-3 text-sm text-slate-400">
            <span>{tasteDna.data.mapped_ratings}/{tasteDna.data.total_ratings} ratings mapped</span>
            <span>·</span>
            <span>Method {tasteDna.data.method_version}</span>
            {tasteDna.data.metadata_status !== 'complete' && <span className="text-amber-300">· Some ratings have no genre metadata</span>}
          </div>
          <div className="mt-8 rounded-3xl border border-white/10 bg-gradient-to-br from-violet/10 to-sakura/5 p-4 sm:p-6">
            <h2 className="px-3 pt-2 text-xl font-semibold">Genre Affinity</h2>
            <TasteRadar dimensions={tasteDna.data.genre_dimensions} />
            <p className="px-3 pb-2 text-center text-xs leading-relaxed text-slate-500">Eight fixed genre signals. 50 is neutral; sparse evidence is shrunk toward the center.</p>
          </div>
          <div className="mt-6 grid gap-5 lg:grid-cols-3">
            <TasteTagGroup title="Story Themes" description="The narrative themes that resonate with you most." dimensions={tasteDna.data.story_themes} />
            <TasteTagGroup title="World & Settings" description="Top 5 worlds, periods, and settings." dimensions={tasteDna.data.worlds_and_settings} />
            <TasteTagGroup title="Character" description="Top 5 cast traits and compositions." dimensions={tasteDna.data.characters} />
          </div>
        </>
      ) : null}
    </section>
  )
}
