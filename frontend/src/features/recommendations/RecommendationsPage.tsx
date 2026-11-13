import { AnimeDetailsModal } from '../animeDetails/components/AnimeDetailsModal'
import { useAnimeDetails } from '../animeDetails/useAnimeDetails'
import type { AnimeSearchResult, RatedAnime } from '../discover/types'
import { RecommendationCard } from './components/RecommendationCard'
import { useRecommendations } from './useRecommendations'

type RecommendationsPageProps = {
  ratings: RatedAnime[]
  onRate: (anime: AnimeSearchResult, rating: number) => void
  onBack: () => void
}

export function RecommendationsPage({ ratings, onRate, onBack }: RecommendationsPageProps) {
  const recommendations = useRecommendations(ratings)
  const animeDetails = useAnimeDetails()

  if (ratings.length < 10) {
    return (
      <section className="py-24 text-center">
        <p className="text-slate-400">Rate at least 10 anime before generating recommendations.</p>
        <button className="mt-6 rounded-xl bg-white px-5 py-3 font-semibold text-ink" type="button" onClick={onBack}>Back to Discover</button>
      </section>
    )
  }

  return (
    <section className="py-16" aria-labelledby="recommendations-heading">
      <button className="text-sm text-violet transition hover:text-white" type="button" onClick={onBack}>← Back to your ratings</button>
      <div className="mt-8 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
        <div><p className="text-sm font-semibold uppercase tracking-[0.28em] text-violet">Your next watch</p><h1 id="recommendations-heading" className="mt-3 text-4xl font-bold sm:text-6xl">Recommendations</h1><p className="mt-4 max-w-2xl text-slate-300">Ranked by the matrix-factorization model from your ratings. Predicted ratings are model estimates, not community scores.</p></div>
        <button className="self-start rounded-xl border border-white/15 bg-white/5 px-4 py-2.5 text-sm transition hover:bg-white/10 disabled:opacity-50 lg:self-auto" type="button" disabled={recommendations.loading} onClick={() => void recommendations.generate()}>Regenerate</button>
      </div>
      {recommendations.loading && <div className="mt-12 grid gap-4 md:grid-cols-2" aria-label="Loading recommendations">{Array.from({ length: 4 }, (_, index) => <div className="h-52 animate-pulse rounded-2xl bg-white/5" key={index} />)}</div>}
      {recommendations.error && <div className="mt-10 rounded-2xl border border-rose-400/20 bg-rose-400/10 p-5 text-rose-200" role="alert"><p>{recommendations.error}</p><button className="mt-3 text-sm underline" type="button" onClick={() => void recommendations.generate()}>Try again</button></div>}
      {recommendations.result && !recommendations.loading && (
        <>
          <div className="mt-8 flex flex-wrap gap-3 text-sm text-slate-400"><span>{recommendations.result.mapped_ratings}/{Math.min(ratings.length, 20)} ratings mapped</span><span>·</span><span>Model {recommendations.result.model_version}</span>{recommendations.result.metadata_status !== 'complete' && <span className="text-amber-300">· Some AniList metadata is unavailable</span>}</div>
          {recommendations.result.unmapped_mal_ids.length > 0 && <p className="mt-3 text-sm text-slate-500">{recommendations.result.unmapped_mal_ids.length} newer or unsupported title{recommendations.result.unmapped_mal_ids.length === 1 ? '' : 's'} could not influence this result.</p>}
          {recommendations.result.items.length === 0 ? <p className="mt-12 rounded-2xl border border-dashed border-white/15 p-10 text-center text-slate-400">No unseen recommendations are available.</p> : <ul className="mt-10 grid gap-4 md:grid-cols-2">{recommendations.result.items.map((item) => <RecommendationCard item={item} onOpen={animeDetails.open} onRate={onRate} key={item.mal_id} />)}</ul>}
        </>
      )}
      <AnimeDetailsModal anime={animeDetails.selected} details={animeDetails.details} loading={animeDetails.loading} error={animeDetails.error} onClose={animeDetails.close} />
    </section>
  )
}
