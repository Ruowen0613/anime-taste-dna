import { AnimeDetailsModal } from '../animeDetails/components/AnimeDetailsModal'
import { useAnimeDetails } from '../animeDetails/useAnimeDetails'
import type { AnimeSearchResult, RatedAnime } from '../discover/types'
import { RatingsPanel } from './components/RatingsPanel'

type MyRatingsPageProps = {
  ratings: RatedAnime[]
  onRate: (anime: AnimeSearchResult, rating: number) => void
  onRemove: (malId: number) => void
  onRecommend: () => void
}

export function MyRatingsPage({ ratings, onRate, onRemove, onRecommend }: MyRatingsPageProps) {
  const animeDetails = useAnimeDetails()

  return (
    <>
      <header className="pb-8 pt-16 sm:pt-24">
        <p className="text-sm font-semibold uppercase tracking-[0.28em] text-violet">Your anime library</p>
        <h1 className="mt-3 text-4xl font-bold sm:text-6xl">My Ratings</h1>
        <p className="mt-4 max-w-2xl text-slate-300">Review, sort, edit, or remove the scores that shape your recommendations.</p>
      </header>
      <RatingsPanel ratings={ratings} onRate={onRate} onRemove={onRemove} onOpen={animeDetails.open} onRecommend={onRecommend} hideVisualTitle />
      <AnimeDetailsModal anime={animeDetails.selected} details={animeDetails.details} loading={animeDetails.loading} error={animeDetails.error} onClose={animeDetails.close} />
    </>
  )
}
