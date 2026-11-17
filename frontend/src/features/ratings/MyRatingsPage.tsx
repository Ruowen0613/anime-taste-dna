import { AnimeDetailsModal } from '../animeDetails/components/AnimeDetailsModal'
import { useAnimeDetails } from '../animeDetails/useAnimeDetails'
import type { AnimeSearchResult, RatedAnime } from '../discover/types'
import { NotInterestedPanel } from './components/NotInterestedPanel'
import { RatingsPanel } from './components/RatingsPanel'
import { useNotInterested } from './useNotInterested'

type MyRatingsPageProps = {
  ratings: RatedAnime[]
  onRate: (anime: AnimeSearchResult, rating: number) => void
  onRemove: (malId: number) => void
  onRecommend: () => void
  onPopularPicks: () => void
}

export function MyRatingsPage({ ratings, onRate, onRemove, onRecommend, onPopularPicks }: MyRatingsPageProps) {
  const animeDetails = useAnimeDetails()
  const notInterested = useNotInterested()

  return (
    <>
      <header className="pb-8 pt-16 sm:pt-24">
        <p className="text-sm font-semibold uppercase tracking-[0.28em] text-violet">Your anime library</p>
        <h1 className="mt-3 text-4xl font-bold sm:text-6xl">My Ratings</h1>
        <p className="mt-4 max-w-2xl text-slate-300">
          Rate up to 100 anime to shape your recommendations. Use{' '}
          <button className="text-violet underline decoration-violet/40 underline-offset-4 transition hover:text-white" type="button" onClick={onPopularPicks}>Popular Picks</button>{' '}
          to search for any title, or browse its Top 100 to get started.
        </p>
      </header>
      <RatingsPanel ratings={ratings} onRate={onRate} onRemove={onRemove} onOpen={animeDetails.open} onRecommend={onRecommend} hideVisualTitle />
      <NotInterestedPanel items={notInterested.items} onOpen={animeDetails.open} onRestore={notInterested.restore} />
      <AnimeDetailsModal anime={animeDetails.selected} details={animeDetails.details} loading={animeDetails.loading} error={animeDetails.error} onClose={animeDetails.close} />
    </>
  )
}
