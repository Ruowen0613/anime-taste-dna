import { AnimeDetailsModal } from '../animeDetails/components/AnimeDetailsModal'
import { useAnimeDetails } from '../animeDetails/useAnimeDetails'
import { RatingsPanel } from './components/RatingsPanel'
import { SearchForm } from './components/SearchForm'
import { SearchResults } from './components/SearchResults'
import { useAnimeSearch } from './useAnimeSearch'
import type { AnimeSearchResult, RatedAnime } from './types'

type DiscoverPageProps = {
  ratings: RatedAnime[]
  onRate: (anime: AnimeSearchResult, rating: number) => void
  onRemove: (malId: number) => void
  onRecommend: () => void
}

export function DiscoverPage({ ratings, onRate, onRemove, onRecommend }: DiscoverPageProps) {
  const search = useAnimeSearch()
  const animeDetails = useAnimeDetails()

  return (
    <>
      <SearchForm query={search.query} searching={search.searching} error={search.error} onQueryChange={search.setQuery} onSubmit={search.submit} />
      <SearchResults results={search.results} ratings={ratings} onRate={onRate} onOpen={animeDetails.open} sortBy={search.sortBy} onSortChange={search.setSortBy} searching={search.searching} />
      <RatingsPanel ratings={ratings} onRate={onRate} onRemove={onRemove} onOpen={animeDetails.open} onRecommend={onRecommend} />
      <AnimeDetailsModal anime={animeDetails.selected} details={animeDetails.details} loading={animeDetails.loading} error={animeDetails.error} onClose={animeDetails.close} />
    </>
  )
}
