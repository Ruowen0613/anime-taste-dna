import { RatingsPanel } from './components/RatingsPanel'
import { SearchForm } from './components/SearchForm'
import { SearchResults } from './components/SearchResults'
import { useAnimeSearch } from './useAnimeSearch'
import { useAnimeDetails } from './useAnimeDetails'
import { useRatings } from './useRatings'

export function DiscoverPage() {
  const search = useAnimeSearch()
  const ratings = useRatings()
  const animeDetails = useAnimeDetails()

  return (
    <>
      <SearchForm query={search.query} searching={search.searching} error={search.error} onQueryChange={search.setQuery} onSubmit={search.submit} />
      <SearchResults results={search.results} ratings={ratings.ratings} onRate={ratings.setRating} onOpen={animeDetails.open} />
      <RatingsPanel ratings={ratings.ratings} onRemove={ratings.removeRating} onOpen={animeDetails.open} />
      <AnimeDetailsModal anime={animeDetails.selected} details={animeDetails.details} loading={animeDetails.loading} error={animeDetails.error} onClose={animeDetails.close} />
    </>
  )
}
import { AnimeDetailsModal } from './components/AnimeDetailsModal'
