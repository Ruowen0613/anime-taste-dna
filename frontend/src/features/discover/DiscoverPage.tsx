import { RatingsPanel } from './components/RatingsPanel'
import { SearchForm } from './components/SearchForm'
import { SearchResults } from './components/SearchResults'
import { useAnimeSearch } from './useAnimeSearch'
import { useRatings } from './useRatings'

export function DiscoverPage() {
  const search = useAnimeSearch()
  const ratings = useRatings()

  return (
    <>
      <SearchForm query={search.query} searching={search.searching} error={search.error} onQueryChange={search.setQuery} onSubmit={search.submit} />
      <SearchResults results={search.results} ratings={ratings.ratings} onRate={ratings.setRating} />
      <RatingsPanel ratings={ratings.ratings} onRemove={ratings.removeRating} />
    </>
  )
}
