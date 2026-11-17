import { useMemo } from 'react'
import { AnimeDetailsModal } from '../animeDetails/components/AnimeDetailsModal'
import { useAnimeDetails } from '../animeDetails/useAnimeDetails'
import { SearchResults } from '../discover/components/SearchResults'
import type { AnimeSearchResult, RatedAnime } from '../discover/types'
import { useAnimeSearch } from '../discover/useAnimeSearch'
import { useNotInterested } from '../ratings/useNotInterested'
import { PopularPickCard } from './components/PopularPickCard'
import { PopularPicksSearch } from './components/PopularPicksSearch'
import { usePopularPicks } from './usePopularPicks'

type PopularPicksPageProps = {
  ratings: RatedAnime[]
  onRate: (anime: AnimeSearchResult, rating: number) => void
  onRemove: (malId: number) => void
  onRecommend: () => void
}

export function PopularPicksPage({ ratings, onRate, onRemove, onRecommend }: PopularPicksPageProps) {
  const search = useAnimeSearch()
  const popular = usePopularPicks()
  const animeDetails = useAnimeDetails()
  const notInterested = useNotInterested()
  const ratingsByMalId = new Map(ratings.map((rating) => [rating.mal_id, rating]))
  const archivedMalIds = new Set(notInterested.malIds)
  const ratedPopularCount = popular.data?.items.filter((item) => ratingsByMalId.has(item.mal_id)).length ?? 0
  const normalizedQuery = search.query.trim().toLocaleLowerCase()
  const filteredItems = useMemo(() => {
    const items = popular.data?.items ?? []
    if (normalizedQuery.length < 3) return items
    return items.filter((item) => item.title.toLocaleLowerCase().includes(normalizedQuery))
  }, [normalizedQuery, popular.data?.items])
  const popularMalIds = new Set(popular.data?.items.map((item) => item.mal_id) ?? [])
  const externalResults = normalizedQuery.length < 3
    ? []
    : search.results.filter((item) => item.mal_id === null || !popularMalIds.has(item.mal_id))

  return (
    <section className="pb-20 pt-16" aria-labelledby="popular-picks-heading">
      <div className="mb-8 flex flex-wrap items-end justify-between gap-5">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.28em] text-violet">Build a stronger taste profile</p>
          <h1 id="popular-picks-heading" className="mt-3 text-4xl font-bold sm:text-6xl">Popular Picks</h1>
          <p className="mt-4 max-w-3xl text-slate-300">Rate any titles you have watched. These are the 100 most-interacted-with anime supported by the current recommendation model.</p>
        </div>
        <div className="flex items-center gap-4">
          <p className="text-sm text-slate-400"><strong className="text-xl text-white">{ratedPopularCount}</strong> / 100 rated</p>
          {ratings.length >= 10 && <button className="rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-ink transition hover:bg-violet hover:text-white" type="button" onClick={onRecommend}>See recommendations</button>}
        </div>
      </div>

      {popular.loading && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-10" aria-label="Loading Popular Picks">
          {Array.from({ length: 100 }, (_, index) => <div className="aspect-[2/3] animate-pulse rounded-xl bg-white/5" key={index} />)}
        </div>
      )}
      {popular.error && <p className="rounded-2xl border border-rose-300/20 bg-rose-300/10 p-5 text-rose-100" role="alert">{popular.error} Make sure FastAPI is running on port 8000.</p>}
      {popular.data && (
        <>
          {popular.data.metadata_status !== 'complete' && <p className="mb-4 text-sm text-amber-200">Some AniList covers or details are temporarily unavailable; you can still rate every model-supported title.</p>}
          <PopularPicksSearch query={search.query} resultCount={filteredItems.length} totalCount={popular.data.items.length} externalResultCount={externalResults.length} searching={search.searching} error={search.error} onQueryChange={search.setQuery} />
          {filteredItems.length === 0 ? (
            <p className="rounded-2xl border border-dashed border-white/15 p-10 text-center text-slate-400">No Popular Picks match “{search.query.trim()}”.</p>
          ) : <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-10">
            {filteredItems.map((anime) => (
              <PopularPickCard
                anime={anime}
                rating={ratingsByMalId.get(anime.mal_id)}
                notInterested={archivedMalIds.has(anime.mal_id)}
                onRate={onRate}
                onRemove={onRemove}
                onNotInterested={notInterested.add}
                onRestoreInterest={(malId) => notInterested.restore([malId])}
                onOpen={animeDetails.open}
                key={anime.mal_id}
              />
            ))}
          </div>}
          {externalResults.length > 0 && (
            <div className="mx-auto mt-14 max-w-6xl">
              <SearchResults
                results={externalResults}
                ratings={ratings}
                onRate={onRate}
                onOpen={animeDetails.open}
                sortBy={search.sortBy}
                onSortChange={search.setSortBy}
                searching={search.searching}
                archivedMalIds={archivedMalIds}
                onRemove={onRemove}
                onNotInterested={notInterested.add}
                onRestoreInterest={(malId) => notInterested.restore([malId])}
              />
            </div>
          )}
        </>
      )}
      <AnimeDetailsModal anime={animeDetails.selected} details={animeDetails.details} loading={animeDetails.loading} error={animeDetails.error} onClose={animeDetails.close} />
    </section>
  )
}
