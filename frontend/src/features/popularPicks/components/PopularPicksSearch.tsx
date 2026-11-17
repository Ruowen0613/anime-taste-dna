type PopularPicksSearchProps = {
  query: string
  resultCount: number
  totalCount: number
  externalResultCount: number
  searching: boolean
  error: string
  onQueryChange: (query: string) => void
}

export function PopularPicksSearch({ query, resultCount, totalCount, externalResultCount, searching, error, onQueryChange }: PopularPicksSearchProps) {
  const active = query.trim().length >= 3

  return (
    <div className="mb-6 flex flex-wrap items-center gap-3">
      <label className="sr-only" htmlFor="popular-picks-search">Search Popular Picks</label>
      <div className="min-w-0 flex-1 sm:max-w-xl">
        <input
          id="popular-picks-search"
          className="w-full rounded-2xl border border-white/15 bg-white/5 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-violet"
          type="search"
          placeholder="Search Popular Picks or any anime…"
          autoComplete="off"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
        />
      </div>
      <p className="text-sm text-slate-400" aria-live="polite">
        {searching ? 'Searching AniList…' : active ? `${resultCount} Popular Pick${resultCount === 1 ? '' : 's'}${externalResultCount ? ` · ${externalResultCount} more` : ''}` : 'Type 3 characters to search'}
        {active && <span className="sr-only"> out of {totalCount}</span>}
      </p>
      {error && <p className="w-full text-sm text-rose-300" role="alert">{error}</p>}
    </div>
  )
}
