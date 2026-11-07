import type { FormEvent } from 'react'

type SearchFormProps = {
  query: string
  searching: boolean
  error: string
  onQueryChange: (query: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}

export function SearchForm({ query, searching, error, onQueryChange, onSubmit }: SearchFormProps) {
  return (
    <header className="py-16 sm:py-24">
      <p className="mb-4 text-sm font-semibold uppercase tracking-[0.25em] text-violet">Discover your taste</p>
      <h1 className="max-w-4xl text-4xl font-bold tracking-tight sm:text-6xl">Rate anime you have watched.</h1>
      <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-300">
        Add 10–20 ratings to unlock recommendations. Your list stays in this browser.
      </p>
      <form className="mt-9 flex max-w-2xl gap-3" onSubmit={onSubmit}>
        <label className="sr-only" htmlFor="anime-search">Search anime</label>
        <input
          id="anime-search"
          className="min-w-0 flex-1 rounded-full border border-white/15 bg-white/5 px-5 py-3 outline-none transition placeholder:text-slate-500 focus:border-violet"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Type at least 3 characters…"
          maxLength={100}
        />
        <button className="rounded-full bg-white px-6 py-3 font-semibold text-ink disabled:opacity-50" disabled={searching || !query.trim()} type="submit">
          {searching ? 'Searching…' : 'Search'}
        </button>
      </form>
      {error && <p className="mt-3 text-sm text-rose-300" role="alert">{error}</p>}
    </header>
  )
}
