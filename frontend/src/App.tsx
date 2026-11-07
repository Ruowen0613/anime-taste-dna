import { ApiStatus } from './features/system/ApiStatus'
import { DiscoverPage } from './features/discover/DiscoverPage'

function App() {
  return (
    <main className="min-h-screen bg-ink text-white">
      <div className="mx-auto max-w-6xl px-6 py-8 sm:px-10">
        <nav className="flex flex-wrap items-center justify-between gap-4" aria-label="Primary navigation">
          <a className="text-lg font-semibold tracking-tight" href="/">
            Anime Taste <span className="text-sakura">DNA</span>
          </a>
          <ApiStatus />
        </nav>
        <DiscoverPage />
      </div>
    </main>
  )
}

export default App
