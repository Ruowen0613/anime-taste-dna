import { useEffect, useState } from 'react'
import { getHealth } from './api/health'

type ApiState = 'checking' | 'online' | 'offline'

function App() {
  const [apiState, setApiState] = useState<ApiState>('checking')

  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then(() => setApiState('online'))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        setApiState('offline')
      })
    return () => controller.abort()
  }, [])

  const statusText = {
    checking: 'Checking API…',
    online: 'API connected',
    offline: 'API offline — start FastAPI on port 8000',
  }[apiState]

  return (
    <main className="relative min-h-screen overflow-hidden bg-ink text-white">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(139,124,255,0.24),transparent_40%),radial-gradient(circle_at_85%_70%,rgba(255,119,168,0.18),transparent_35%)]" />
      <div className="relative mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-8 sm:px-10">
        <nav className="flex items-center justify-between" aria-label="Primary navigation">
          <a className="text-lg font-semibold tracking-tight" href="/">
            Anime Taste <span className="text-sakura">DNA</span>
          </a>
          <span
            className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300"
            aria-live="polite"
          >
            <span className={`mr-2 inline-block h-2 w-2 rounded-full ${apiState === 'online' ? 'bg-emerald-400' : 'bg-amber-400'}`} />
            {statusText}
          </span>
        </nav>

        <section className="flex flex-1 items-center py-20">
          <div className="max-w-3xl">
            <p className="mb-5 text-sm font-semibold uppercase tracking-[0.25em] text-violet">Your taste, decoded</p>
            <h1 className="text-5xl font-bold leading-[1.05] tracking-tight sm:text-7xl">
              Find anime that feels made for <span className="bg-gradient-to-r from-sakura to-violet bg-clip-text text-transparent">you.</span>
            </h1>
            <p className="mt-7 max-w-2xl text-lg leading-8 text-slate-300 sm:text-xl">
              Rate what you have watched. A real recommendation model learns your preferences, finds your next favorites, and maps your unique Anime Taste DNA.
            </p>
            <div className="mt-10 flex flex-wrap gap-4">
              <button className="rounded-full bg-white px-6 py-3 font-semibold text-ink transition hover:bg-slate-200" type="button" disabled>
                Start rating soon
              </button>
              <a className="rounded-full border border-white/15 px-6 py-3 font-semibold text-white transition hover:bg-white/10" href="#roadmap">
                See the MVP
              </a>
            </div>
          </div>
        </section>

        <section id="roadmap" className="grid gap-4 border-t border-white/10 py-8 text-sm text-slate-300 sm:grid-cols-3">
          <p><strong className="block text-white">01 · Discover</strong> Search and rate 10–20 anime.</p>
          <p><strong className="block text-white">02 · Recommend</strong> Get an ML-ranked personal Top 10.</p>
          <p><strong className="block text-white">03 · Decode</strong> Explore an interpretable taste profile.</p>
        </section>
      </div>
    </main>
  )
}

export default App
