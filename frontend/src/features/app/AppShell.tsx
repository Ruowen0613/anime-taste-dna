import { useState } from 'react'
import { DiscoverPage } from '../discover/DiscoverPage'
import { PreferenceQuizModal } from '../preferenceQuiz/components/PreferenceQuizModal'
import { usePreferenceQuiz } from '../preferenceQuiz/usePreferenceQuiz'
import { RecommendationsPage } from '../recommendations/RecommendationsPage'
import { useRatings } from '../ratings/useRatings'
import { ApiStatus } from '../system/ApiStatus'

type View = 'discover' | 'recommendations'

export function AppShell() {
  const [view, setView] = useState<View>('discover')
  const [quizOpen, setQuizOpen] = useState(false)
  const ratings = useRatings()
  const quiz = usePreferenceQuiz()

  return (
    <main className="min-h-screen bg-ink text-white">
      <div className="mx-auto max-w-6xl px-6 py-8 sm:px-10">
        <nav className="flex flex-wrap items-center justify-between gap-4" aria-label="Primary navigation">
          <button className="text-lg font-semibold tracking-tight" type="button" onClick={() => setView('discover')}>
            Anime Taste <span className="text-sakura">DNA</span>
          </button>
          <div className="flex items-center gap-3">
            <button className="rounded-full border border-sakura/30 bg-sakura/10 px-4 py-2 text-sm text-sakura transition hover:bg-sakura/20 hover:text-white" type="button" onClick={() => setQuizOpen(true)}>Taste quiz</button>
            {ratings.ratings.length >= 10 && (
              <button className="rounded-full border border-violet/40 bg-violet/10 px-4 py-2 text-sm text-violet transition hover:bg-violet/20 hover:text-white" type="button" onClick={() => setView(view === 'discover' ? 'recommendations' : 'discover')}>
                {view === 'discover' ? 'Recommendations' : 'Discover'}
              </button>
            )}
            <ApiStatus />
          </div>
        </nav>
        {view === 'discover' ? (
          <DiscoverPage ratings={ratings.ratings} onRate={ratings.setRating} onRemove={ratings.removeRating} onRecommend={() => setView('recommendations')} />
        ) : (
          <RecommendationsPage ratings={ratings.ratings} preferences={quiz.preferences} onRate={ratings.setRating} onBack={() => setView('discover')} />
        )}
      </div>
      {quizOpen && (
        <PreferenceQuizModal
          excludedMalIds={[...ratings.ratings.map(({ mal_id }) => mal_id), ...quiz.seenMalIds]}
          completedRounds={quiz.preferences.length}
          onSubmit={quiz.recordPreference}
          onSkip={quiz.recordSeen}
          onClose={() => setQuizOpen(false)}
        />
      )}
    </main>
  )
}
