import { useState } from 'react'
import { PreferenceQuizModal } from '../preferenceQuiz/components/PreferenceQuizModal'
import { usePreferenceQuiz } from '../preferenceQuiz/usePreferenceQuiz'
import { PopularPicksPage } from '../popularPicks/PopularPicksPage'
import { MyRatingsPage } from '../ratings/MyRatingsPage'
import { RecommendationsPage } from '../recommendations/RecommendationsPage'
import { useRatings } from '../ratings/useRatings'
import { ApiStatus } from '../system/ApiStatus'

type View = 'ratings' | 'popular' | 'recommendations'
const TASTE_QUIZ_ENABLED = import.meta.env.VITE_ENABLE_TASTE_QUIZ === 'true'

export function AppShell() {
  const [view, setView] = useState<View>('popular')
  const [quizOpen, setQuizOpen] = useState(false)
  const ratings = useRatings()
  const quiz = usePreferenceQuiz()

  return (
    <main className="min-h-screen bg-ink text-white">
      <div className={`mx-auto px-6 py-8 sm:px-10 ${view === 'popular' ? 'max-w-[1800px]' : 'max-w-6xl'}`}>
        <nav className="flex flex-wrap items-center justify-between gap-4" aria-label="Primary navigation">
          <button className="text-lg font-semibold tracking-tight" type="button" onClick={() => setView('popular')}>
            Anime Taste <span className="text-sakura">DNA</span>
          </button>
          <div className="flex items-center gap-3">
            <button aria-current={view === 'popular' ? 'page' : undefined} className={`rounded-full border px-4 py-2 text-sm transition ${view === 'popular' ? 'border-violet bg-violet/20 text-white' : 'border-white/15 bg-white/5 text-slate-300 hover:border-violet/50 hover:text-white'}`} type="button" onClick={() => setView('popular')}>Popular Picks</button>
            <button aria-current={view === 'ratings' ? 'page' : undefined} className={`rounded-full border px-4 py-2 text-sm transition ${view === 'ratings' ? 'border-violet bg-violet/20 text-white' : 'border-white/15 bg-white/5 text-slate-300 hover:border-violet/50 hover:text-white'}`} type="button" onClick={() => setView('ratings')}>My Ratings</button>
            {TASTE_QUIZ_ENABLED && <button className="rounded-full border border-sakura/30 bg-sakura/10 px-4 py-2 text-sm text-sakura transition hover:bg-sakura/20 hover:text-white" type="button" onClick={() => setQuizOpen(true)}>Taste quiz</button>}
            {ratings.ratings.length >= 10 && (
              <button
                aria-current={view === 'recommendations' ? 'page' : undefined}
                className={`rounded-full border px-4 py-2 text-sm transition ${view === 'recommendations' ? 'border-violet bg-violet/20 text-white' : 'border-violet/40 bg-violet/10 text-violet hover:bg-violet/20 hover:text-white'}`}
                type="button"
                onClick={() => setView('recommendations')}
              >
                Recommendations
              </button>
            )}
            <ApiStatus />
          </div>
        </nav>
        {view === 'popular' ? (
          <PopularPicksPage ratings={ratings.ratings} onRate={ratings.setRating} onRemove={ratings.removeRating} onRecommend={() => setView('recommendations')} />
        ) : view === 'ratings' ? (
          <MyRatingsPage ratings={ratings.ratings} onRate={ratings.setRating} onRemove={ratings.removeRating} onRecommend={() => setView('recommendations')} onPopularPicks={() => setView('popular')} />
        ) : (
          <RecommendationsPage ratings={ratings.ratings} preferences={quiz.preferences} onRate={ratings.setRating} onBack={() => setView('ratings')} />
        )}
      </div>
      {TASTE_QUIZ_ENABLED && quizOpen && (
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
