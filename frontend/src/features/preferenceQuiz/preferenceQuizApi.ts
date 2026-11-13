import { API_BASE_URL } from '../../api/health'
import type { QuizOption } from './types'

export async function getQuizOptions(excludedMalIds: number[], signal?: AbortSignal): Promise<QuizOption[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/preference-quiz/options`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ excluded_mal_ids: excludedMalIds }),
    signal,
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(payload?.detail ?? 'Could not load taste quiz options.')
  }
  return ((await response.json()) as { items: QuizOption[] }).items
}
