import { useEffect, useState } from 'react'
import { getHealth } from '../../api/health'

type ApiState = 'checking' | 'online' | 'offline'

const statusText: Record<ApiState, string> = {
  checking: 'Checking API…',
  online: 'API connected',
  offline: 'API offline — start FastAPI on port 8000',
}

export function ApiStatus() {
  const [state, setState] = useState<ApiState>('checking')

  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then(() => setState('online'))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        setState('offline')
      })
    return () => controller.abort()
  }, [])

  return (
    <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300" aria-live="polite">
      <span className={`mr-2 inline-block h-2 w-2 rounded-full ${state === 'online' ? 'bg-emerald-400' : 'bg-amber-400'}`} />
      {statusText[state]}
    </span>
  )
}
