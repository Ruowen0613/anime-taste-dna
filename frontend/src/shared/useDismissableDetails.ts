import { useEffect, useRef } from 'react'

export function useDismissableDetails<T extends HTMLElement>() {
  const containerRef = useRef<T>(null)

  useEffect(() => {
    const closeOpenDetails = (exceptTarget?: EventTarget | null) => {
      containerRef.current?.querySelectorAll<HTMLDetailsElement>('details[open]').forEach((details) => {
        if (!(exceptTarget instanceof Node) || !details.contains(exceptTarget)) {
          details.removeAttribute('open')
        }
      })
    }
    const handlePointerDown = (event: PointerEvent) => closeOpenDetails(event.target)
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') closeOpenDetails()
    }
    document.addEventListener('pointerdown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  return containerRef
}
