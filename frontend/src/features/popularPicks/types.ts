import type { AnimeSearchResult } from '../discover/types'

export type PopularPick = AnimeSearchResult & {
  mal_id: number
  model_interactions: number
}

export type PopularPicksResponse = {
  model_version: string
  metadata_status: 'complete' | 'partial' | 'unavailable'
  items: PopularPick[]
}
