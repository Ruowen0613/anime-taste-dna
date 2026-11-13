export type RecommendationItem = {
  mal_id: number
  title: string
  rank: number
  predicted_rating: number
  anilist_id: number | null
  cover_image_url: string | null
  year: number | null
  format: string | null
  genres: string[]
  average_score: number | null
}

export type ArchivedRecommendation = Pick<RecommendationItem, 'mal_id' | 'title'>

export type RecommendationResponse = {
  model_version: string
  mapped_ratings: number
  mapped_preferences: number
  pairwise_refinement_active: boolean
  unmapped_mal_ids: number[]
  metadata_status: 'complete' | 'partial' | 'unavailable'
  items: RecommendationItem[]
}
