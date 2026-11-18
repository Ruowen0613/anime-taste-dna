export type TasteContribution = {
  mal_id: number
  title: string
  rating: number
}

export type TasteDimension = {
  label: string
  score: number
  support: number
  average_rating: number | null
  contributing_titles: TasteContribution[]
}

export type TasteDnaResponse = {
  method_version: string
  mapped_ratings: number
  total_ratings: number
  metadata_status: 'complete' | 'partial' | 'unavailable'
  unmapped_mal_ids: number[]
  genre_dimensions: TasteDimension[]
  story_themes: TasteDimension[]
  worlds_and_settings: TasteDimension[]
  characters: TasteDimension[]
}
