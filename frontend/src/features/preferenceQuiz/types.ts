export type QuizOption = {
  mal_id: number
  anilist_id: number
  title: string
  cover_image_url: string | null
  year: number | null
  format: string | null
  genres: string[]
  summary: string | null
}

export type PairwisePreference = {
  most_mal_id: number
  least_mal_id: number
}
