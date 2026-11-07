export type AnimeSearchResult = {
  mal_id: number | null
  anilist_id: number
  title: string
  cover_image_url: string | null
  year: number | null
  format: string | null
  genres: string[]
}

export type RatedAnime = AnimeSearchResult & { mal_id: number; rating: number }

export type AnimeReview = {
  id: number
  summary: string
  helpful_count: number
  rating_count: number
  user_name: string
  user_avatar_url: string | null
}

export type AnimeDetails = AnimeSearchResult & {
  description: string | null
  average_score: number | null
  episodes: number | null
  duration_minutes: number | null
  status: string | null
  source: string | null
  banner_image_url: string | null
  anilist_url: string
  reviews: AnimeReview[]
}
