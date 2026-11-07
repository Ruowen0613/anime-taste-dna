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
