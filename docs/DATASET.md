# Dataset

## Selected MVP snapshot

- Name: Anime Recommendations Database
- Kaggle reference: `CooperUnion/anime-recommendations-database`
- Source page: <https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database>
- Version: 1
- Kaggle last-updated timestamp: 2016-12-21
- License: CC0: Public Domain
- Archive SHA-256: `49595ab309d1d773ae66464379dc23d813466199816ae6b24e38bc73b8f2cd12`

This snapshot was chosen for the first reproducible MVP because it is compact, uses MyAnimeList IDs directly, and contains anonymous explicit user ratings. Raw files are downloaded into `data/raw/`, which is Git-ignored.

Run:

```bash
./scripts/download_cooperunion_dataset.sh
```

## Raw files

### `anime.csv`

12,294 data rows with columns:

```text
anime_id,name,genre,type,episodes,rating,members
```

`anime_id` is the canonical MyAnimeList ID. The aggregate `rating` and `members` columns are metadata only; model fitting and popularity statistics must use training interactions rather than these full-snapshot aggregates.

### `rating.csv`

7,813,737 data rows with columns:

```text
user_id,anime_id,rating
```

- `user_id` is an anonymous generated identifier.
- `anime_id` is a MyAnimeList ID.
- Explicit scores are integers from 1 through 10.
- `-1` means watched without an assigned score. These rows must not be treated as a rating of 1. The initial explicit-feedback baseline and matrix-factorization model will exclude them while reporting their count separately.

## Known limitations

- The catalog and interactions are old, so recently released anime cannot receive matrix-factorization item embeddings in the first artifact.
- The snapshot has no trustworthy event timestamps, so the MVP cannot make a chronological train/test split. The split must be deterministic and user-aware, and this limitation must be reported with evaluation results.
- Users and interactions reflect MyAnimeList participation and are not representative of every anime viewer.
- Popular titles receive substantially more interactions, introducing exposure and popularity bias.
- Title metadata is not used for runtime identity joins. MAL IDs remain canonical and AniList mappings must be explicit.

The dataset snapshot, filters, seeds, row counts, and fingerprints must be recorded in every exported model artifact.
