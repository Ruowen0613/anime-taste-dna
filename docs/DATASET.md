# Dataset

## Selected MVP snapshot

- Name: MyAnimeList User Ratings + Anime Dataset
- Kaggle reference: `andrewgatchalian/myanimelist-user-ratings`
- Source page: <https://www.kaggle.com/datasets/andrewgatchalian/myanimelist-user-ratings>
- Version: 3
- Snapshot/update date: 2023-12-08
- License reported by Kaggle: MIT
- Archive SHA-256: `1a86c041de4505dc651c7c6c3bde6d07ba9a695cf3367725a095cb8eef57fbfd`

This snapshot was selected because it uses MyAnimeList IDs, contains explicit user scores and timestamps, includes completed and dropped titles, and has a substantially newer catalog than the superseded 2016 development snapshot.

Run:

```bash
./scripts/download_myanimelist_2023_dataset.sh
```

The compressed archive is about 340 MB. The script verifies its checksum and extracts only the two required files, totaling about 4.04 GB, into Git-ignored `data/raw/`.

## Raw files

### `anime_titles.csv`

59,655 data rows before validation with columns:

```text
anime_id,title,mean,genres,studios,synopsis,media_type,num_episodes
```

`anime_id` is the canonical MyAnimeList ID. Aggregate `mean` is metadata only and must not be used to fit or evaluate the training-only popularity baseline.

### `anime_user_ratings.csv`

44,379,880 data rows before validation with columns:

```text
user_id,anime_id,title,user_status,user_score,user_eps_watched,user_rewatch,updated_at
```

- `user_id` contains a public MyAnimeList username in the raw source. Preprocessing must map it immediately to a deterministic internal integer and must never export, log, or commit the original value.
- `anime_id` is a MyAnimeList ID.
- `user_score` contains non-zero explicit scores.
- `user_status` is limited by the publisher to completed and dropped interactions.
- `updated_at` enables deterministic time-aware held-out evaluation.

The unused `username_list_full.csv` is intentionally not extracted. The separate one-hot genre file is also omitted because `anime_titles.csv` already contains genre metadata.

## Known limitations

- The snapshot is current only through November/December 2023, so newer anime will still require a later versioned offline retraining run.
- Users were sampled from recently-online MyAnimeList accounts, which creates activity and platform-selection bias.
- Raw user identifiers are public usernames rather than anonymous IDs. They must be irreversibly removed during preprocessing.
- Completed and dropped interactions do not represent all exposure, plan-to-watch, or currently-watching behavior.
- Popular titles receive substantially more interactions, introducing exposure and popularity bias.
- Title strings are never used for runtime joins. MAL IDs remain canonical and AniList mappings must be explicit.

The dataset reference, version, checksum, filters, seeds, row counts, time-split protocol, and fingerprints must be recorded in every exported model artifact.
