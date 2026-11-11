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

The file contains 22,590 CSV records before validation. Its synopsis fields contain embedded newlines, so line-counting tools report a misleading larger number. Columns:

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
- The published description emphasizes completed and dropped interactions, but the actual audit also finds watching and on-hold rows. Status-specific filtering must use observed data rather than the description alone.
- `updated_at` enables deterministic time-aware held-out evaluation.

The unused `username_list_full.csv` is intentionally not extracted. The separate one-hot genre file is also omitted because `anime_titles.csv` already contains genre metadata.

## Executed audit

The versioned report is stored at `ml/reports/dataset-audit-v3.json`. It was produced by chunk-scanning the full ratings CSV; no raw usernames or username mappings are present in the report.

Key observed facts:

- 44,379,880 of 44,379,880 rating rows pass score, MAL ID, user-presence, and timestamp parsing checks.
- There are 18,145 users and 16,139 rated MAL IDs.
- No duplicate user–item rows were found.
- 166 timestamps predate MyAnimeList's 2004 launch and must be treated as invalid/sentinel values for chronological splitting.
- 1,765,739 interactions across 1,247 MAL IDs have no row in the supplied title metadata and require explicit resolution or filtering.
- Actual statuses include completed, dropped, watching, and on-hold.
- The median user has 2,593 interactions, demonstrating strong active-user selection bias.

## Executed preprocessing

The reproducible preprocessing command is:

```bash
cd ml
anime-taste-preprocess --config configs/preprocess.toml
```

The versioned, sanitized result is stored at
`ml/reports/preprocessing-report-v1.json`. Large Parquet outputs remain Git-ignored under
`data/processed/v1/`.

The current processed dataset:

- keeps only `completed` and `dropped` ratings dated 2004-01-01 or later;
- drops interactions whose MAL ID is absent from the supplied title metadata;
- removes 27 duplicate metadata rows by exact MAL ID, keeping the first source row;
- iteratively requires at least 25 retained ratings per user and 20 per item;
- contains 41,529,539 ratings from 18,095 anonymous users across 14,260 anime;
- uses contiguous `user_index` and `item_index` values and exports no raw username or mapping.

The generated files are `ratings.parquet`, `items.parquet`, and
`preprocessing_report.json`. Their hashes are recorded in the versioned report. The 116 MB
processed directory can be regenerated from the verified raw snapshot rather than committed.

## Known limitations

- The snapshot is current only through November/December 2023, so newer anime will still require a later versioned offline retraining run.
- Users were sampled from recently-online MyAnimeList accounts, which creates activity and platform-selection bias.
- Raw user identifiers are public usernames rather than anonymous IDs. They must be irreversibly removed during preprocessing.
- Completed and dropped interactions do not represent all exposure, plan-to-watch, or currently-watching behavior.
- Popular titles receive substantially more interactions, introducing exposure and popularity bias.
- Title strings are never used for runtime joins. MAL IDs remain canonical and AniList mappings must be explicit.

The dataset reference, version, checksum, filters, seeds, row counts, time-split protocol, and fingerprints must be recorded in every exported model artifact.
