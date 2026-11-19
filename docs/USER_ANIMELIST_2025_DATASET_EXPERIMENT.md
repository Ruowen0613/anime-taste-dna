# User Animelist 2025 Dataset Pilot

## Decision

The 2025 cross-platform dataset is promising enough for further work, but it does not replace
`signed-rating-hybrid-v2`. A resource-bounded pilot passed its internal relevance and behavior
gates. Production acceptance remains disabled because the source has no timestamps or platform
provenance, and its random-target protocol is not directly comparable with the current model's
chronological 2023 evaluation.

## Source audit

The downloaded `mramazan/User-Animelist-Dataset` snapshot contains 148,170,496 rating rows and
20,237 anime metadata rows. The raw files are Git-ignored and have these fingerprints:

- `ratings.csv`: SHA-256
  `2ee00f83262cd4cdb2d4a0f1661545307ccc956b01314df0d5e5232433339e68`
- `animes.csv`: SHA-256
  `5bec78930e6e7633156046338947954c7e619d9615116bec13d9fc4e04221175`

The executed chunked audit found:

- 148,064,215 valid 1–10 integer ratings and 106,281 zero-score rows;
- 1,774,493 users and 20,237 items represented by valid ratings;
- complete, one-to-one metadata coverage for all dataset-local anime IDs;
- 3,384,712 duplicate rows beyond the first across 2,757,008 user-item groups;
- 57,178 duplicate groups whose ratings conflict;
- no timestamps and no field identifying whether a row came from MAL, AniList, or Kitsu.

Dataset-local `animeID` is not a MAL ID. The pipeline extracts the canonical MAL ID from
`mal_url`; only one row happens to have matching values.

## Cleaning and split

Exact duplicate ratings are collapsed. If duplicate rows for one user-item pair disagree, the
entire pair is dropped because the source provides neither a timestamp nor platform provenance to
resolve it safely.

The pilot uses a deterministic hash sample so it fits comfortably on a 16 GB development machine:

| Group | Users | Rows |
| --- | ---: | ---: |
| Global training | 106,222 | 8,688,154 |
| Validation history / target | 1,369 | 274,902 / 67,375 |
| Test history / target | 1,357 | 268,235 / 65,806 |

The three user groups are pairwise disjoint. Because timestamps are absent, validation and test
targets are deterministic random 20% samples rather than future interactions. Each held-out user
has at least 80 history ratings and 10 targets before metric eligibility filtering.

## Model results

The pilot trained a 16-dimensional Weighted ALS model for two iterations using ratings of at least
8 as positive interactions. Validation selected fold-in regularization 100 and popularity blend 8.
No test parameters were tuned.

On 1,354 untouched eligible test users, averaged across 5, 10, 20, 40, and 80 observed ratings:

| Model | Recall@10 | NDCG@10 |
| --- | ---: | ---: |
| Interaction-count popularity | 0.05435157 | 0.11395029 |
| Weighted ALS hybrid | 0.06347489 | 0.13132668 |
| ALS + signed all-rating residual | **0.06548930** | **0.13531862** |

Weighted ALS improved over popularity by 16.8% in Recall@10 and 15.2% in NDCG@10. Adding the
validation-selected signed residual (regularization 1, weight 0.1) improved over ALS by another
3.2% and 3.0%, respectively. Against popularity, the complete candidate improved Recall@10 by
20.5% and NDCG@10 by 18.8%.

The value of personalization increased with more user evidence. The signed candidate's test
Recall@10 rose from 0.05436348 at five ratings to 0.08779210 at eighty; NDCG@10 rose from
0.11373555 to 0.17904308. At ten ratings, flipping high scores to low scores left only 43.18% of
the Top 10 unchanged, and popularity overlap was 56.48%, so the candidate passed the behavior
gates.

## Limitations and next step

These absolute metrics must not be compared directly with `signed-rating-hybrid-v2`: the source
population, catalog, dimensions, and target-splitting protocol differ. The pilot also uses only a
deterministic subset of the 1.77 million users.

The next justified experiment is to obtain or construct a source with platform namespace and
timestamps, or to train the same pilot protocol on a matched deterministic sample of the 2023
dataset. A matched-protocol comparison is required before considering a production migration.

Executed evidence is stored in:

- `ml/reports/user-animelist-2025-audit-v1.json`
- `ml/reports/user-animelist-2025-pilot-preparation-v1.json`
- `ml/reports/strict-cold-start-user-animelist-2025-pilot-popularity-v1.json`
- `ml/reports/weighted-als-user-animelist-2025-pilot-validation-v1.json`
- `ml/reports/weighted-als-user-animelist-2025-pilot-test-v1.json`
- `ml/reports/signed-hybrid-user-animelist-2025-pilot-validation-v1.json`
- `ml/reports/signed-hybrid-user-animelist-2025-pilot-test-v1.json`
