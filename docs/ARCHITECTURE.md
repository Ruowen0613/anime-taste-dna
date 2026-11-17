# Architecture

## System overview

```text
Browser (React) ──REST──> FastAPI ──read──> versioned model artifacts
      │                     │
      │                     └──GraphQL──> AniList metadata/search
      └── localStorage

Kaggle ratings + metadata ──> preprocessing ──> training/evaluation
                                      └──────> exported artifacts
```

The deployable system has one static frontend and one stateless API. Data preparation and training are offline jobs, never request-time work.

## Components

- `frontend/`: React/TypeScript SPA, rating state in local storage, REST client, Popular Picks/My Ratings/Recommendations/Taste DNA views.
- `backend/`: FastAPI application, request validation, AniList adapter, artifact loader, cold-start inference, ranking, explanation/Taste DNA services.
- `ml/`: dataset preparation, split creation, baseline, PyTorch training, fold-in/evaluation, and artifact export.
- `scripts/`: repeatable developer and data utility entry points.
- `docs/`: decisions, data/evaluation documentation, and deployment notes.

Frontend product behavior is grouped under `frontend/src/features/<feature>/`, with each feature owning its components, hooks, API adapter, and types. Backend behavior follows the same boundary under `backend/app/features/<feature>/`, with feature-owned routers, schemas, and services. Application entry files only configure infrastructure and compose these features.

## Frontend/backend communication

The frontend reads `VITE_API_BASE_URL` and sends JSON over HTTPS. Contracts are versioned under `/api/v1`; Pydantic response/request models are the source of truth and matching TypeScript types are maintained at the client boundary. Browser ratings use a schema-versioned local-storage key. CORS allows only configured frontend origins outside local development.

Errors use a stable shape such as `{ "detail": "..." }`; the UI provides loading, empty, and retry states. AniList tokens, if ever needed, remain server-side.

## Dataset preprocessing

1. Download a documented Kaggle snapshot outside Git into `data/raw/`.
2. Validate columns, types, rating range, duplicate policy, and missing identifiers.
3. Filter users/items using documented minimum-interaction thresholds.
4. Resolve canonical MAL IDs and build explicit contiguous training indices.
5. Fetch/cache AniList mappings by MAL ID; preserve unresolved records as missing rather than title-matching silently.
6. Write deterministic processed tables, ID maps, metadata version/hash, and a preprocessing report to ignored artifact directories.
7. Create seeded user-aware splits used by every model.

Preprocessing parameters must be configuration, not hidden notebook state.

## Offline training and evaluation

The production candidate uses fixed implicit Weighted ALS item factors with two cold-start signals:
the original positive-interaction/popularity score and a small signed explicit-rating residual. The
residual maps every 1–10 rating around neutral midpoint 5.5, so low ratings repel related item
factors instead of disappearing from the user profile. Global factors train only on the strict
training-user group; validation and test users are disjoint. Seeds, hyperparameters, filters, and
dataset fingerprints accompany each run.

Evaluation simulates genuinely new users: expose 5, 10, 20, 40, or 80 ratings from users absent from global training, fold in a user representation with frozen item parameters, and rank the complete train-supported catalog while excluding only exposed ratings. Models share users, targets, candidates, and relevance rules. Report Recall@10, NDCG@10, catalog coverage, popularity overlap, eligible-user counts, and limitations.

Export a compact, versioned bundle containing item factors, interaction counts, MAL-ID maps, the
supported mask, both fixed fold-in configurations, blend parameters, titles, and model version. Raw
training data and historical user factors are not required by the API.

## Online inference

At process startup, FastAPI loads and validates one artifact bundle. For a recommendation request it:

1. validates 10–100 unique MAL IDs and 1–10 ratings;
2. resolves known item indices and reports/handles unmapped inputs;
3. solves both the positive implicit ALS fold-in and a regularized signed fold-in over up to 100 explicit ratings while item factors stay frozen;
4. reduces the popularity blend after 10 mapped ratings using `base_blend * min(1, 10 / mapped_ratings)`, standardizes the ALS/popularity score, adds the validation-selected signed residual, masks rated/unavailable titles, and selects Top K;
5. batch-enriches ranked MAL IDs with runtime AniList metadata while preserving the model ranking;
6. returns a typed validation error when too few inputs map to the supported model universe.

Inference is bounded and CPU-friendly. The artifact is immutable per deployment, making replicas stateless and results traceable to a model version.

### Experimental pairwise taste feedback

`POST /api/v1/preference-quiz/options` samples model-supported items, enriches a candidate pool through
one AniList MAL-ID query, and greedily selects four options that add distinct genres. The browser
stores each `most_mal_id > least_mal_id` choice separately from watched ratings. The evaluated
experimental fold-in represents each pair as a weighted item-factor difference row with no
user-bias term; item embeddings and all global model parameters remain frozen. Least-interest
choices are excluded from ranking.

Validation selected pairwise weight 0.0: every positive candidate reduced NDCG@10 on the common
eligible cohort. Production therefore records the choices and applies explicit least-interest
exclusions but does not alter the user vector. A future pairwise-aware model must beat rating-only
fold-in under the same Recall@10/NDCG@10 protocol before this signal can affect ranking scores.

## Anime metadata integration

AniList GraphQL provides search, covers, titles, descriptions, genres, tags, format, and season/year. Search results should request `idMal` and use the MAL ID as the bridge to the ratings dataset. Persist an offline mapping table with `mal_id`, `anilist_id`, mapping status, and metadata timestamp. Runtime metadata may be cached in memory with bounded TTL; rate limits and partial outages must produce clear degraded behavior.

## Proposed API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Lightweight process health check |
| GET | `/api/v1/anime/search?q=...&limit=...` | Proxied/cached AniList search with IDs normalized |
| POST | `/api/v1/anime/metadata` | Batch-enrich up to 100 canonical MAL IDs for local library cards |
| GET | `/api/v1/anime/{anilist_id}` | Lazy-loaded description, community score, metadata, and public review summaries |
| GET | `/api/v1/popular-picks` | Return the model universe's 100 most-interacted-with titles for direct rating |
| POST | `/api/v1/recommendations` | Validate ratings and return ranked unseen anime, explanations, and model version |
| POST | `/api/v1/preference-quiz/options` | Return four model-supported, metadata-enriched quiz options |
| POST | `/api/v1/taste-dna` | Return metadata-derived dimension scores and supporting titles |
| GET | `/api/v1/model/info` | Expose safe model version and training/evaluation metadata |

The implemented `POST /api/v1/recommendations` accepts
`{ ratings: [{ mal_id, rating }], preferences: [{ most_mal_id, least_mal_id }], excluded_mal_ids: [], limit: 10 }`. It requires 10–100 unique positive MAL IDs with
integer ratings from 1–10 and a limit from 1–20. At least five ratings must map to supported model
items. The response is
`{ model_version, mapped_ratings, mapped_preferences, unmapped_mal_ids, metadata_status, items: [{ mal_id, anilist_id, title, rank, predicted_rating, cover_image_url, year, format, genres, average_score }] }`.
`predicted_rating` is nullable: the signed hybrid production engine returns `null` because its score
is a ranking signal, not a calibrated 1–10 rating. The service loads both fold-in regularizations,
item factors/counts, blend parameters, titles, supported-item mask, and MAL
mapping from the versioned artifact; it never loads historical user parameters. Unmapped ratings
are reported explicitly and never title-matched. The API sends ranked MAL IDs to AniList in one
`idMal_in` query and joins only on the returned `idMal`. `metadata_status` is `complete`, `partial`,
or `unavailable`; an AniList outage never discards the model results.

The implemented search response is `{ items: AnimeSearchResult[] }`, where each item includes `anilist_id`, nullable `mal_id`, title, cover URL, year, format, genres, AniList average score, and popularity count. Its `sort` query parameter accepts `popularity` (default), `score`, `relevance`, or `newest`; ordering is performed by AniList before limiting results. A null `mal_id` is never replaced by a title match; the frontend shows that result but disables rating it.

AniList access uses one process-scoped async HTTP client with keep-alive connection pooling. Anime
metadata is cached for six hours; season relations and expanded franchise exclusions are cached for
24 hours with bounded entry counts. Recommendation enrichment starts the candidate-relation query
and provisional Top-K metadata query concurrently, then fetches only metadata made newly necessary
by season deduplication. The model ranker remains independent of AniList availability.

The recommendations client submits every locally rated or archived MAL ID as an explicit ranking
exclusion and displays the returned Top 10. Scoring an already-watched recommendation persists it through the shared
ratings feature and triggers a fresh request, so the title disappears and the next unseen candidate
fills its position. When more than 100 local ratings exist, the latest 100 are used for fold-in while
the complete local rated-ID set remains excluded from the displayed results.

The client also identifies archived titles separately as `not_interested_mal_ids`. The API expands
those IDs through AniList's explicit `PREQUEL`/`SEQUEL` relations before ranking, so every season in
the discovered chain is excluded without relying on title matching. It over-fetches model candidates,
groups them by the same relation graph, and keeps only the highest-ranked season from each franchise
in the final Top 10. If AniList relations are temporarily unavailable, ranking degrades to exact-ID
exclusion and preserves model order.
“Not interested” archives are also versioned in browser local storage. They filter ranking
candidates without becoming synthetic low ratings, so dismissing a title does not distort the
collaborative-filtering user vector. My Ratings owns a collapsed card-based manager where users can
review metadata and restore individual or selected archived titles. Legacy title-only records are
enriched through the bounded `POST /api/v1/anime/metadata` batch contract when the manager opens.
The Recommendations view only ranks and collects feedback; archiving a recommendation returns the
viewport to the page header while a replacement is generated.

## Deployment

- **Vercel:** builds `frontend/`; injects public `VITE_API_BASE_URL` pointing to Cloud Run.
- **Cloud Run:** runs the backend Docker image containing API code and the selected compact inference artifact. The build verifies the artifact against the committed size, SHA-256, schema, and model-version manifest. Uvicorn exposes the configured port and replicas remain stateless.
- **GitHub Actions:** pull requests run frontend lint/type-check/build plus Python lint/tests. After
  those checks pass on `main`, a keyless Workload Identity Federation workflow restores the
  versioned model from private GCS, builds and pushes an immutable backend image, deploys it to
  Cloud Run, and smoke-tests the resulting service.
- **Artifacts/data:** large datasets, experimental checkpoints, and the production artifact stay outside Git. The release image is built only after the selected artifact has been restored to its ignored versioned path; the Docker build enforces the committed manifest before packaging it. The resulting immutable image, rather than an unverified loose artifact, is promoted to Cloud Run.
