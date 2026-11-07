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

- `frontend/`: React/TypeScript SPA, rating state in local storage, REST client, Discover/Recommendations/Taste DNA views.
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

The popularity model computes training-only item statistics with shrinkage/minimum support. Matrix factorization learns user/item embeddings and biases from training ratings with validation-based model selection. Seeds, hyperparameters, filters, and dataset fingerprints accompany each run.

Evaluation simulates new users: expose 5, 10, or 20 training-history ratings, fold in a user representation with frozen item parameters, and rank a documented candidate set containing held-out relevant items. Both models share splits/candidates. Report Recall@10 and NDCG@10 with eligible-user counts and limitations.

Export a compact, versioned bundle containing item embeddings/biases, item-index maps, rating normalization, popularity fallback, metadata lookup, model/config version, and integrity metadata. Raw training data and training-only user embeddings are not required by the API.

## Online inference

At process startup, FastAPI loads and validates one artifact bundle. For a recommendation request it:

1. validates 10–20 unique MAL IDs and 1–10 ratings;
2. resolves known item indices and reports/handles unmapped inputs;
3. initializes and optimizes a regularized user vector (and optional bias) while item parameters stay frozen;
4. scores candidate items, masks rated/unavailable titles, and selects Top K;
5. attaches cached metadata and evidence-based explanations;
6. uses the popularity artifact as an explicit fallback when too few inputs map.

Inference is bounded and CPU-friendly. The artifact is immutable per deployment, making replicas stateless and results traceable to a model version.

## Anime metadata integration

AniList GraphQL provides search, covers, titles, descriptions, genres, tags, format, and season/year. Search results should request `idMal` and use the MAL ID as the bridge to the ratings dataset. Persist an offline mapping table with `mal_id`, `anilist_id`, mapping status, and metadata timestamp. Runtime metadata may be cached in memory with bounded TTL; rate limits and partial outages must produce clear degraded behavior.

## Proposed API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Lightweight process health check |
| GET | `/api/v1/anime/search?q=...&limit=...` | Proxied/cached AniList search with IDs normalized |
| GET | `/api/v1/anime/{anilist_id}` | Lazy-loaded description, community score, metadata, and public review summaries |
| POST | `/api/v1/recommendations` | Validate ratings and return ranked unseen anime, explanations, and model version |
| POST | `/api/v1/taste-dna` | Return metadata-derived dimension scores and supporting titles |
| GET | `/api/v1/model/info` | Expose safe model version and training/evaluation metadata |

`POST /api/v1/recommendations` will accept `{ ratings: [{ mal_id, rating }], limit: 10 }`. Final schemas should be committed and tested before frontend integration.

The implemented search response is `{ items: AnimeSearchResult[] }`, where each item includes `anilist_id`, nullable `mal_id`, title, cover URL, year, format, genres, AniList average score, and popularity count. Its `sort` query parameter accepts `popularity` (default), `score`, `relevance`, or `newest`; ordering is performed by AniList before limiting results. A null `mal_id` is never replaced by a title match; the frontend shows that result but disables rating it.

## Deployment

- **Vercel:** builds `frontend/`; injects public `VITE_API_BASE_URL` pointing to Cloud Run.
- **Cloud Run:** builds a backend Docker image containing API code and a selected compact inference artifact. It runs Uvicorn, exposes the configured port, and scales stateless replicas.
- **GitHub Actions:** on pull requests/pushes, installs locked dependencies and runs frontend lint/type-check/build plus Python lint/tests. Deployment workflows are added after environments and credentials exist.
- **Artifacts/data:** large datasets and experimental checkpoints stay outside Git (local ignored paths or cloud object storage). A small production artifact may be downloaded during image build or release using checksums; the exact mechanism is decided before deployment.
