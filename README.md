# Anime Taste DNA

A personalized anime recommendation app that uses offline-trained collaborative filtering, cold-start user fold-in, and interpretable metadata-derived taste profiles.

This repository contains an end-to-end recommendation slice: AniList-backed discovery, browser-persisted ratings, a strictly evaluated Weighted ALS and popularity hybrid, cold-start fold-in inference, and an enriched Top 10 recommendations view.

## Live deployment

- Frontend: <https://anime-taste-dna.vercel.app>
- API health: <https://anime-taste-dna-api-197238956100.northamerica-northeast2.run.app/health>
- API documentation: <https://anime-taste-dna-api-197238956100.northamerica-northeast2.run.app/docs>

## Repository layout

```text
frontend/             React + TypeScript + Vite + Tailwind
backend/              FastAPI service and tests
ml/                   Preprocessing, training, evaluation, and artifact export
docs/                 Architecture documentation
scripts/              Future repeatable utility scripts
.github/workflows/    Basic continuous integration
```

## Prerequisites

- Node.js 20.19+ (or Node.js 22+) and npm
- Python 3.11+

## Frontend

```bash
cd frontend
npm install
cp ../.env.example .env.local
npm run dev
```

Open <http://localhost:5173>. With the backend running at port 8000, the landing page displays its health status. The default API URL is already `http://localhost:8000`; the env file is optional locally.

Useful checks:

```bash
npm run lint
npm run typecheck
npm run build
```

## Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
uvicorn app.main:app --reload --port 8000
```

Then visit <http://localhost:8000/health> or <http://localhost:8000/docs>. The frontend uses `GET /api/v1/anime/search`; AniList results without a MyAnimeList ID remain visible but cannot be rated because MAL ID is the canonical model item identity. After running the ML export, `POST /api/v1/recommendations` loads the default artifact from `data/artifacts/weighted-als-hybrid-v1/weighted_als_hybrid.npz`. Set `MODEL_ARTIFACT_PATH` to override it.

### Production backend image

The production model is ignored by Git, so restore the verified artifact at
`data/artifacts/weighted-als-hybrid-v1/weighted_als_hybrid.npz` before building from the repository root:

```bash
docker build -f backend/Dockerfile -t anime-taste-dna-backend:weighted-als-v1 .
docker run --rm -p 8080:8080 anime-taste-dna-backend:weighted-als-v1
```

The image build validates the artifact's byte size, SHA-256, model version, required arrays, and
item dimensions against `backend/model/weighted-als-hybrid-v1.json`. A missing, stale, or corrupt
model therefore fails the build instead of reaching Cloud Run.

Run checks from `backend/`:

```bash
ruff check .
pytest
```

## Current status

- Implemented: AniList title search, normalized search cards, 1–10 rating add/edit/remove, versioned local-storage persistence, and lazy-loaded anime detail modals with AniList community scores and public review summaries.
- Implemented in ML: strict held-out-user evaluation, popularity and bias baselines, explicit MF/BPR diagnostics, Weighted ALS, popularity-aware reranking, and a gated versioned inference artifact.
- Implemented in the API: typed recommendation contract, Weighted ALS artifact validation/loading, MAL ID mapping, bounded implicit fold-in/hybrid ranking, rated-item filtering, explicit unmapped-ID reporting, and batched AniList metadata enrichment with graceful degradation.
- Implemented in the frontend: a model-ranked Top 10 view clearly separated from AniList community scores, responsive cards, retry/regenerate states, reusable anime detail modals, in-place scoring of already-watched recommendations, and browser-local “not interested” archives with automatic replacement and selective restore.
- Implemented experimentally: a four-title, genre-diverse Taste Quiz with short AniList summaries. Most/least-interest choices are stored separately from ratings; least-interest choices are excluded from recommendations.
- Not implemented: Cloud Run/Vercel release and Taste DNA.
- Evaluated: the current linear pairwise fold-in did not improve validation ranking, so its production weight is 0. Quiz choices are retained for future pairwise-aware model experiments and are never converted into fake ratings.
- Hidden by default: set `VITE_ENABLE_TASTE_QUIZ=true` only for experimental development; the production-facing navigation does not expose it.

## ML status

On one frozen strict test of 1,809 users absent from global training, the Weighted ALS hybrid improved Recall@10 by about 10.7% and NDCG@10 by about 7.4% over interaction-count popularity while increasing 10-rating catalog coverage from 14 to 83 items. Raw data and generated model files are Git-ignored; sanitized reports are committed under `ml/reports/`. See [docs/EVALUATION.md](docs/EVALUATION.md) for exact metrics, protocol, and limitations.

See [SPEC.md](SPEC.md) for requirements and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the planned system design.

## Dataset

The first MVP uses the 2023 MyAnimeList User Ratings + Anime Dataset snapshot with explicit ratings and timestamps keyed by MyAnimeList ID. Download and verify it with:

```bash
./scripts/download_myanimelist_2023_dataset.sh
```

Raw data stays outside Git under `data/raw/`. See [docs/DATASET.md](docs/DATASET.md) for the exact version, checksum, schema, processing rules, and limitations.
