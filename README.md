# Anime Taste DNA

A personalized anime recommendation app that uses offline-trained collaborative filtering, cold-start user fold-in, and interpretable metadata-derived taste profiles.

This repository contains an end-to-end recommendation slice: AniList-backed discovery, browser-persisted ratings, offline-trained matrix factorization, cold-start fold-in inference, and an enriched Top 10 recommendations view.

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

Then visit <http://localhost:8000/health> or <http://localhost:8000/docs>. The frontend uses `GET /api/v1/anime/search`; AniList results without a MyAnimeList ID remain visible but cannot be rated because MAL ID is the canonical model item identity. After running the ML export, `POST /api/v1/recommendations` loads the default artifact from `data/artifacts/matrix-factorization-v1/matrix_factorization.npz`. Set `MODEL_ARTIFACT_PATH` to override it.

Run checks from `backend/`:

```bash
ruff check .
pytest
```

## Current status

- Implemented: AniList title search, normalized search cards, 1–10 rating add/edit/remove, versioned local-storage persistence, and lazy-loaded anime detail modals with AniList community scores and public review summaries.
- Implemented in ML: reproducible download/audit, anonymous preprocessing, chronological splits, evaluated popularity baseline, PyTorch matrix factorization, regularized cold-start fold-in, and a versioned inference artifact.
- Implemented in the API: typed recommendation contract, artifact validation/loading, MAL ID mapping, bounded fold-in/ranking, rated-item filtering, explicit unmapped-ID reporting, and batched AniList metadata enrichment with graceful degradation.
- Implemented in the frontend: a model-backed Top 10 view with predicted ratings clearly separated from AniList community scores, responsive cards, retry/regenerate states, reusable anime detail modals, in-place scoring of already-watched recommendations, and browser-local “not interested” archives with automatic replacement and selective restore.
- Implemented experimentally: a four-title, genre-diverse Taste Quiz with short AniList summaries. Most/least-interest choices are stored separately from ratings; least-interest choices are excluded from recommendations.
- Not implemented: production artifact packaging/deployment and Taste DNA.
- Evaluated: the current linear pairwise fold-in did not improve validation ranking, so its production weight is 0. Quiz choices are retained for future pairwise-aware model experiments and are never converted into fake ratings.

## ML status

The executed matrix-factorization model beats the shared popularity baseline by roughly 43% on Recall@10 and 38% on NDCG@10 under the documented test protocol. Raw data and generated model files are Git-ignored; sanitized reports are committed under `ml/reports/`. See [docs/EVALUATION.md](docs/EVALUATION.md) for exact metrics and limitations.

See [SPEC.md](SPEC.md) for requirements and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the planned system design.

## Dataset

The first MVP uses the 2023 MyAnimeList User Ratings + Anime Dataset snapshot with explicit ratings and timestamps keyed by MyAnimeList ID. Download and verify it with:

```bash
./scripts/download_myanimelist_2023_dataset.sh
```

Raw data stays outside Git under `data/raw/`. See [docs/DATASET.md](docs/DATASET.md) for the exact version, checksum, schema, processing rules, and limitations.
