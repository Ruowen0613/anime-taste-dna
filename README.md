# Anime Taste DNA

A personalized anime recommendation app that uses offline-trained collaborative filtering, cold-start user fold-in, and interpretable metadata-derived taste profiles.

This repository currently contains the first usable Discover slice: AniList-backed search, explicit MAL/AniList identities, browser-persisted 1–10 ratings, a FastAPI health/search API, and the project foundation. Recommendation training and recommendation results are not implemented yet.

## Repository layout

```text
frontend/             React + TypeScript + Vite + Tailwind
backend/              FastAPI service and tests
ml/                   Future preprocessing, training, evaluation, artifacts
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

Then visit <http://localhost:8000/health> or <http://localhost:8000/docs>. The frontend uses `GET /api/v1/anime/search`; AniList results without a MyAnimeList ID remain visible but cannot be rated because MAL ID is the canonical model item identity.

Run checks from `backend/`:

```bash
ruff check .
pytest
```

## Current status

- Implemented: AniList title search, normalized search cards, 1–10 rating add/edit/remove, versioned local-storage persistence, and lazy-loaded anime detail modals with AniList community scores and public review summaries.
- Not implemented: dataset preparation, popularity baseline, matrix factorization, fold-in inference, offline evaluation, recommendations, and Taste DNA.

## ML status

The ML package is structural only. Future commands will preprocess a versioned Kaggle ratings snapshot, train popularity and PyTorch matrix-factorization models offline, evaluate cold-start performance, and export a compact inference bundle. Raw data and model files are Git-ignored.

See [SPEC.md](SPEC.md) for requirements and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the planned system design.

## Dataset

The first MVP uses the CC0 CooperUnion Anime Recommendations Database snapshot with anonymous explicit ratings keyed by MyAnimeList ID. Download and verify it with:

```bash
./scripts/download_cooperunion_dataset.sh
```

Raw data stays outside Git under `data/raw/`. See [docs/DATASET.md](docs/DATASET.md) for the exact version, checksum, schema, processing rules, and limitations.
