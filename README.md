# Anime Taste DNA

A personalized anime recommendation app that uses offline-trained collaborative filtering, cold-start user fold-in, and interpretable metadata-derived taste profiles.

This repository currently contains the project foundation: a React landing page, a FastAPI health endpoint, architecture/product documentation, and organized ML placeholders. Recommendation training and the full product flow are intentionally not implemented yet.

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

Then visit <http://localhost:8000/health> or <http://localhost:8000/docs>.

Run checks from `backend/`:

```bash
ruff check .
pytest
```

## ML status

The ML package is structural only. Future commands will preprocess a versioned Kaggle ratings snapshot, train popularity and PyTorch matrix-factorization models offline, evaluate cold-start performance, and export a compact inference bundle. Raw data and model files are Git-ignored.

See [SPEC.md](SPEC.md) for requirements and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the planned system design.
