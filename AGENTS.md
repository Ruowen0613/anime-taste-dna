# Anime Taste DNA — Agent Guide

## Mission and deadline

Build a deployed, portfolio-quality MVP by **October 31, 2026**. The product must use a real, evaluated recommender—not LLM-generated recommendations. Favor a working vertical slice and sound ML methodology over optional features.

## Stack

- Frontend: React, TypeScript, Vite, Tailwind CSS, Recharts.
- Backend: Python 3.11+, FastAPI, Pydantic; one REST service.
- ML: PyTorch, pandas, NumPy, scikit-learn; offline training and lightweight online inference.
- Metadata: AniList GraphQL API, with explicit MyAnimeList-to-AniList ID mapping.
- Deployment: Vercel (frontend), Docker/Google Cloud Run (backend), GitHub Actions.

## Architecture conventions

- Keep `frontend/`, `backend/`, and `ml/` independently runnable. Training code must never run in an API request.
- Define and document API schemas before wiring UI features. Use typed Pydantic models and matching TypeScript types.
- Put offline data preparation, training, and evaluation in `ml/`; load only versioned inference artifacts in `backend/`.
- Treat MyAnimeList ID as the canonical dataset item ID and store AniList ID mappings explicitly. Never join titles heuristically at request time.
- Persist MVP ratings in browser local storage. Do not add a database or authentication without a documented need.
- Read configuration from environment variables. Never commit secrets, raw datasets, or model checkpoints.

## Coding and testing

- Prefer small, typed modules and straightforward functions over framework layers or premature abstractions.
- TypeScript must pass `tsc`; Python public interfaces should have type hints.
- Add tests for API contracts, preprocessing, split logic, recommendation filtering/ranking, fold-in inference, and metrics.
- Before handoff, run relevant lint, type-check, build, and tests; report exact results and anything not run.
- Preserve existing work and keep unrelated changes out of the task.

## ML correctness

- Implement popularity as the baseline and PyTorch matrix factorization as Model B.
- Fit preprocessing and models only on training data. Use deterministic, documented train/validation/test splits and fixed seeds.
- Evaluate both models with Recall@10 and NDCG@10 at 5, 10, and 20 observed ratings. Exclude observed/rated items from candidates.
- Cold-start users should be folded into fixed item embeddings with regularized optimization (or an equivalently justified method).
- Never invent metrics or report unexecuted results. Record dataset version, filters, seeds, protocol, and limitations.
- Taste DNA labels must derive from metadata such as genres/tags. Do not assign semantic names to latent factors.

## Workflow

1. Check `SPEC.md`, `docs/ARCHITECTURE.md`, and the current milestone.
2. Make the smallest coherent change toward the MVP.
3. Update contracts/docs when behavior changes.
4. Add or update focused tests, then run relevant checks.
5. Summarize changes, evidence, risks, and the next small task.

## Scope restrictions

Until the MVP is complete, do not add PostgreSQL, authentication, Kubernetes, Terraform, microservices, LLM recommendations, real-time training, social features, or a mobile app. Content-based or hybrid ranking and UMAP are optional only after the baseline MVP is working and evaluated.
