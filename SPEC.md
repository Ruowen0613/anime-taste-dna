# Anime Taste DNA — Product Requirements

## Overview and goals

Anime Taste DNA is a personalized anime discovery web app. A visitor rates at least 10 watched titles, can strengthen the profile through a model-supported Popular Picks catalog, and receives a machine-learning-generated Top 10.

The MVP should be functional, deployed, reproducible, and credible as an AI/Applied Computing portfolio project by **October 31, 2026**. Success means a polished end-to-end product backed by honest offline evaluation—not an LLM or a disguised popularity list.

## MVP scope

- Search anime through AniList-backed metadata.
- Add, change, and remove 1–10 ratings; retain them in browser local storage.
- Generate Top 10 recommendations using a trained collaborative-filtering model with cold-start fold-in.
- Provide a sufficiently popular/highly rated baseline for comparison and fallback.
- Exclude rated anime and show metadata-grounded explanations.
- Display an interpretable Taste DNA visualization derived from genres/tags.
- Evaluate both recommenders offline and document the protocol and limitations.

## Main screens

### Popular Picks

- Show the 100 most-interacted-with titles supported by the model for direct rating.
- Use one type-ahead search to filter those titles and surface AniList results outside the Top 100.
- Let visitors rate 1–10 or mark a title not interested without searching first.

### My Ratings

- Show every locally rated title with cover, title, year/format, and genres.
- Edit or remove any rating and sort the collection.
- Manage not-interested titles in a collapsed card section and selectively restore them to the recommendation pool.
- Indicate progress toward the minimum recommended input count and open recommendations.

### Recommendations

- Request and show 10 unseen recommendations.
- Show cover, title, genres, and a carefully labeled model/relevance score if useful.
- Explain recommendations using evidence such as shared genres, tags, or similarity to highly rated input titles.
- Return to Discover to revise ratings and regenerate results.

### Taste DNA

- Visualize normalized preference scores for prominent genres/themes, initially with a radar or bar chart.
- Explain how the profile was calculated and identify rated titles contributing to each score.
- Never interpret anonymous latent dimensions as human concepts.

## User journey

1. The visitor lands on Popular Picks and rates familiar titles or searches the AniList catalog.
2. They manage at least 10 local ratings in My Ratings; ratings remain available after refresh.
3. They request recommendations; the frontend submits item IDs and ratings.
4. The backend validates input, folds the new user into fixed item embeddings, ranks unseen items, and enriches the Top 10 with metadata.
5. The visitor reviews explanations, adjusts ratings if desired, and explores Taste DNA.

## Functional requirements

- FR-1: Search accepts a non-empty query and returns consistently identified anime metadata.
- FR-2: Ratings accept integer values 1–10 and are uniquely keyed by canonical anime ID.
- FR-3: Local ratings survive reload and can be edited/deleted.
- FR-4: Recommendation requests accept 10–100 ratings, reduce popularity influence as mapped evidence grows beyond 10 ratings, and return no rated item.
- FR-5: Results are deterministic for a fixed artifact, request, and seed where optimization permits.
- FR-6: Explanations cite metadata or computable item/user similarity; they must not make unsupported causal claims.
- FR-7: Taste DNA scores are derived reproducibly from ratings and metadata.
- FR-8: `GET /health` reports service availability.
- FR-9: Missing ID mappings and unavailable metadata degrade gracefully without corrupt joins.

## ML requirements

### Popularity baseline

Rank anime using a documented score that balances mean rating and support, after a minimum-interaction threshold or Bayesian shrinkage. Fit all statistics on training data only.

### Matrix factorization

Train PyTorch user and item embeddings (with biases as appropriate) on historical explicit ratings. Export item parameters plus ID mappings and normalization metadata for inference. For a new user, optimize a regularized user vector/bias while item parameters remain frozen. Remove observed items before Top-K selection.

### Evaluation

- Use reproducible user-aware train/validation/test splits with no future/held-out rating leakage.
- Simulate cold start with 5, 10, and 20 observed ratings per eligible test user and rank held-out relevant items among an explicitly documented candidate set.
- Report Recall@10 and NDCG@10 for popularity and matrix factorization under identical splits/candidates.
- Save configuration, seed, dataset/filter versions, aggregate results, and limitations. Never fabricate unavailable results.

## Non-functional requirements

- Responsive, accessible UI with keyboard-operable controls and meaningful loading/error/empty states.
- Typed, versionable REST contracts; clear separation between offline training and online inference.
- API validation and bounded inputs; no credentials in client bundles or source control.
- Reproducible local setup and training commands; CI checks frontend and backend.
- Backend container suitable for Cloud Run; frontend configurable for Vercel API URL.
- Reasonable MVP latency target: cached search and recommendation responses should normally complete within 2 seconds, excluding upstream outages/cold starts.

## Acceptance criteria

- A new visitor can rate at least 10 anime, refresh without losing ratings, and receive exactly 10 unseen recommendations when enough mapped candidates exist.
- Search and result cards display real AniList metadata with explicit MAL/AniList identity handling.
- Recommendations come from the exported recommender artifact; no LLM supplies the ranking.
- Taste DNA displays reproducible metadata-derived preference scores.
- Popularity and matrix-factorization results include Recall@10 and NDCG@10 for 5/10/20 observed-rating settings using the documented leakage-safe protocol.
- Frontend build/type-check and backend/ML tests pass in CI; setup and deployment instructions are current.
- The deployed frontend and Cloud Run API complete the primary journey on desktop and mobile.

## Milestones

- **Oct 8–11:** Repository setup, React UI shell, FastAPI service, mock search/rating flow.
- **Oct 12–18:** Data preprocessing, popularity baseline, PyTorch matrix factorization, cold-start fold-in.
- **Oct 19–25:** API/UI/model integration, Taste DNA, offline evaluation.
- **Oct 26–30:** Testing, deployment, documentation, polish.
- **Oct 31:** Buffer and final verification.

## Explicitly out of scope

Authentication, server-side user profiles, PostgreSQL, social features, comments, watchlist synchronization, payments, mobile apps, Kubernetes, Terraform, multiple backend services, online/incremental training, and LLM-generated rankings. Content-based/hybrid ranking and embedding projections are stretch goals only.
