# Machine learning workspace

This package holds offline-only recommendation work. No data preparation or training runs in the FastAPI process. The first completed stage is a chunked, privacy-preserving audit of the 2023 ratings snapshot.

- `configs/`: reproducible preprocessing, training, and evaluation configuration.
- `src/anime_taste_dna/`: importable preprocessing, model, inference, and metric modules.
- `tests/`: focused ML correctness tests.

Raw data belongs in root `data/`, while generated reports, runs, and checkpoints belong in ignored data/artifact directories.

## Dataset audit

Install the package in an isolated environment and run the chunked audit from `ml/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
anime-taste-audit --config configs/preprocess.toml
```

The audit validates schemas, ratings, timestamps, MAL ID coverage, status counts, exact duplicate user-item pairs, and interaction-count distributions. Raw usernames are mapped to internal integers in memory and never written to the report.

The sanitized, reproducible result is committed at `reports/dataset-audit-v3.json`. Large working files and processed datasets remain under Git-ignored `data/`.

## Preprocessing

Build the anonymous, filtered Parquet dataset after auditing:

```bash
anime-taste-preprocess --config configs/preprocess.toml
```

This keeps completed and dropped explicit ratings from 2004 onward, resolves exact metadata duplicates by MAL ID, iteratively enforces configured user/item interaction minimums, assigns contiguous internal indices, and writes `ratings.parquet`, `items.parquet`, and a fingerprinted report under `data/processed/v1/`.

## Dataset split

Create the shared leakage-safe train, validation, and test files:

```bash
anime-taste-split --config configs/preprocess.toml
```

The split is chronological within each anonymous user. The latest 10% is test data, the preceding 10% is validation data, and the earlier history is training data. Counts are rounded down with at least one validation and test interaction, while retaining at least 20 training interactions. Equal timestamps are ordered by item index so repeated runs are deterministic without a random seed.

## Popularity baseline

Fit the train-only Bayesian popularity baseline, select shrinkage on validation, and evaluate once on test:

```bash
anime-taste-popularity --config configs/popularity.toml
```

The candidate universe contains only items supported by train data. Evaluation treats held-out ratings of 8 or higher as relevant and excludes all user history preceding the target split. The generated inference table and full report remain under Git-ignored `data/artifacts/popularity-v1/`; a sanitized report is committed after an executed run.

## Matrix factorization

Train the personalized PyTorch model and evaluate cold-start fold-in:

```bash
anime-taste-mf --config configs/matrix_factorization.toml
```

The model learns user/item factors and biases from train only, selects its checkpoint by validation RMSE, freezes item parameters, and solves a regularized user vector from the most recent 5, 10, or 20 pre-test ratings. Test ranking covers the full item universe after filtering prior user history. The inference artifact exports item parameters and MAL ID mappings without historical user factors.

## Pairwise Taste Quiz evaluation

Evaluate the experimental pairwise protocol against the fixed exported model:

```bash
anime-taste-pairwise-eval --config configs/pairwise_quiz.toml
```

The command selects pairwise weight on validation only, then compares 0, 5, and 10 Quiz rounds on
one common test cohort. Its versioned report is `reports/pairwise-quiz-v1.json`.

## Personalization audit

Audit whether the exported model actually changes rankings between users, rather than merely
reproducing item popularity:

```bash
anime-taste-personalization-audit --config configs/personalization_audit.toml
```

The audit evaluates the fixed artifact on one common test cohort and reports catalog coverage,
recommendation concentration, user-factor norms, and Top-10 overlap with item-bias-only and
interaction-only inference ablations. Its versioned report is
`reports/personalization-audit-v1.json`.
