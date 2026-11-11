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
