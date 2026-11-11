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
