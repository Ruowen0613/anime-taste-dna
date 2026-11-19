# Machine learning workspace

This package holds offline-only recommendation work. No data preparation or training runs in the FastAPI process. The first completed stage is a chunked, privacy-preserving audit of the 2023 ratings snapshot.

- `configs/`: reproducible preprocessing, training, and evaluation configuration.
- `src/anime_taste_dna/`: importable preprocessing, model, inference, and metric modules.
- `tests/`: focused ML correctness tests.

Raw data belongs in root `data/`, while generated reports, runs, and checkpoints belong in ignored data/artifact directories.

## 2025 dataset pilot

Audit and prepare the isolated cross-platform pilot:

```bash
anime-taste-audit-user-animelist-2025 --config configs/user_animelist_2025_audit.toml
anime-taste-prepare-user-animelist-2025 --config configs/user_animelist_2025_pilot.toml
```

Then run its popularity baseline, Weighted ALS validation/frozen test, and signed-residual
validation/frozen test using the corresponding `*_user_animelist_2025_pilot*.toml` configs. The
pilot is deliberately blocked from production acceptance; its source has no timestamps or platform
provenance. See `docs/USER_ANIMELIST_2025_DATASET_EXPERIMENT.md` for executed results.

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

## Strict new-user evaluation

Create pairwise-disjoint global-training, validation-user, and test-user groups without overwriting
the historical per-user chronological split:

```bash
anime-taste-user-holdout-split --config configs/user_holdout_split.toml
```

Then fit and select the validation-only popularity baseline:

```bash
anime-taste-strict-cold-start --config configs/strict_cold_start.toml
```

The evaluator ranks the complete train-supported catalog and excludes only the 5, 10, or 20 ratings
actually exposed to the model. Hidden user history is not used for global training, fold-in, or
candidate filtering. Versioned reports are `reports/strict-user-holdout-split-v1.json` and
`reports/strict-cold-start-popularity-v1.json`; test users remain untouched.

Retrain explicit MF using only global-training users and select its checkpoint directly by strict
validation-user ranking:

```bash
anime-taste-strict-mf --config configs/strict_mf.toml
```

The executed validation report is `reports/strict-cold-start-mf-v1.json`. This experiment does not
export an artifact or access held-out test users.

Test whether additional new-user ratings reduce the need for strong fold-in regularization on one
common validation cohort:

```bash
anime-taste-strict-mf --config configs/strict_mf_rating_count.toml
```

This validation-only experiment compares 10, 20, 40, and 80 observed ratings across regularization
0.1 through 10,000. Its report is `reports/strict-mf-rating-count-validation-v1.json`.

Train Weighted ALS and select a popularity-aware hybrid ranker on strict validation users:

```bash
anime-taste-weighted-als --config configs/weighted_als.toml
```

The broad and refined validation searches are recorded in
`reports/strict-cold-start-weighted-als-v1.json` and
`reports/strict-cold-start-weighted-als-hybrid-refinement-v1.json`. These commands do not access
test users or export a production artifact.

After freezing the validation winner, run the one-time strict test and gated artifact export:

```bash
anime-taste-weighted-als-test --config configs/weighted_als_test.toml
```

The executed result is `reports/strict-cold-start-weighted-als-test-v1.json`. The fixed hybrid beat
interaction-count popularity on both test Recall@10 and NDCG@10, so the command exported
`data/artifacts/weighted-als-hybrid-v1/weighted_als_hybrid.npz`. Generated artifacts remain
Git-ignored.

Evaluate explicit low ratings as confidence-weighted negative fold-in signals without retraining the
global model or touching test users:

```bash
anime-taste-negative-feedback --config configs/negative_feedback.toml
```

The executed report is `reports/negative-feedback-validation-v1.json`. The measured improvement was
negligible, so this behavior is not enabled in production.

## Signed explicit-rating hybrid

Validate a bounded signed residual on top of the fixed Weighted ALS production score:

```bash
anime-taste-signed-hybrid-validation --config configs/signed_hybrid_validation.toml
```

The selection protocol requires Recall@10/NDCG@10 together with three product-behavior gates:
changing high ratings to 1 must materially change Top 10, recommendations must not reproduce the
pure popularity Top 10, and catalog coverage must exceed 1% of supported items. After freezing the
winner, run the final test once:

```bash
anime-taste-signed-hybrid-test --config configs/signed_hybrid_test.toml
```

The frozen candidate passed both ranking metrics against interaction-count popularity and all
behavior gates. Export its derived artifact without re-running evaluation:

```bash
anime-taste-export-signed-hybrid \
  --source-artifact ../data/artifacts/weighted-als-hybrid-v1/weighted_als_hybrid.npz \
  --signed-test-report reports/signed-hybrid-test-v1.json \
  --rating-scale-test-report reports/rating-scale-test-v1.json \
  --output-artifact ../data/artifacts/signed-rating-hybrid-v2/signed_rating_hybrid.npz
```

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

Diagnose fold-in strength without using test targets:

```bash
anime-taste-fold-in-diagnostic --config configs/fold_in_diagnostic.toml
```

This command compares regularization candidates on one common validation cohort and records ranking
metrics, user-factor norms, interaction-score variance, catalog coverage, and overlap with the
item-bias-only ranking in `reports/fold-in-diagnostic-v1.json`.

Compare explicit-MF optimizer settings without touching test targets or exporting a production
artifact:

```bash
anime-taste-mf-training-diagnostic --config configs/mf_training_diagnostic.toml
```

The experiment retrains each declared variant, selects fold-in regularization by validation
NDCG@10, and compares its full ranking with its own item-bias-only ranking. Results are written to
`reports/mf-training-diagnostic-v1.json`.

## BPR Model C experiment

Train the first ranking-aware candidate and evaluate it on validation only:

```bash
anime-taste-bpr --config configs/bpr.toml
```

The command trains BPR from high-rated train interactions, samples only items absent from each
user's train history as negatives, selects its checkpoint on a fixed validation cohort, and then
runs one full validation evaluation. It does not read test targets or export a production artifact.
The first ridge fold-in report is `reports/bpr-validation-v1.json`. The current config uses a
pairwise logistic fold-in aligned with BPR training and writes
`reports/bpr-pairwise-fold-in-v1.json`. Both candidates were rejected on validation and never
accessed test targets.

## Full-rating pairwise MF experiment

Train and validate the observed-only, full-rating pairwise candidate:

```bash
anime-taste-rating-pairwise --config configs/rating_pairwise.toml
```

The model compares titles rated by the same user, weights each pair by its exact 1–10 rating gap,
and never treats an unrated title as disliked. It evaluates 10, 20, 40, and 80-rating cold-start
profiles on strict validation only. The executed report is
`reports/rating-pairwise-validation-v1.json`; this candidate was rejected and is not a production
artifact.
