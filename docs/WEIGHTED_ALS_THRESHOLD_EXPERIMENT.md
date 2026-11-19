# Weighted ALS positive-threshold experiment

## Question

Does treating ratings of 6 or 7 as positive implicit feedback improve strict cold-start ranking
relative to the production threshold of 8?

## Protocol

- Dataset: `v2-user-holdout` strict validation-user split.
- Test users: not accessed.
- Fixed seed: 42.
- Model: 32-dimensional Weighted ALS, four iterations, alpha 10, training regularization 1.
- Selection: validation NDCG@10, then Recall@10, over the existing popularity-blend grid.
- Evaluation: the same 1,808 eligible users at 5, 10, and 20 observed ratings; held-out ratings
  of at least 8 remain the relevance definition.

## Results

| Positive threshold | Mean Recall@10 | Mean NDCG@10 | Selected iteration | Popularity blend |
|---:|---:|---:|---:|---:|
| 8 (current baseline) | 0.03042765 | 0.60390053 | 2 | 32 |
| 7 | 0.03059096 | 0.60563623 | 2 | 12 |
| 6 | 0.02973656 | 0.59568588 | 4 | 6 |

Threshold 7 improved mean Recall@10 by about 0.54% and mean NDCG@10 by about 0.29% relative to
threshold 8. The gain was concentrated at five observed ratings. At 20 ratings, threshold 7 was
worse on both metrics, and its selected Top-10 catalog coverage was lower at every observation
count (79/62/38 items versus 100/86/51).

Threshold 6 was worse than threshold 8 on both mean metrics. Treating 6 as positive feedback is
therefore rejected under this protocol.

## Decision

Do not change production from threshold 8 yet. Threshold 7 is a promising but very small
validation-only improvement with a coverage tradeoff and mixed results by rating count. Before any
test access or production replacement, run a focused validation refinement that preserves the
confidence assigned to 8/9/10 while giving 7 a weaker weight, and add an explicit minimum-coverage
acceptance criterion.

## Weak 7 rating follow-up

A second validation-only experiment kept the production confidence for ratings 8, 9, and 10
fixed at 11, 21, and 31. It added rating 7 with confidence 1, 3, 5, or 7. Confidence 1 is the
unchanged baseline because it contributes no extra implicit weight.

| Rating 7 confidence | Mean Recall@10 | Mean NDCG@10 | Coverage at 5/10/20 ratings |
|---:|---:|---:|---:|
| 1 (baseline) | 0.03043054 | 0.60390053 | 100 / 86 / 51 |
| 3 | 0.03043054 | 0.60485079 | 76 / 52 / 29 |
| 5 | 0.03053948 | 0.60523514 | 54 / 42 / 26 |
| 7 | 0.03054504 | 0.60602640 | 46 / 41 / 26 |

Confidence 7 improved mean Recall@10 by about 0.39% and mean NDCG@10 by about 0.35% relative to
the threshold-8 baseline. The improvement was concentrated at five observed ratings. At 20
ratings, Recall@10 and NDCG@10 were both lower than the baseline, and catalog coverage was roughly
halved or worse at every observation count.

This establishes that rating 7 contains useful positive information, but does not justify changing
the production ALS component by itself. The production signed residual already treats rating 7 as
positive. Any replacement must be evaluated as a complete signed hybrid and meet an explicit
coverage gate before test access or deployment.

Full reports:

- `ml/reports/strict-cold-start-weighted-als-hybrid-refinement-v1.json`
- `ml/reports/weighted-als-threshold-7-validation-v1.json`
- `ml/reports/weighted-als-threshold-6-validation-v1.json`
- `ml/reports/weighted-als-weak-7-confidence-{1,3,5,7}-validation-v1.json`
