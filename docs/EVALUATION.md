# Offline recommendation evaluation

## Protocol

Both models use the same deterministic per-user chronological train, validation, and test split.
Only train interactions fit model parameters. Ratings of 8 or higher in the target split count as
relevant. Ranking covers every train-supported item after removing the user's history available
before the target split, and metrics are macro-averaged across eligible users.

The cold-start conditions expose the 5, 10, or 20 most recent interactions before test. Matrix
factorization freezes item parameters and solves a regularized user factor and bias from those
ratings. Popularity is non-personalized, so its ranking is intentionally unchanged across the
three conditions. Validation chooses model checkpoints and fold-in settings; test is used only
after those choices are fixed.

## Executed results

| Model | Observed | Eligible users | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Bayesian popularity | 5 | 18,086 | 0.03076363 | 0.37968674 |
| Bayesian popularity | 10 | 18,086 | 0.03076363 | 0.37968674 |
| Bayesian popularity | 20 | 18,086 | 0.03076363 | 0.37968674 |
| Matrix factorization | 5 | 18,086 | 0.04385212 | 0.52429491 |
| Matrix factorization | 10 | 18,086 | 0.04394219 | 0.52467443 |
| Matrix factorization | 20 | 18,086 | 0.04412245 | 0.52577783 |

At 5, 10, and 20 observations, matrix factorization improves Recall@10 over popularity by 42.55%,
42.84%, and 43.42%, respectively. NDCG@10 improves by 38.09%, 38.19%, and 38.48%.

The 32-dimensional explicit-feedback model minimizes rating MSE. Validation RMSE selected epoch 1;
later epochs lowered train RMSE while validation RMSE worsened. Fold-in regularization was selected
without test access: among candidates within 0.001 of the best mean validation NDCG@10, choose the
one with highest mean validation Recall@10. This selected regularization 1,000.

## Limitations

- The source snapshot ends in 2023 and overrepresents highly active MyAnimeList users.
- Historical non-interaction is not evidence of dislike, so metrics measure recovery of known
  high ratings rather than causal preference or exposure.
- Chronological test behavior is affected by release timing and catalog popularity.
- The improvement from 5 to 20 observations is real but small. More ratings should not be presented
  as guaranteeing materially better recommendations without further model and protocol work.
- Only one matrix-factorization architecture and training configuration has been executed so far.
  Broader hyperparameter experiments must remain validation-only.

## Executed Taste Quiz proxy evaluation

The pairwise experiment evaluates the same fixed `matrix-factorization-v1` item artifact with 10
observed ratings plus 0, 5, or 10 simulated Quiz rounds. A Quiz pair is constructed only from
chronologically earlier history, requires a rating gap of at least two points, and never accesses
the target split. All conditions use the same users who can supply 10 pairs. Historical rating
differences are only a proxy for pre-watch interest, which remains an important limitation.

Validation compared pairwise weights 0, 0.05, 0.1, 0.25, 0.75, 2, and 5. Weight 0 produced the
highest validation NDCG@10, so it was selected before test evaluation. Stronger weights increased
preference satisfaction but consistently reduced ranking quality.

| Quiz rounds | Eligible test users | Recall@10 | NDCG@10 | Mean Top-10 replacements |
|---:|---:|---:|---:|---:|
| 0 | 18,081 | 0.04395434 | 0.52481952 | — |
| 5 | 18,081 | 0.04395434 | 0.52481952 | 0.0 |
| 10 | 18,081 | 0.04395434 | 0.52481952 | 0.0 |

The evidence does not support using the current linear pairwise constraint to update the online
user vector. Production weight is therefore 0. Quiz choices remain separate feedback; least-interest
items may be explicitly excluded, but the system does not claim that Quiz improves accuracy. A
pairwise-aware training objective or hybrid model requires a new validation experiment.

## Personalization audit

The ranking metrics above establish that matrix factorization beats the popularity baseline, but
they do not establish that its recommendations vary meaningfully between users. A separate audit
therefore measured catalog coverage, recommendation concentration, user-vector magnitude, and
Top-10 overlap against inference ablations on the same 18,086-user test cohort.

At 10 observed ratings, the full model recommended only 62 of 14,075 supported items across all
users (0.44% catalog coverage). Its mean Top-10 overlap with an item-bias-only ranking was 98.21%,
while overlap with the interaction-only ranking was 0.23%. The mean folded-in user-factor norm was
0.0108. Increasing observations from 10 to 20 changed only 1.95% of Top-10 slots on average.

These results show that the current model's strong Recall@10 and NDCG@10 mostly come from learned
item bias rather than useful user-specific ordering. The interaction-only ablation produced much
broader coverage (4,742 items, 33.69%) but very poor ranking quality (Recall@10 0.00123522 and
NDCG@10 0.01121872), so simply removing item bias is not a viable fix. The next model experiment
must be selected on validation data and report both ranking quality and personalization diagnostics.

Machine-readable executed reports are stored in `ml/reports/`, including
`personalization-audit-v1.json`. No metric in this document is an estimate or an unexecuted target.

## Online artifact

The exported `matrix-factorization-v1` NPZ is approximately 2.16 MB and contains item factors,
item biases, the global mean, selected fold-in regularization, supported-item mask, MAL IDs, and
basic titles. It deliberately excludes historical user factors and identifiers. A local FastAPI
contract run mapped 10 ratings, returned 10 unseen items, and completed in approximately 31 ms.
This local timing is evidence for bounded inference behavior, not a production latency guarantee.
