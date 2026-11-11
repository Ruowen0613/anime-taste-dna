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

Machine-readable executed reports are stored in `ml/reports/`. No metric in this document is an
estimate or an unexecuted target.
