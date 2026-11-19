# Genre Metadata Hybrid Experiment

## Question

Can a genre-based user profile improve the production signed hybrid ranking without sacrificing
accuracy, catalog coverage, or personalization?

## Protocol

- Strict held-out validation users only; test users were not accessed.
- The production Weighted ALS, signed residual, and rating-count-aware popularity decay stayed fixed.
- Static item genres were converted to IDF-weighted vectors using train-supported items only.
- A signed user genre profile was built from exposed 1–10 ratings around neutral midpoint 5.5.
- The standardized genre score was added as a small reranking residual.
- Selection used mean NDCG@10 after Recall@10, coverage, and popularity-overlap gates.

The offline item table supplied 77 retained genre features and covered 100% of train-supported
items after the minimum-frequency filter.

## Results

| Metadata weight | Mean Recall@10 | Mean NDCG@10 | Mean coverage | Popularity overlap |
|---:|---:|---:|---:|---:|
| 0.0 | 0.03052464 | 0.60689268 | 251.0 | 62.21% |
| 0.0001 | 0.03054951 | 0.60683350 | 250.4 | 62.26% |
| 0.0003 | 0.03055691 | 0.60693618 | 250.4 | 62.35% |
| 0.001 | 0.03052409 | 0.60569744 | 250.2 | 62.14% |
| 0.003 | 0.02995652 | 0.59638792 | 267.4 | 59.45% |
| 0.010 | 0.02712977 | 0.53825468 | 340.0 | 36.87% |

The selected global weight was 0.0003. Relative to no metadata signal, it improved mean Recall@10
by about 0.11% and mean NDCG@10 by about 0.007%, while slightly reducing coverage and increasing
popularity overlap. The gain is not practically meaningful.

By rating count, the best NDCG weight was zero for 5, 10, and 20 observations, 0.0003 for 40, and
0.001 for 80. That schedule would improve mean NDCG by about 0.09% and mean Recall by about 0.26%,
but would also increase popularity overlap. This remains too small and mixed to justify test access
or production integration.

## Decision

Do not add this genre residual to production. Genres are useful for explanations and Taste DNA but
are too coarse to identify a specific future high-rated title from the full catalog. A future
metadata experiment should use richer tags or learned text representations and must be evaluated
under the same leakage-safe protocol.

Full report: `ml/reports/metadata-hybrid-validation-v1.json`.
