# Rating Count Aware Signed Weight Experiment

## Question

Should the signed explicit-rating residual receive more weight as a user supplies more ratings?

## Protocol

- Strict held-out validation users only; test users were not accessed.
- Same 1,801 users were eligible at 5, 10, 20, 40, and 80 observed ratings.
- The production Weighted ALS artifact, signed regularization, and linear popularity decay stayed
  fixed.
- Signed weights 0.03, 0.04, 0.05, 0.06, 0.075, 0.10, 0.15, and 0.20 were compared.
- Selection maximized NDCG@10 after gates for Recall@10, catalog coverage, and popularity overlap.

## Results

| Observed ratings | Fixed weight | Selected weight | Fixed NDCG@10 | Selected NDCG@10 |
|---:|---:|---:|---:|---:|
| 5 | 0.05 | 0.05 | 0.54444306 | 0.54444306 |
| 10 | 0.05 | 0.05 | 0.58301074 | 0.58301074 |
| 20 | 0.05 | 0.05 | 0.61222057 | 0.61222057 |
| 40 | 0.05 | 0.04 | 0.63258090 | 0.63484221 |
| 80 | 0.05 | 0.03 | 0.66220815 | 0.66290574 |

Across all five rating counts, the schedule changed mean Recall@10 from 0.03052464 to 0.03058576
and mean NDCG@10 from 0.60689268 to 0.60748446. Mean catalog coverage changed from 251.0 to
246.6 items, while mean popularity overlap changed from 62.21% to 62.34%.

The hypothesis that signed weight should increase with more evidence was not supported. The fixed
0.05 weight remained optimal through 20 ratings, and validation selected slightly lower weights at
40 and 80 ratings. The aggregate ranking gain was about 0.10% NDCG and 0.20% Recall, with slightly
lower coverage and slightly higher popularity overlap. This is too small and mixed to justify test
access or a production change.

Full report: `ml/reports/signed-weight-schedule-validation-v1.json`.
