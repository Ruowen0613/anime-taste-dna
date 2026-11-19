# Two Stage Candidate Reranker Experiment

## Question

Can a learned pairwise reranker improve the production model by reordering only its Top 200
candidates rather than replacing full-catalog retrieval?

## Protocol

- The production signed hybrid generated Top 200 candidates.
- The reranker was fitted on 3,000 global-training users using their final 20 interactions as
  pseudo-future targets; validation and test users were excluded from fitting.
- Training produced 159,650 pairwise examples from 15,965 retrieved positive targets.
- Features included the production score, its ALS/popularity and signed components, item
  popularity, positive and signed embedding similarity, item-factor norm, and observed-rating
  count.
- A 16-unit PyTorch MLP learned pairwise order over candidates.
- Validation covered the same 1,801 strict users at 10, 20, 40, and 80 observed ratings.
- Residual weights from 0 through 0.5 were evaluated. Weight 0 exactly preserves the production
  ranking and must be beaten before the reranker can be accepted.
- Test users were not accessed.

## Results

The fixed production candidate generator achieved mean Recall@10 0.03133163 and mean NDCG@10
0.62250454 on the common validation slices.

Every epoch selected residual weight 0. At epoch 1, the best nonzero residual weight was 0.001 and
reduced Recall@10 to 0.03131358 and NDCG@10 to 0.62187020. Later epochs continued to reduce
training pairwise loss but further degraded validation ranking. A standalone reranker without the
production-score residual guard was substantially worse, so it was rejected before final reporting.

## Decision

Do not integrate this reranker or access the test set. The current pseudo-target pairwise task
mostly teaches the model to demote popular hard negatives, but that behavior does not improve
future high-rating retrieval. The strong production score already contains the useful signal from
these item embeddings and popularity statistics.

Further reranker work requires genuinely new features or labels, such as offline AniList tags,
synopsis representations, or updated interaction data. Recombining the existing production score,
item factors, and popularity is unlikely to yield a material gain.

Full report: `ml/reports/candidate-reranker-validation-v1.json`.
