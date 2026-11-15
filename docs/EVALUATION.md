# Offline recommendation evaluation

## Strict new-user protocol (current model-selection protocol)

An audit found that the original chronological protocol placed every user in train, validation, and
test. It separated target interactions in time, but it did not measure a user who was absent from
global item-embedding training. It also excluded each user's full hidden history from candidates
while exposing only 5, 10, or 20 ratings to fold-in. Those results remain reproducible historical
evidence, but they are no longer sufficient for selecting the next production model.

The strict protocol assigns users, with seed 42, to 14,477 global-training users, 1,809 validation
users, and 1,809 untouched test users. The groups are pairwise disjoint. Held-out users retain the
chronologically latest 20% of interactions as targets, with at least 20 earlier interactions
available for 5/10/20-rating experiments. Global model parameters may use only training users.
Candidate ranking covers all train-supported items and excludes only the ratings actually exposed
to the model; hidden history is neither used for fold-in nor silently removed from candidates.

The first validation-only baseline evaluated 1,808 eligible validation users. A broad Bayesian
shrinkage search reached a plateau at shrinkage 250,000, but pure train interaction count was still
stronger. The count baseline is therefore the strict protocol's baseline to beat:

| Observed ratings | Recall@10 | NDCG@10 | Catalog coverage |
|---:|---:|---:|---:|
| 5 | 0.02738187 | 0.55981055 | 13 items |
| 10 | 0.02748630 | 0.56191260 | 13 items |
| 20 | 0.02777205 | 0.56586980 | 16 items |

High NDCG alongside low Recall is possible because held-out targets contain many relevant items:
NDCG measures how well the ten returned positions are ordered, while Recall divides hits by all
relevant target items. The baseline is accurate but extremely concentrated, so model comparisons
must also report catalog coverage. Test users remain untouched.

### Strict explicit-MF validation

The 32-dimensional explicit-MSE model was retrained using only the 14,477 global-training users.
Four epochs were checkpointed directly by strict held-out-user ranking across 5, 10, and 20 exposed
ratings. Fold-in regularization 10, 100, 1,000, and 10,000 was selected on validation only; test
users were not loaded. Epoch 1 with regularization 10,000 was best:

| Model | Mean Recall@10 | Mean NDCG@10 | 10-rating coverage |
|---|---:|---:|---:|
| Interaction-count popularity | 0.02755 | 0.56253 | 13 items |
| Strict explicit MF | 0.01610 | 0.34806 | 12 items |

At 10 ratings, full MF achieved NDCG@10 0.34737630 versus 0.34723988 for its item-bias-only
ablation, a gain of only 0.00013642. The selected user-factor norm was 0.00110736. Lower
regularization increased coverage but sharply reduced ranking quality, and later epochs worsened
both ranking metrics despite lower train RMSE. Under the corrected protocol, explicit MF neither
beats popularity nor provides material personalization. The historical claim that MF beats
popularity applies only to the original transductive protocol and must not be described as strict
new-user performance.

### Weighted ALS and popularity hybrid validation

Weighted implicit ALS used ratings of at least 8 as positive preferences, confidence
`1 + 10 * (rating - 7)`, 32 dimensions, and the same normal equations for offline alternating
updates and held-out-user fold-in. Pure ALS improved personalization but remained below the count
baseline. A validation-only hybrid then standardized each user's ALS scores and added a weighted,
standardized log interaction-count score.

A broad search selected training iteration 2, fold-in regularization 100, and initially reached the
blend boundary. A preregistered refinement over blend weights 4–64 produced an interior optimum at
32; weight 64 was worse. The fixed validation winner was:

| Model | Mean Recall@10 | Mean NDCG@10 | 10-rating coverage |
|---|---:|---:|---:|
| Interaction-count popularity | 0.02755 | 0.56253 | 13 items |
| Weighted ALS + popularity | 0.03043 | 0.60390 | 86 items |

The hybrid also improved with additional observations: NDCG@10 was 0.58966041, 0.60511835, and
0.61692283 at 5, 10, and 20 ratings. This is the first candidate to beat the strict popularity
baseline on both ranking metrics while increasing coverage. Its iteration, training parameters,
fold-in regularization, and blend weight are now frozen before the first and only test-user run.

### Frozen strict test result

The frozen candidate was retrained deterministically for two iterations and evaluated once on all
1,809 untouched test users. No test-time parameter search occurred. The same users, targets,
complete train-supported catalog, relevance threshold, and exposed-rating exclusions were used for
the count baseline and hybrid.

| Model | Mean Recall@10 | Mean NDCG@10 | 10-rating coverage |
|---|---:|---:|---:|
| Interaction-count popularity | 0.02719940 | 0.56787919 | 14 items |
| Weighted ALS + popularity | 0.03011625 | 0.60966969 | 83 items |

The hybrid improved Recall@10 by approximately 10.7% and NDCG@10 by approximately 7.4% while
expanding 10-rating coverage almost sixfold. Its NDCG@10 increased from 0.59900206 at five ratings
to 0.62035020 at twenty ratings. Both predeclared metrics beat the baseline, so
`weighted-als-hybrid-v1` passed the export gate. The exported artifact contains fixed item factors,
interaction counts, canonical MAL IDs, titles, supported-item mask, and all inference parameters;
historical user factors and identifiers are excluded.

### Explicit low-rating fold-in experiment

After freezing the production candidate, a validation-only experiment tested whether ratings below
the positive threshold should actively shape the user vector. Positive ratings retained the
production confidence rule. Ratings at or below thresholds 3–7 were added as confidence-weighted
implicit targets with preference `p=0`; negative confidence multipliers 1, 5, 10, 20, and 40 were
searched. Global item factors, popularity blend, users, candidates, and targets stayed fixed, and
test users were not accessed again.

The selected setting used threshold 6 and multiplier 10, but improved mean Recall@10 by only
0.00000692 and mean NDCG@10 by only 0.00007377. Ten-rating catalog coverage remained 86 items.
This is not a practically meaningful improvement, so explicit low ratings are not enabled in the
production fold-in. They remain candidate exclusions. Future negative-feedback work should use a
model trained jointly with signed or pairwise preferences, and must establish a material validation
effect before receiving a new independent evaluation.

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

### Validation-only fold-in diagnosis

A follow-up sweep tested fold-in regularization 0.1, 1, 10, 100, and 1,000 on one common validation
cohort without accessing test targets. At 10 observed ratings, item-bias-only achieved Recall@10
0.02908499 and NDCG@10 0.27925718. The exported regularization of 1,000 achieved Recall@10
0.02919120 and NDCG@10 0.27848268: a Recall gain of only 0.00010621 and an NDCG loss of 0.00077450.

| Fold-in regularization | User-factor norm | Interaction score std | Coverage | Bias overlap | Recall@10 | NDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 2.3873 | 1.0130 | 35.51% | 1.66% | 0.00172473 | 0.01568440 |
| 1 | 1.6174 | 0.6947 | 31.42% | 4.74% | 0.00386249 | 0.03438100 |
| 10 | 0.5721 | 0.2544 | 12.74% | 33.22% | 0.01749664 | 0.15968201 |
| 100 | 0.1001 | 0.0457 | 1.08% | 84.02% | 0.02826578 | 0.26922216 |
| 1,000 | 0.0112 | 0.0051 | 0.31% | 98.33% | 0.02919120 | 0.27848268 |

Lower regularization recovers diversity but makes ranking substantially worse, and more observations
make the weakly regularized trials worse rather than better. The failure is therefore not fixed by
choosing a different fold-in value: the learned interaction geometry does not generalize to new-user
ranking. The training curve supports this diagnosis. Train RMSE fell from 1.4325 to 1.2494 over
epochs 1–3 while validation RMSE rose from 1.6777 to 1.7139, so epoch 1 is a legitimate early-stop
result rather than a reporting bug. The next experiment should change training regularization and
checkpoint selection before introducing a different model objective.

### Explicit-MF training variants

Three additional models were trained without test access using learning rate 0.005 and AdamW weight
decay 0.00001, 0.0001, or 0.001. All three again selected epoch 1 by validation RMSE and produced
nearly identical results. The lowest-decay variant improved validation RMSE from the original
model's 1.6777 to 1.5952, but its best validation ranking used fold-in regularization 1,000 and
achieved only Recall@10 0.02180459 and NDCG@10 0.20279198. Its own item-bias-only ranking achieved
Recall@10 0.02252185 and NDCG@10 0.20109621.

The interaction term therefore added 0.00169577 NDCG but lost 0.00071726 Recall, retained 91.93%
Top-10 overlap with bias-only, and covered only 55 supported items. Improving rating RMSE did not
improve Top-N ranking, while practical AdamW weight-decay changes had no meaningful effect. This is
evidence of objective mismatch rather than a fold-in hyperparameter mistake. Further tuning of the
same explicit-MSE setup is not justified before testing a ranking-aware objective.

## BPR Model C validation experiment

The first ranking-aware candidate trained 32-dimensional Bayesian Personalized Ranking matrix
factorization on 12,679,964 train interactions rated 8 or higher. Each positive was paired with a
uniformly sampled train-supported item absent from that user's train history. A fixed cohort of
2,048 validation users selected the checkpoint by NDCG@10; epoch 1 won even though pairwise training
loss continued to fall through epoch 4. Test targets were never loaded.

For cold-start validation, observed ratings were centered and folded into the fixed BPR item factors
with ridge regression. On the full 18,078-user validation cohort, regularization 100 produced the
best NDCG@10:

| Model | Recall@10 | NDCG@10 | Catalog coverage |
|---|---:|---:|---:|
| Current explicit MF, reg 1,000 | 0.02919120 | 0.27848268 | 0.31% |
| BPR Model C, reg 100 | 0.01585189 | 0.18092936 | 1.71% |

BPR increased coverage but substantially reduced both ranking metrics, so this candidate is rejected
without test evaluation or artifact export. The falling BPR loss alongside worsening ranking also
shows that offline pair classification alone is not a sufficient checkpoint signal. A likely design
weakness is the mismatch between pairwise historical-user training and rating-regression cold-start
fold-in; a follow-up must optimize the new-user vector with the same pairwise objective before BPR
can be judged conclusively.

### Pairwise cold-start follow-up

The follow-up removed that mismatch. It fixed BPR item parameters and optimized each new user vector
with regularized pairwise logistic loss. Observed rating gaps of at least two points supplied direct
preference pairs; high-rated observations were also paired with deterministic sampled items absent
from the user's observed history. Eight batched Newton updates solved the 32-dimensional user vector.

Checkpoint monitoring used a fixed seeded random sample of 2,048 validation users rather than the
first users in index order. On the full validation cohort, regularization 10 was selected:

| Fold-in | Recall@10 | NDCG@10 | Catalog coverage |
|---|---:|---:|---:|
| BPR ridge | 0.01585189 | 0.18092936 | 1.71% |
| BPR pairwise | 0.02055292 | 0.21774257 | 15.43% |
| Current explicit MF | 0.02919120 | 0.27848268 | 0.31% |

Matching training and fold-in objectives materially improved both BPR accuracy and diversity, but
it still failed the predeclared requirement to beat the current MF on Recall@10 and NDCG@10. The BPR
route is therefore stopped at validation: no test evaluation, artifact export, or production change
is justified. Future work should test a different model family or candidate/reranking strategy rather
than continue tuning this BPR implementation.

Machine-readable executed reports are stored in `ml/reports/`, including
`personalization-audit-v1.json` and `fold-in-diagnostic-v1.json`. No metric in this document is an
estimate or an unexecuted target. The explicit-MF training comparison is stored in
`mf-training-diagnostic-v1.json`; the rejected BPR candidate is stored in
`bpr-validation-v1.json`, and its pairwise fold-in follow-up is stored in
`bpr-pairwise-fold-in-v1.json`.

## Online artifact

The exported `matrix-factorization-v1` NPZ is approximately 2.16 MB and contains item factors,
item biases, the global mean, selected fold-in regularization, supported-item mask, MAL IDs, and
basic titles. It deliberately excludes historical user factors and identifiers. A local FastAPI
contract run mapped 10 ratings, returned 10 unseen items, and completed in approximately 31 ms.
This local timing is evidence for bounded inference behavior, not a production latency guarantee.
