# Theory index

Every result listed here is stated in English, with assumptions and proof,
on the linked page, and machine-checked in Lean 4 / Mathlib under
[`proofs/`](../../proofs/README.md). No proof uses `sorry`. Run
`lake build` in `proofs/` to check them.

## Metrics

| Result | Statement | English | Lean |
|---|---|---|---|
| Balance range | $0 \le \beta \le 2 - 2/K$, attained by a single partition | [evaluation.md](../evaluation.md#2-partition-quality) | `PartitionBalance.balance_le`, `balance_of_single_partition` |

## Reconstruction F1 under partitioning ([E08](../../results/E08-f1-decomposition/README.md))

| Result | English | Lean file |
|---|---|---|
| Law-of-total-expectation decomposition $\mathbb{E}[F] = S(1 - (1-\alpha)\delta)$ | [E08 §Setup](../../results/E08-f1-decomposition/README.md#setup) | [`PartitioningF1.lean`](../../proofs/VertexVoyageProofs/PartitioningF1.lean) |
| Node-averaged (macro) decomposition | [E08 §Macro-averaging](../../results/E08-f1-decomposition/README.md#macro-averaging-f1-score) | [`PartitioningF1NodeAveraged.lean`](../../proofs/VertexVoyageProofs/PartitioningF1NodeAveraged.lean) |
| Chance floor of a global pair budget | [E08 §Chance floor](../../results/E08-f1-decomposition/README.md#the-chance-floor-a-global-pair-budget-makes-density-the-no-skill-baseline) | [`PartitioningF1ChanceFloor.lean`](../../proofs/VertexVoyageProofs/PartitioningF1ChanceFloor.lean) |
| Erdős–Rényi within-block density equals $p$ | [E08 §Erdős–Rényi sweep](../../results/E08-f1-decomposition/README.md#the-erdősrényi-sweep-does-q1q8-depend-on-average-degree) | [`ErdosRenyi.lean`](../../proofs/VertexVoyageProofs/ErdosRenyi.lean), [`ErdosRenyiPartitions.lean`](../../proofs/VertexVoyageProofs/ErdosRenyiPartitions.lean), [`ErdosRenyiChanceFloor.lean`](../../proofs/VertexVoyageProofs/ErdosRenyiChanceFloor.lean) |

## Partition alignment ([E10](../../results/E10-partition-alignment/README.md))

All in [`PartitionAlignment.lean`](../../proofs/VertexVoyageProofs/PartitionAlignment.lean).

| # | Result | English | Lean name(s) |
|---|---|---|---|
| 1 | Two runs differing by a linear map; effect of normalisation | [theory](partition-alignment.md#theorem-1-when-two-runs-differ-by-a-linear-map-and-what-normalisation-does-to-it) | `sgns_score_gl_invariant`, `normalised_linear_is_linear_only_if_conformal` |
| 2 | A random isometry is as far as a random direction | [theory](partition-alignment.md#theorem-2-a-random-isometry-is-as-far-as-a-random-direction) | `norm_isometry_sub_sq` |
| 3 | The residual block can represent any linear map | [theory](partition-alignment.md#theorem-3-the-residual-block-can-represent-any-linear-map) | `relu_sub_relu_neg` |
| 4 | The alignment signal lives in cross-partition pairs | [theory](partition-alignment.md#theorem-4-the-alignment-signal-lives-in-cross-partition-pairs) | `contrastiveLoss_comp_isometry` |
| 5 | Geometry of the unit sphere | [theory](partition-alignment.md#theorem-5-geometry-of-the-unit-sphere) | `norm_sub_sq_of_unit` |
| 6 | Collapse and frustration of the contrastive loss | [theory](partition-alignment.md#theorem-6-collapse-and-frustration-of-the-contrastive-loss) | `contrastiveLoss_const` |
| 7 | Why normalisation is needed | [theory](partition-alignment.md#theorem-7-why-normalisation-is-needed) | `contrastive_hinge_scale` |
| 8 | Orthogonal alignment is a maximum-cosine problem | [theory](partition-alignment.md#theorem-8-orthogonal-alignment-is-a-maximum-cosine-problem) | `procrustes_objective_of_unit` |
| 9 | Merging replicas | [theory](partition-alignment.md#theorem-9-merging-replicas) | `replica_spread_of_unit` |
| 10 | Headroom | [theory](partition-alignment.md#theorem-10-headroom) | `alignment_gain_le` |
| 11 | Angle between two term gradients from three norms | [log](../../results/E10-partition-alignment/log/2026-11-07-contrastive-citeseer.md#theorem-11-the-angle-between-two-term-gradients-is-read-from-three-norms) | `norm_add_sq_of_terms` |
| 12 | Mean positive cosine from the attraction term | [log](../../results/E10-partition-alignment/log/2026-11-07-contrastive-citeseer.md#theorem-12-mean-positive-cosine-from-the-attraction-term) | `sum_norm_sub_sq_of_unit` |
| 13 | No cross-partition reconstruction caps recall at $1 - \delta$ | [log](../../results/E10-partition-alignment/log/2026-11-07-contrastive-citeseer.md#theorem-13-no-cross-partition-reconstruction-caps-recall-at-1---delta) | `recall_le_within_share` |
| 14 | Micro recall splits by scope | [log](../../results/E10-partition-alignment/log/2026-11-07-contrastive-astroph.md#theorem-14-micro-recall-splits-by-scope) | `recall_split_by_scope` |
| 15 | Shares of the inverse-fraction weighting | [log](../../results/E10-partition-alignment/log/2026-11-07-cross-weighted-contrastive-astroph.md#theorem-15-shares-of-the-inverse-fraction-weighting) | `total_weight_of_inverse_fraction` |
| 16 | Cancellation ratio of several terms | [log](../../results/E10-partition-alignment/log/2026-11-07-cross-weighted-contrastive-astroph.md#theorem-16-cancellation-ratio-of-several-terms) | `cancellation_ratio_of_terms_le_one` |
| 17 | A triplet with an orthogonal negative is a cosine threshold | [log](../../results/E10-partition-alignment/log/2026-11-07-cross-weighted-triplet-citeseer.md#theorem-17-a-triplet-with-an-orthogonal-negative-is-a-cosine-threshold) | `norm_sub_of_orthogonal_unit` |
| 18 | A triplet with a correlated negative | [log](../../results/E10-partition-alignment/log/2026-11-07-cross-weighted-triplet-astroph.md#theorem-18-a-triplet-with-a-correlated-negative) | `norm_sub_of_unit_eq_sqrt` |

The Lean names in this table are the first ones cited after each statement.
Some theorems use further lemmas, which are named on the linked page.

## Not formalised

- $\mathbb{E}[\delta] = (P-1)/P$ for random partitioning: linearity of
  expectation over edges, see [evaluation.md](../evaluation.md#2-partition-quality).
