# Uniform triplet loss on CITESEER analysis

Source: one run of `vv test` on CITESEER with the triplet aggregator, logged in `VertexVoyage/vv.log` (2026-11-07, 19:49 to 19:50). The settings are those of [Cross-weighted triplet loss on CITESEER analysis](2026-11-07-cross-weighted-triplet-citeseer.md#cross-weighted-triplet-loss-on-citeseer-analysis) (margin $\alpha = 0.5$, unit output scale, $P = 4$ [label propagation](../../../doc/glossary.md#label-propagation), bilinear decoder, 100 aligner epochs, validation off), with one change: `alignment_link_weighting = uniform` (log: `Aligner link weighting uniform`). This is item 1 of that section's next runs. It isolates the effect of the $1/\delta$ weighting on the triplet loss. As before, every run is a **single run with one seed**, on its own unseeded LPA partition.

## Setup

| Quantity | Triplet, uniform | Triplet, weighted | Contrastive, weighted | Contrastive, uniform |
|---|---|---|---|---|
| $\lvert E_p \rvert$ | $856, 783, 803, 767$; sum $3209$ | sum $3244$ | sum $3169$ | sum $3190$ |
| $\lvert E_{\text{cut}} \rvert$, $\delta$ | $874$, $0.2141$ | $839$, $0.2055$ | $914$, $0.2239$ | $893$, $0.2187$ |
| Cross share of the objective | $\delta = 0.2141$ | $1/(2 - \delta) = 0.5573$ | $0.5630$ | $\delta = 0.2187$ |
| node2vec partition F1 | $0.390, 0.418, 0.382, 0.389$ | $0.363, 0.419, 0.388, 0.347$ | $0.351, 0.485, 0.359, 0.415$ | $0.400, 0.422, 0.418, 0.369$ |
| HeaRT test edges cut | $300/453 = 66.2\%$ | $57.8\%$ | $64.0\%$ | $63.8\%$ |
| Decoder train / validation loss | $0.461$ / $0.464$ | $0.470$ / $0.461$ | $0.207$ / $0.246$ | $0.170$ / $0.199$ |

- All partitions have $816$ nodes. The aligner saw all $4083$ links over $3149$ replica rows, and the $8166 = 2 \cdot 4083$ logged pairs are every link in both directions.
- The $9788$ warnings are again the vocabulary warnings, one per partition for each node outside it. No other warning fired.
- The node2vec partition F1 values are within the range of the earlier runs, so the starting replicas are of similar quality.

## What the uniform log can and cannot show

With uniform weighting, the objective is the plain triplet mean over all $8166$ pairs, logged as a single `ranking` term. There is no within/cross split of the loss or of the gradient, so the cosine $\cos(g_{\text{w}}, g_{\text{c}})$ and $\rho$ of the weighted runs cannot be computed. What remains is a bound.

**Bound on the cut hinge.** Let $T_{\text{w}}, T_{\text{c}} \geq 0$ be the (unlogged) within and cut means of the hinge. A mean over a disjoint union is the count-weighted mean of the part means. The cut pairs are $2 \cdot 874 = 1748$ of the $8166$ pairs, a share of exactly $\delta$, so
$$ L = (1 - \delta)\, T_{\text{w}} + \delta\, T_{\text{c}}. $$
This is the same decomposition as [Theorem 15](2026-11-07-cross-weighted-contrastive-astroph.md#theorem-15-shares-of-the-inverse-fraction-weighting) (d), with shares $1 - \delta$ and $\delta$ instead of the inverse-fraction ones. Because $T_{\text{w}} \geq 0$, it follows that $T_{\text{c}} \leq L / \delta$.

**Consistency check at epoch 1.** Epoch 1 is evaluated before the first optimiser step, so it measures the replicas moved only by the randomly initialised residual `vectors + second(relu(first(vectors)))`, which is not aligned. With the scope means of the weighted triplet run at epoch 1 ($T_{\text{w}} = 0.0077$, $T_{\text{c}} = 0.5926$), the decomposition predicts $0.7859 \cdot 0.0077 + 0.2141 \cdot 0.5926 = 0.133$. The logged value is $0.1327$. The two runs have different partitions, so this agreement says that the unaligned within and cut hinges are much the same from partition to partition.

## Training dynamics

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 15 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $L$ | 0.1327 | 0.1193 | 0.1078 | 0.0989 | 0.0938 | 0.0875 | 0.0770 | 0.0700 | 0.0549 | 0.0461 | 0.0311 | 0.0246 | 0.0186 | 0.0163 | 0.0142 | 0.0147 | 0.0127 | 0.0126 |
| $\lVert g \rVert$ | 0.0240 | 0.0214 | 0.0204 | 0.0252 | 0.0220 | 0.0240 | 0.0192 | 0.0214 | 0.0177 | 0.0169 | 0.0135 | 0.0206 | 0.0136 | 0.0119 | 0.0144 | 0.0134 | 0.0151 | 0.0163 |

- **The loss falls $90.5\%$**, from $0.1327$ to $0.0126$. It first goes below $0.05$ at epoch 17 and below $0.02$ at epoch 50. Its minimum is $0.0109$ at epoch 92, and over epochs 86 to 100 it moves within $[0.0109, 0.0138]$: a noisy plateau, as in the weighted run, from the one resampled negative per triplet.
- **The cut triplets are mostly satisfied by the end.** By the bound, $T_{\text{c}} \leq 0.0126 / 0.2141 = 0.059$ at epoch 100, against $T_{\text{c}} \leq 0.620$ at epoch 1. The weighted run ended at $T_{\text{c}} = 0.051$. So even with $2.6\times$ less weight on the cut pairs ($0.214$ against $0.557$), the cut hinge ends at most slightly above the weighted run's.
- **No collapse, no vanishing gradient.** The total norm stays between $0.0115$ (epoch 72) and $0.0252$ (epoch 4), median $0.0147$ over epochs 51 to 100. As in the weighted run, it does not decay with the loss: the hinge gradient of an active triplet has fixed size, and only the number of active triplets shrinks. The median is about $3.3\times$ lower than the weighted run's $0.049$. This fits the cut term, which drives the training, getting $2.6\times$ less weight.

## Downstream metrics

**Graph reconstruction ($k = 4083$ closest pairs).**

| Quantity | **Triplet, uniform** | Triplet, weighted | Contrastive, weighted | Contrastive, uniform |
|---|---|---|---|---|
| Reconstructed cut pairs / true | 0 / 0 | 0 / 0 | 31 / 3 | 0 / 0 |
| Within true positives | **2116** of 3209 | 1789 of 3244 | 1593 of 3169 | 1722 of 3190 |
| Micro recall | **0.518** | 0.438 | 0.391 | 0.422 |
| Within recall $r_w$ | **0.659** | 0.551 | 0.503 | 0.540 |
| Within pair F1 | **0.580** | 0.488 | 0.441 | 0.474 |
| Mean partition pair F1 | **0.580** | 0.492 | 0.448 | 0.480 |
| Global macro F1 | **0.671** | 0.596 | 0.536 | 0.544 |
| Macro within recall | **0.813** | 0.726 | 0.624 | 0.631 |

- **The best CITESEER reconstruction so far, by a wide margin.** Micro recall rose by $+0.080$ over the weighted triplet run and by $+0.096$ over the best contrastive run. Global macro F1 is $0.671$. $2962$ of the $3145$ nodes with true neighbours get a reconstructed neighbour.
- **Still no cut pair.** All $4083$ reconstructed pairs are within partitions (`reconstructed-edge partition share` within $= 1.000$). By [Theorem 13](2026-11-07-contrastive-citeseer.md#theorem-13-no-cross-partition-reconstruction-caps-recall-at-1---delta), the micro recall is capped at $1 - \delta = 0.786$, and the run reaches $66\%$ of that cap. The whole gain is in the within scope.
- **Per partition:**

| Partition | True edges | Reconstructed | Reconstructed / true | Recall | Precision | Pair F1 |
|---|---|---|---|---|---|---|
| 0 | 856 | 924 | 1.08 | 0.609 | 0.564 | 0.585 |
| 1 | 783 | 1018 | 1.30 | 0.676 | 0.520 | 0.587 |
| 2 | 803 | 1037 | 1.29 | 0.676 | 0.524 | 0.590 |
| 3 | 767 | 1104 | 1.44 | 0.682 | 0.474 | 0.559 |

The partitions are balanced, so there is no size bias: every partition gets $1.08$ to $1.44$ times its true edge count, and the pair F1 values are within $0.031$ of each other.

**Link prediction, HeaRT (500 negatives per positive, all 453 test edges).**

| Edges | **Triplet, uniform**: MRR / Hits@1 / Hits@3 / Hits@10 / mean rank | Triplet, weighted: MRR | Contrastive, weighted: MRR | Contrastive, uniform: MRR |
|---|---|---|---|---|
| All (453) | **0.212** / 0.126 / 0.234 / 0.400 / 105.3 | 0.171 | 0.124 | 0.147 |
| Cross-partition (300) | **0.107** / 0.040 / 0.113 / 0.257 / 133.2 | 0.101 | 0.063 | 0.072 |
| Within-partition (153) | **0.380** / 0.222 / 0.477 / 0.673 / 52.4 | 0.273 | 0.215 | 0.291 |
| Earlier $P = 4$ baseline, 10 runs | 0.138 ± 0.007 / 0.081 ± 0.010 / 0.155 ± 0.010 / 0.245 ± 0.013 | | | |

- **Best on every row.** Against the 10-run baseline: MRR $+10.5$, Hits@1 $+4.5$, Hits@3 $+7.9$ and Hits@10 $+11.9$ standard deviations. The weighted triplet run was $+4.7$ on MRR.
- **The gain over the weighted triplet run is almost all within.** Within MRR $+0.107$ ($0.273 \to 0.380$), cross MRR $+0.006$ ($0.101 \to 0.107$). With $153$ to $191$ within test edges and a reciprocal-rank standard deviation of about $0.35$, one within MRR has standard error about $0.027$, so the within gain is about $2.8$ standard errors of a difference. With $262$ to $300$ cut edges, one cross MRR has standard error about $0.012$, so the cross difference is within noise.
- **Mix correction.** This run has the largest cut share of test edges ($66.2\%$). Cut edges rank worse, so the mix works against this run. Recomposed at a common $64\%$ cut share, $0.64 \cdot 0.107 + 0.36 \cdot 0.380 = 0.205$, against $0.163$ (triplet weighted), $0.151$ (contrastive uniform) and $0.118$ (contrastive weighted). The lead over the weighted triplet run, $+0.042$, is about $4.2$ run-to-run standard deviations of a single-run difference ($\sqrt 2 \cdot 0.007 = 0.010$).
- The count-weighted mean of the two scope MRRs is $(300 \cdot 0.107 + 153 \cdot 0.380)/453 = 0.199$, against the logged $0.212$. This gap ($0.012$) is larger than in the weighted run ($0.003$). Each scope's HeaRT call draws its own negatives, so the scope rows are not an exact partition of the overall row. The recomposed values are approximate to about this amount.

**Link prediction, thresholded.** The three evaluations share the $453$ negatives:

| Scope | Positives | Recall | True positives | Precision | F1 | False positives |
|---|---|---|---|---|---|---|
| All | 453 | 0.815 | 369 | 0.614 | 0.700 | 232 |
| Cross partitions | 300 | 0.773 | 232 | 0.500 | 0.607 | 232 |
| Within partitions | 153 | 0.895 | 137 | 0.371 | 0.525 | 232 |

- $232 + 137 = 369$, and every scope has $232$ false positives. The specificity is $221/453 = 0.488$, and the accuracy $(369 + 221)/906 = 0.651$ matches the log.
- **This is the same operating point as the weighted triplet run** (specificity $0.483$, decoder loss $0.47$). The weighting does not change it, so it is a property shared by both triplet runs. It was attributed to the unit output scale; [Aligner start, training and output scale on AstroPh at $P = 1$ analysis](2026-11-08-aligner-start-astroph-p1.md#aligner-start-training-and-output-scale-on-astroph-at-p--1-analysis) later shows that the triplet training causes it. Full F1 is $0.700$, against $0.691$ (triplet weighted), $0.679$ and $0.684$ (contrastive).

## Effect of the weighting on the triplet loss

| Metric | Triplet, uniform | Triplet, weighted | Difference |
|---|---|---|---|
| Cross share of the objective | 0.214 | 0.557 | $\times 2.6$ |
| Cut hinge at epoch 100 | $\leq 0.059$ | 0.051 | at most slightly worse |
| Micro recall / macro F1 | 0.518 / 0.671 | 0.438 / 0.596 | $+0.080$ / $+0.075$ |
| HeaRT MRR all / cross / within | 0.212 / 0.107 / 0.380 | 0.171 / 0.101 / 0.273 | $+0.041$ / $+0.006$ / $+0.107$ |
| Mix-corrected MRR at 64% cut | 0.205 | 0.163 | $+0.042$ |
| Specificity | 0.488 | 0.483 | none |

1. **On CITESEER the $1/\delta$ weighting hurts the triplet loss**, as it hurt the contrastive loss (MRR $0.147 \to 0.124$). Without it, every within metric is better, and the cross metrics are the same within noise.
2. **The weighting buys nothing for the cut pairs here.** The uniform objective already satisfies almost all cut triplets (bound $0.059$ against $0.051$), and in neither run do cut pairs enter the reconstruction. Cross MRR differs by $0.006$.
3. **What the weighting costs is the within ordering.** It gives the within term weight $0.443$ instead of $0.786$. The within triplets that stay active are those against close negatives ([Theorem 18](2026-11-07-cross-weighted-triplet-astroph.md#theorem-18-a-triplet-with-a-correlated-negative)), and their gradient is what keeps hard within negatives away from the anchor. With less weight on them, and a stronger cut pull disturbing the within geometry, within ranking and reconstruction both get worse. This explains the within gain, but it is not proven: the uniform log has no per-scope terms, so it cannot be checked directly.
4. **Confounds:** the four runs have different LPA partitions (with $\delta$ between $0.206$ and $0.224$), and the HeaRT test mixes differ. The node2vec partition F1 values are similar, and the mix correction keeps the ordering.
5. **On AstroPh the weighting helped the contrastive loss** (MRR $+0.040$), and its effect on the triplet loss there is not yet measured. The two datasets differ in $\delta$ ($0.21$ against $0.09$), so the weighting multiplies the cut share by $2.6$ on CITESEER and by $5.9$ on AstroPh.

## Next runs

1. Run uniform triplet on AstroPh, which completes the 2×2 (loss × weighting) on both datasets.
2. Log the per-scope terms under uniform weighting as well. With `UniformObjective` this needs an extra no-grad split of the loss by scope, for logging only. It would replace the bound $T_{\text{c}} \leq L/\delta$ with the measured value and allow checking point 3 of the section above.
3. Swap the output scales (contrastive with `unit`, triplet with `replica_norm`). The two triplet runs share the low specificity, which points at the scale.
4. Repeat uniform and weighted triplet over at least 5 seeds with a fixed LPA seed, so that the $+0.042$ MRR difference is measured on matched partitions.

