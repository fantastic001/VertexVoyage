# Cross-weighted contrastive loss on AstroPh analysis

Source: one run of `vv test` on AstroPh, logged in `VertexVoyage/vv.log` (2026-11-07, 16:30 to 16:43). The settings are those of [Contrastive loss on AstroPh analysis](2026-11-07-contrastive-astroph.md#contrastive-loss-on-astroph-analysis), including `VERTEX_VOYAGE_ALIGNMENT_LINK_MEMORY=200000`, with one change: `alignment_link_weighting = inverse_cross_fraction` (the new default). Cross-partition links are weighted by $1/\delta$, as proposed in the AstroPh section's next runs (item 2). The question is whether this improves on the uniform run of that section. Both are **single runs with one seed each**, and label propagation is unseeded, so the two runs also have different partitions.

## Weighted objective

Let $E^+ = E^+_{\text{within}} \sqcup E^+_{\text{cross}}$ be the positive links, with $n = \lvert E^+ \rvert$, $c = \lvert E^+_{\text{cross}} \rvert$, $0 < c < n$ and $\delta = c / n$. With $RF = 1$, as here, $E^+_{\text{cross}} = E^+_{\text{cut}}$. Write $S_{\text{w}} = \sum_{E^+_{\text{within}}} D^2$ and $S_{\text{c}} = \sum_{E^+_{\text{cross}}} D^2$. The weighted attraction gives every within link weight $1$ and every cross link weight $1/\delta$, and divides by the total weight:

$$ A_\delta = \frac{S_{\text{w}} + S_{\text{c}} / \delta}{(n - c) + c / \delta} $$

`InverseCrossFractionObjective` in `vertex_voyage/aggregation/alignment_objective.py` computes this. It calls the loss once per scope and multiplies the scope's terms by the scope share (Theorem 15). The repulsion of each call is a mean over uniformly sampled pairs, so both scope repulsions estimate the same expectation, and their share-weighted sum still estimates it.

## Theorem 15 (shares of the inverse-fraction weighting)

**Assumptions:** $n, c \in \mathbb{R}$ with $0 < c < n$, $\delta = c / n$, and $S_{\text{w}}, S_{\text{c}} \in \mathbb{R}$.

**Statement:**

- (a) The total weight is $(n - c) + c / \delta = 2n - c$.
- (b) The cross scope has share $\dfrac{c / \delta}{2n - c} = \dfrac{1}{2 - \delta}$ and the within scope has share $\dfrac{n - c}{2n - c} = \dfrac{1 - \delta}{2 - \delta}$.
- (c) $\dfrac{1}{2 - \delta} \geq \dfrac12$: the cross links carry at least half of the weight, however few of them there are.
- (d) $$ A_\delta = \frac{1 - \delta}{2 - \delta} \cdot \frac{S_{\text{w}}}{n - c} + \frac{1}{2 - \delta} \cdot \frac{S_{\text{c}}}{c} $$ The weighted attraction is the share-weighted mix of the two scope means.

**Proof:**

1. (a) $c / \delta = c / (c/n) = n$ because $c \neq 0$ and $n \neq 0$ (both are positive). Hence the total is $n - c + n = 2n - c$, which is positive because $c < n$.
2. (b) The cross share is $n / (2n - c)$. Divide numerator and denominator by $n > 0$: $1 / (2 - c/n) = 1/(2 - \delta)$. The within share is $(n - c)/(2n - c)$; divide by $n$: $(1 - \delta)/(2 - \delta)$.
3. (c) $0 < \delta < 1$ because $0 < c < n$. So $0 < 2 - \delta \leq 2$, and taking reciprocals of positive numbers reverses the order: $1/(2 - \delta) \geq 1/2$.
4. (d) By (a) the denominator of $A_\delta$ is $2n - c$, and $S_{\text{c}} / \delta = n S_{\text{c}} / c$. The right side equals $\frac{n - c}{2n - c} \cdot \frac{S_{\text{w}}}{n - c} + \frac{n}{2n - c} \cdot \frac{S_{\text{c}}}{c} = \frac{S_{\text{w}} + n S_{\text{c}}/c}{2n - c}$, using $n - c > 0$ and $c > 0$ to cancel. This is the left side. $\blacksquare$

Lean: `total_weight_of_inverse_fraction`, `cross_share_of_inverse_fraction`, `within_share_of_inverse_fraction`, `half_le_cross_share`, `weighted_mean_of_scopes` in section `CrossWeighting`.

**Consequences:**

- Uniform weighting gives the cross scope share $\delta$. The inverse-fraction weighting gives it $1/(2 - \delta)$, so the factor by which the cut pairs gain weight is $1/(\delta(2 - \delta))$.
- The scope means $S_{\text{w}}/(n - c)$ and $S_{\text{c}}/c$ are mean squared distances on the unit sphere. By [Theorem 12](2026-11-07-contrastive-citeseer.md#theorem-12-mean-positive-cosine-from-the-attraction-term), $1 - \tfrac12 S_{\text{c}}/c$ is the mean cosine of the cut pairs. This is the direct measure of alignment progress that the earlier runs could not log.

**Example (this run):** $n = 178299$ and $c = 14785$, so $\delta = 0.082923$. The cross share is $1/1.917077 = 0.521627$ and the within share is $0.917077/1.917077 = 0.478373$. These match the logged `weight 0.521627` and `weight 0.478373`. Under uniform weighting the cut pairs would have share $0.0829$, so they gain a factor of $0.5216/0.0829 = 6.3$.

## Theorem 16 (cancellation ratio of several terms)

**Assumptions:** $H$ is a [normed space](../../../doc/glossary.md#normed-space), $s$ is a finite index set, $g_i \in H$ for $i \in s$, and $\sum_{i \in s} \lVert g_i \rVert > 0$.

**Statement:** $\rho = \dfrac{\lVert \sum_{i \in s} g_i \rVert}{\sum_{i \in s} \lVert g_i \rVert} \leq 1$.

**Proof:** the triangle inequality, extended to finite sums by induction on $s$, gives $\lVert \sum g_i \rVert \leq \sum \lVert g_i \rVert$. Divide by the positive denominator. $\blacksquare$

Lean: `cancellation_ratio_of_terms_le_one`.

**Consequence:** with four terms (attraction and repulsion, within and cross) the log no longer gives the angle between attraction and repulsion: Theorem 11 (b) needs the norm of each side, and only the four term norms and the total are logged. $\rho$ remains a scale-free measure of how much of the gradient survives the summation. For the same gradients, a four-term $\rho$ is at most the two-term $\rho$, because its denominator is at least as large. So the $\rho$ values below are not directly comparable with the uniform run.

## Setup

| Quantity | This run (weighted) | Uniform run | Source line |
|---|---|---|---|
| $\lvert V_p \rvert$ | $12429, 2115, 2114, 2114$ | $12216, 2186, 2185, 2185$ | `Partition size` |
| $\lvert E_p \rvert$ | $131402, 10464, 9868, 11780$; sum $163514$ | sum $162074$ | `Partition number of edges` |
| $\lvert E_{\text{cut}} \rvert$, $\delta$ | $14785$, $0.0829$ | $16225$, $0.0910$ | `cross-partition fraction 0.082923` |
| Share of nodes / within edges in partition 0 | $66.2\%$ / $80.4\%$ | $65.1\%$ / $79.0\%$ | derived |
| node2vec partition F1 | $0.676, 0.607, 0.561, 0.610$ | $0.675, 0.535, 0.669, 0.565$ | `Best achieved F1 score` |
| Links trained, replica rows | $178299$, $18643$ | $178299$, $18627$ | `Aligner trained on` |
| Scope shares | within $0.4784$, cross $0.5216$ | within $0.909$, cross $0.091$ (implicit) | term weights |

$\lvert E \rvert = 178299$, $\lvert E_{\text{test}} \rvert = 19811$ and every other setting are the same in both runs. The $56312$ vocabulary warnings are again one per partition for each node outside it: $(18772 - 12430) + (18772 - 2116) + 2 \times (18772 - 2115) = 56312$.

## Training dynamics

All gradient norms are norms of the **weighted** terms, $\nabla_\theta (w_s \cdot \text{mean}_s)$. $\bar c_{\text{w}} = 1 - A_{\text{w}}/2$ and $\bar c_{\text{c}} = 1 - A_{\text{c}}/2$ are the mean cosines of within and cut pairs (Theorem 12), computed from the unweighted scope means $A_{\text{w}}$ and $A_{\text{c}}$. $L$ is the weighted loss, so it is **not** comparable with the $L$ of the uniform run.

| Epoch | $L$ | $A_{\text{w}}$ | $A_{\text{c}}$ | $R_{\text{w}}$ | $R_{\text{c}}$ | $\lVert g_{A,\text{w}} \rVert$ | $\lVert g_{A,\text{c}} \rVert$ | $\lVert g_{R,\text{w}} \rVert$ | $\lVert g_{R,\text{c}} \rVert$ | $\lVert g \rVert$ | $\rho$ | $\bar c_{\text{w}}$ | $\bar c_{\text{c}}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.1179 | 0.3579 | 1.8118 | 0.0017 | 0.0016 | 0.2437 | 0.9147 | 0.0026 | 0.0028 | 1.0860 | 0.933 | 0.821 | 0.094 |
| 2 | 0.6950 | 0.2276 | 1.1029 | 0.0108 | 0.0109 | 0.1821 | 0.7792 | 0.0368 | 0.0409 | 0.8593 | 0.827 | 0.886 | 0.449 |
| 3 | 0.4342 | 0.1204 | 0.5424 | 0.0939 | 0.0935 | 0.1022 | 0.4113 | 0.1436 | 0.1566 | 0.2238 | 0.275 | 0.940 | 0.729 |
| 4 | 0.4068 | 0.0671 | 0.2617 | 0.2385 | 0.2380 | 0.0557 | 0.1902 | 0.1709 | 0.1861 | 0.1418 | 0.235 | 0.966 | 0.869 |
| 5 | 0.4384 | 0.0486 | 0.1595 | 0.3318 | 0.3320 | 0.0392 | 0.1119 | 0.1634 | 0.1783 | 0.2030 | 0.412 | 0.976 | 0.920 |
| 6 | 0.4544 | 0.0436 | 0.1257 | 0.3682 | 0.3679 | 0.0350 | 0.0891 | 0.1595 | 0.1739 | 0.2186 | 0.478 | 0.978 | 0.937 |
| 8 | 0.4327 | 0.0507 | 0.1321 | 0.3400 | 0.3392 | 0.0418 | 0.1049 | 0.1657 | 0.1809 | 0.2141 | 0.434 | 0.975 | 0.934 |
| 10 | 0.3888 | 0.0743 | 0.1953 | 0.2500 | 0.2526 | 0.0634 | 0.1660 | 0.1663 | 0.1802 | 0.1504 | 0.261 | 0.963 | 0.902 |
| 15 | 0.3622 | 0.1575 | 0.3290 | 0.1145 | 0.1160 | 0.1233 | 0.2589 | 0.1112 | 0.1213 | 0.1389 | 0.226 | 0.921 | 0.836 |
| 20 | 0.3396 | 0.1321 | 0.2674 | 0.1377 | 0.1363 | 0.1025 | 0.1945 | 0.1272 | 0.1388 | 0.0416 | 0.074 | 0.934 | 0.866 |
| 30 | 0.3177 | 0.1169 | 0.2671 | 0.1219 | 0.1228 | 0.0883 | 0.1616 | 0.1130 | 0.1233 | 0.0246 | 0.051 | 0.942 | 0.866 |
| 40 | 0.2989 | 0.1182 | 0.2467 | 0.1121 | 0.1150 | 0.0845 | 0.1452 | 0.1054 | 0.1156 | 0.0182 | 0.040 | 0.941 | 0.877 |
| 50 | 0.2787 | 0.1104 | 0.2355 | 0.1038 | 0.1024 | 0.0772 | 0.1296 | 0.0976 | 0.1054 | 0.0146 | 0.036 | 0.945 | 0.882 |
| 60 | 0.2653 | 0.1101 | 0.2258 | 0.0947 | 0.0950 | 0.0751 | 0.1203 | 0.0895 | 0.0978 | 0.0104 | 0.027 | 0.945 | 0.887 |
| 70 | 0.2530 | 0.1089 | 0.2141 | 0.0892 | 0.0892 | 0.0730 | 0.1124 | 0.0839 | 0.0921 | 0.0083 | 0.023 | 0.946 | 0.893 |
| 80 | 0.2429 | 0.1083 | 0.2040 | 0.0850 | 0.0845 | 0.0715 | 0.1048 | 0.0792 | 0.0862 | 0.0067 | 0.020 | 0.946 | 0.898 |
| 90 | 0.2364 | 0.1081 | 0.1957 | 0.0824 | 0.0829 | 0.0700 | 0.0991 | 0.0753 | 0.0821 | 0.0061 | 0.019 | 0.946 | 0.902 |
| 100 | 0.2296 | 0.1078 | 0.1880 | 0.0801 | 0.0798 | 0.0691 | 0.0942 | 0.0727 | 0.0789 | 0.0053 | 0.017 | 0.946 | 0.906 |

Summary over all 100 epochs:

- **The cut pairs are pulled together.** Their mean cosine rises from $\bar c_{\text{c}} = 0.094$ at epoch 1, close to the $0$ expected for unaligned partitions ([Theorem 2](../../../doc/theory/partition-alignment.md#theorem-2-a-random-isometry-is-as-far-as-a-random-direction)), to $0.940$ at epoch 7 ($A_{\text{c}}$ minimum $0.1204$), and ends at $0.906$. The within pairs go from $0.821$ to $0.946$, the same final value as in the uniform run.
- **The weighting does what Theorem 15 says.** At epoch 1 the cut attraction gradient is $3.75\times$ the within one. It stays the larger of the two in every one of the 100 epochs and is still $1.36\times$ the within one at epoch 100, its smallest ratio.
- **The two repulsion estimates agree.** From epoch 3 on, $R_{\text{w}}$ and $R_{\text{c}}$ differ by at most $2.6\%$ of their value (for example $0.2385$ against $0.2380$ at epoch 4). The largest gap, $7.2\%$, occurs at epochs 1 and 2, where both are below $0.011$. This confirms empirically that both scope repulsions estimate the same expectation.
- **A deeper collapse.** $R$ peaks at $0.368$ at epoch 6, $220$ times its epoch-1 value, against a peak of $0.195$ in the uniform run. The strong pull on cut pairs contracts the embedding further before repulsion takes over.
- **The standoff persists.** From epoch 30 on, $\rho \in [0.017, 0.054]$, with the minimum at epoch 100, and the sum of attraction norms is $1.02$ to $1.10$ times the sum of repulsion norms from epoch 50. The smallest total norm is $5.3 \cdot 10^{-3}$ at epoch 100, and no vanishing-gradient warning fired.
- **Not converged.** $L$ falls by $3.9\%$ over epochs 86 to 100 ($0.2389 \to 0.2296$) and reaches its minimum at the last epoch.

## Downstream metrics

**Graph reconstruction ($k = 178299$ closest pairs).**

| Scope | Uniform run: reconstructed / true positives | Weighted run: reconstructed / true positives | Weighted precision | Weighted recall |
|---|---|---|---|---|
| Within partitions | 178 106 / 60 119 | 158 282 / 48 935 | 0.309 | 0.299 |
| Cross partitions | 193 / 6 | 20 017 / 481 | 0.024 | **0.0325** |
| All | 178 299 / 60 125 | 178 299 / 49 416 | 0.277 | 0.277 |

- **Cut recall rose $88$-fold**, from $0.00037$ to $0.0325$. The reconstruction now contains $20017$ cut pairs, $11.2\%$ of $k$, against $0.11\%$ before. Their precision, $0.024$, is $146\times$ the chance level $0.000165$, but only $2.4\%$ of them are true edges.
- **Global reconstruction got worse.** By [Theorem 14](2026-11-07-contrastive-astroph.md#theorem-14-micro-recall-splits-by-scope) the recall is $(1 - \delta) r_w + \delta r_c = 0.9171 \cdot 0.2993 + 0.0829 \cdot 0.0325 = 0.2745 + 0.0027 = 0.277$, against $0.337$ in the uniform run. The cut gain adds $0.0027$. The within loss, $r_w$ falling from $0.371$ to $0.299$, costs about $0.066$. Weighted F1 fell from $0.410$ to $0.357$, macro F1 from $0.409$ to $0.383$, and the number of nodes with at least one reconstructed neighbour from $13103$ to $12097$.
- **Per partition:**

| Partition | True edges | Reconstructed | Reconstructed / true | Recall | Pair F1 | Uniform run: reconstructed / true, recall |
|---|---|---|---|---|---|---|
| 0 | 131 402 | 43 483 | 0.33 | 0.151 | 0.227 | 0.71, 0.238 |
| 1 | 10 464 | 41 460 | 3.96 | 0.899 | 0.362 | 3.34, 0.885 |
| 2 | 9 868 | 40 826 | 4.14 | 0.911 | 0.355 | 1.68, 0.847 |
| 3 | 11 780 | 32 513 | 2.76 | 0.908 | 0.483 | 2.68, 0.879 |

The within loss is concentrated in the giant partition. Its share of the reconstructed pairs fell from $0.71$ to $0.33$ of its true edges, and its recall from $0.238$ to $0.151$. The small partitions now take $2.8$ to $4.1$ times their share. The deeper contraction at epochs 4 to 8 made the small partitions even denser relative to the giant one, so the size bias of the global top-$k$ grew.

**Link prediction (HeaRT, 500 negatives per positive, 1000 sampled positives per row, bilinear decoder).**

| Edges | Uniform run: MRR / Hits@1 / Hits@10 | Weighted run: MRR | Mean rank | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|
| All test edges | 0.343 / 0.228 / 0.578 | **0.383** | 29.9 | 0.266 | 0.418 | 0.647 |
| Cross-partition | 0.155 / 0.080 / 0.302 | **0.175** | 104.5 | 0.090 | 0.186 | 0.349 |
| Within-partition | 0.333 / 0.219 / 0.557 | **0.414** | 15.9 | 0.274 | 0.476 | 0.703 |
| Earlier $P = 4$ baseline, 10 runs | 0.452 ± 0.022 / 0.326 ± 0.023 / 0.697 ± 0.019 | | | | | |

- **Every ranking metric improved**: MRR by $+0.040$ overall, $+0.020$ cross and $+0.081$ within, and Hits@10 by $+0.069$ overall.
- The cross-partition MRR is $12.9\times$ chance ($0.0136$). The within/cross MRR ratio is $2.36$, against $2.15$ in the uniform run.
- Against the 10-run baseline (most likely no alignment), the gap narrowed from $-5.0$ to $-3.1$ standard deviations on MRR, from $-4.3$ to $-2.6$ on Hits@1, from $-5.3$ to $-3.7$ on Hits@3 and from $-6.2$ to $-2.6$ on Hits@10. The weighted run is still clearly below the baseline.
- The test edges are cut in $2012 / 19811 = 10.2\%$ of cases, against $\delta = 8.3\%$.
- The thresholded "cross partitions" and "within partitions" `Precision`/`Recall`/`F1`/`Accuracy` lines are again copies of the full-model numbers ($0.9012 / 0.9137 / 0.9074 / 0.9067$) because of the `lp_full_eval` cache key, and are not valid.

**How much of the improvement is noise.** Two sources of variation separate the runs.

- *Test sampling:* each HeaRT row ranks $1000$ sampled positives. The reciprocal rank lies in $[0, 1]$; with Hits@1 near $0.27$ its standard deviation is about $0.38$, so the standard error of one MRR is about $0.38/\sqrt{1000} = 0.012$. The difference of two independent MRRs has standard error about $0.017$, which puts the overall gain $+0.040$ at about $2.3$ standard errors and the within gain $+0.081$ at about $4.7$.
- *Training and partitioning:* the 10-run baseline has a run-to-run standard deviation of $0.022$ on MRR, so the difference of two single runs has standard deviation about $\sqrt{2} \cdot 0.022 = 0.031$. Against that, the MRR gain is $1.3$ standard deviations and the Hits@10 gain is $0.069 / (\sqrt 2 \cdot 0.019) = 2.6$.

So the link-prediction improvement is consistent across all ranking metrics and both scopes, but with one run per setting it is suggestive, not established.

## Is there an improvement?

| Criterion | Uniform | Weighted | Verdict |
|---|---|---|---|
| Cut pairs aligned in training ($\bar c_{\text{c}}$ at epoch 100) | not logged | 0.906, from 0.094 | the weighting aligns cut pairs |
| Cut pairs in reconstruction (recall) | 0.00037 | 0.0325 | improved $88\times$, still small |
| Global reconstruction (weighted F1) | 0.410 | 0.357 | **worse** |
| Within reconstruction recall | 0.371 | 0.299 | **worse**, in the giant partition |
| HeaRT MRR all / cross / within | 0.343 / 0.155 / 0.333 | 0.383 / 0.175 / 0.414 | improved on every row |
| Hits@10 all | 0.578 | 0.647 | improved |
| Gap to the no-alignment baseline (MRR) | $-5.0$ sd | $-3.1$ sd | smaller, still large |

1. **For link prediction, yes**, and consistently: every ranking metric improved in both scopes, and the MRR gap to the baseline shrank by $37\%$ (from $0.109$ to $0.069$). The size of the gain is within about two run-to-run standard deviations, so it needs repeated runs to confirm.
2. **For alignment itself, yes**: the training loss directly shows the cut pairs going from random ($\bar c_{\text{c}} = 0.094$) to nearly aligned ($0.906$), and cut pairs now enter the reconstruction.
3. **For graph reconstruction, no.** The pull on cut pairs comes with a deeper contraction that worsens the size bias of the global top-$k$ and costs the giant partition a third of its recall. The gain in cut recall is worth $0.0027$ of global recall, which by Theorem 14 is at most $\delta r_c$, and the within loss is worth $0.066$.
4. **The frustration of [Theorem 6](../../../doc/theory/partition-alignment.md#theorem-6-collapse-and-frustration-of-the-contrastive-loss) is not removed.** It is shifted: the net gradient still keeps only about $2\%$ of the term magnitudes at the end. The aligned cut cosine, $0.906$, sits below the within cosine, $0.946$, because the uniformly sampled negatives push back on both equally.
5. **Even with the weighting, contrastive alignment is still below no alignment for link prediction** on AstroPh with $P = 4$ LPA ($-3.1$ sd on MRR).

## Next runs

1. Repeat both weightings over at least 5 seeds on the same partitioning, or fix the LPA seed, so that the $+0.040$ MRR gain has error bars that are independent of the partition.
2. Run no alignment and Procrustes on the same partitioning, so that the remaining gap to the baseline is measured on matched runs.
3. Normalise the reconstruction per partition, or use a balanced partitioner, to see whether the reconstruction loss is caused only by the size bias, which the deeper contraction amplified.
4. Raise `alignment_train_steps`, since $L$ was still falling at epoch 100, and log $\bar c_{\text{c}}$ against $\bar c_{\text{w}}$ to see whether the cut cosine keeps closing the gap.
5. Add a geometry-preserving term (for example $\lVert W(u,p) - z_{u,p} \rVert^2$ on within pairs) to limit the contraction at epochs 4 to 8, which is what cost the giant partition its recall.
6. Enable validation (`VERTEX_VOYAGE_ALIGNMENT_VALIDATION=true`) and compare the validation $\bar c_{\text{c}}$ with the training one, to check that the cut alignment generalises to held-out cut edges.

