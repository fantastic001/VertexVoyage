# Cross-weighted triplet loss on CITESEER analysis

Source: one run of `vv test` on CITESEER with the **triplet** aggregator (`aggregator: triplet`), logged in `VertexVoyage/vv.log` (2026-11-07, 18:22 to 18:23). The other settings are those of [Cross-weighted contrastive loss on CITESEER analysis](2026-11-07-cross-weighted-contrastive-citeseer.md#cross-weighted-contrastive-loss-on-citeseer-analysis): $P = 4$, [label propagation](../../../doc/glossary.md#label-propagation), bilinear decoder, 100 aligner epochs, validation off, and `alignment_link_weighting = inverse_cross_fraction`. The positive links are weighted by $1/\delta$, not uniformly. It is compared below with the two contrastive CITESEER runs: uniform and weighted. All three are **single runs with one seed each**, on three different unseeded LPA partitions.

## Loss and settings that differ from the contrastive runs

- **Loss:** the [triplet loss](../../../doc/glossary.md#triplet-loss) of [Losses, term by term](../../../doc/theory/partition-alignment.md#losses-term-by-term), $\max(D(a, p) - D(a, q) + \alpha, 0)$, with margin $\alpha = 0.5$ (`alignment_triplet_margin`). `triplet_loss` uses every link in both directions as (anchor, positive) and draws one uniformly random replica as the negative $q$. Hence the logged pair counts are twice the link counts: $6488 = 2 \cdot 3244$ within and $1678 = 2 \cdot 839$ cross.
- **Terms:** there is no separate attraction and repulsion. Each scope has a single `ranking` term, so the pull towards the positive and the push from the negative are inside the same term, and the logged per-scope gradient is already their net. With two terms (`ranking within`, `ranking cross`), [Theorem 11](2026-11-07-contrastive-citeseer.md#theorem-11-the-angle-between-two-term-gradients-is-read-from-three-norms) applies again and gives the angle between the within and cross gradients. That angle is a different quantity from the attraction/repulsion angle of the contrastive runs.
- **Output scale:** `TripletAlignmentAggregator` uses `UNIT_OUTPUT_SCALE`, so the merged embeddings passed to the link-prediction decoder are **unit vectors**. The contrastive aggregator uses `REPLICA_NORM_OUTPUT_SCALE`, which keeps the mean replica norm. The decoder input is therefore not on the same scale, and this confounds the link-prediction comparison.

## Setup

| Quantity | Triplet, weighted | Contrastive, weighted | Contrastive, uniform |
|---|---|---|---|
| $\lvert E_p \rvert$ | $927, 789, 760, 768$; sum $3244$ | sum $3169$ | sum $3190$ |
| $\lvert E_{\text{cut}} \rvert$, $\delta$ | $839$, $0.2055$ | $914$, $0.2239$ | $893$, $0.2187$ |
| Cross share $1/(2 - \delta)$ and gain over uniform ([Theorem 15](2026-11-07-cross-weighted-contrastive-astroph.md#theorem-15-shares-of-the-inverse-fraction-weighting)) | $0.5573$, $\times 2.71$ | $0.5630$, $\times 2.52$ | $\delta$, $\times 1$ |
| node2vec partition F1 | $0.363, 0.419, 0.388, 0.347$ | $0.351, 0.485, 0.359, 0.415$ | $0.400, 0.422, 0.418, 0.369$ |
| HeaRT test edges cut | $262/453 = 57.8\%$ | $290/453 = 64.0\%$ | $289/453 = 63.8\%$ |
| Decoder train / validation loss | $0.470$ / $0.461$ | $0.207$ / $0.246$ | $0.170$ / $0.199$ |

The logged `weight 0.557254` equals $1/(2 - 0.205486)$, as Theorem 15 (b) predicts. All partitions have $816$ nodes, and the $9788$ vocabulary warnings are again one per partition for each node outside it.

## Theorem 17 (a triplet with an orthogonal negative is a cosine threshold)

A uniformly random unit vector in $\mathbb{R}^d$ has expected inner product $0$ with any fixed anchor, so "orthogonal negative" is the typical case for the uniformly sampled $q$ (Theorem 2). This theorem says what a zero triplet term then requires of the positive pair.

**Assumptions:** $H$ is a real [inner product space](../../../doc/glossary.md#inner-product-space); $a, p, q \in H$ with $\lVert a \rVert = \lVert p \rVert = \lVert q \rVert = 1$ and $\langle a, q \rangle = 0$; and $\alpha \in \mathbb{R}$ with $0 \leq \alpha \leq \sqrt 2$.

**Statement:**

- (a) $\lVert a - q \rVert = \sqrt 2$.
- (b) $$ \max\big(\lVert a - p \rVert - \lVert a - q \rVert + \alpha, 0\big) = 0 \iff \langle a, p \rangle \geq 1 - \frac{(\sqrt 2 - \alpha)^2}{2} $$

**Proof:**

1. (a) By [Theorem 5](../../../doc/theory/partition-alignment.md#theorem-5-geometry-of-the-unit-sphere) (a), $\lVert a - q \rVert^2 = 2 - 2\langle a, q \rangle = 2$. A norm is non-negative, so $\lVert a - q \rVert = \sqrt 2$.
2. (b) By (a), the maximum is $0$ exactly when $\lVert a - p \rVert - \sqrt 2 + \alpha \leq 0$, that is $\lVert a - p \rVert \leq \sqrt 2 - \alpha$.
3. Both sides of that inequality are non-negative: the left because it is a norm, the right because $\alpha \leq \sqrt 2$. Squaring is increasing on non-negative numbers, so it is equivalent to $\lVert a - p \rVert^2 \leq (\sqrt 2 - \alpha)^2$.
4. By Theorem 5 (a), $\lVert a - p \rVert^2 = 2 - 2 \langle a, p \rangle$, so the condition is $2 - 2\langle a, p \rangle \leq (\sqrt 2 - \alpha)^2$, which rearranges to $\langle a, p \rangle \geq 1 - (\sqrt 2 - \alpha)^2 / 2$. $\blacksquare$

Lean: `norm_sub_of_orthogonal_unit`, `triplet_hinge_zero_iff_of_orthogonal_negative` in section `TripletOrthogonalNegative`.

**Consequence:** with $\alpha = 0.5$, a triplet against a typical negative is satisfied as soon as $\cos(a, p) \geq 1 - 0.9142^2/2 = 0.582$ (distance $\leq 0.914$). The contrastive attraction keeps pulling until $\cos = 1$; the triplet loss stops pulling a pair at about $0.58$ against typical negatives, and keeps pulling only against negatives that happen to lie close to the anchor.

**Example:** $a = (1, 0, 0)$, $q = (0, 0, 1)$, $p = (0.6, 0.8, 0)$. Then $\langle a, p \rangle = 0.6 \geq 0.582$, $\lVert a - p \rVert = \sqrt{0.16 + 0.64} = 0.894 \leq 0.914$, and the hinge is $\max(0.894 - 1.414 + 0.5, 0) = \max(-0.020, 0) = 0$. With $p = (0.5, 0.866, 0)$, $\langle a, p \rangle = 0.5 < 0.582$, $\lVert a - p \rVert = 1$, and the hinge is $1 - 1.414 + 0.5 = 0.086 > 0$.

## Training dynamics

$T_{\text{w}}$ and $T_{\text{c}}$ are the unweighted scope means of the triplet hinge. $L$ is the weighted loss $0.4427\, T_{\text{w}} + 0.5573\, T_{\text{c}}$. The gradient norms are of the weighted terms. $\cos(g_{\text{w}}, g_{\text{c}})$ is from Theorem 11 (b), and $\rho = \lVert g \rVert / (\lVert g_{\text{w}} \rVert + \lVert g_{\text{c}} \rVert)$.

| Epoch | $L$ | $T_{\text{w}}$ | $T_{\text{c}}$ | $\lVert g_{\text{w}} \rVert$ | $\lVert g_{\text{c}} \rVert$ | $\lVert g \rVert$ | $\cos(g_{\text{w}}, g_{\text{c}})$ | $\rho$ | $\lVert g_{\text{c}} \rVert / \lVert g_{\text{w}} \rVert$ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.3336 | 0.0077 | 0.5926 | 0.00330 | 0.0656 | 0.0656 | -0.042 | 0.951 | 19.9 |
| 2 | 0.2941 | 0.0082 | 0.5213 | 0.00466 | 0.0511 | 0.0499 | -0.303 | 0.895 | 11.0 |
| 3 | 0.2736 | 0.0095 | 0.4834 | 0.00968 | 0.0394 | 0.0399 | -0.066 | 0.814 | 4.1 |
| 4 | 0.2624 | 0.0119 | 0.4615 | 0.01242 | 0.0503 | 0.0515 | -0.024 | 0.821 | 4.0 |
| 5 | 0.2514 | 0.0098 | 0.4434 | 0.01145 | 0.0556 | 0.0576 | 0.075 | 0.859 | 4.9 |
| 6 | 0.2363 | 0.0092 | 0.4167 | 0.00786 | 0.0490 | 0.0519 | 0.294 | 0.912 | 6.2 |
| 8 | 0.2083 | 0.0076 | 0.3678 | 0.00334 | 0.0564 | 0.0569 | 0.104 | 0.951 | 16.9 |
| 10 | 0.1769 | 0.0075 | 0.3114 | 0.00290 | 0.0485 | 0.0485 | -0.037 | 0.943 | 16.7 |
| 15 | 0.1361 | 0.0082 | 0.2376 | 0.00430 | 0.0537 | 0.0546 | 0.154 | 0.940 | 12.5 |
| 20 | 0.1096 | 0.0075 | 0.1907 | 0.00309 | 0.0449 | 0.0454 | 0.123 | 0.946 | 14.5 |
| 30 | 0.0826 | 0.0065 | 0.1430 | 0.00306 | 0.0532 | 0.0538 | 0.168 | 0.956 | 17.4 |
| 40 | 0.0683 | 0.0063 | 0.1177 | 0.00259 | 0.0406 | 0.0411 | 0.160 | 0.952 | 15.7 |
| 50 | 0.0545 | 0.0058 | 0.0932 | 0.00279 | 0.0465 | 0.0461 | -0.162 | 0.936 | 16.7 |
| 60 | 0.0512 | 0.0064 | 0.0868 | 0.00202 | 0.0607 | 0.0607 | -0.005 | 0.968 | 30.0 |
| 70 | 0.0431 | 0.0062 | 0.0725 | 0.00282 | 0.0522 | 0.0520 | -0.104 | 0.945 | 18.5 |
| 80 | 0.0398 | 0.0059 | 0.0667 | 0.00365 | 0.0573 | 0.0573 | -0.051 | 0.939 | 15.7 |
| 90 | 0.0310 | 0.0057 | 0.0511 | 0.00363 | 0.0636 | 0.0641 | 0.102 | 0.953 | 17.5 |
| 100 | 0.0314 | 0.0069 | 0.0508 | 0.00337 | 0.0442 | 0.0436 | -0.220 | 0.916 | 13.1 |

Summary over all 100 epochs:

- **The cut triplets are almost all satisfied by the end.** $T_{\text{c}}$ falls from $0.593$ to $0.051$, a drop of $91\%$. It first goes below $0.1$ at epoch 47 and has its minimum, $0.0495$, at epoch 96. By Theorem 17, a cut triplet against a typical negative is satisfied once the cut pair has cosine $\geq 0.582$. The log does not give the cut-pair cosine itself, since the triplet hinge is not a function of the positive distance alone.
- **The within triplets are satisfied from the start.** $T_{\text{w}}$ stays between $0.0045$ (epoch 83) and $0.0119$ (epoch 4). CITESEER's within neighbours start near cosine $0.975$ (see the contrastive weighted run), far above the threshold of Theorem 17. The small rise at epochs 3 to 5 is the disturbance from the cut pull.
- **No tug-of-war.** The within and cross gradients are close to orthogonal: $\cos(g_{\text{w}}, g_{\text{c}}) \in [-0.303, 0.315]$ over all epochs (minimum at epoch 2, maximum at epoch 63), and $\rho \in [0.814, 0.982]$. Between scopes, almost nothing cancels. The contrastive runs reached $\rho \approx 0.01$ to $0.1$ between attraction and repulsion. That is a different pair of terms, but it shows that the triplet objective does not end in the frustrated standoff of [Theorem 6](../../../doc/theory/partition-alignment.md#theorem-6-collapse-and-frustration-of-the-contrastive-loss). A satisfied triplet contributes zero gradient, whereas a contrastive positive keeps pulling until it coincides.
- **The cross term drives the training.** $\lVert g_{\text{c}} \rVert / \lVert g_{\text{w}} \rVert$ is between $4.0$ (epoch 4) and $37.6$ (epoch 38).
- **No collapse.** There is no repulsion term to spike. The total norm stays between $0.035$ (epoch 81) and $0.066$ (epoch 1), with median $0.049$ over epochs 51 to 100. It does not decay with the loss because the hinge gradient of an active triplet has fixed magnitude. Only the number of active triplets shrinks. No vanishing-gradient warning fired.
- **Near a noisy plateau.** Over epochs 86 to 100, $L$ moves within $[0.0301, 0.0354]$, with its minimum $0.0301$ at epoch 96. The noise comes from the one random negative per triplet, resampled every epoch.

## Downstream metrics

**Graph reconstruction ($k = 4083$ closest pairs).**

| Quantity | Triplet, weighted | Contrastive, weighted | Contrastive, uniform |
|---|---|---|---|
| Reconstructed cut pairs / true | **0 / 0** | 31 / 3 | 0 / 0 |
| Within true positives | 1789 of 3244 | 1593 of 3169 | 1722 of 3190 |
| Micro recall (Theorem 14) | **0.438** | 0.391 | 0.422 |
| Within recall $r_w$ | 0.551 | 0.503 | 0.540 |
| Within pair F1 | **0.488** | 0.441 | 0.474 |
| Mean partition pair F1 | **0.492** | 0.448 | 0.480 |
| Global macro F1 | **0.596** | 0.536 | 0.544 |
| Macro within recall | 0.726 | 0.624 | 0.631 |

- **No cut pair is reconstructed.** By [Theorem 13](2026-11-07-contrastive-citeseer.md#theorem-13-no-cross-partition-reconstruction-caps-recall-at-1---delta) the micro recall is capped at $1 - \delta = 0.795$, and the observed $0.438$ is all within. This is consistent with Theorem 17: the triplet loss is content with cut pairs at moderate cosine, so they never compete with the within pairs near $0.975$ for the $k$ closest places.
- Nevertheless, **this is the best CITESEER reconstruction so far** on every within metric. The triplet loss leaves within geometry almost untouched ($T_{\text{w}}$ is near zero throughout and its gradient is small), while the contrastive weighted run squeezed within pairs to cosine $0.995$.
- Per partition:

| Partition | True edges | Reconstructed | Reconstructed / true | Recall | Pair F1 |
|---|---|---|---|---|---|
| 0 | 927 | 831 | 0.90 | 0.448 | 0.472 |
| 1 | 789 | 1264 | 1.60 | 0.608 | 0.468 |
| 2 | 760 | 807 | 1.06 | 0.574 | 0.556 |
| 3 | 768 | 1181 | 1.54 | 0.596 | 0.470 |

**Link prediction, HeaRT (500 negatives per positive, all 453 test edges).**

| Edges | Triplet, weighted: MRR / Hits@1 / Hits@3 / Hits@10 / mean rank | Contrastive, weighted: MRR | Contrastive, uniform: MRR |
|---|---|---|---|
| All (453) | **0.171** / 0.080 / 0.190 / 0.358 / 112.2 | 0.124 | 0.147 |
| Cross-partition (262) | **0.101** / 0.031 / 0.122 / 0.221 / 142.3 | 0.063 | 0.072 |
| Within-partition (191) | 0.273 / 0.157 / 0.298 / 0.545 / 71.1 | 0.215 | **0.291** |
| Earlier $P = 4$ baseline, 10 runs | 0.138 ± 0.007 / 0.081 ± 0.010 / 0.155 ± 0.010 / 0.245 ± 0.013 | | |

- **Best overall and cross-partition ranking of all CITESEER runs.** Cross MRR is $0.101$, $7.4\times$ chance ($0.0136$) and $1.4\times$ the best contrastive run. The within MRR, $0.273$, is between the two contrastive runs.
- Against the 10-run baseline (most likely no alignment): MRR $+4.7$, Hits@3 $+3.5$ and Hits@10 $+8.7$ standard deviations, with Hits@1 at the baseline ($-0.2$). This is the first CITESEER alignment run that is clearly above the baseline on MRR. The uniform contrastive run was $+1.3$, the weighted contrastive run $-2.0$.
- **The mix of test edges differs.** Here $57.8\%$ of the test edges are cut, against $64\%$ in the contrastive runs. The overall MRR over a disjoint union of scopes is the count-weighted mean of the scope MRRs; this is Theorem 14 (a) with reciprocal ranks in place of true positives. It holds only approximately here, because each scope's HeaRT call draws its own negatives: $(262 \cdot 0.101 + 191 \cdot 0.273)/453 = 0.174$ against the logged $0.171$. Recomposed at a common $64\%$ cut share, the triplet run gives $0.64 \cdot 0.101 + 0.36 \cdot 0.273 = 0.163$, against $0.151$ for the uniform contrastive run and $0.118$ for the weighted one. So the lead holds after correcting the mix, but shrinks.
- **Noise:** with the baseline's run-to-run standard deviation of $0.007$, a single-run difference has standard deviation about $0.010$. The mix-corrected lead over the uniform contrastive run, $+0.012$, is about $1.2$ of those; the lead over the weighted contrastive run, $+0.045$, is about $4.5$.

**Link prediction, thresholded.** The three evaluations share the $453$ negatives, as in the contrastive weighted section:

| Scope | Positives | Recall | True positives | Precision | False positives |
|---|---|---|---|---|---|
| All | 453 | 0.801 | 363 | 0.608 | 234 |
| Cross partitions | 262 | 0.763 | 200 | 0.461 | 234 |
| Within partitions | 191 | 0.853 | 163 | 0.411 | 234 |

- $200 + 163 = 363$, and each scope's false-positive count, $\text{TP}/\text{precision} - \text{TP}$, is $234$. The specificity is therefore $219/453 = 0.483$, against $0.857$ for the contrastive weighted run. Accuracy, $(363 + 219)/906 = 0.642$, matches the log.
- The decoder sits at a very different operating point: high recall, low specificity. This fits the much higher decoder loss ($0.470$ against $0.207$). A bilinear logit $x^\top W y$ of unit vectors is bounded by the operator norm of $W$, so the decoder cannot separate the classes as sharply as with replica-norm-scaled inputs. This is a hypothesis; the output scale was not varied. **Refuted later:** in [Aligner start, training and output scale on AstroPh at $P = 1$ analysis](2026-11-08-aligner-start-astroph-p1.md#aligner-start-training-and-output-scale-on-astroph-at-p--1-analysis), unit vectors without training keep specificity $0.965$, and only triplet training brings it down to $0.52$.
- The full F1 is $0.691$, against $0.679$ (contrastive weighted) and $0.684$ (contrastive uniform). Because the operating point is different, the cross/within recall gap ($0.763$ against $0.853$) is not comparable with the contrastive weighted gap ($0.472$ against $0.791$). Ranking metrics such as HeaRT are the fair comparison.

## Reading

1. **On CITESEER, weighted triplet alignment beats weighted contrastive alignment on every downstream metric**: reconstruction micro recall $0.438$ against $0.391$, macro F1 $0.596$ against $0.536$, and MRR $0.171$ against $0.124$. It is also at least as good as uniform contrastive on all of them except within-partition MRR.
2. **The mechanism is consistent with the theory of this page.** The triplet hinge stops pulling once a pair is ordered by the margin (Theorem 17), so the near-coincident within pairs exert no force and keep their node2vec geometry. Cut pairs are pulled only until they beat a typical random negative. There is no collapse and no attraction/repulsion standoff. The contrastive loss pulls every positive towards coincidence and gets the frustrated standoff of Theorem 6.
3. **The same property explains the zero cut pairs in reconstruction.** A loss that orders distances does not make cut pairs as close as within pairs, which is what top-$k$ reconstruction would need. The synthetic check showed the same thing: triplet alignment left replica cosines at $0.19$ to $0.34$ while matching the AUC of contrastive alignment.
4. **The link-prediction gain may partly come from the unit output scale**, not from the loss: the triplet aggregator feeds unit vectors to the decoder. The run cannot separate the two. On AstroPh at $P = 1$ the matched runs of [Aligner start, training and output scale on AstroPh at $P = 1$ analysis](2026-11-08-aligner-start-astroph-p1.md#aligner-start-training-and-output-scale-on-astroph-at-p--1-analysis) show that unit normalisation alone lowers the MRR by $0.017$, so there it is not a source of gain.
5. **The 1/δ weighting looks harmless here**, unlike in the contrastive run. It increases the share of the cut triplets ($\times 2.71$), and because satisfied triplets have no gradient, the within geometry is not squeezed. There is no uniform triplet run to measure the weighting's own effect.

## Next runs

1. Run the triplet aggregator with `alignment_link_weighting=uniform` on CITESEER, so that the effect of the weighting on the triplet loss is isolated.
2. Run the contrastive aggregator with `alignment_output_scale=unit`, and the triplet aggregator with `replica_norm`, to separate the effect of the loss from the effect of the output scale on link prediction.
3. Repeat the triplet run on AstroPh with the same settings, to check whether its advantage over contrastive holds when $\delta$ is small and the partitions are unbalanced.
4. Log the mean cosine of cut pairs for the triplet loss too (one extra no-grad pass over the positive distances), so that Theorem 17's threshold can be checked against the data.
5. Repeat each configuration over at least 5 seeds with a fixed LPA seed. All comparisons in this page so far are single-run.

