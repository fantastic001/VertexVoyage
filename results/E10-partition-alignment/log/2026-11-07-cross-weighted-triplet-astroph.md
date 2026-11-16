# Cross-weighted triplet loss on AstroPh analysis

Source: one run of `vv test` on AstroPh with the **triplet** aggregator (`aggregator: triplet`), logged in `VertexVoyage/vv.log` (2026-11-07, 18:58 to 19:11). The settings are those of [Cross-weighted contrastive loss on AstroPh analysis](2026-11-07-cross-weighted-contrastive-astroph.md#cross-weighted-contrastive-loss-on-astroph-analysis): $P = 4$, [label propagation](../../../doc/glossary.md#label-propagation), bilinear decoder, 100 aligner epochs, validation off, every edge used as a positive link, and `alignment_link_weighting = inverse_cross_fraction`. The loss, the single `ranking` term per scope, and the unit output scale are as described in [Cross-weighted triplet loss on CITESEER analysis](2026-11-07-cross-weighted-triplet-citeseer.md#loss-and-settings-that-differ-from-the-contrastive-runs) (margin $\alpha = 0.5$). It is compared with the two contrastive AstroPh runs, uniform ([Contrastive loss on AstroPh analysis](2026-11-07-contrastive-astroph.md#contrastive-loss-on-astroph-analysis)) and weighted. All three are **single runs with one seed each**, on three different unseeded LPA partitions. This is item 3 of the CITESEER triplet section's next runs.

This run is also the first AstroPh run after the `lp_full_eval` cache fix, so its thresholded cross and within lines are valid. The contrastive AstroPh runs have only valid full-model thresholded numbers.

## Setup

| Quantity | Triplet, weighted | Contrastive, weighted | Contrastive, uniform |
|---|---|---|---|
| $\lvert V_p \rvert$ | $12277, 2165, 2165, 2165$ | $12429, 2115, 2114, 2114$ | $12216, 2186, 2185, 2185$ |
| $\lvert E_p \rvert$ | $129227, 10814, 11234, 11304$; sum $162579$ | sum $163514$ | sum $162074$ |
| $\lvert E_{\text{cut}} \rvert$, $\delta$ | $15720$, $0.0882$ | $14785$, $0.0829$ | $16225$, $0.0910$ |
| Share of nodes / within edges in partition 0 | $65.4\%$ / $79.5\%$ | $66.2\%$ / $80.4\%$ | $65.1\%$ / $79.0\%$ |
| node2vec partition F1 | $0.676, 0.561, 0.639, 0.581$ | $0.676, 0.607, 0.561, 0.610$ | $0.675, 0.535, 0.669, 0.565$ |
| Replica rows | $18608$ | $18643$ | $18627$ |
| Cross share $1/(2 - \delta)$ and gain over uniform ([Theorem 15](2026-11-07-cross-weighted-contrastive-astroph.md#theorem-15-shares-of-the-inverse-fraction-weighting)) | $0.5231$, $\times 5.93$ | $0.5216$, $\times 6.29$ | $\delta$, $\times 1$ |
| HeaRT test edges cut | $2170 / 19811 = 10.95\%$ | $2012 / 19811 = 10.16\%$ | $2206 / 19811 = 11.14\%$ |

- The logged `weight 0.523058` equals $1/(2 - 0.088167)$, as Theorem 15 (b) predicts.
- The pair counts are twice the link counts, because `triplet_loss` uses every link in both directions: $325158 = 2 \cdot 162579$ within and $31440 = 2 \cdot 15720$ cross.
- The $56312$ vocabulary warnings are again one per partition for each node outside it: $(18772 - 12278) + 3 \cdot (18772 - 2166) = 56312$.
- The three runs differ in their partitions by at most $1.5\%$ in $\delta$ and $1.6$ points in the size of partition 0. The cut share of the test edges differs by less than one point, so, unlike CITESEER, the HeaRT rows need no mix correction.

## Theorem 18 (a triplet with a correlated negative)

[Theorem 17](2026-11-07-cross-weighted-triplet-citeseer.md#theorem-17-a-triplet-with-an-orthogonal-negative-is-a-cosine-threshold) assumed that the negative is orthogonal to the anchor. A negative drawn from the same region of the embedding, for example from the same partition, can have a positive cosine with the anchor. This theorem drops the orthogonality assumption.

**Assumptions:** $H$ is a real [inner product space](../../../doc/glossary.md#inner-product-space); $a, p, q \in H$ with $\lVert a \rVert = \lVert p \rVert = \lVert q \rVert = 1$; $\kappa = \langle a, q \rangle$ is the cosine of the negative to the anchor; and $\alpha \in \mathbb{R}$ is the margin. Write $h = \max(\lVert a - p \rVert - \lVert a - q \rVert + \alpha, 0)$ for the triplet hinge.

**Statement:**

- (a) $\lVert a - q \rVert = \sqrt{2 - 2\kappa}$.
- (b) If $\alpha \leq \lVert a - q \rVert$, then $$ h = 0 \iff \langle a, p \rangle \geq 1 - \frac{(\lVert a - q \rVert - \alpha)^2}{2} = 1 - \frac{(\sqrt{2 - 2\kappa} - \alpha)^2}{2}. $$
- (c) If $\lVert a - q \rVert < \alpha$, then $h > 0$ for **every** $p$.
- (d) If $\alpha \geq 0$ and $\kappa > 1 - \alpha^2 / 2$, then $\lVert a - q \rVert < \alpha$, so by (c) $h > 0$ for every $p$.

**Proof:**

1. (a) By [Theorem 5](../../../doc/theory/partition-alignment.md#theorem-5-geometry-of-the-unit-sphere) (a), $\lVert a - q \rVert^2 = 2 - 2\kappa$. A norm is non-negative, and the square root of the square of a non-negative number is that number, so $\lVert a - q \rVert = \sqrt{2 - 2\kappa}$.
2. (b), "$\Rightarrow$": $h = 0$ and $h \geq \lVert a - p \rVert - \lVert a - q \rVert + \alpha$ give $\lVert a - p \rVert \leq \lVert a - q \rVert - \alpha$. Both sides are non-negative, the right one by the assumption $\alpha \leq \lVert a - q \rVert$, so squaring keeps the order: $\lVert a - p \rVert^2 \leq (\lVert a - q \rVert - \alpha)^2$. By Theorem 5 (a), $\lVert a - p \rVert^2 = 2 - 2\langle a, p \rangle$. Substituting and rearranging gives $\langle a, p \rangle \geq 1 - (\lVert a - q \rVert - \alpha)^2 / 2$.
3. (b), "$\Leftarrow$": the same substitution turns the assumption into $\lVert a - p \rVert^2 \leq (\lVert a - q \rVert - \alpha)^2$. On non-negative numbers, $x^2 \leq y^2$ implies $x \leq y$, so $\lVert a - p \rVert \leq \lVert a - q \rVert - \alpha$. Hence $\lVert a - p \rVert - \lVert a - q \rVert + \alpha \leq 0$, and the maximum with $0$ is $0$.
4. (c) $\lVert a - p \rVert \geq 0$, so $\lVert a - p \rVert - \lVert a - q \rVert + \alpha \geq \alpha - \lVert a - q \rVert > 0$. The maximum is at least this first argument, so $h > 0$.
5. (d) By Theorem 5 (a), $\lVert a - q \rVert^2 = 2 - 2\kappa < 2 - 2(1 - \alpha^2/2) = \alpha^2$. Both $\lVert a - q \rVert$ and $\alpha$ are non-negative, so $\lVert a - q \rVert < \alpha$. $\blacksquare$

Lean: `norm_sub_of_unit_eq_sqrt`, `triplet_hinge_zero_iff_of_unit`, `triplet_hinge_pos_of_close_negative`, `norm_sub_lt_of_inner_gt` in section `TripletCorrelatedNegative`.

**Consequences:**

- Theorem 17 is the case $\kappa = 0$, where $\sqrt{2 - 2\kappa} = \sqrt 2$.
- The threshold $1 - (\sqrt{2 - 2\kappa} - \alpha)^2 / 2$ grows with $\kappa$: a closer negative demands a closer positive. Equivalently, a positive pair at cosine $c_p$ (distance $\sqrt{2 - 2c_p}$) is in an active triplet exactly when $\sqrt{2 - 2\kappa} < \sqrt{2 - 2 c_p} + \alpha$.
- By (d), with $\alpha = 0.5$, any negative with $\kappa > 0.875$ makes the triplet active **whatever the positive is**. Such triplets give a gradient that no alignment can remove.

**Example ($\alpha = 0.5$):**

- With $\kappa = 0.5$, $\lVert a - q \rVert = 1$, and the threshold is $1 - 0.5^2/2 = 0.875$. A positive at cosine $0.82$, about the starting within cosine on AstroPh in the weighted contrastive run, gives $\lVert a - p \rVert = \sqrt{0.36} = 0.6$ and $h = 0.6 - 1 + 0.5 = 0.1 > 0$. Against an orthogonal negative the same positive gives $h = \max(0.6 - 1.414 + 0.5, 0) = 0$.
- A positive at cosine $0.82$ is active exactly when $\sqrt{2 - 2\kappa} < 1.1$, that is $\kappa > 0.395$.
- With $\kappa = 0.9$, $\lVert a - q \rVert = \sqrt{0.2} = 0.447 < 0.5$. Even $p = a$ gives $h = 0 - 0.447 + 0.5 = 0.053 > 0$.

## Training dynamics

The columns are as in the CITESEER triplet section: $T_{\text{w}}$ and $T_{\text{c}}$ are the unweighted scope means of the hinge, $L = 0.4769\, T_{\text{w}} + 0.5231\, T_{\text{c}}$, the gradient norms are those of the weighted terms, $\cos(g_{\text{w}}, g_{\text{c}})$ is from [Theorem 11](2026-11-07-contrastive-citeseer.md#theorem-11-the-angle-between-two-term-gradients-is-read-from-three-norms) (b), and $\rho = \lVert g \rVert / (\lVert g_{\text{w}} \rVert + \lVert g_{\text{c}} \rVert)$.

| Epoch | $L$ | $T_{\text{w}}$ | $T_{\text{c}}$ | $\lVert g_{\text{w}} \rVert$ | $\lVert g_{\text{c}} \rVert$ | $\lVert g \rVert$ | $\cos(g_{\text{w}}, g_{\text{c}})$ | $\rho$ | $\lVert g_{\text{c}} \rVert / \lVert g_{\text{w}} \rVert$ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.3355 | 0.0547 | 0.5916 | 0.03202 | 0.0854 | 0.0960 | 0.163 | 0.817 | 2.67 |
| 2 | 0.2910 | 0.0533 | 0.5077 | 0.03114 | 0.0754 | 0.0743 | -0.241 | 0.697 | 2.42 |
| 3 | 0.2594 | 0.0580 | 0.4429 | 0.03815 | 0.0631 | 0.0636 | -0.290 | 0.628 | 1.65 |
| 4 | 0.2374 | 0.0593 | 0.3998 | 0.04059 | 0.0637 | 0.0672 | -0.230 | 0.644 | 1.57 |
| 5 | 0.2199 | 0.0566 | 0.3688 | 0.03991 | 0.0690 | 0.0642 | -0.406 | 0.589 | 1.73 |
| 6 | 0.2060 | 0.0543 | 0.3444 | 0.03724 | 0.0585 | 0.0484 | -0.566 | 0.506 | 1.57 |
| 8 | 0.1848 | 0.0498 | 0.3079 | 0.03120 | 0.0508 | 0.0422 | -0.558 | 0.515 | 1.63 |
| 10 | 0.1686 | 0.0490 | 0.2777 | 0.02752 | 0.0465 | 0.0374 | -0.592 | 0.506 | 1.69 |
| 15 | 0.1423 | 0.0461 | 0.2300 | 0.01636 | 0.0314 | 0.0247 | -0.624 | 0.518 | 1.92 |
| 20 | 0.1261 | 0.0459 | 0.1993 | 0.01949 | 0.0179 | 0.0173 | -0.572 | 0.464 | 0.92 |
| 30 | 0.1070 | 0.0445 | 0.1639 | 0.01503 | 0.0189 | 0.0100 | -0.851 | 0.294 | 1.26 |
| 40 | 0.0946 | 0.0439 | 0.1409 | 0.01418 | 0.0153 | 0.0082 | -0.849 | 0.277 | 1.08 |
| 50 | 0.0851 | 0.0439 | 0.1226 | 0.01382 | 0.0148 | 0.0073 | -0.873 | 0.254 | 1.07 |
| 60 | 0.0785 | 0.0437 | 0.1103 | 0.01399 | 0.0125 | 0.0074 | -0.852 | 0.277 | 0.90 |
| 70 | 0.0740 | 0.0440 | 0.1013 | 0.01442 | 0.0132 | 0.0088 | -0.798 | 0.321 | 0.91 |
| 80 | 0.0702 | 0.0437 | 0.0944 | 0.01359 | 0.0151 | 0.0075 | -0.869 | 0.261 | 1.11 |
| 90 | 0.0669 | 0.0435 | 0.0884 | 0.01336 | 0.0220 | 0.0116 | -0.899 | 0.327 | 1.65 |
| 100 | 0.0640 | 0.0435 | 0.0827 | 0.01404 | 0.0146 | 0.0084 | -0.830 | 0.292 | 1.04 |

Summary over all 100 epochs:

- **The cut triplets are progressively satisfied.** $T_{\text{c}}$ falls from $0.592$ to $0.083$, a drop of $86\%$. It first goes below $0.2$ at epoch 20 and below $0.1$ at epoch 72, and its minimum is at the last epoch. The epoch-1 value matches CITESEER's ($0.593$), as expected when cut pairs start unaligned ([Theorem 2](../../../doc/theory/partition-alignment.md#theorem-2-a-random-isometry-is-as-far-as-a-random-direction)).
- **The within triplets have a floor.** $T_{\text{w}}$ starts at $0.055$, $7\times$ CITESEER's $0.0077$, and only falls to $0.0435$ ($-20\%$, minimum $0.0431$ at epoch 68). It is between $0.0431$ and $0.0445$ from epoch 30 on. On CITESEER the within pairs start near cosine $0.975$; on AstroPh the weighted contrastive run measured $0.821$. By Theorem 18 a positive at cosine $0.82$ is active against every negative with $\kappa > 0.395$. In the giant partition, with $65\%$ of the replicas, such negatives are plausible, and negatives with $\kappa > 0.875$ keep the triplet active whatever the alignment does (Theorem 18 (d)). This is an explanation consistent with the data, not a measurement: the log has no negative cosines.
- **Within and cross gradients oppose each other, unlike on CITESEER.** $\cos(g_{\text{w}}, g_{\text{c}})$ is $+0.163$ at epoch 1, turns negative from epoch 2, and stays in $[-0.933, -0.744]$ from epoch 30 on (minimum at epoch 57). $\rho$ falls from $0.817$ to $[0.199, 0.360]$ over epochs 30 to 100. On CITESEER the same angle stayed in $[-0.30, 0.32]$ and $\rho \geq 0.81$. Pulling the cut pairs together disturbs the ordering of within triplets, and the active within triplets resist. The pull is balanced: $\lVert g_{\text{c}} \rVert / \lVert g_{\text{w}} \rVert$ is between $0.84$ (epoch 65) and $1.65$ from epoch 30 on, and the cross gradient is the larger one in $82$ of the $100$ epochs.
- **This is a milder standoff than the contrastive one.** The net gradient keeps $20$ to $36\%$ of the term magnitudes, against $1.7$ to $5.4\%$ from epoch 30 in the weighted contrastive run and $0.9$ to $5.6\%$ in the uniform one. Those $\rho$ are between different pairs of terms (Theorem 16), but in all three runs they measure how much of the gradient survives the summation. The loss keeps falling: $T_{\text{c}}$ keeps decreasing while $T_{\text{w}}$ stays flat. The contrastive runs were also still falling at epoch 100, but slowly ($3.9\%$ over epochs 86 to 100 in the weighted run, $1.5\%$ in the uniform one).
- **No collapse.** There is no repulsion term. The total norm falls from $0.096$ to a minimum of $5.75 \cdot 10^{-3}$ (epoch 56), with median $7.9 \cdot 10^{-3}$ over epochs 51 to 100. No vanishing-gradient warning fired.
- **Not converged.** Over epochs 86 to 100, $L$ falls by $6.6\%$ ($0.0685 \to 0.0640$) and $T_{\text{c}}$ by $9.2\%$ ($0.0911 \to 0.0827$). Both are at their minimum at epoch 100.

## Downstream metrics

**Graph reconstruction ($k = 178299$ closest pairs).** The counts are the pair probabilities of `Pair probability F1 report` multiplied by the pair counts.

| Scope | Pairs | True edges | Reconstructed | True positives | Precision | Recall |
|---|---|---|---|---|---|---|
| Within partitions | 82 383 816 | 162 579 | 170 973 | 53 145 | 0.311 | 0.327 |
| Cross partitions | 90 661 290 | 15 720 | 7 326 | 1 335 | 0.182 | **0.0849** |
| All | 173 045 106 | 178 299 | 178 299 | 54 480 | 0.306 | 0.306 |

| Quantity | Triplet, weighted | Contrastive, weighted | Contrastive, uniform |
|---|---|---|---|
| Reconstructed cut pairs / true positives | 7326 / **1335** | 20017 / 481 | 193 / 6 |
| Cut recall $r_c$ / cut precision | **0.0849** / **0.182** | 0.0325 / 0.024 | 0.00037 / 0.031 |
| Within recall $r_w$ | 0.327 | 0.299 | **0.371** |
| Within pair F1 | 0.319 | 0.304 | **0.353** |
| Micro recall (Theorem 14) | 0.306 | 0.277 | **0.337** |
| Weighted F1 | **0.416** | 0.357 | 0.410 |
| Global macro F1 | **0.533** | 0.383 | 0.409 |
| Nodes with a reconstructed neighbour | **16136** of 18604 | 12097 | 13103 of 18624 |
| Mean partition pair F1 | 0.357 | 0.357 | **0.449** |

- **The best cut reconstruction so far.** $1335$ true cut edges are recovered, $2.8\times$ the weighted contrastive run and $222\times$ the uniform one. The cut precision, $0.182$, is $1051\times$ chance ($15720 / 90661290 = 0.000173$). On CITESEER the triplet run reconstructed no cut pair at all. The CITESEER section explained that by Theorem 17: cut pairs stop at moderate cosine, while within pairs sit near $0.975$. On AstroPh the within pairs are less tight (about $0.82$ at the start) and stay partly unsatisfied, so the bar for entering the top $k$ is lower and cut pairs can reach it. The log has no pair cosines to confirm this.
- **By [Theorem 14](2026-11-07-contrastive-astroph.md#theorem-14-micro-recall-splits-by-scope)** the micro recall is $(1 - \delta) r_w + \delta r_c = 0.9118 \cdot 0.3269 + 0.0882 \cdot 0.0849 = 0.2981 + 0.0075 = 0.306$. The cut part is $0.0075$, $2.8\times$ that of the weighted contrastive run, and still small because $\delta$ is small. The micro recall is between the two contrastive runs. The uniform run is better on it because its $r_w$ is higher.
- **Macro F1 rose most** ($0.533$, against $0.409$ and $0.383$). $16136$ nodes get at least one reconstructed neighbour, against $13103$ and $12097$, so the reconstruction is spread over many more nodes. The macro within recall is $0.558 \pm 0.411$, against $0.427 \pm 0.413$ in the uniform run. The macro cross recall is $0.0121 \pm 0.0817$, and $1.5\%$ of each node's reconstructed edges cross partitions (`reconstructed-edge partition share`), against $0.04\%$ in the uniform run.
- **Per partition:**

| Partition | True edges | Reconstructed | Reconstructed / true | Recall | Pair F1 | Contrastive weighted: reconstructed / true, recall | Contrastive uniform: reconstructed / true, recall |
|---|---|---|---|---|---|---|---|
| 0 | 129 227 | 33 327 | 0.26 | 0.168 | 0.267 | 0.33, 0.151 | 0.71, 0.238 |
| 1 | 10 814 | 50 350 | 4.66 | 0.927 | 0.328 | 3.96, 0.899 | 3.34, 0.885 |
| 2 | 11 234 | 29 981 | 2.67 | 0.970 | 0.528 | 4.14, 0.911 | 1.68, 0.847 |
| 3 | 11 304 | 57 315 | 5.07 | 0.930 | 0.306 | 2.76, 0.908 | 2.68, 0.879 |

The size bias of the global top-$k$ is the strongest of the three runs. Partition 0 gets only $0.26$ of its true edge count, and the small partitions $2.7$ to $5.1$ times theirs. Partition 0's recall, $0.168$, is between the two contrastive runs. Its precision is $0.652$, so the pairs it does get are mostly right. The micro within metrics are dominated by partition 0, which holds $91.5\%$ of the within pairs (`weight=0.914697`).

**Link prediction, HeaRT (500 negatives per positive, 1000 sampled positives per row, bilinear decoder).**

| Edges | Triplet, weighted: MRR / Hits@1 / Hits@3 / Hits@10 / mean rank | Contrastive, weighted: MRR / Hits@1 / Hits@3 / Hits@10 | Contrastive, uniform: MRR / Hits@1 / Hits@3 / Hits@10 |
|---|---|---|---|
| All test edges | **0.513** / **0.394** / **0.577** / **0.725** / 19.7 | 0.383 / 0.266 / 0.418 / 0.647 | 0.343 / 0.228 / 0.375 / 0.578 |
| Cross-partition | **0.282** / **0.144** / **0.343** / **0.553** / 49.2 | 0.175 / 0.090 / 0.186 / 0.349 | 0.155 / 0.080 / 0.164 / 0.302 |
| Within-partition | **0.543** / **0.421** / **0.605** / **0.758** / 14.7 | 0.414 / 0.274 / 0.476 / 0.703 | 0.333 / 0.219 / 0.363 / 0.557 |
| Earlier $P = 4$ baseline, 10 runs | 0.452 ± 0.022 / 0.326 ± 0.023 / 0.517 ± 0.027 / 0.697 ± 0.019 | | |

- **Best on every ranking metric and every scope** of all AstroPh runs in this page. MRR rose by $+0.130$ overall, $+0.107$ cross and $+0.129$ within over the weighted contrastive run.
- **The first AstroPh alignment run above the baseline**: MRR $+2.8$, Hits@1 $+3.0$, Hits@3 $+2.2$ and Hits@10 $+1.5$ standard deviations of the 10-run table. The contrastive runs were at $-3.1$ (weighted) and $-5.0$ (uniform) on MRR.
- Cross-partition MRR is $20.8\times$ chance ($0.0136$), against $12.9\times$ and $11.4\times$. The within/cross MRR ratio is $1.92$, against $2.36$ and $2.15$: the gap between the scopes is the smallest so far.
- The scope mix is nearly the same in all three runs, and the count-weighted mean of the scope MRRs, $0.1095 \cdot 0.282 + 0.8905 \cdot 0.543 = 0.514$, matches the logged $0.513$. (This is Theorem 14 (a) with reciprocal ranks in place of true positives; it holds only approximately because each scope's HeaRT call draws its own negatives.)
- **Noise.** With Hits@1 near $0.39$, the standard deviation of one reciprocal rank is about $0.42$, so the standard error of one MRR over $1000$ sampled positives is about $0.013$, and that of a difference about $0.019$. The $+0.130$ gain over the weighted contrastive run is about $7$ such standard errors, and about $4.2$ run-to-run standard deviations ($\sqrt 2 \cdot 0.022 = 0.031$). It is far larger than the gain from the $1/\delta$ weighting ($+0.040$).

**Link prediction, thresholded.** The three evaluations share the $19811$ negatives. The false-positive count of each scope, $\text{TP}/\text{precision} - \text{TP}$, is $9503 \pm 2$.

| Scope | Positives | Recall | True positives | Precision | F1 | Accuracy |
|---|---|---|---|---|---|---|
| All | 19811 | 0.986 | 19532 | 0.673 | 0.800 | 0.753 |
| Cross partitions | 2170 | 0.934 | 2026 | 0.176 | 0.296 | 0.561 |
| Within partitions | 17641 | 0.992 | 17505 | 0.648 | 0.784 | 0.743 |

- $2026 + 17505 = 19531$, which equals the full model's $19532$ up to rounding. Each accuracy follows from $(\text{TP} + 19811 - 9503) / (\text{positives} + 19811)$.
- **The specificity is $1 - 9503/19811 = 0.520$**, against $0.889$ (uniform) and $0.900$ (weighted) for the contrastive runs. Their full-model precision / recall / F1 / accuracy were $0.891 / 0.908 / 0.899 / 0.898$ and $0.901 / 0.914 / 0.907 / 0.907$. So at the threshold, triplet is **worse** (F1 $0.800$, accuracy $0.753$), while it ranks far better. This is the same operating point as the CITESEER triplet run (high recall, specificity $0.48$), with a high decoder loss ($0.4295$ train, $0.4300$ validation; on CITESEER $0.470$). The unit output scale was suspected as the cause (see the CITESEER triplet section). **Refuted later:** in [Aligner start, training and output scale on AstroPh at $P = 1$ analysis](2026-11-08-aligner-start-astroph-p1.md#aligner-start-training-and-output-scale-on-astroph-at-p--1-analysis), unit vectors without training keep specificity $0.965$; the triplet training causes the low specificity.
- The cut recall at the threshold is $0.934$, against $0.992$ within. The precision rows are not comparable between scopes, because the same $9503$ false positives are divided over very different positive counts.

## Reading

1. **On AstroPh, weighted triplet alignment is the first alignment that beats the no-alignment baseline on link prediction** (MRR $0.513$ against $0.452 \pm 0.022$), and it beats both contrastive runs on every ranking metric in both scopes. This is a single run, but the gain over the contrastive runs is several times the run-to-run variation.
2. **It also reconstructs cut edges best** ($r_c = 0.085$, $1335$ true cut edges) and has the best macro F1 and weighted F1. Its micro recall stays below the uniform contrastive run, because the top-$k$ size bias against the giant partition is the strongest of the three runs.
3. **The CITESEER explanation carries over only in part.** On CITESEER the within triplets were satisfied and the two scopes did not interact. On AstroPh the within triplets have a floor of active triplets (Theorem 18) and the within and cross gradients oppose each other ($\cos \approx -0.85$). The standoff is partial, though: $20$ to $36\%$ of the gradient survives, against $1$ to $5\%$ for contrastive, and the cut loss keeps falling.
4. **The two datasets disagree on cut reconstruction**: zero cut pairs on CITESEER, the most so far on AstroPh. Both are consistent with a loss that only orders distances. Whether cut pairs enter the top $k$ depends on how tight the within pairs are, which is a property of the data, not of the loss.
5. **The thresholded metrics are worse** than for contrastive (specificity $0.52$ against $0.89$ to $0.90$), on both datasets. Together with the high decoder loss, this pointed at the unit output scale. **Refuted later** by [Aligner start, training and output scale on AstroPh at $P = 1$ analysis](2026-11-08-aligner-start-astroph-p1.md#aligner-start-training-and-output-scale-on-astroph-at-p--1-analysis): the cause is the triplet training, not the unit vectors.

## Next runs

1. Swap the output scales (`alignment_output_scale=unit` for contrastive, `replica_norm` for triplet) on AstroPh, where the ranking and thresholded metrics now point in opposite directions. This separates the effect of the loss from that of the scale.
2. Run the triplet aggregator with `alignment_link_weighting=uniform` on AstroPh. With $\delta = 0.088$ the weighting multiplies the cut share by $5.9$, so its effect should be larger here than on CITESEER.
3. Raise `alignment_train_steps`: $T_{\text{c}}$ was still falling by $9\%$ over the last 15 epochs.
4. Log the cosine of each sampled negative to its anchor ($\kappa$) and of each positive pair, split by scope, to test Theorem 18's explanation of the within floor and of the opposing gradients.
5. Normalise the reconstruction per partition, since the size bias against partition 0 is now the main thing limiting micro recall.
6. Repeat over at least 5 seeds with a fixed LPA seed, together with no alignment on the same partitions, to turn the $+2.8$ sd over the historical baseline into a matched comparison.

