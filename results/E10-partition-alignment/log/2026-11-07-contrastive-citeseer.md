# Contrastive loss on CITESEER analysis

Source: one run of `vv test` on CITESEER with the contrastive aggregator, logged in `VertexVoyage/vv.log` (2026-11-07). Every number below is either read directly from that log or derived from logged numbers by one of the identities proved in this section. This is **one run with one seed**, so no number here has a standard deviation.

## Setup

Let $G = (V, E)$ be the CITESEER [graph](../../../doc/glossary.md#graph) after removing the link-prediction test edges, and let $E_{\text{test}}$ be the removed edges.

| Quantity | Value | Source line |
|---|---|---|
| $\lvert V \rvert$ | $3264$ | `Total number of nodes` |
| $\lvert E \rvert$ | $4083$ | `Total number of edges` |
| $\lvert E_{\text{test}} \rvert$ | $453$ | `Removed 453 edges` |
| $P$ | $4$, [label propagation](../../../doc/glossary.md#label-propagation) | `use_lpa: True`, `partitions: 4` |
| $\lvert V_p \rvert$ | $816$ for each $p$, so $RF = 1$ | `Partition size: 816` |
| $\lvert E_p \rvert$ | $853, 815, 724, 798$; sum $3190$ | `Partition number of edges` |
| $\lvert E_{\text{cut}} \rvert$ | $4083 - 3190 = 893$ | derived |
| $\delta = \lvert E_{\text{cut}} \rvert / \lvert E \rvert$ | $0.2187$ | `delta: 0.218712` |
| [node2vec](../../../doc/glossary.md#node2vec) | $d = 50$, $p = 0.5$, $q = 0.25$, 10 epochs | `Initialized Node2Vec` |
| Aligner | $h = 2d = 100$, 100 epochs, [Adam](../../../doc/glossary.md#adam-optimization), learning rate $0.01$, margin $m = 1$ | defaults of `AlignmentSettings` |
| Positive pairs $\lvert E^+ \rvert$ | $4083$ | `Aligner trained on 4083 links` |
| Negative pairs $\lvert E^- \rvert$ | $4083$ per epoch, resampled uniformly from the $3129$ replica rows | `repulsion ... over 4083 pairs` |
| Validation | **off**: `validated on 0`, `validation loss not computed` | final aligner line |

Facts that follow from the table:

- With $RF = 1$ the set $E^+_{\text{rep}}$ of [Pair sets](../../../doc/theory/partition-alignment.md#pair-sets) is empty, so $E^+ = E^+_{\text{intra}} \cup E^+_{\text{cut}}$ with $\lvert E^+_{\text{intra}} \rvert = 3190$ and $\lvert E^+_{\text{cut}} \rvert = 893$.
- By [Theorem 4](../../../doc/theory/partition-alignment.md#theorem-4-the-alignment-signal-lives-in-cross-partition-pairs), only the $893$ cut pairs ($21.9\%$ of the positives) carry information about how the partitions are placed relative to each other. The other $78.1\%$ only reshape each partition internally.
- The whole stream arrived as **one buffer** (`Updating aggregator on 4083 edges`, logged once), so the aligner was trained exactly once, for 100 epochs.
- The $9788$ lines `Node ... not found in Word2Vec vocabulary` are expected and not a defect: $9788 = 4 \times 2447$, which is one line per partition for every node outside that partition.

## Training dynamics

Notation for epoch $t$:

- $L^{(t)} = A^{(t)} + R^{(t)}$ is the logged `loss`. $A^{(t)} = \frac{1}{\lvert E^+ \rvert} \sum_{E^+} D^2$ is the attraction mean and $R^{(t)} = \frac{1}{\lvert E^- \rvert} \sum_{E^-} \max(0, m - D)^2$ the repulsion mean. Both are means, so $L$ is the mean-normalised form of $L_m$ from [Losses, term by term](../../../doc/theory/partition-alignment.md#losses-term-by-term).
- $\theta$ is the vector of all aligner parameters. $g_A = \nabla_\theta A$, $g_R = \nabla_\theta R$ and $g = g_A + g_R = \nabla_\theta L$. The logged numbers are the Euclidean [norms](../../../doc/glossary.md#normed-space) $\lVert g_A \rVert$ (`attraction`), $\lVert g_R \rVert$ (`repulsion`) and $\lVert g \rVert$ (`total`).
- $\cos(g_A, g_R)$ is the cosine of the angle between the two term gradients, and $\rho = \lVert g \rVert / (\lVert g_A \rVert + \lVert g_R \rVert)$ is the **cancellation ratio**. Theorem 11 shows how both are computed from the three logged norms.
- $\bar c^{(t)} = 1 - A^{(t)}/2$ is the mean cosine over positive pairs (Theorem 12).

| Epoch | $L$ | $A$ | $R$ | $\lVert g_A \rVert$ | $\lVert g_R \rVert$ | $\lVert g \rVert$ | $\cos(g_A, g_R)$ | $\rho$ | $\bar c$ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.4325 | 0.4220 | 0.0105 | 0.4105 | 0.0195 | 0.3939 | -0.860 | 0.916 | 0.789 |
| 2 | 0.2816 | 0.2602 | 0.0215 | 0.3400 | 0.0733 | 0.2701 | -0.964 | 0.653 | 0.870 |
| 3 | 0.2406 | 0.1350 | 0.1056 | 0.1902 | 0.2734 | 0.0886 | -0.991 | 0.191 | 0.933 |
| 4 | 0.2662 | 0.0937 | 0.1726 | 0.1296 | 0.3152 | 0.1873 | -0.992 | 0.421 | 0.953 |
| 5 | 0.2574 | 0.0945 | 0.1629 | 0.1283 | 0.3038 | 0.1770 | -0.993 | 0.410 | 0.953 |
| 7 | 0.2170 | 0.1391 | 0.0778 | 0.1826 | 0.2133 | 0.0414 | -0.990 | 0.105 | 0.930 |
| 10 | 0.2175 | 0.1819 | 0.0356 | 0.2287 | 0.1141 | 0.1182 | -0.984 | 0.345 | 0.909 |
| 20 | 0.1722 | 0.1275 | 0.0447 | 0.1611 | 0.1488 | 0.0221 | -0.993 | 0.071 | 0.936 |
| 30 | 0.1484 | 0.1065 | 0.0420 | 0.1273 | 0.1269 | 0.0155 | -0.993 | 0.061 | 0.947 |
| 40 | 0.1308 | 0.0903 | 0.0405 | 0.1051 | 0.1111 | 0.0138 | -0.993 | 0.064 | 0.955 |
| 50 | 0.1158 | 0.0800 | 0.0357 | 0.0917 | 0.0954 | 0.0105 | -0.994 | 0.056 | 0.960 |
| 60 | 0.1045 | 0.0723 | 0.0322 | 0.0812 | 0.0842 | 0.0101 | -0.993 | 0.061 | 0.964 |
| 70 | 0.0988 | 0.0657 | 0.0331 | 0.0728 | 0.0767 | 0.0119 | -0.989 | 0.079 | 0.967 |
| 80 | 0.0893 | 0.0613 | 0.0280 | 0.0677 | 0.0675 | 0.0103 | -0.988 | 0.076 | 0.969 |
| 90 | 0.0817 | 0.0551 | 0.0266 | 0.0598 | 0.0669 | 0.0137 | -0.983 | 0.108 | 0.972 |
| 100 | 0.0835 | 0.0530 | 0.0305 | 0.0616 | 0.0578 | 0.0168 | -0.962 | 0.141 | 0.973 |

The smallest total norm is $\lVert g \rVert = 9.54 \cdot 10^{-3}$ at epoch 57. The median over epochs 51 to 100 is $1.19 \cdot 10^{-2}$. No epoch triggered the vanishing-gradient warning (threshold $10^{-7}$).

## Theorem 11 (the angle between two term gradients is read from three norms)

**Assumptions:**

- $H$ is a real [inner product space](../../../doc/glossary.md#inner-product-space) with norm $\lVert x \rVert = \sqrt{\langle x, x \rangle}$. Here $H = \mathbb{R}^{\lvert \theta \rvert}$, the parameter space of the aligner.
- $a, r \in H$ with $a \neq 0$ and $r \neq 0$. Here $a = g_A$ and $r = g_R$.
- For (e): a real number $\rho$ with $\lVert a + r \rVert \leq \rho (\lVert a \rVert + \lVert r \rVert)$.

**Statement:**

- (a) $\lVert a + r \rVert^2 = \lVert a \rVert^2 + \lVert r \rVert^2 + 2 \langle a, r \rangle$.
- (b) $\cos(a, r) = \frac{\langle a, r \rangle}{\lVert a \rVert \lVert r \rVert} = \frac{\lVert a + r \rVert^2 - \lVert a \rVert^2 - \lVert r \rVert^2}{2 \lVert a \rVert \lVert r \rVert}$
- (c) $\big\lvert \lVert a \rVert - \lVert r \rVert \big\rvert \leq \lVert a + r \rVert \leq \lVert a \rVert + \lVert r \rVert$.
- (d) $\rho = \lVert a + r \rVert / (\lVert a \rVert + \lVert r \rVert) \leq 1$.
- (e) $$ \cos(a, r) \leq \frac{\rho^2 (\lVert a \rVert + \lVert r \rVert)^2 - \lVert a \rVert^2 - \lVert r \rVert^2}{2 \lVert a \rVert \lVert r \rVert} $$

**Proof:**

1. (a) Expand by bilinearity and symmetry of the inner product: $\langle a + r, a + r \rangle = \langle a, a \rangle + \langle a, r \rangle + \langle r, a \rangle + \langle r, r \rangle = \lVert a \rVert^2 + 2 \langle a, r \rangle + \lVert r \rVert^2$.
2. (b) $a \neq 0$ and $r \neq 0$ give $\lVert a \rVert > 0$ and $\lVert r \rVert > 0$, so $\lVert a \rVert \lVert r \rVert > 0$ and both quotients are defined. Solve (a) for $\langle a, r \rangle = \tfrac12 (\lVert a + r \rVert^2 - \lVert a \rVert^2 - \lVert r \rVert^2)$ and divide by $\lVert a \rVert \lVert r \rVert$.
3. (c) The right inequality is the triangle inequality. For the left one, apply the reverse triangle inequality $\lvert \lVert x \rVert - \lVert y \rVert \rvert \leq \lVert x - y \rVert$ to $x = a$, $y = -r$, and use $\lVert -r \rVert = \lVert r \rVert$ and $a - (-r) = a + r$.
4. (d) $\lVert a \rVert + \lVert r \rVert > 0$ by step 2, so dividing the right inequality of (c) by it keeps the direction.
5. (e) Both sides of the assumption are non-negative because $\lVert a + r \rVert \geq 0$, so squaring keeps the direction: $\lVert a + r \rVert^2 \leq \rho^2 (\lVert a \rVert + \lVert r \rVert)^2$. Substitute this upper bound into the numerator of (b); the denominator $2 \lVert a \rVert \lVert r \rVert$ is positive, so the quotient can only grow. $\blacksquare$

Lean: `norm_add_sq_of_terms`, `cosine_of_term_norms`, `norm_add_le_sum_of_term_norms`, `abs_sub_term_norms_le_norm_add`, `cancellation_ratio_le_one`, `cosine_le_of_cancellation` in section `GradientCancellation`.

**Consequences:**

- $\rho = 1$ exactly when the two gradients point the same way, and $\rho$ reaches its lower bound $\lvert \lVert a \rVert - \lVert r \rVert \rvert / (\lVert a \rVert + \lVert r \rVert)$ exactly when they point in opposite directions. When $\lVert a \rVert \approx \lVert r \rVert$ that lower bound is $\approx 0$.
- (e) turns a small $\rho$ into a certified upper bound on the cosine, so the angle can be checked without storing either gradient.

**Example (epoch 50):** $\lVert a \rVert = 0.09172$, $\lVert r \rVert = 0.09535$, $\lVert a + r \rVert = 0.01046$. Then $\lVert a \rVert^2 = 0.008413$, $\lVert r \rVert^2 = 0.009092$, $\lVert a + r \rVert^2 = 0.000109$ and $2 \lVert a \rVert \lVert r \rVert = 0.017491$, so by (b)

$$ \cos(g_A, g_R) = \frac{0.000109 - 0.008413 - 0.009092}{0.017491} = -0.994 $$

The angle is $\arccos(-0.994) = 173.7°$. The cancellation ratio is $\rho = 0.01046 / 0.18707 = 0.056$, close to its lower bound $\lvert 0.09172 - 0.09535 \rvert / 0.18707 = 0.019$ from (c). The logged norms have seven significant digits, so the rounding error in the cosine is far below the third decimal.

## Theorem 12 (mean positive cosine from the attraction term)

**Assumptions:** $H$ is a real inner product space, $S$ is a finite index set with $n = \lvert S \rvert > 0$, and $x_i, y_i \in H$ with $\lVert x_i \rVert = \lVert y_i \rVert = 1$ for all $i \in S$.

**Statement:**

$$ \sum_{i \in S} \lVert x_i - y_i \rVert^2 = 2n - 2 \sum_{i \in S} \langle x_i, y_i \rangle, \qquad \text{so} \qquad \frac1n \sum_{i \in S} \langle x_i, y_i \rangle = 1 - \frac{1}{2n} \sum_{i \in S} \lVert x_i - y_i \rVert^2 $$

**Proof:**

1. For each $i$, [Theorem 5](../../../doc/theory/partition-alignment.md#theorem-5-geometry-of-the-unit-sphere) (a) gives $\lVert x_i - y_i \rVert^2 = 2 - 2 \langle x_i, y_i \rangle$.
2. Sum over $S$: the constant $2$ summed $n$ times is $2n$, and $\sum 2 \langle x_i, y_i \rangle = 2 \sum \langle x_i, y_i \rangle$ by linearity of the finite sum.
3. Divide by $2n > 0$ and rearrange. $\blacksquare$

Lean: `sum_norm_sub_sq_of_unit`.

**Consequence:** the aligner outputs are unit vectors (`ResidualAligner.forward` normalises them), so $\bar c = 1 - A/2$ is the exact mean cosine over the positive pairs of that epoch.

**Example:** at epoch 1, $A = 0.4220$ gives $\bar c = 1 - 0.2110 = 0.789$. At epoch 100, $A = 0.0530$ gives $\bar c = 0.973$.

## Theorem 13 (no cross-partition reconstruction caps recall at $1 - \delta$)

**Assumptions:**

- A reconstruction $\hat E$ is scored against the true edges $E = E_{\text{within}} \sqcup E_{\text{cut}}$, a disjoint union with $w = \lvert E_{\text{within}} \rvert$, $c = \lvert E_{\text{cut}} \rvert$ and $w + c > 0$.
- Micro recall is $\text{TP} / (w + c)$, where $\text{TP} = \lvert \hat E \cap E \rvert$.
- No reconstructed edge crosses a partition, so $\hat E \cap E \subseteq E_{\text{within}}$ and $\text{TP} \leq w$.

**Statement:** $\text{TP} / (w + c) \leq 1 - c / (w + c) = 1 - \delta$.

**Proof:**

1. Multiply both sides by $w + c > 0$; the inequality becomes $\text{TP} \leq (w + c) - c = w$.
2. This is the third assumption. $\blacksquare$

Lean: `recall_le_within_share` in section `CutRecall`.

**Example:** CITESEER here has $w = 3190$ and $c = 893$, so micro recall is at most $3190 / 4083 = 0.781$ whatever happens inside the partitions.

## Downstream metrics

**Graph reconstruction.** The global reconstruction connects the $k = \lvert E \rvert = 4083$ closest pairs of merged embeddings $w_u$.

| Scope | Pairs | $p(\text{true edge})$ | $p(\text{reconstructed edge})$ | $p(\text{true positive})$ | F1 |
|---|---|---|---|---|---|
| Within partitions | 1 330 080 | 0.002398 | 0.003070 | 0.001295 | 0.4735 |
| Cross partitions | 3 544 923 | 0.000252 | **0.000000** | 0.000000 | 0.0000 |

- **Not one of the 4083 closest pairs crosses a partition**, even though the aligner was trained on $893$ cut edges. The aligned partitions stay separated in the common space, and every cut edge is closer to some within pair, so by Theorem 13 the micro recall is at most $0.781$.
- The macro view agrees: within recall $0.631 \pm 0.432$, cross recall $0.000 \pm 0.000$, and cross precision undefined (`nan`) because the denominator is $0$.
- Global macro F1 is $0.544$. Per-partition node2vec F1 before merging was $0.400, 0.422, 0.418, 0.369$; this is a different denominator and should not be compared directly.

**Link prediction (HeaRT, 500 negatives per positive, bilinear decoder).** Under a scorer that ignores the input, the rank of the positive is uniform on $\{1, \dots, 501\}$, so the chance levels are $\mathbb{E}[1/\text{rank}] = H_{501}/501 \approx 0.0136$ (with $H_n$ the $n$-th harmonic number), mean rank $251$ and Hits@$k$ $= k/501$.

| Edges | Count | MRR | Mean rank | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|
| All test edges | 453 | 0.1468 | 119.3 | 0.062 | 0.174 | 0.309 |
| Cross-partition | 289 | 0.0720 | 156.5 | 0.024 | 0.059 | 0.163 |
| Within-partition | 164 | 0.2908 | 53.3 | 0.146 | 0.335 | 0.579 |
| Chance | | 0.0136 | 251 | 0.002 | 0.006 | 0.020 |
| Earlier CITESEER $P = 4$ bilinear LPA, 10 runs ([link-prediction evaluation](../../E09-link-prediction/README.md#bilinear-decoder)) | | 0.138 ± 0.007 | | 0.081 ± 0.010 | 0.155 ± 0.010 | 0.245 ± 0.013 |

- Cross-partition MRR is $5.3\times$ chance and Hits@10 is $8\times$ chance, so the decoder does extract some cross-partition signal. Within-partition MRR is still $4\times$ the cross MRR.
- **Test edges are cut far more often than training edges:** $289 / 453 = 63.8\%$ of the test edges are cross-partition, against $\delta = 21.9\%$ of the training edges. Label propagation runs on the training graph and cuts where training edges are sparse, which is where the removed edges were. The cross-partition rows therefore dominate the overall metrics.
- The earlier 10-run table was produced before the aggregator registry existed (2026-10-13), so it is most likely the no-alignment baseline. Against it, this single run has a higher MRR (0.147 against $0.138 \pm 0.007$) and Hits@10 (0.309 against $0.245 \pm 0.013$), but a lower Hits@1 (0.062 against $0.081 \pm 0.010$). One seed cannot separate this from run-to-run variation; this needs repeated runs.
- The thresholded `Precision`, `Recall`, `F1` and `Accuracy` printed for "cross partitions" and "within partitions" are **not valid** in this log. All three evaluations call `run("lp_full_eval", ...)` with the same key, so `PersistedRun` returns the cached full-model result (`Key lp_full_eval found in memory`), and the three blocks print identical numbers. The HeaRT ranking metrics above are computed separately and are valid.

## Reading

1. **No vanishing gradient, but an almost exact tug-of-war.** Over all 100 epochs each term gradient has a norm between $0.019$ and $0.41$, more than $10^5$ times the threshold, so in the sense of the [vanishing gradient problem](../../../doc/glossary.md#vanishing-gradient-problem) nothing vanishes. From epoch 3 to epoch 100, however, $\cos(g_A, g_R) \leq -0.953$ in every epoch (angle above $162°$; the maximum $-0.953$ is at epoch 98), and from epoch 20 on $\rho \in [0.054, 0.177]$: typically about $90\%$ of the gradient magnitude of the two terms cancels. This is the frustration of [Theorem 6](../../../doc/theory/partition-alignment.md#theorem-6-collapse-and-frustration-of-the-contrastive-loss) seen directly in parameter space. The aligner sits near a compromise where pulling positives closer pushes random pairs inside the margin by the same amount.
2. **Three phases.**
   - Epochs 1 to 2: attraction dominates ($\lVert g_A \rVert / \lVert g_R \rVert = 21$ at epoch 1). The aligner contracts the embedding and $\bar c$ rises from $0.789$ to $0.870$.
   - Epochs 3 to 5: the contraction overshoots. $R$ rises sixteenfold, from $0.0105$ to $0.173$, which means a large share of random pairs now have cosine above $1 - m^2/2 = 0.5$ ([Theorem 5](../../../doc/theory/partition-alignment.md#theorem-5-geometry-of-the-unit-sphere) (c)). This is the start of the collapse of Theorem 6 (a). The repulsion gradient takes over and the total loss rises at epoch 4.
   - Epochs 6 to 100: a slow joint descent along the narrow direction the two terms do not oppose. $L$ falls from $0.23$ to $0.08$ and $\bar c$ rises to $0.973$. Over epochs 91 to 100 $L$ moves within $[0.079, 0.084]$ with no clear trend. That is a plateau at this learning rate, with noise from the negatives being resampled every epoch. [Adam](../../../doc/glossary.md#adam-optimization) divides each coordinate by a running estimate of its gradient magnitude, so the small $\lVert g \rVert$ does not by itself mean small steps.
3. **The descent is spent on within-partition geometry.** $\bar c = 0.973$ averages over all positives, and $78\%$ of them are intra pairs, which by Theorem 4 cannot align partitions. The reconstruction shows that the cut pairs were not closed: no cut pair is among the $4083$ nearest pairs. The run does not record separate attraction terms for intra and cut pairs, so it cannot say how far the cut pairs moved.
4. **The vanishing-gradient threshold detects the wrong failure here.** An absolute threshold on $\lVert g \rVert$ would need to be about $10^{-2}$ to fire, and that value is not transferable across datasets or model sizes. The scale-free signal is $\rho$ together with $\cos(g_A, g_R)$ from Theorem 11.
5. **No validation evidence.** Validation was off (`validated on 0`), so this run says nothing about overfitting of the aligner. See [validation loss](../../../doc/glossary.md#validation-loss).

## Next runs

1. Rerun with `VERTEX_VOYAGE_ALIGNMENT_VALIDATION=true` (and optionally `VERTEX_VOYAGE_ALIGNMENT_VALIDATION_FRACTION`) to get a per-epoch validation loss on held-out edges.
2. Log the attraction separately for $E^+_{\text{intra}}$ and $E^+_{\text{cut}}$, and the mean cosine of cut pairs (Theorem 12 applied to $E^+_{\text{cut}}$ only), so that alignment progress is measured on the pairs that carry it.
3. Train on $E^+_{\text{cut}}$ only, or weight it by $1/\delta$, and check whether any cross pair enters the reconstruction.
4. Log $\cos(g_A, g_R)$ and $\rho$ per epoch, and warn on a sustained $\rho < 0.1$ instead of (or in addition to) the absolute norm threshold.
5. Lower the learning rate or decay it, since $L$ plateaued with noise over the last 10 epochs instead of converging, and sweep $m \in \{0.5, 1, \sqrt{2}\}$, which covers the meaningful cosine thresholds of [Theorem 5](../../../doc/theory/partition-alignment.md#theorem-5-geometry-of-the-unit-sphere) (c). A smaller $m$ shrinks the region where repulsion is active and should weaken the opposition of the two terms.
6. Fix the `lp_full_eval` cache key in `vertex_voyage/plugins/test_commands.py` so that the cross-partition and within-partition thresholded metrics are actually computed.
7. Repeat over at least 10 seeds so that the comparison with the 10-run no-alignment baseline has error bars.

