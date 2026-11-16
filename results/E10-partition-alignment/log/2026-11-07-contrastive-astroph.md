# Contrastive loss on AstroPh analysis

Source: one run of `vv test` on AstroPh, logged in `VertexVoyage/vv.log` (2026-11-07, 15:04 to 15:17). The settings are those of [Contrastive loss on CITESEER analysis](2026-11-07-contrastive-citeseer.md#contrastive-loss-on-citeseer-analysis): contrastive aggregator, $P = 4$, [label propagation](../../../doc/glossary.md#label-propagation), bilinear decoder, validation off. The one change is `VERTEX_VOYAGE_ALIGNMENT_LINK_MEMORY=200000`. The default memory of $20000$ links is smaller than $\lvert E \rvert$ for AstroPh and would have silently dropped most edges. The notation $A, R, g_A, g_R, g, \rho, \bar c$ and Theorems 11 to 13 are those of the CITESEER section. This is **one run with one seed**, so no number here has a standard deviation.

## Setup

Let $G = (V, E)$ be the AstroPh [graph](../../../doc/glossary.md#graph) after removing the test edges $E_{\text{test}}$.

| Quantity | AstroPh | CITESEER, for comparison | Source line |
|---|---|---|---|
| $\lvert V \rvert$ | $18772$ | $3264$ | `Total number of nodes` |
| $\lvert E \rvert$ | $178299$ | $4083$ | `Total number of edges` |
| $\lvert E_{\text{test}} \rvert$ | $19811$ | $453$ | `Removed ... edges` |
| $\lvert V_p \rvert$ | $12216, 2186, 2185, 2185$ | $816$ each | `Partition size` |
| $\lvert E_p \rvert$ | $127970, 10945, 11500, 11659$; sum $162074$ | sum $3190$ | `Partition number of edges` |
| $\lvert E_{\text{cut}} \rvert$ | $178299 - 162074 = 16225$ | $893$ | derived |
| $\delta$ | $0.0910$ | $0.2187$ | `delta: 0.090999` |
| [node2vec](../../../doc/glossary.md#node2vec) | $d = 50$, $p = 2$, $q = 0.25$; partition F1 $0.675, 0.535, 0.669, 0.565$ | $d = 50$, $p = 0.5$, $q = 0.25$ | `New best` |
| Aligner | $h = 100$, 100 epochs, [Adam](../../../doc/glossary.md#adam-optimization), learning rate $0.01$, $m = 1$ | same | defaults |
| Positive pairs $\lvert E^+ \rvert$ | $178299$, i.e. **every** edge | $4083$, every edge | `Aligner trained on 178299 links` |
| Negative pairs per epoch | $178299$, from $18627$ replica rows | $4083$, from $3129$ rows | same line |
| Validation | off (`validated on 0`) | off | same line |

Facts that follow from the table:

- **The partitions are very unbalanced.** Label propagation put $65.1\%$ of the nodes and $79.0\%$ of the covered edges into partition 0. The other three partitions have about $2185$ nodes each. LPA does not balance partition sizes, and this run shows it.
- With $RF = 1$ the set $E^+_{\text{rep}}$ is empty, so $E^+ = E^+_{\text{intra}} \cup E^+_{\text{cut}}$ with $\lvert E^+_{\text{intra}} \rvert = 162074$ and $\lvert E^+_{\text{cut}} \rvert = 16225$. By [Theorem 4](../../../doc/theory/partition-alignment.md#theorem-4-the-alignment-signal-lives-in-cross-partition-pairs) only the cut pairs, $9.1\%$ of the positives, carry information about how the partitions sit relative to each other. On CITESEER the share was $21.9\%$.
- All edges arrived in one buffer (`Updating aggregator on 178299 edges`, logged once), so the aligner was trained once, for 100 epochs.
- The $56312$ lines `Node ... not found in Word2Vec vocabulary` are expected, one per partition for each node outside it: $(18772 - 12217) + (18772 - 2187) + 2 \times (18772 - 2186) = 56312$, with the vocabulary sizes from `Trained Word2Vec model`.

## Training dynamics

| Epoch | $L$ | $A$ | $R$ | $\lVert g_A \rVert$ | $\lVert g_R \rVert$ | $\lVert g \rVert$ | $\cos(g_A, g_R)$ | $\rho$ | $\bar c$ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.4679 | 0.4658 | 0.0021 | 0.5869 | 0.0068 | 0.5807 | -0.905 | 0.978 | 0.767 |
| 2 | 0.2780 | 0.2554 | 0.0226 | 0.3849 | 0.1309 | 0.2553 | -0.993 | 0.495 | 0.872 |
| 3 | 0.2619 | 0.1246 | 0.1373 | 0.1883 | 0.3202 | 0.1373 | -0.988 | 0.270 | 0.938 |
| 4 | 0.2858 | 0.0913 | 0.1945 | 0.1288 | 0.3201 | 0.1938 | -0.989 | 0.432 | 0.954 |
| 5 | 0.2753 | 0.0900 | 0.1853 | 0.1207 | 0.2991 | 0.1797 | -0.994 | 0.428 | 0.955 |
| 6 | 0.2543 | 0.1025 | 0.1519 | 0.1329 | 0.2673 | 0.1357 | -0.995 | 0.339 | 0.949 |
| 7 | 0.2409 | 0.1221 | 0.1188 | 0.1547 | 0.2288 | 0.0832 | -0.980 | 0.217 | 0.939 |
| 8 | 0.2365 | 0.1426 | 0.0939 | 0.1784 | 0.2013 | 0.0607 | -0.956 | 0.160 | 0.929 |
| 10 | 0.2311 | 0.1713 | 0.0598 | 0.2174 | 0.1715 | 0.0707 | -0.961 | 0.182 | 0.914 |
| 15 | 0.2098 | 0.1502 | 0.0595 | 0.2062 | 0.1886 | 0.0234 | -0.997 | 0.059 | 0.925 |
| 20 | 0.2049 | 0.1238 | 0.0811 | 0.1708 | 0.2122 | 0.0467 | -0.994 | 0.122 | 0.938 |
| 30 | 0.1933 | 0.1255 | 0.0679 | 0.1552 | 0.1579 | 0.0129 | -0.997 | 0.041 | 0.937 |
| 40 | 0.1860 | 0.1272 | 0.0588 | 0.1474 | 0.1367 | 0.0122 | -0.999 | 0.043 | 0.936 |
| 50 | 0.1802 | 0.1189 | 0.0613 | 0.1310 | 0.1323 | 0.0046 | -0.9994 | 0.018 | 0.941 |
| 60 | 0.1764 | 0.1154 | 0.0610 | 0.1245 | 0.1263 | 0.0039 | -0.9996 | 0.016 | 0.942 |
| 70 | 0.1733 | 0.1134 | 0.0600 | 0.1211 | 0.1224 | 0.0033 | -0.9997 | 0.014 | 0.943 |
| 80 | 0.1706 | 0.1118 | 0.0588 | 0.1180 | 0.1182 | 0.0024 | -0.9998 | 0.010 | 0.944 |
| 90 | 0.1681 | 0.1103 | 0.0578 | 0.1151 | 0.1151 | 0.0022 | -0.9998 | 0.009 | 0.945 |
| 100 | 0.1666 | 0.1083 | 0.0582 | 0.1117 | 0.1133 | 0.0026 | -0.9998 | 0.012 | 0.946 |

Summary over all 100 epochs (not only the rows above):

- The smallest total norm is $\lVert g \rVert = 2.10 \cdot 10^{-3}$ at epoch 96, and the median over epochs 51 to 100 is $2.87 \cdot 10^{-3}$. No epoch triggered the vanishing-gradient warning and no term was marked `(vanishing)`. The term norms never fell below $6.8 \cdot 10^{-3}$, the epoch-1 repulsion.
- From epoch 2 on, $\cos(g_A, g_R) \leq -0.952$, with the maximum $-0.952$ at epoch 9. From epoch 30 on, $\cos(g_A, g_R) \leq -0.9967$. From epoch 50 on it is at most $-0.99927$ (at epoch 54), an angle above $177.8°$.
- From epoch 30 on, $\rho \in [0.009, 0.056]$, with the minimum $0.009$ at epoch 96.
- Over epochs 86 to 100, $L$ goes from $0.1692$ to $0.1666$, a fall of $1.6\%$ in 15 epochs. The minimum is $0.1664$ at epoch 99.
- $A$ is smallest at epoch 5 ($0.0900$) and **ends higher**, at $0.1083$. $\bar c$ reaches $0.955$ at epoch 5 and ends at $0.946$.

**Example (Theorem 11 at epoch 90):** $\lVert g_A \rVert = 0.1150813$, $\lVert g_R \rVert = 0.1150734$, $\lVert g \rVert = 0.002171296$. Then $\lVert g_A \rVert^2 = 0.0132437$, $\lVert g_R \rVert^2 = 0.0132419$, $\lVert g \rVert^2 = 0.0000047$ and $2 \lVert g_A \rVert \lVert g_R \rVert = 0.0264856$, so

$$ \cos(g_A, g_R) = \frac{0.0000047 - 0.0132437 - 0.0132419}{0.0264856} = -0.99982 $$

The angle is $178.9°$. Theorem 11 (c) gives the lower bound $\lvert \lVert g_A \rVert - \lVert g_R \rVert \rvert / (\lVert g_A \rVert + \lVert g_R \rVert) = 3.4 \cdot 10^{-5}$ for $\rho$, and the observed $\rho$ is $0.0094$: the two term gradients are equal in length to five digits and almost exactly opposite. The norms are logged to seven significant digits, so the rounding error in the cosine is below $10^{-6}$.

## Theorem 14 (micro recall splits by scope)

[Theorem 13](2026-11-07-contrastive-citeseer.md#theorem-13-no-cross-partition-reconstruction-caps-recall-at-1---delta) assumed that no reconstructed edge crosses a partition. On AstroPh a few do, so here is the general form.

**Assumptions:**

- The true edges split as $E = E_{\text{within}} \sqcup E_{\text{cut}}$, a disjoint union with $w = \lvert E_{\text{within}} \rvert > 0$ and $c = \lvert E_{\text{cut}} \rvert > 0$.
- $\text{TP}_w = \lvert \hat E \cap E_{\text{within}} \rvert$ and $\text{TP}_c = \lvert \hat E \cap E_{\text{cut}} \rvert$, so that $\text{TP} = \text{TP}_w + \text{TP}_c$.
- The within recall is $r_w = \text{TP}_w / w$, the cut recall is $r_c = \text{TP}_c / c$, and $\delta = c / (w + c)$.
- For (b): $\text{TP}_w \leq w$.

**Statement:**

- (a) $\dfrac{\text{TP}_w + \text{TP}_c}{w + c} = (1 - \delta) \, r_w + \delta \, r_c$.
- (b) $\dfrac{\text{TP}_w + \text{TP}_c}{w + c} \leq (1 - \delta) + \delta \, r_c$.

**Proof:**

1. (a) $\frac{w}{w + c} \cdot \frac{\text{TP}_w}{w} = \frac{\text{TP}_w}{w + c}$ because $w > 0$, and likewise $\frac{c}{w + c} \cdot \frac{\text{TP}_c}{c} = \frac{\text{TP}_c}{w + c}$ because $c > 0$. Add the two, and note $\frac{w}{w + c} = 1 - \delta$.
2. (b) $\text{TP}_w \leq w$ and $w > 0$ give $r_w \leq 1$. The weight $1 - \delta = w / (w + c)$ is non-negative, so $(1 - \delta) r_w \leq 1 - \delta$. Substitute into (a). $\blacksquare$

Lean: `recall_split_by_scope`, `recall_le_within_share_add_cut_recall` in section `CutRecall`.

**Consequence:** with $r_c = 0$, (b) is Theorem 13. A cut recall of $r_c$ raises the ceiling by only $\delta \, r_c$, so when $\delta$ is small, cut recall barely moves the global recall even if alignment is perfect.

**Example:** on AstroPh $w = 162074$, $c = 16225$, $\text{TP}_w = 60119$ and $\text{TP}_c = 6$, so $\delta = 0.0910$, $r_w = 0.3709$ and $r_c = 0.00037$. By (a) the recall is $0.9090 \cdot 0.3709 + 0.0910 \cdot 0.00037 = 0.3372$, matching `Weighted F1: recall=0.337267`. The cut part contributes $3.4 \cdot 10^{-5}$ of it. By (b) the ceiling is $0.9090$.

## Downstream metrics

**Graph reconstruction ($k = \lvert E \rvert = 178299$ closest pairs).** Edge counts are pair count times logged probability.

| Scope | Pairs | True edges | Reconstructed | True positives | Precision | Recall |
|---|---|---|---|---|---|---|
| Within partitions | 81 769 465 | 162 074 | 178 106 | 60 119 | 0.338 | 0.371 |
| Cross partitions | 91 647 911 | 16 225 | 193 | 6 | 0.031 | 0.00037 |
| All | 173 417 376 | 178 299 | 178 299 | 60 125 | 0.337 | 0.337 |

- **Cut pairs are practically absent from the reconstruction.** Of the $178299$ closest pairs only $193$ cross a partition, $0.11\%$, and $6$ of those are true edges. A random cut pair is a true edge with probability $0.000177$, so the cross precision of $0.031$ is $176\times$ chance, but it applies to only 193 pairs, and $16219$ of the $16225$ cut edges are missed. Training on all $16225$ cut edges did not make any substantial number of them nearest pairs.
- The macro view agrees: cross recall $0.00003 \pm 0.0019$ against within recall $0.427 \pm 0.413$, and $99.96\%$ of each node's reconstructed edges stay inside its partition (`reconstructed-edge partition share`).
- $13103$ of the $18624$ nodes with true neighbours get at least one reconstructed neighbour, so $5521$ nodes get none. The global macro F1 is $0.409$.

**Per partition.** The reconstruction is global, so the partitions compete for the $k$ reconstructed pairs.

| Partition | $\lvert V_p \rvert$ | True edges | Reconstructed | Reconstructed / true | True positives | Recall | Pair F1 |
|---|---|---|---|---|---|---|---|
| 0 | 12216 | 127 970 | 90 916 | 0.71 | 30 445 | 0.238 | 0.278 |
| 1 | 2186 | 10 945 | 36 588 | 3.34 | 9 690 | 0.885 | 0.408 |
| 2 | 2185 | 11 500 | 19 352 | 1.68 | 9 738 | 0.847 | 0.631 |
| 3 | 2185 | 11 659 | 31 250 | 2.68 | 10 246 | 0.879 | 0.478 |

- The three small partitions take $1.7$ to $3.3$ times as many reconstructed pairs as they have true edges and recover $85$ to $89\%$ of their edges. The giant partition gets $0.71$ times its share and recovers $23.8\%$. In the common space, pairs inside a small partition are closer than pairs inside the giant one, so the global top-$k$ favours the small partitions. This comes from the unbalanced partition sizes, not from the loss.
- Partition 0 carries $91.2\%$ of the within pairs (`weight=0.912434`), so its recall of $0.238$ dominates $r_w = 0.371$.

**Link prediction (HeaRT, 500 negatives per positive, bilinear decoder).** `heart_benchmark` samples $\min(\lvert \text{positives} \rvert, 1000)$ positives, so each row is measured on $1000$ sampled test edges. On CITESEER every row used all its test edges. Chance levels are those of the CITESEER section: MRR $H_{501}/501 = 0.0136$, mean rank $251$, Hits@$k$ $= k/501$.

| Edges | Count (sampled) | MRR | Mean rank | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|
| All test edges | 19811 (1000) | 0.3427 | 31.0 | 0.228 | 0.375 | 0.578 |
| Cross-partition | 2206 (1000) | 0.1545 | 111.1 | 0.080 | 0.164 | 0.302 |
| Within-partition | 17605 (1000) | 0.3328 | 22.0 | 0.219 | 0.363 | 0.557 |
| Chance | | 0.0136 | 251 | 0.002 | 0.006 | 0.020 |
| Earlier AstroPh $P = 4$ bilinear LPA, 10 runs ([link-prediction evaluation](../../E09-link-prediction/README.md#bilinear-decoder)) | | 0.452 ± 0.022 | | 0.326 ± 0.023 | 0.517 ± 0.027 | 0.697 ± 0.019 |

- Cross-partition MRR is $11.4\times$ chance, so the decoder does find cross-partition signal. Within MRR is only $2.2\times$ the cross MRR. On CITESEER that ratio was $4.0\times$.
- Test edges are cut slightly more often than training edges, $2206 / 19811 = 11.1\%$ against $\delta = 9.1\%$. The bias has the same direction as on CITESEER ($63.8\%$ against $21.9\%$) but is much weaker.
- **Against the earlier 10-run table**, most likely the no-alignment baseline (see the CITESEER section), every ranking metric of this run is far lower: MRR by $5.0$ standard deviations of that table, Hits@1 by $4.3$, Hits@3 by $5.3$ and Hits@10 by $6.2$. That is outside run-to-run variation, even allowing for the fact that this is one seed against a historical table. On CITESEER the same comparison was mixed (MRR $+1.3$ sd).
- The thresholded `Precision`/`Recall`/`F1`/`Accuracy` lines for "cross partitions" and "within partitions" are again cached copies of the full-model numbers ($0.8907 / 0.9083 / 0.8994 / 0.8984$). This is the `lp_full_eval` key collision from the CITESEER section, and those lines are not valid.

## Reading

1. **The tug-of-war ends in a near-exact standoff.** From epoch 50 on, the two term gradients are equal in length to within $2.1\%$ and point in opposite directions to within $2.2°$ (smallest angle $177.8°$). The net step keeps $0.9$ to $2.0\%$ of the term magnitudes ($\rho \in [0.009, 0.020]$). On CITESEER $\rho$ stayed at $0.05$ to $0.18$. Nothing vanishes in the sense of the [vanishing gradient problem](../../../doc/glossary.md#vanishing-gradient-problem), since both terms stay above $0.11$. The model is at a stationary point of $L$ where attraction and repulsion balance: the frustration of [Theorem 6](../../../doc/theory/partition-alignment.md#theorem-6-collapse-and-frustration-of-the-contrastive-loss), now realised almost exactly.
2. **Four phases.**
   - Epoch 1: attraction dominates by a factor of $86$ ($\lVert g_A \rVert / \lVert g_R \rVert$).
   - Epochs 2 to 5: contraction and overshoot. $R$ peaks at epoch 4, rising $93$-fold from $0.0021$ to $0.1945$: a large share of random pairs now have cosine above $1 - m^2/2 = 0.5$ ([Theorem 5](../../../doc/theory/partition-alignment.md#theorem-5-geometry-of-the-unit-sphere) (c)). $A$ reaches its minimum $0.090$ at epoch 5.
   - Epochs 6 to 10: rebound. Repulsion pushes the embedding apart again, $A$ rises to $0.171$ and $\bar c$ falls from $0.955$ to $0.914$.
   - Epochs 11 to 100: a slow joint descent into the standoff. $L$ falls from $0.23$ to $0.167$. $A$ never returns to its epoch-5 value and ends at $0.108$ ($\bar c = 0.946$).
3. **The plateau is much higher than on CITESEER** ($L = 0.167$ against $0.08$). The positive pairs end further apart ($\bar c = 0.946$ against $0.973$), and the repulsion stays higher ($R = 0.058$ against $0.031$). Hypothesis, not tested by this log: AstroPh's neighbourhoods are larger and more clustered (average clustering $0.51$ in the giant partition), so pulling them into tight blobs drags more random pairs inside the margin. That is the frustration mechanism of Theorem 6 at a larger scale.
4. **Using all edges did not create alignment.** The aligner saw every one of the $16225$ cut edges, yet only $6$ of them are reconstructed. By Theorem 4 the $162074$ intra pairs, $91\%$ of the attraction term, cannot align partitions, and they outweigh the cut pairs $10$ to $1$ in the gradient.
5. **Link prediction is clearly worse than the baseline, and within edges suffer too.** Within-partition MRR is $0.333$, and $88.9\%$ of the test edges are within partitions. The baseline table has no within/cross split, but its overall MRR of $0.452$ is above this run's within MRR, so this run cannot reach the baseline even on within edges alone. The contrastive geometry does not preserve node2vec geometry inside partitions (Theorem 6, consequence: blobs instead of node2vec neighbourhoods), and the bilinear decoder pays for it. Cross-partition MRR is $11\times$ chance, so some cross signal is there, but it does not make up for the within loss.
6. **Partition imbalance dominates reconstruction** independently of alignment: three small partitions are over-reconstructed by up to $3.3\times$ and the giant one has recall $0.238$.
7. **No validation evidence**, as on CITESEER. See [validation loss](../../../doc/glossary.md#validation-loss).

## Comparison with CITESEER

| Quantity | CITESEER | AstroPh |
|---|---|---|
| $\delta$ (training edges) | 0.219 | 0.091 |
| Cut share of test edges | 0.638 | 0.111 |
| Share of edges seen by the aligner | 1.000 | 1.000 |
| Partition sizes | balanced, $4 \times 816$ | $12216$ and $3 \times \approx 2185$ |
| $\bar c$ at epoch 1 / 100 | 0.789 / 0.973 | 0.767 / 0.946 |
| Epoch of minimum $A$ | 98 (near the end) | 5 (then $A$ rises again) |
| $\max \cos(g_A, g_R)$ after the overshoot | $-0.953$ (epochs 3 to 100) | $-0.952$ (epochs 2 to 100); $-0.99927$ from epoch 50 |
| $\rho$ range, epochs 30 to 100 | $[0.054, 0.177]$ | $[0.009, 0.056]$ |
| $\min \lVert g \rVert$ | $9.5 \cdot 10^{-3}$ | $2.1 \cdot 10^{-3}$ |
| $L$ at the end | noisy plateau, $[0.079, 0.084]$ | slow plateau, $0.1692 \to 0.1666$ over 15 epochs |
| Reconstructed cut pairs / true among them | 0 / 0 | 193 / 6 |
| Cut recall $r_c$ | 0 | 0.00037 |
| HeaRT MRR all / cross / within | 0.147 / 0.072 / 0.291 | 0.343 / 0.155 / 0.333 |
| MRR against the 10-run baseline | $+1.3$ sd | $-5.0$ sd |

Across both datasets, contrastive alignment ends with the two gradients pointing in nearly opposite directions. In neither run does it make cut pairs competitive with within pairs in the reconstruction. On AstroPh it also clearly hurts link prediction.

## Next runs (AstroPh specific)

1. Rerun with `VERTEX_VOYAGE_ALIGNMENT_VALIDATION=true` to see whether the standoff level of $L$ is also the validation optimum.
2. Train the attraction on $E^+_{\text{cut}}$ only, or weight it by $1/\delta \approx 11$, so that the alignment signal is not outweighed $10$ to $1$ by intra pairs. Log the attraction split into intra and cut parts.
3. Lower the margin ($m = 0.5$) or sample fewer negatives per positive, and check whether $\rho$ stays above $0.05$ and $\bar c$ stops falling after epoch 5.
4. Use a balanced partitioner (for example [balanced label propagation](../../../doc/glossary.md#balanced-label-propagation)), or normalise the reconstruction per partition, to separate the size effect from the alignment effect.
5. Repeat with no alignment and with Procrustes on the same seed and partitioning, so that the $-5.0$ sd MRR gap is measured on matched runs and not against a historical table.
6. Make `alignment_link_memory` default to unbounded (or to $\lvert E \rvert$) for static partitions, and log a warning whenever links are dropped from the deque, so that a too-small memory cannot silently truncate training again.

