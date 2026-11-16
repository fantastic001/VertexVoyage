# Mean against triplet on AstroPh at $P = 4$ analysis

Source: one run of `vv test` on AstroPh with $P = 4$ and the mean aggregator, logged in `VertexVoyage/vv.log` (2026-11-08, 03:45 to 04:08), together with R5 and R6 of [Aligner training on AstroPh at $P = 4$ analysis](2026-11-08-aligner-training-astroph-p4.md#aligner-training-on-astroph-at-p--4-analysis). It is item 1 of that section's next runs. The new run is **R7**. Each configuration is a **single run**.

## What the three runs compare

As in the previous section, the four LPA partitions are disjoint in nodes, and every node $u$ has one replica $z_{u,p(u)}$. At $P = 4$ the mean of one replica is that replica, so

| Run | Aggregated vector of $u$ |
|---|---|
| R7, mean | $z_{u,p(u)}$ |
| R6, triplet, 0 steps | $N(z_{u,p(u)})$ |
| R5, triplet, 100 steps | $W(u, p(u))$ after 100 triplet steps |

R7 → R6 changes only the norm, and R6 → R5 only the direction, by training. These are the same two steps as R4 → R3 → R1 at $P = 1$. The six runs R1 to R7 (without R2) form a matched $2 \times 3$ design: $P \in \{1, 4\}$ crossed with {mean, 0 steps, 100 steps}.

**Setup.** R7 logs `seed 42`, LPA seed 42, the partition sizes $12249, 2175, 2174, 2174$ with $129242, 10964, 10600, 11853$ within edges, and node2vec partition F1 $0.675, 0.571, 0.553, 0.632$. All of these equal R5 and R6 to all printed digits. It logs no aligner configuration, and its aggregation takes $0.11$ s against $2.38$ s (R6) and $5.58$ s (R5). The decoder train / validation loss is $0.2043$ / $0.2124$.

## Downstream metrics

**HeaRT link prediction** ($19811$ positive test edges, $2124$ of them cut, cut share $0.1072$). "Recomposed" is $0.1072 \cdot \text{cross} + 0.8928 \cdot \text{within}$, as in the previous section.

| Scope | R7 mean: MRR / Hits@1 / Hits@10 / mean rank | R6 0 steps: MRR | R5 100 steps: MRR | test3 mean, $P = 4$ |
|---|---|---|---|---|
| All, logged | 0.474 / 0.344 / 0.727 / 27.6 | 0.537 | **0.554** | $0.444 \pm 0.027$ |
| All, recomposed | 0.476 | 0.530 | **0.538** | – |
| Cross-partition | 0.047 / 0.004 / 0.120 / 126.0 | 0.023 | **0.251** | 0.040 |
| Within-partition | 0.527 / 0.395 / 0.770 / 14.1 | **0.591** | 0.573 | 0.503 |

- **R7 reproduces test3 mean** ($z = +1.1$; cross $0.047$ against $0.040$, within $0.527$ against $0.503$). So the matched triplet gain over mean can be compared with test3's.
- **Without alignment, cut edges are near chance either way**: $3.4\times$ chance with the norm (R7), $1.7\times$ without (R6). Hits@1 on cut edges is $0.004$ and $0.002$.
- **The unexpected result: the normalised vectors rank within edges far better than the raw ones.** Within MRR is $0.591$ for $N(z)$ against $0.527$ for $z$ ($+0.064$, about $3.5$ standard errors of a difference of $0.018$). Within Hits@1 rises from $0.395$ to $0.468$. At $P = 1$ the same step cost $0.017$.

**Decomposition of triplet's gain over mean at $P = 4$.** With $M_{\text{R}k}$ the MRR of run R$k$, adding and subtracting $M_{\text{R}6}$ gives
$$ \underbrace{M_{\text{R}5} - M_{\text{R}7}}_{\text{gain of triplet over mean}} = \underbrace{M_{\text{R}6} - M_{\text{R}7}}_{\text{unit normalisation}} + \underbrace{M_{\text{R}5} - M_{\text{R}6}}_{\text{triplet training}}. $$

| Term | All, logged | All, recomposed | Cross | Within |
|---|---|---|---|---|
| Unit normalisation, R6 − R7 | $+0.063$ ($79\%$) | $+0.054$ ($87\%$) | $-0.024$ | $+0.064$ |
| Triplet training, R5 − R6 | $+0.017$ ($21\%$) | $+0.008$ ($13\%$) | $+0.228$ | $-0.018$ |
| Gain of triplet over mean, R5 − R7 | $+0.080$ | $+0.062$ | $+0.204$ | $+0.046$ |

- **Most of triplet's overall gain over mean at $P = 4$ is the normalisation, not the alignment.** The training does the whole cut-edge gain ($+0.228$), but on AstroPh cut edges are $10.7\%$ of the test set, and the training costs within edges $0.018$.
- **Example.** Out of mean's $0.476$ (recomposed), normalising the node2vec vectors gives $0.530$, and 100 triplet steps then give $0.538$.
- **The matched gain is larger than test3's** ($+0.080$ against $+0.033$). A single-run difference of triplet and mean has a test3 standard deviation of $\sqrt{0.027^2 + 0.042^2} = 0.050$, so the excess $0.047$ is about one standard deviation.

**The $2 \times 3$ design.** MRR; at $P = 1$ every test edge is within, and the $P = 4$ "all" row is recomposed.

| | Mean | 0 steps | 100 steps | Normalisation | Training |
|---|---|---|---|---|---|
| $P = 1$ (R4, R3, R1) | **0.687** | 0.670 | 0.618 | $-0.017$ | $-0.052$ |
| $P = 4$, all (R7, R6, R5) | 0.476 | 0.530 | **0.538** | $+0.054$ | $+0.008$ |
| $P = 4$, within | 0.527 | **0.591** | 0.573 | $+0.064$ | $-0.018$ |
| $P = 4$, cross | 0.047 | 0.023 | **0.251** | $-0.024$ | $+0.228$ |

- **The normalisation changes sign with $P$**: it costs $0.017$ at $P = 1$ and gains $0.064$ within at $P = 4$.
- **The training changes sign too**, but only through the cut edges: it costs within ranking at both $P$ ($-0.052$, $-0.018$).

**Hypothesis for the sign of the normalisation, not tested here.** At $P = 1$ there is one node2vec model, so all norms are on one scale, and the norm carries information about the node (its walk frequency, which grows with degree). At $P = 4$ there are four models, trained on graphs of $12249$ and about $2175$ nodes, and nothing makes their norms comparable. The decoder is one bilinear form $x^\top W y$ for all partitions. Its logit scales with $\lVert x \rVert \lVert y \rVert$, so a partition whose model has larger norms gets larger logits for its non-edges too, and the HeaRT negatives of a node $u$ are scored against those of other partitions. Unit vectors put all partitions on one scale and lose the within-model information, which at $P = 1$ is worth $0.017$. This predicts that dividing each replica by the mean norm of its partition, $z_{u,p} / \overline{\lVert z \rVert}_p$, beats both mean and $N(z)$ at $P = 4$, and equals mean at $P = 1$, where it only rescales all vectors by one constant.

**Thresholded metrics and decoder.** As before, $\text{FP} = R \cdot n_+ (1/\text{precision} - 1)$ is the same in every scope: $1849$ in R7, against $2002$ (R6) and $9519$ (R5).

| Quantity | R7 mean | R6 0 steps | R5 100 steps |
|---|---|---|---|
| Precision / recall / F1, all | **0.905** / 0.886 / **0.895** | 0.897 / 0.875 / 0.886 | 0.673 / **0.987** / 0.800 |
| Cross: precision / recall | **0.345** / 0.459 | 0.251 / 0.316 | 0.170 / **0.915** |
| Specificity / accuracy | **0.907** / **0.896** | 0.899 / 0.887 | 0.520 / 0.753 |
| Decoder train / validation loss | **0.204** / **0.212** | 0.260 / 0.259 | 0.419 / 0.420 |

- **The decoder fits the raw vectors best and ranks them worst.** R7 has the lowest decoder loss and the best threshold metrics, but the lowest MRR. The decoder is trained against random negatives, while HeaRT ranks each positive against $500$ hard negatives of its own source node. The norm separates random pairs well and the hard negatives badly. This fits the hypothesis above, but does not prove it.
- The low specificity of the triplet aggregator is again the training's: both untrained aggregators are at $0.90$.

**Graph reconstruction** ($k = 178299$ closest pairs).

| Quantity | R7 mean | R6 0 steps | R5 100 steps |
|---|---|---|---|
| Macro precision / recall / F1 | 0.397 / 0.303 / 0.343 | 0.345 / 0.296 / 0.319 | **0.562** / **0.435** / **0.490** |
| S (within F1, micro over pairs) | 0.179 | 0.168 | **0.275** |
| Within true positives, $r_w$ | 30466, 0.187 | 28647, 0.176 | **46802**, **0.288** |
| Reconstructed cut pairs | 0 | 0 | 12 |
| Weighted F1, Jaccard | 0.238, 0.083 | 0.207, 0.078 | **0.364**, **0.132** |
| Nodes with a reconstructed neighbour | 10194 | 9219 | **14737** |

| Partition | True edges | R7: reconstructed / true, recall, precision | R6 | R5 |
|---|---|---|---|---|
| 0 | 129 242 | 0.054, 0.046, 0.858 | 0.03, 0.029, 0.892 | **0.15**, **0.119**, 0.770 |
| 1 | 10 964 | 5.43, 0.732, 0.135 | 5.49, 0.746, 0.136 | 4.41, **0.957**, 0.217 |
| 2 | 10 600 | 5.98, 0.720, 0.120 | 6.16, 0.735, 0.119 | 6.61, **0.933**, 0.141 |
| 3 | 11 853 | 4.09, 0.745, 0.182 | 4.10, 0.749, 0.183 | 3.37, **0.935**, 0.277 |

- **For reconstruction the normalisation costs a little** (macro F1 $-0.024$, against $-0.049$ at $P = 1$), and **the training gains a lot** ($+0.171$). Triplet's reconstruction gain over mean at $P = 4$ ($+0.147$) is the training's.
- **The size bias is already in the raw node2vec vectors.** In R7, partition 0 gets $0.054$ of its true count ($6972$ of $178299$ pairs, $3.9\%$). So the bias of R6 is not created by the normalisation, which only strengthens it ($0.054 \to 0.03$). It comes from the four node2vec models themselves. This weakens the size-bias hypothesis of the previous section in its cosine form; a per-partition scale is still the most direct candidate.

## Reading

1. **On AstroPh at $P = 4$, the matched gain of triplet over mean is $+0.080$ MRR ($+0.062$ recomposed), and four fifths of it is the unit normalisation.** The training gives the cut-edge gain ($+0.228$) and costs within ranking ($-0.018$).
2. **Triplet with 0 steps is a strong baseline.** It costs nothing to train ($2.4$ s), is within $0.008$ (recomposed) of triplet, and is the best of the three on within edges. Every comparison of an aligner against mean should include it.
3. **The norm helps at $P = 1$ and hurts at $P = 4$.** The hypothesis is that norms of separately trained models are on different scales, which one bilinear decoder cannot undo.
4. **The decoder loss and the threshold metrics prefer mean, the ranking prefers the normalised aggregators.** The decoder loss is not a proxy for link prediction quality.
5. **For reconstruction, the training is what helps, through the size bias against partition 0**, which is already present in the raw vectors.

## Next runs

1. **Separate the norm scale from the norm.** Add an output scale that divides each replica by the mean norm of its partition, $z_{u,p} / \overline{\lVert z \rVert}_p$, and run it at $P = 1$ and $P = 4$. The hypothesis predicts mean's MRR at $P = 1$ ($0.687$) and above $0.591$ within at $P = 4$. This needs code: the current scales are `unit` and `replica_norm`. As a cheap first check, log the mean and standard deviation of $\lVert z_{u,p} \rVert$ per partition.
2. **`replica_norm` for triplet at $P = 4$.** If the hypothesis holds, it should lose against `unit` at $P = 4$, the opposite of what was expected from $P = 1$.
3. **Check whether the normalisation explains test3 elsewhere.** Run mean and triplet with 0 steps at $P = 4$ on CITESEER, Enron and AS-Oregon. In test3, triplet beat mean on CITESEER and AS-Oregon. If 0 steps reaches most of that gain, the gains of the aligner in this page have to be restated against it.
4. **Make the training spend less on within edges** (anchor term $\lambda \lVert W - N(z) \rVert^2$, smaller margin, fewer steps) at $P = 4$, against R5 and R6. The aim is to keep the $+0.228$ cross and lose less than $0.018$ within. With item 1, the anchor could pull towards the partition-scaled vector instead of $N(z)$.
5. **Repeat R5, R6 and R7 with 5 seeds**, with R1, R3 and R4. Every difference above rests on one run; within MRR differences of $0.018$ are about one standard error.

