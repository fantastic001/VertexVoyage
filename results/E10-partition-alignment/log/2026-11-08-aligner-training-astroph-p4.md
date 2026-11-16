# Aligner training on AstroPh at $P = 4$ analysis

Source: two runs of `vv test` on AstroPh with $P = 4$ and the triplet aggregator, logged in `VertexVoyage/vv.log` (2026-11-08). The run names continue those of [Aligner start, training and output scale on AstroPh at $P = 1$ analysis](2026-11-08-aligner-start-astroph-p1.md#aligner-start-training-and-output-scale-on-astroph-at-p--1-analysis).

| Run | Aggregator | Time |
|---|---|---|
| R5 | triplet, identity start, 100 steps | 01:35 to 01:58 |
| R6 | triplet, identity start, 0 steps | 02:10 to 02:33 |

These runs are item 2 of that section's next runs: they measure how much of the triplet aggregator at $P = 4$ is alignment. Each configuration is a **single run**.

## What the two runs compare

Let $p(u)$ be the partition of node $u$, and $z_{u,p}$ the node2vec vector of $u$ trained on partition $p$. The four LPA partitions are disjoint in nodes: their sizes sum to $\lvert V \rvert = 18772$. So every node has at most one replica, and the aggregated vector of $u$ is

| Run | Aggregated vector of $u$ |
|---|---|
| R6, 0 steps | $N(z_{u,p(u)}) = z_{u,p(u)} / \lVert z_{u,p(u)} \rVert$ |
| R5, 100 steps | $W(u, p(u))$ after 100 triplet steps |

In R6 each partition stays in its own node2vec frame, and nothing relates the frames to each other. R6 is therefore the **unaligned control**, and R6 → R5 changes only the training, as R3 → R1 does at $P = 1$. With R1 and R3, the four runs form a $2 \times 2$ design: $P \in \{1, 4\}$ crossed with $\{0, 100\}$ steps.

**Example.** Take a cut edge $(u, v)$ with $p(u) = 0$ and $p(v) = 1$. In R6, the cosine of $N(z_{u,0})$ and $N(z_{v,1})$ compares coordinates of two independently trained models. By [Theorem 2](../../../doc/theory/partition-alignment.md#theorem-2-a-random-isometry-is-as-far-as-a-random-direction), it then carries no more information than the cosine with a random direction.

## Setup

| Quantity | R5 | R6 |
|---|---|---|
| Shared random state, LPA seed | 42, 42 | 42, 42 |
| $\lvert V_p \rvert$ | $12249, 2175, 2174, 2174$ | same |
| $\lvert E_p \rvert$ | $129242, 10964, 10600, 11853$; sum $162659$ | same |
| $\lvert E_{\text{cut}} \rvert$, $\delta$ | $15640$, $0.0877$ | same |
| Share of nodes / within edges in partition 0 | $65.3\%$ / $79.5\%$ | same |
| node2vec partition F1 | $0.675, 0.571, 0.553, 0.632$ | same |
| Aligner seed, `initialisation` | 1514671486, `identity` | 1514671486, `identity` |
| `train_steps` | 100 | 0 (`loss not computed`) |
| Links / pairs / replica rows | $178299$ / $356598$ / $18657$ | same |
| HeaRT test edges cut | $2124 / 19811 = 10.72\%$ | same |
| Aggregation time | $5.58$ s | $2.38$ s |
| Decoder train / validation loss | $0.4187$ / $0.4202$ | $0.2596$ / $0.2591$ |

- The other aligner settings are those of R1 to R3: triplet loss with margin $0.5$, uniform weighting, `effective_output_scale = unit`, `link_memory` $= 10^6$, `learning_rate` $= 0.01$, `negative_pool` $= 256$, `hidden_width_factor` $= 2$, no validation split.
- The partition and the node2vec models are identical in both runs (equal sizes and F1 to all printed digits). **R5 and R6 are the first matched pair at $P > 1$ in this page.**
- $\delta = 15640 / 178299 = 0.0877$ is the cut share of the links, and $2 \cdot 15640 = 31280$ of the $356598$ pairs are cut pairs.

## Training dynamics

Each epoch is one optimiser step over all pairs, and the logged loss is computed before the step. The uniform-run row is the run of [Uniform triplet loss on AstroPh analysis](2026-11-07-uniform-triplet-astroph.md#uniform-triplet-loss-on-astroph-analysis): random start and another, unseeded partition.

| Epoch | 1 | 2 | 5 | 10 | 15 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| $L$, R5 ($P = 4$, identity) | 0.0914 | 0.0821 | 0.0698 | 0.0583 | 0.0539 | 0.0424 | 0.0381 |
| $\lVert g \rVert$, R5 | 0.0507 | 0.0251 | 0.0139 | 0.0103 | 0.0064 | 0.0025 | 0.0026 |
| $L$, uniform run ($P = 4$, random) | 0.1042 | 0.0850 | 0.0714 | 0.0600 | 0.0544 | 0.0422 | 0.0375 |
| $L$, R1 ($P = 1$, identity) | 0.0556 | 0.0420 | 0.0336 | 0.0303 | 0.0291 | 0.0271 | 0.0263 |

- **The loss falls $58\%$**, from $0.0914$ to $0.0381$, with its minimum $0.0379$ at epoch 98. Over epochs 86 to 100 it moves by $-2.3\%$ ($0.0390 \to 0.0381$).
- **The start again matters little.** The identity start begins $12\%$ below the random start of the uniform run ($0.0914$ against $0.1042$). At the tabulated epochs from 15 on, the two curves differ by at most $0.0006$. The runs differ in partition too, so this only supports the $P = 1$ result of R1 and R2.
- **The cut pairs raise the starting loss.** At the identity start, $L$ is $0.0914$ at $P = 4$ against $0.0556$ at $P = 1$. With the decomposition $L = (1 - \delta)\, T_{\text{w}} + \delta\, T_{\text{c}}$ of [Uniform triplet loss on AstroPh analysis](2026-11-07-uniform-triplet-astroph.md#uniform-triplet-loss-on-astroph-analysis), and *assuming* that the within mean $T_{\text{w}}$ at the start equals the $P = 1$ value $0.0556$, the cut mean is
$$ T_{\text{c}} \approx \frac{0.0914 - 0.9123 \cdot 0.0556}{0.0877} = 0.46. $$
  The assumption is rough, since the small partitions have their own node2vec models. The weighted triplet run measured $T_{\text{c}} = 0.59$ at a random start.
- No vanishing-gradient warning fired (threshold $10^{-7}$).

## Downstream metrics

**HeaRT link prediction** ($19811$ positive test edges, $2124$ of them cut). The test3 columns are the 10-run AstroPh $P = 4$ groups of [Link prediction, all groups](2026-11-07-test3-ten-run-experiments.md#link-prediction-all-groups).

| Scope | R6, 0 steps: MRR / Hits@1 / Hits@10 / mean rank | R5, 100 steps: MRR / Hits@1 / Hits@10 / mean rank | R5 − R6, MRR | Uniform run: MRR | test3 mean | test3 triplet |
|---|---|---|---|---|---|---|
| All | 0.537 / 0.418 / 0.739 / 29.9 | **0.554** / 0.422 / **0.795** / **15.3** | $+0.017$ | 0.556 | $0.444 \pm 0.027$ | $0.477 \pm 0.042$ |
| Cross-partition | 0.023 / 0.002 / 0.050 / 169.0 | **0.251** / **0.119** / **0.533** / **60.6** | $+0.228$ | 0.246 | 0.040 | 0.129 |
| Within-partition | **0.591** / **0.468** / 0.804 / 12.5 | 0.573 / 0.440 / 0.806 / 12.1 | $-0.018$ | 0.581 | 0.503 | 0.535 |

- **Without training, cut edges are ranked at chance.** R6 has cross MRR $0.023$, $1.7\times$ the chance level $0.0136$, and cross Hits@10 $0.050$. This is what [Theorem 2](../../../doc/theory/partition-alignment.md#theorem-2-a-random-isometry-is-as-far-as-a-random-direction) predicts for unaligned frames.
- **The training aligns.** It raises cross MRR $11\times$, to $18\times$ chance, and cross Hits@10 from $0.050$ to $0.533$. This is far above the noise of one MRR (about $0.0095$ for cross, see [Uniform triplet loss on AstroPh analysis](2026-11-07-uniform-triplet-astroph.md#uniform-triplet-loss-on-astroph-analysis)).
- **The training costs within ranking**: $-0.018$ MRR and $-0.028$ Hits@1. One within MRR has a standard error of about $0.013$, so a difference of two has about $0.018$, and the loss is about one standard error. Within Hits@10 and mean rank do not move.
- **Overall, the two effects nearly cancel.** The logged overall row samples its own $1000$ positives. Recomposing it at the cut share $0.1072$ of the test edges gives
$$ \text{R5: } 0.1072 \cdot 0.251 + 0.8928 \cdot 0.573 = 0.538, \qquad \text{R6: } 0.1072 \cdot 0.023 + 0.8928 \cdot 0.591 = 0.530. $$
  The gain $+0.008$ is the cross gain $0.1072 \cdot 0.228 = +0.024$ minus the within loss $0.8928 \cdot 0.018 = 0.016$. **On AstroPh at $P = 4$, two thirds of what the alignment gains on cut edges is spent on within edges.**
- **R5 reproduces the uniform run** within $0.008$ in every scope, with another start and another partition.
- R6 is $+3.4$ sd above test3 mean ($0.444 \pm 0.027$). The comparison is not matched: test3 used older code and unseeded partitions, and mean keeps the node2vec norm. A mean run on this partition is needed (next runs). It was done later: mean scores $0.474$, and the normalisation, not the code, explains R6's lead.

**The $2 \times 2$ design.** MRR; at $P = 1$ every test edge is within.

| | 0 steps | 100 steps | Effect of the training |
|---|---|---|---|
| $P = 1$ (R3, R1) | 0.670 | 0.618 | $-0.052$ |
| $P = 4$, all (R6, R5) | 0.537 | 0.554 | $+0.017$ (recomposed $+0.008$) |
| $P = 4$, within | 0.591 | 0.573 | $-0.018$ |
| $P = 4$, cross | 0.023 | 0.251 | $+0.228$ |
| Partition loss, $P = 1$ all − $P = 4$ all | $0.133$ | $0.064$ | |

- **The training helps only where there are cut edges.** At $P = 1$ it costs $0.052$; at $P = 4$ it gains $0.017$ overall.
- **It costs within ranking less at $P = 4$** ($0.018$ against $0.052$). The $P = 4$ within edges are a subset ($17687$) of the $P = 1$ test edges, and their 0-step MRR is already lower ($0.591$ against $0.670$). With one run each, this is an observation, not a result.

**Thresholded metrics and decoder.** The three scopes share the $19811$ negatives, and $\text{FP} = R \cdot n_+ (1/\text{precision} - 1)$ is the same in every scope up to rounding: $2002$ in R6, $9519$ in R5. Specificity is $1 - \text{FP} / 19811$.

| Scope | R6: precision / recall / F1 | R5: precision / recall / F1 |
|---|---|---|
| All | **0.897** / 0.875 / **0.886** | 0.673 / **0.987** / 0.800 |
| Cross partitions | **0.251** / 0.316 / 0.280 | 0.170 / **0.915** / **0.286** |
| Within partitions | **0.893** / 0.942 / **0.917** | 0.649 / **0.996** / 0.786 |
| Specificity / accuracy | **0.899** / **0.887** | 0.520 / 0.753 |

- **Training moves the decoder to the triplet operating point at $P = 4$ too**: specificity $0.899 \to 0.520$, the value of R1, R2 and the uniform run ($0.525$). The location of the low specificity in the training, found at $P = 1$, holds at $P = 4$.
- **The unaligned frames already cost the decoder.** R6's specificity is below R3's ($0.899$ against $0.965$) and its decoder loss is twice R3's ($0.260$ against $0.127$). At the threshold it accepts only $31.6\%$ of the cut edges.
- **Threshold and ranking disagree.** R6 has the higher F1 and accuracy, R5 the higher MRR. The thresholded metrics measure the decoder's operating point, the MRR the ordering.

**Graph reconstruction** ($k = 178299$ closest pairs).

| Quantity | R6, 0 steps | R5, 100 steps | Uniform run |
|---|---|---|---|
| Macro precision / recall / F1 | 0.345 / 0.296 / 0.319 | **0.562** / **0.435** / **0.490** | – / – / 0.489 |
| S (within F1, micro over pairs) | 0.168 | **0.275** | 0.273 |
| Within true positives, $r_w$ | 28647, 0.176 | **46802**, **0.288** | 46569, 0.286 |
| Reconstructed cut pairs / true positives | 0 / 0 | 12 / 1 | 5 / 0 |
| Weighted F1, Jaccard | 0.207, 0.078 | **0.364**, **0.132** | 0.362, – |
| Nodes with a reconstructed neighbour | 9219 of 18655 | **14737** of 18655 | 14628 of 18637 |

| Partition | True edges | R6: reconstructed / true, recall, precision | R5: reconstructed / true, recall, precision |
|---|---|---|---|
| 0 | 129 242 | **0.03**, 0.029, 0.892 | 0.15, 0.119, 0.770 |
| 1 | 10 964 | 5.49, 0.746, 0.136 | 4.41, 0.957, 0.217 |
| 2 | 10 600 | 6.16, 0.735, 0.119 | 6.61, 0.933, 0.141 |
| 3 | 11 853 | 4.10, 0.749, 0.183 | 3.37, 0.935, 0.277 |

- **At $P = 4$ the training improves reconstruction**: macro F1 $+0.171$, S $+0.106$. At $P = 1$ it worsened both ($-0.060$ and $-0.167$).
- **Neither run reconstructs cut edges** (R5: $1$ of $15640$). By [Theorem 13](2026-11-07-contrastive-citeseer.md#theorem-13-no-cross-partition-reconstruction-caps-recall-at-1---delta), the micro recall is capped at $1 - \delta = 0.912$. The micro recall equals $r_w \cdot 162659 / 178299$: $0.161$ (R6), $0.262$ (R5).
- **The gain is in the size bias.** In R6, partition 0 holds $79.5\%$ of the within edges but gets $4251$ of the $178299$ pairs ($2.4\%$), $0.03$ of its true count. The three small partitions take the other $97.6\%$. Training raises partition 0 to $19910$ pairs ($0.15$ of its true count), and the recall of the small partitions from $0.74$ to $0.93$–$0.96$.
- **Hypothesis, not tested here.** Without alignment each partition keeps its own cosine scale, and node2vec on a small partition puts neighbours at higher cosine than on partition 0. The global top $k$ is then filled from the small partitions. The triplet loss, with one margin for all partitions, may put the partitions on a more common scale. Logging the per-partition cosine of true edges would test this. [Mean against triplet on AstroPh at $P = 4$ analysis](2026-11-08-mean-vs-triplet-astroph-p4.md#mean-against-triplet-on-astroph-at-p--4-analysis) later shows that the bias is already in the raw node2vec vectors.

## Reading

1. **At $P = 4$, the triplet training is the alignment.** Without it cut edges are at chance (cross MRR $0.023$); with it they reach $18\times$ chance ($0.251$).
2. **The alignment is nearly paid for by within ranking on AstroPh.** Cross $+0.228$, within $-0.018$; mix-recomposed, the overall gain is $+0.008$ ($+0.017$ logged). With one run, the overall gain is not established.
3. **The low specificity comes from the training at both $P = 1$ and $P = 4$.**
4. **For reconstruction the training helps at $P = 4$, by reducing the bias against the large partition**, unlike at $P = 1$, where there is no such bias to reduce.
5. **The start does not matter at $P = 4$ either**, as far as the comparison with the uniform run can show.

## Next runs

1. **Mean at $P = 4$ on the same partition** (LPA seed 42). It measures the effect of the normalisation at $P = 4$ and completes a matched comparison of triplet against mean, which test3 could not give.
   - *Status:* done, see [Mean against triplet on AstroPh at $P = 4$ analysis](2026-11-08-mean-vs-triplet-astroph-p4.md#mean-against-triplet-on-astroph-at-p--4-analysis). The normalisation gives four fifths of triplet's gain over mean.
2. **The candidates for keeping the training close to the start** (anchor term, smaller margin, fewer steps), now at $P = 4$ against R5 and R6. The aim is to keep the cross gain of $0.228$ and lose less than $0.018$ within.
3. **Log the per-partition cosine of true edges** for R5 and R6, to test the size-bias hypothesis, or normalise the reconstruction per partition.
4. **Repeat R5 and R6 with 5 seeds**, together with R1, R3 and R4.

