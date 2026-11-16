# Uniform triplet loss on AstroPh analysis

Source: one run of `vv test` on AstroPh with the triplet aggregator, logged in `VertexVoyage/vv.log` (2026-11-07, 20:12 to 20:26). The log also still contains the CITESEER run of the previous section (19:49 to 19:50), and only the AstroPh part is used here. The settings are those of [Cross-weighted triplet loss on AstroPh analysis](2026-11-07-cross-weighted-triplet-astroph.md#cross-weighted-triplet-loss-on-astroph-analysis), with one change: `alignment_link_weighting = uniform` (log: `Aligner link weighting uniform`). This is item 2 of that section's next runs. Together with [Uniform triplet loss on CITESEER analysis](2026-11-07-uniform-triplet-citeseer.md#uniform-triplet-loss-on-citeseer-analysis), it completes the comparison of loss and weighting on both datasets. As before, every run is a **single run with one seed**, on its own unseeded LPA partition.

## Setup

| Quantity | Triplet, uniform | Triplet, weighted | Contrastive, weighted | Contrastive, uniform |
|---|---|---|---|---|
| $\lvert V_p \rvert$ | $12481, 2097, 2097, 2097$ | $12277, 2165, 2165, 2165$ | $12429, 2115, 2114, 2114$ | $12216, 2186, 2185, 2185$ |
| $\lvert E_p \rvert$ | $129561, 10984, 11145, 10982$; sum $162672$ | sum $162579$ | sum $163514$ | sum $162074$ |
| $\lvert E_{\text{cut}} \rvert$, $\delta$ | $15627$, $0.0876$ | $15720$, $0.0882$ | $14785$, $0.0829$ | $16225$, $0.0910$ |
| Cross share of the objective | $\delta = 0.0876$ | $1/(2 - \delta) = 0.5231$ | $0.5216$ | $\delta = 0.0910$ |
| Share of nodes / within edges in partition 0 | $66.5\%$ / $79.6\%$ | $65.4\%$ / $79.5\%$ | $66.2\%$ / $80.4\%$ | $65.1\%$ / $79.0\%$ |
| node2vec partition F1 | $0.676, 0.629, 0.626, 0.636$ | $0.676, 0.561, 0.639, 0.581$ | $0.676, 0.607, 0.561, 0.610$ | $0.675, 0.535, 0.669, 0.565$ |
| Replica rows | $18642$ | $18608$ | $18643$ | $18627$ |
| HeaRT test edges cut | $2067/19811 = 10.43\%$ | $10.95\%$ | $10.16\%$ | $11.14\%$ |
| Decoder train / validation loss | $0.4191$ / $0.4193$ | $0.4295$ / $0.4300$ | not recorded | not recorded |

- The $356598 = 2 \cdot 178299$ logged pairs are every link in both directions, of which $2 \cdot 15627 = 31254$ are cut pairs, a share of exactly $\delta$.
- The small partitions have higher node2vec F1 ($0.626$ to $0.636$) than in the weighted triplet run ($0.561$ to $0.639$). The starting replicas are therefore at least as good here.

## What the uniform log can show

As on CITESEER, the uniform objective is logged as one `ranking` term, with no per-scope split. With $T_{\text{w}}, T_{\text{c}} \geq 0$ the unlogged scope means of the hinge, the decomposition of the previous section gives
$$ L = (1 - \delta)\, T_{\text{w}} + \delta\, T_{\text{c}}, \qquad T_{\text{w}} \leq \frac{L}{1 - \delta}, \qquad T_{\text{c}} \leq \frac{L}{\delta}. $$

- **Epoch-1 check.** The scope means of the weighted triplet run at epoch 1 ($T_{\text{w}} = 0.0547$, $T_{\text{c}} = 0.5916$) predict $0.9124 \cdot 0.0547 + 0.0876 \cdot 0.5916 = 0.1018$. The logged value is $0.1042$. Here too the unaligned hinges are much the same from partition to partition.
- **The within floor is lower.** At epoch 100, $T_{\text{w}} \leq 0.0375 / 0.9124 = 0.0411$. The weighted triplet run had $T_{\text{w}} \in [0.0431, 0.0445]$ from epoch 30 on, so with uniform weighting the within hinge ends **below** that floor. This is an inequality on logged values, not an estimate. The weighted run's floor is therefore not a property of AstroPh alone ([Theorem 18](2026-11-07-cross-weighted-triplet-astroph.md#theorem-18-a-triplet-with-a-correlated-negative)): part of it was held up by the cut pull.
- **The cut bound is uninformative.** $T_{\text{c}} \leq 0.0375 / 0.0876 = 0.43$ at epoch 100. Since $\delta$ is small, the loss says little about the cut pairs. The downstream metrics below say more.

## Training dynamics

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 15 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $L$ | 0.1042 | 0.0850 | 0.0796 | 0.0761 | 0.0714 | 0.0682 | 0.0634 | 0.0600 | 0.0544 | 0.0512 | 0.0467 | 0.0438 | 0.0422 | 0.0410 | 0.0397 | 0.0389 | 0.0379 | 0.0375 |
| $\lVert g \rVert$ | 0.0673 | 0.0341 | 0.0318 | 0.0351 | 0.0254 | 0.0175 | 0.0162 | 0.0142 | 0.0085 | 0.0065 | 0.0045 | 0.0027 | 0.0020 | 0.0029 | 0.0036 | 0.0021 | 0.0030 | 0.0052 |

- **The loss falls $64\%$**, from $0.1042$ to $0.0375$. It is below $0.1$ from epoch 2 and below $0.05$ from epoch 22. It is at its minimum at the last epoch and still falling: $-2.8\%$ over epochs 86 to 100 ($0.0386 \to 0.0375$).
- **The gradient becomes small.** The total norm falls from $0.067$ to a minimum of $1.9 \cdot 10^{-3}$ (epoch 57), with median $2.5 \cdot 10^{-3}$ over epochs 51 to 100. The weighted run had median $7.9 \cdot 10^{-3}$, $3.2\times$ higher, with the cut term giving half of it. With uniform weighting the cut pairs carry $8.8\%$ of the objective, and the within triplets, mostly satisfied, give little gradient. No vanishing-gradient warning fired; the threshold is $10^{-7}$.
- The log has a single term, so there is no within/cross angle. Whether the opposition of the two scopes seen in the weighted run ($\cos \approx -0.85$) also occurs here cannot be read from the log.

## Downstream metrics

**Graph reconstruction ($k = 178299$ closest pairs).**

| Quantity | Triplet, uniform | Triplet, weighted | Contrastive, weighted | Contrastive, uniform |
|---|---|---|---|---|
| Reconstructed cut pairs / true positives | 5 / 0 | 7326 / **1335** | 20017 / 481 | 193 / 6 |
| Cut recall $r_c$ | 0 | **0.0849** | 0.0325 | 0.00037 |
| Within true positives | 46569 of 162672 | 53145 of 162579 | 48935 of 163514 | **60119** of 162074 |
| Within recall $r_w$ | 0.286 | 0.327 | 0.299 | **0.371** |
| Within pair F1 | 0.273 | 0.319 | 0.304 | **0.353** |
| Micro recall | 0.261 | 0.306 | 0.277 | **0.337** |
| Weighted F1 | 0.362 | **0.416** | 0.357 | 0.410 |
| Global macro F1 | 0.489 | **0.533** | 0.383 | 0.409 |
| Nodes with a reconstructed neighbour | 14628 of 18637 | **16136** of 18604 | 12097 | 13103 of 18624 |

- **No cut edge is reconstructed.** Only $5$ of the $178299$ reconstructed pairs cross partitions, and none of them is a true edge. Without the weighting, AstroPh behaves like CITESEER: the triplet loss satisfies cut triplets at a moderate cosine ([Theorem 17](2026-11-07-cross-weighted-triplet-citeseer.md#theorem-17-a-triplet-with-an-orthogonal-negative-is-a-cosine-threshold)), and the cut pairs never reach the closest $k$. By [Theorem 13](2026-11-07-contrastive-citeseer.md#theorem-13-no-cross-partition-reconstruction-caps-recall-at-1---delta), the micro recall is then capped at $1 - \delta = 0.912$, and $0.261 = 0.9124 \cdot 0.2863$ is all within.
- **The within reconstruction is also worse than with the weighting** ($r_w$ $0.286$ against $0.327$). This is unlike CITESEER, where removing the weighting raised $r_w$ from $0.551$ to $0.659$. The difference is in partition 0:

| Partition | True edges | Reconstructed | Reconstructed / true | Recall | Precision | Pair F1 | Triplet weighted: reconstructed / true, recall |
|---|---|---|---|---|---|---|---|
| 0 | 129 561 | 19 571 | **0.15** | 0.116 | 0.769 | 0.202 | 0.26, 0.168 |
| 1 | 10 984 | 44 029 | 4.01 | 0.945 | 0.236 | 0.377 | 4.66, 0.927 |
| 2 | 11 145 | 59 132 | 5.31 | 0.953 | 0.180 | 0.302 | 2.67, 0.970 |
| 3 | 10 982 | 55 562 | 5.06 | 0.958 | 0.189 | 0.316 | 5.07, 0.930 |

- **The size bias is the strongest of all AstroPh runs.** Partition 0 holds $79.6\%$ of the within edges but gets only $11.0\%$ of the reconstructed pairs ($0.15$ of its true count). Its precision is high ($0.769$), so the pairs it gets are mostly right, but its recall is $0.116$. The three small partitions get $4.0$ to $5.3$ times their count, and recall above $0.94$ there cannot compensate, since together they hold only $20\%$ of the within edges. The small partitions are each internally tighter than the giant one relative to the global top-$k$, and the uniform objective, dominated by within pairs, tightens them further.

**Link prediction, HeaRT (500 negatives per positive, 1000 sampled positives per row, bilinear decoder).**

| Edges | **Triplet, uniform**: MRR / Hits@1 / Hits@3 / Hits@10 / mean rank | Triplet, weighted: MRR / Hits@1 / Hits@10 | Contrastive, weighted: MRR | Contrastive, uniform: MRR |
|---|---|---|---|---|
| All test edges | **0.556** / **0.433** / **0.620** / **0.798** / 18.5 | 0.513 / 0.394 / 0.725 | 0.383 | 0.343 |
| Cross-partition | 0.246 / 0.122 / 0.288 / 0.515 / 63.0 | **0.282** / **0.144** / **0.553** | 0.175 | 0.155 |
| Within-partition | **0.581** / **0.456** / **0.653** / **0.816** / 12.8 | 0.543 / 0.421 / 0.758 | 0.414 | 0.333 |
| Earlier $P = 4$ baseline, 10 runs | 0.452 ± 0.022 / 0.326 ± 0.023 / 0.517 ± 0.027 / 0.697 ± 0.019 | | | |

- **The best overall and within-partition ranking on AstroPh.** Against the 10-run baseline: MRR $+4.7$, Hits@1 $+4.7$, Hits@3 $+3.8$ and Hits@10 $+5.3$ standard deviations. The weighted triplet run was $+2.8$ on MRR.
- **The weighting traded within for cross.** Over the weighted triplet run, within MRR rose by $+0.039$ and cross MRR fell by $-0.036$. With $1000$ sampled positives per row, one MRR has a standard error of about $0.013$ (within) and $0.0095$ (cross, where Hits@1 is near $0.13$ and the reciprocal-rank standard deviation about $0.30$). The within gain is about $2.1$ standard errors of a difference and the cross loss about $2.7$. Both are suggestive, not established.
- **The overall gain is mostly the test mix.** About $90\%$ of the test edges are within, so the within gain dominates the overall row. Recomposing both runs at the weighted run's cut share of $10.95\%$: uniform $0.1095 \cdot 0.246 + 0.8905 \cdot 0.581 = 0.544$, weighted $0.1095 \cdot 0.282 + 0.8905 \cdot 0.543 = 0.514$. The corrected gain, $+0.030$, is about $1$ run-to-run standard deviation of a single-run difference ($0.031$).
- Cross-partition MRR is $18.1\times$ chance ($0.0136$), against $20.8\times$ with the weighting. The within/cross MRR ratio is $2.36$, against $1.92$: without the weighting, the gap between the scopes is wider again.

**Link prediction, thresholded.** The three evaluations share the $19811$ negatives. Each scope has $9420 \pm 2$ false positives.

| Scope | Positives | Recall | True positives | Precision | F1 | Accuracy |
|---|---|---|---|---|---|---|
| All | 19811 | 0.987 | 19546 | 0.675 | 0.801 | 0.756 |
| Cross partitions | 2067 | 0.912 | 1886 | 0.167 | 0.282 | 0.561 |
| Within partitions | 17744 | 0.995 | 17661 | 0.652 | 0.788 | 0.747 |

- $1886 + 17661 = 19547$, the full model's $19546$ up to rounding. The specificity is $1 - 9420/19811 = 0.525$, the same operating point as the weighted triplet run ($0.520$). On CITESEER, too, the weighting did not change the operating point. On both datasets it is a property of the triplet aggregator, not of the weighting. [Aligner start, training and output scale on AstroPh at $P = 1$ analysis](2026-11-08-aligner-start-astroph-p1.md#aligner-start-training-and-output-scale-on-astroph-at-p--1-analysis) later locates it in the triplet training, not in the unit output scale.

## Effect of the weighting on the triplet loss, both datasets

| Metric | CITESEER uniform | CITESEER weighted | AstroPh uniform | AstroPh weighted |
|---|---|---|---|---|
| $\delta$; cut share of the objective | 0.214; 0.214 | 0.206; 0.557 | 0.088; 0.088 | 0.088; 0.523 |
| Cut true positives in reconstruction | 0 | 0 | 0 | **1335** |
| Within recall $r_w$ | **0.659** | 0.551 | 0.286 | **0.327** |
| Global macro F1 | **0.671** | 0.596 | 0.489 | **0.533** |
| HeaRT MRR all (mix-corrected) | **0.205** | 0.163 | **0.544** | 0.514 |
| HeaRT MRR cross | 0.107 | 0.101 | 0.246 | **0.282** |
| HeaRT MRR within | **0.380** | 0.273 | **0.581** | 0.543 |
| Specificity at the threshold | 0.488 | 0.483 | 0.525 | 0.520 |

1. **Link prediction on within edges is better without the weighting on both datasets** (CITESEER $+0.107$, AstroPh $+0.039$). Giving the within triplets their natural share ($1 - \delta$) instead of $(1 - \delta)/(2 - \delta)$ helps the ordering inside partitions.
2. **On cut edges the datasets disagree.** On CITESEER the weighting changes cross MRR by $0.006$ (noise) and reconstructs no cut edges either way. On AstroPh, where it multiplies the cut share by $6.0$ instead of $2.6$, it raises cross MRR by $0.036$ and is the only setting that reconstructs cut edges ($1335$). The weighting matters for the cut pairs only when $\delta$ is small enough that, unweighted, they are a minor part of the objective.
3. **Reconstruction disagrees too.** On CITESEER, with balanced partitions, the uniform run reconstructs best. On AstroPh the weighted run does, because the uniform run has the strongest size bias against partition 0 (it gets $0.15$ of its true count, against $0.26$). With one run each, it is not yet clear whether this is an effect of the weighting or of the partition. The partitions are similar in size ($12481$ against $12277$ nodes in partition 0).
4. **Overall MRR prefers uniform on both datasets, and AstroPh's overall ranking is dominated by within edges** ($\approx 90\%$ of test edges). Mix-corrected, the AstroPh gain is about one run-to-run standard deviation; the CITESEER gain about $4$.
5. **The best AstroPh link prediction so far is uniform triplet** (MRR $0.556$, $+4.7$ sd over the baseline). The best AstroPh cut-edge recovery, both in reconstruction and in cross MRR, is weighted triplet. Neither setting wins on both, so the choice depends on whether cut edges matter for the task.

## Next runs

1. Repeat uniform and weighted triplet on AstroPh over at least 5 seeds with a fixed LPA seed. The AstroPh differences here (within $+0.039$, cross $-0.036$, overall $+0.030$ corrected) are each one to three standard errors, and the partitions differ between runs.
2. Try an intermediate weighting, for example a cut share between $\delta$ and $1/(2 - \delta)$ (a power $\delta^{-\beta}$ with $0 < \beta < 1$). On AstroPh the two extremes trade within against cross ranking.
3. Log the per-scope terms under uniform weighting (no-grad split, logging only), so that $T_{\text{w}}$ and $T_{\text{c}}$ are measured instead of bounded, and the within/cross gradient angle can be compared with the weighted run.
4. Normalise the reconstruction per partition. On AstroPh the size bias against partition 0 is the main limit on reconstruction in every run, and it is strongest here.
5. Swap the output scales, since every triplet run shares specificity $\approx 0.5$ and every contrastive run has $\approx 0.89$.

