# Cross-weighted contrastive loss on CITESEER analysis

Source: one run of `vv test` on CITESEER, logged in `VertexVoyage/vv.log` (2026-11-07, 17:32 to 17:33). The settings are those of [Contrastive loss on CITESEER analysis](2026-11-07-contrastive-citeseer.md#contrastive-loss-on-citeseer-analysis) (the uniform run), with `alignment_link_weighting = inverse_cross_fraction`, as in [Cross-weighted contrastive loss on AstroPh analysis](2026-11-07-cross-weighted-contrastive-astroph.md#cross-weighted-contrastive-loss-on-astroph-analysis). This run also uses the fixed `lp_full_eval_cross` and `lp_full_eval_within` cache keys, so it is the first CITESEER run whose thresholded cross-partition and within-partition link-prediction metrics are valid. Both runs are **single runs with one seed**, and the unseeded label propagation gives them different partitions.

## Setup

| Quantity | This run (weighted) | Uniform run | Source line |
|---|---|---|---|
| $\lvert V_p \rvert$ | $816$ each | $816$ each | `Partition size` |
| $\lvert E_p \rvert$ | $831, 829, 760, 749$; sum $3169$ | $853, 815, 724, 798$; sum $3190$ | `Partition number of edges` |
| $\lvert E_{\text{cut}} \rvert$, $\delta$ | $914$, $0.2239$ | $893$, $0.2187$ | `cross-partition fraction 0.223855` |
| node2vec partition F1 | $0.351, 0.485, 0.359, 0.415$ | $0.400, 0.422, 0.418, 0.369$ | `Best achieved F1 score` |
| Links trained, replica rows | $4083$, $3123$ | $4083$, $3129$ | `Aligner trained on` |
| Scope shares | within $0.4370$, cross $0.5630$ | within $0.781$, cross $0.219$ (implicit) | term weights |
| HeaRT test edges, cross share | $453$, $290/453 = 64.0\%$ | $453$, $289/453 = 63.8\%$ | `total cross partition edges` |

**Theorem 15 on CITESEER:** $n = 4083$ and $c = 914$, so $\delta = 0.223855$. The cross share is $1/(2 - \delta) = 1/1.776145 = 0.563017$, matching the logged `weight 0.563017`. The cut pairs gain a factor of $0.5630/0.2239 = 2.52$, against $6.3$ on AstroPh. A large $\delta$ leaves less room for re-weighting: by Theorem 15 (b) the gain is $1/(\delta(2 - \delta))$, which falls towards $1$ as $\delta \to 1$.

The $9788 = 4 \times (3264 - 817)$ vocabulary warnings are again one per partition for each node outside it.

## Training dynamics

Notation as in the AstroPh weighted section: the gradient norms are of the weighted terms, $L$ is the weighted loss (not comparable with the uniform run), and $\bar c_{\text{w}}, \bar c_{\text{c}}$ are the mean cosines of within and cut pairs ([Theorem 12](2026-11-07-contrastive-citeseer.md#theorem-12-mean-positive-cosine-from-the-attraction-term)).

| Epoch | $L$ | $A_{\text{w}}$ | $A_{\text{c}}$ | $R_{\text{w}}$ | $R_{\text{c}}$ | $\lVert g_{A,\text{w}} \rVert$ | $\lVert g_{A,\text{c}} \rVert$ | $\lVert g_{R,\text{w}} \rVert$ | $\lVert g_{R,\text{c}} \rVert$ | $\lVert g \rVert$ | $\rho$ | $\bar c_{\text{w}}$ | $\bar c_{\text{c}}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9510 | 0.0494 | 1.6327 | 0.0107 | 0.0098 | 0.0291 | 1.0282 | 0.0098 | 0.0122 | 1.0312 | 0.955 | 0.975 | 0.184 |
| 2 | 0.5733 | 0.0321 | 0.9453 | 0.0279 | 0.0263 | 0.0212 | 0.7555 | 0.0479 | 0.0589 | 0.6705 | 0.759 | 0.984 | 0.527 |
| 3 | 0.4011 | 0.0174 | 0.4634 | 0.1328 | 0.1324 | 0.0113 | 0.3857 | 0.1364 | 0.1750 | 0.0985 | 0.139 | 0.991 | 0.768 |
| 4 | 0.4221 | 0.0097 | 0.2432 | 0.2780 | 0.2831 | 0.0058 | 0.1905 | 0.1445 | 0.1854 | 0.1391 | 0.264 | 0.995 | 0.878 |
| 5 | 0.4538 | 0.0068 | 0.1726 | 0.3545 | 0.3530 | 0.0037 | 0.1274 | 0.1326 | 0.1706 | 0.1755 | 0.404 | 0.997 | 0.914 |
| 6 | 0.4581 | 0.0056 | 0.1587 | 0.3683 | 0.3647 | 0.0029 | 0.1163 | 0.1270 | 0.1660 | 0.1769 | 0.429 | 0.997 | 0.921 |
| 8 | 0.4341 | 0.0051 | 0.2116 | 0.3147 | 0.3111 | 0.0025 | 0.1610 | 0.1239 | 0.1595 | 0.1342 | 0.300 | 0.997 | 0.894 |
| 10 | 0.4153 | 0.0055 | 0.3234 | 0.2255 | 0.2348 | 0.0026 | 0.2359 | 0.1124 | 0.1447 | 0.0849 | 0.171 | 0.997 | 0.838 |
| 15 | 0.3677 | 0.0082 | 0.3950 | 0.1450 | 0.1391 | 0.0040 | 0.2637 | 0.1040 | 0.1325 | 0.0616 | 0.122 | 0.996 | 0.802 |
| 20 | 0.3415 | 0.0099 | 0.3792 | 0.1237 | 0.1236 | 0.0048 | 0.2739 | 0.1126 | 0.1447 | 0.0484 | 0.090 | 0.995 | 0.810 |
| 30 | 0.2927 | 0.0093 | 0.3257 | 0.1078 | 0.1032 | 0.0041 | 0.2260 | 0.1031 | 0.1341 | 0.0347 | 0.074 | 0.995 | 0.837 |
| 40 | 0.2564 | 0.0088 | 0.2894 | 0.0922 | 0.0875 | 0.0037 | 0.1902 | 0.0883 | 0.1157 | 0.0268 | 0.067 | 0.996 | 0.855 |
| 50 | 0.2295 | 0.0086 | 0.2555 | 0.0817 | 0.0821 | 0.0035 | 0.1608 | 0.0745 | 0.1007 | 0.0242 | 0.071 | 0.996 | 0.872 |
| 60 | 0.2039 | 0.0088 | 0.2269 | 0.0711 | 0.0733 | 0.0035 | 0.1406 | 0.0656 | 0.0876 | 0.0206 | 0.069 | 0.996 | 0.887 |
| 70 | 0.1849 | 0.0092 | 0.2035 | 0.0674 | 0.0656 | 0.0036 | 0.1254 | 0.0579 | 0.0743 | 0.0212 | 0.081 | 0.995 | 0.898 |
| 80 | 0.1676 | 0.0095 | 0.1865 | 0.0631 | 0.0549 | 0.0037 | 0.1196 | 0.0515 | 0.0644 | 0.0233 | 0.097 | 0.995 | 0.907 |
| 90 | 0.1563 | 0.0098 | 0.1666 | 0.0564 | 0.0596 | 0.0038 | 0.1045 | 0.0496 | 0.0682 | 0.0220 | 0.097 | 0.995 | 0.917 |
| 100 | 0.1467 | 0.0101 | 0.1540 | 0.0540 | 0.0567 | 0.0040 | 0.1023 | 0.0451 | 0.0575 | 0.0277 | 0.133 | 0.995 | 0.923 |

Summary over all 100 epochs:

- **The within pairs start almost coincident.** $\bar c_{\text{w}} = 0.975$ at epoch 1, against $0.821$ on AstroPh. Node2vec on the small, sparse CITESEER partitions ($175$ to $190$ connected components each, average degree $1.84$ to $2.04$) already places neighbours nearly on top of each other. The within pairs end at $0.995$.
- **The uniform run was already dominated by cut pairs.** At epoch 1, $A_{\text{w}} = 0.049$ and $A_{\text{c}} = 1.633$. With uniform weights these mix into $(3169 \cdot 0.049 + 914 \cdot 1.633)/4083 = 0.404$, close to the uniform run's epoch-1 attraction of $0.422$ (on a different partition). The cut pairs, $22\%$ of the links, contributed about $90\%$ of the attraction even without weighting. Their squared distance is $33$ times that of within pairs.
- **The within attraction becomes negligible.** $\lVert g_{A,\text{w}} \rVert$ is between $24$ and $93$ times smaller than $\lVert g_{A,\text{c}} \rVert$ in every epoch (minimum ratio $24.4$ at epoch 98, maximum $92.9$ at epoch 11). Except at epoch 1, it is also the smallest term norm. The within pairs are held only by repulsion and by the skip connection of the aligner.
- **The cut pairs are pulled together, rebound, and are still converging.** $\bar c_{\text{c}}$ goes from $0.184$ at epoch 1 to $0.921$ at epoch 6, falls back to $0.802$ at epoch 15, and rises again to $0.923$ at epoch 100, where $A_{\text{c}}$ reaches its minimum.
- **A deep collapse.** $R$ peaks at $0.368$ at epoch 6, $34$ times its epoch-1 value of $0.0107$. On the uniform run the peak was $0.173$.
- **The two repulsion estimates agree less closely than on AstroPh.** From epoch 3 on they differ by at most $15.8\%$ (at epoch 51). The cut call samples only $914$ negatives, against $14785$ on AstroPh, so its estimate is noisier, as expected.
- **No standoff as tight as on AstroPh.** From epoch 20 on $\rho \in [0.061, 0.137]$, the minimum at epoch 56 and the maximum at epoch 98. The smallest total norm is $0.0169$ at epoch 88. No vanishing-gradient warning fired.
- **Not converged.** $L$ falls by $9.7\%$ over epochs 86 to 100 ($0.1624 \to 0.1467$), with its minimum at epoch 99.

## Downstream metrics

**Graph reconstruction ($k = \lvert E \rvert = 4083$ closest pairs).**

| Scope | Uniform run: reconstructed / true positives | Weighted run: reconstructed / true positives | Weighted precision | Weighted recall |
|---|---|---|---|---|
| Within partitions | 4083 / 1722 | 4052 / 1593 | 0.393 | 0.503 |
| Cross partitions | 0 / 0 | 31 / 3 | 0.097 | **0.0033** |
| All | 4083 / 1722 | 4083 / 1596 | 0.391 | 0.391 |

(The uniform within true positives, $1722$, are $p(\text{true positive}) \cdot \text{pairs} = 0.001295 \cdot 1330080$.)

- **Cut pairs barely enter the reconstruction:** $31$ cut pairs, $0.76\%$ of $k$, with $3$ true edges, so the cut recall is $0.0033$. That is $372\times$ chance precision, but tiny in absolute terms. Training brought the cut pairs to a mean cosine of $0.923$, and the within pairs sit at $0.995$. The reconstruction takes the $k$ closest pairs, so almost every within pair is still closer than almost every cut pair. On AstroPh the gap was $0.946$ against $0.906$, and $20017$ cut pairs entered.
- **Global reconstruction got worse.** By [Theorem 14](2026-11-07-contrastive-astroph.md#theorem-14-micro-recall-splits-by-scope) the micro recall is $(1 - \delta) r_w + \delta r_c = 0.7761 \cdot 0.5027 + 0.2239 \cdot 0.0033 = 0.3902 + 0.0007 = 0.391$. The uniform run had $1722/4083 = 0.422$. Within pair F1 fell from $0.474$ to $0.441$ and global macro F1 from $0.544$ to $0.536$. $2712$ of $3114$ nodes with true neighbours get a reconstructed neighbour.
- **Per partition** (all partitions have $816$ nodes, so there is no size bias here):

| Partition | True edges | Reconstructed | Reconstructed / true | Recall | Pair F1 | Uniform run pair F1 |
|---|---|---|---|---|---|---|
| 0 | 831 | 1457 | 1.75 | 0.505 | 0.367 | 0.413 |
| 1 | 829 | 990 | 1.19 | 0.560 | 0.510 | 0.489 |
| 2 | 760 | 880 | 1.16 | 0.470 | 0.435 | 0.511 |
| 3 | 749 | 725 | 0.97 | 0.470 | 0.478 | 0.507 |

  The mean pair F1 over partitions fell from $0.480$ to $0.448$. The partitions are not the same in both runs, so only the mean is comparable.

**Link prediction, HeaRT (500 negatives per positive; all $453$ test edges are used, because $453 < 1000$).**

| Edges | Uniform run: MRR / Hits@1 / Hits@3 / Hits@10 / mean rank | Weighted run: MRR / Hits@1 / Hits@3 / Hits@10 / mean rank |
|---|---|---|
| All (453) | 0.147 / 0.062 / 0.174 / 0.309 / 119.3 | **0.124** / 0.055 / 0.124 / 0.254 / 122.1 |
| Cross-partition (290; 289 in the uniform run) | 0.072 / 0.024 / 0.059 / 0.163 / 156.5 | **0.063** / 0.014 / 0.055 / 0.152 / 153.2 |
| Within-partition (163; 164 in the uniform run) | 0.291 / 0.146 / 0.335 / 0.579 / 53.3 | **0.215** / 0.110 / 0.239 / 0.423 / 68.8 |
| Earlier $P = 4$ baseline, 10 runs | 0.138 ± 0.007 / 0.081 ± 0.010 / 0.155 ± 0.010 / 0.245 ± 0.013 | |

- **Every ranking metric got worse** except the cross-partition mean rank: MRR by $-0.023$ overall, $-0.009$ cross and $-0.076$ within. The largest loss is on within-partition edges.
- Against the 10-run baseline (most likely no alignment), MRR moved from $+1.3$ to $-2.0$ standard deviations of that table, Hits@1 from $-1.9$ to $-2.6$, Hits@3 from $+1.9$ to $-3.1$, and Hits@10 from $+4.9$ to $+0.7$. The uniform run was above the baseline on three of four metrics; the weighted run is below it on three of four.
- **Noise:** the baseline's run-to-run standard deviation of MRR is $0.007$, so the difference of two single runs has standard deviation about $\sqrt 2 \cdot 0.007 = 0.010$. The overall MRR loss of $0.023$ is about $2.3$ such standard deviations. This is suggestive, not established.

**Link prediction, thresholded (now valid).** All three evaluations score the same $453$ negative edges, so they share one false-positive count. From the logged numbers:

| Scope | Positives | Recall (true positive rate) | True positives | Precision | False positives | Accuracy |
|---|---|---|---|---|---|---|
| All | 453 | 0.587 | 266 | 0.804 | 65 | 0.722 |
| Cross partitions | 290 | 0.472 | 137 | 0.678 | 65 | 0.707 |
| Within partitions | 163 | 0.791 | 129 | 0.665 | 65 | 0.839 |

- The true positives add up, $137 + 129 = 266$, and each scope's false-positive count, $\text{TP}/\text{precision} - \text{TP}$, is $65$. That gives a specificity of $388/453 = 0.857$.
- By [Theorem 14](2026-11-07-contrastive-astroph.md#theorem-14-micro-recall-splits-by-scope) (a) applied to the test edges, the full recall is $\tfrac{290}{453} \cdot 0.472 + \tfrac{163}{453} \cdot 0.791 = 0.587$, matching the log.
- **Read only the recall per scope.** Precision and accuracy of a scope combine its own true positives with the false positives of all $453$ negatives, so they depend on how many positives the scope has. The within precision ($0.665$) is below the cross precision ($0.678$) only because the within scope has fewer positives, not because within predictions are worse.
- At the decoder's threshold, a held-out cut edge is detected with probability $0.472$ and a within edge with probability $0.791$. This is the first direct, valid measurement of the cross-partition gap for thresholded prediction. There is no uniform-run counterpart, because those lines were cache copies.
- Full-model F1 is $0.679$, against $0.684$ in the uniform run, a difference within noise.

## Is there an improvement?

| Criterion | Uniform | Weighted | Verdict |
|---|---|---|---|
| Cut pairs aligned in training ($\bar c_{\text{c}}$ at epoch 100) | not logged | 0.923, from 0.184 | the weighting aligns cut pairs |
| Cut pairs in reconstruction (recall) | 0 | 0.0033 | marginally better, still negligible |
| Global micro recall | 0.422 | 0.391 | **worse** |
| Mean partition pair F1 | 0.480 | 0.448 | **worse** |
| HeaRT MRR all / cross / within | 0.147 / 0.072 / 0.291 | 0.124 / 0.063 / 0.215 | **worse** on every row |
| MRR against the no-alignment baseline | $+1.3$ sd | $-2.0$ sd | **worse** |

**No.** On CITESEER, weighting cut edges by $1/\delta$ aligns the cut pairs in the training loss, but every downstream metric is worse or unchanged. The within-partition link prediction loses the most. This is the opposite of AstroPh.

## Why CITESEER and AstroPh respond differently

| Quantity | CITESEER | AstroPh |
|---|---|---|
| $\delta$ | 0.224 | 0.083 |
| Cut gain from the weighting, $1/(\delta(2 - \delta))$ | 2.5 | 6.3 |
| $\bar c_{\text{w}}$ at epoch 1 / 100 | 0.975 / 0.995 | 0.821 / 0.946 |
| $\bar c_{\text{c}}$ at epoch 1 / 100 | 0.184 / 0.923 | 0.094 / 0.906 |
| Within / cut attraction gradient, epoch 100 | 1 : 25 | 1 : 1.4 |
| Peak repulsion $R$ (weighted / uniform run) | 0.368 / 0.173 | 0.368 / 0.195 |
| Cut pairs in reconstruction | 31 | 20017 |
| $\Delta$ MRR all / within (weighted − uniform) | $-0.023$ / $-0.076$ | $+0.040$ / $+0.081$ |
| $\Delta$ micro recall | $-0.031$ | $-0.060$ |

Hypotheses consistent with these numbers (none of them is tested by a controlled run):

1. **CITESEER's within pairs leave no room.** They start at cosine $0.975$, so the within attraction has almost no gradient, and the weighted objective is almost entirely cut attraction against repulsion. The deeper contraction at epochs 4 to 8 (repulsion peak $0.368$, twice the uniform run) then squeezes within neighbourhoods further, to $0.995$. That costs the decoder resolution among within pairs, which is consistent with the within MRR falling from $0.291$ to $0.215$. On AstroPh the within pairs keep a real attraction gradient (ratio $1 : 1.4$) and end at $0.946$, so their geometry is preserved.
2. **Reconstruction is a ranking, so cut pairs enter only if they beat within pairs.** On CITESEER the gap $0.995 - 0.923 = 0.072$ keeps almost all cut pairs out. On AstroPh the gap is $0.946 - 0.906 = 0.040$ on a much larger graph, and thousands enter, mostly wrong ones.
3. **The weighting matters less when $\delta$ is large.** On CITESEER the cut pairs already contributed about $90\%$ of the uniform attraction, so the $2.5\times$ re-weighting mainly strengthened the collapse rather than adding signal that was missing.

In both datasets, global reconstruction got worse with the weighting. The link-prediction effect has opposite signs, each from one run against one run. **The weighting should not stay the default** without the repeated, matched runs below.

## Next runs

1. Repeat uniform and weighted over at least 5 seeds with a fixed LPA seed on both datasets, so that the opposite-sign MRR effects get error bars.
2. Try a softer weighting: cut weight $\delta^{-\beta}$ with $\beta \in \{0.25, 0.5\}$ ($\beta = 1$ is this run, $\beta = 0$ is uniform). On CITESEER the cut pairs already dominate the uniform attraction.
3. Reduce the collapse directly: fewer negatives per positive, a smaller margin $m$, or a geometry-preserving term on within pairs. Then check whether $\bar c_{\text{w}}$ stays below about $0.98$ and the within MRR recovers.
4. Rank cut pairs separately in the reconstruction (per-scope top-$k$, for example $k_{\text{c}} = \delta k$), so that cut alignment is measured without competing against near-coincident within pairs.
5. Report the thresholded per-scope recall, now valid, for every future run, and leave out the per-scope precision and accuracy, which depend on the positive count of the scope.
6. Consider resetting the default `alignment_link_weighting` to `uniform` until item 1 is done.

