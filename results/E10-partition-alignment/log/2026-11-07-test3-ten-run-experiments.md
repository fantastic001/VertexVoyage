# Earlier 10-run experiments (test3) analysis

Source: `VertexVoyage/test3/`, one directory per (dataset, $P$, aggregator) group, each with $10$ runs (`0.out` to `9.out`) of `vv test`, run between 2026-10-14 and 2026-10-23. `VertexVoyage/spec.yml` groups them, and `VertexVoyage/note.md` has the CITESEER MRR table. All numbers below were extracted again from the `.out` files and agree with `note.md` where it reports them. The four `CITESEER-*-bilinear_new` directories are empty and are ignored.

## What was run

| Dataset | $\lvert V \rvert$ / $\lvert E \rvert$ / test edges | $P$ | Aggregators (10 runs each) |
|---|---|---|---|
| CITESEER | $3264$ / $4083$ / $453$ | 1, 2, 4, 8 | mean, Procrustes, triplet, unlabelled `bilinear` |
| AstroPh | $18772$ / $178299$ / $19811$ | 1, 2, 4, 8 | mean, triplet, contrastive |
| Enron | $36692$ / $165448$ / $18383$ | 1, 2, 4, 8 | mean, triplet |
| AS-Oregon | $11806$ / $34903$ / $3878$ | 1, 2, 4, 8 | mean, triplet |

All runs use [label propagation](../../../doc/glossary.md#label-propagation) partitions with $RF = 1$, [node2vec](../../../doc/glossary.md#node2vec) per partition, the bilinear decoder and HeaRT with $500$ negatives. AS-Oregon appears as `AS-Oregon` (mean) and `AS_Oregon` (triplet); both have $34903$ edges and $3878$ test edges, so they are the same dataset.

**Aggregators.**

- *mean*: averages the replicas of a node. With $RF = 1$ every node has one replica, so mean is **no alignment**: the partition embeddings are used as they are.
- *Procrustes*: an [orthogonal Procrustes](../../../doc/glossary.md#orthogonal-procrustes-problem) map per partition (Theorem 8). It keeps norms.
- *triplet*, *contrastive*: the residual aligner of [Approach](../README.md#approach), with the losses of [Losses, term by term](../../../doc/theory/partition-alignment.md#losses-term-by-term). Triplet uses the unit output scale and contrastive the replica-norm one.
- `bilinear` (no suffix, CITESEER only): run on 2026-10-14 between 00:58 and 01:34, before commit `86f1dc1`. It is discussed separately below.

**Code version.** Apart from `bilinear`, every run started after commit `86f1dc1` (2026-10-14 13:29). The next commit in the repository is from 2026-10-24. Against the current code, the aligner differs in:

| Setting | test3 runs | Runs analysed in the sections above |
|---|---|---|
| `alignment_link_memory` | $20000$ (most recent links kept) | $1000000$ (AstroPh: `200000` in the first contrastive run, then the new default) |
| Link weighting | uniform (the only option) | uniform or `inverse_cross_fraction` |
| Epochs, learning rate, margin, hidden width | $100$, $0.01$, $\alpha = 0.5$, $2d$ | same |
| Per-scope thresholded lines | invalid (`lp_full_eval` cache bug) | valid after the fix |

$20000$ links cover all of CITESEER's $4083$ training edges, but only $57.3\%$ of AS-Oregon's $34903$, $11.2\%$ of AstroPh's $178299$ and $12.1\%$ of Enron's $165448$. The thresholded per-scope lines in every test3 file are copies of the full-model line, so only the full-model thresholded metrics are used.

**Statistics.** For each group, $\bar x \pm s$ is the mean and sample [standard deviation](../../../doc/glossary.md#standard-deviation) over the $10$ runs. Two groups $a$ and $b$ are compared with the Welch statistic $t = (\bar x_a - \bar x_b) / \sqrt{s_a^2/10 + s_b^2/10}$ (see [t-test](../../../doc/glossary.md#t-test)). With $10$ runs per group, $\lvert t \rvert > 2.1$ corresponds to $p < 0.05$ two-sided. Chance MRR is $H_{501}/501 = 0.0136$, as before.

## Link prediction, all groups

HeaRT MRR, mean ± standard deviation over 10 runs. "Cross" and "within" are the scope rows. At $P = 1$ there are no cut edges.

| Dataset | $P$ | mean: all / cross / within | triplet: all / cross / within | other |
|---|---|---|---|---|
| CITESEER | 1 | 0.263 ± 0.007 / – / 0.264 | 0.259 ± 0.016 / – / 0.262 | Procrustes 0.268 ± 0.016 |
| | 2 | 0.167 ± 0.018 / 0.025 / 0.285 | **0.273 ± 0.022** / 0.189 / 0.338 | Procrustes 0.208 ± 0.012 / 0.069 / 0.328 |
| | 4 | 0.122 ± 0.013 / 0.015 / 0.303 | **0.272 ± 0.018** / 0.179 / 0.418 | Procrustes 0.207 ± 0.015 / 0.092 / 0.398 |
| | 8 | 0.106 ± 0.007 / 0.015 / 0.329 | **0.282 ± 0.015** / 0.194 / 0.507 | Procrustes 0.217 ± 0.014 / 0.139 / 0.426 |
| AstroPh | 1 | **0.683 ± 0.011** / – / 0.680 | 0.539 ± 0.017 / – / 0.530 | contrastive 0.239 ± 0.013 |
| | 2 | 0.491 ± 0.021 / 0.069 / 0.548 | 0.488 ± 0.014 / 0.128 / 0.516 | contrastive 0.207 ± 0.014 |
| | 4 | 0.444 ± 0.027 / 0.040 / 0.503 | 0.477 ± 0.042 / 0.129 / 0.535 | contrastive 0.237 ± 0.011 / 0.108 / 0.259 |
| | 8 | 0.381 ± 0.019 / 0.030 / 0.431 | **0.495 ± 0.019** / 0.159 / 0.538 | contrastive 0.235 ± 0.011 |
| Enron | 1 | **0.517 ± 0.016** / – / 0.514 | 0.374 ± 0.017 / – / 0.370 | |
| | 2 | **0.375 ± 0.036** / 0.070 / 0.434 | 0.301 ± 0.010 / 0.076 / 0.335 | |
| | 4 | **0.350 ± 0.009** / 0.044 / 0.410 | 0.276 ± 0.016 / 0.098 / 0.315 | |
| | 8 | **0.323 ± 0.031** / 0.036 / 0.385 | 0.297 ± 0.018 / 0.119 / 0.343 | |
| AS-Oregon | 1 | **0.250 ± 0.010** / – / 0.252 | 0.214 ± 0.015 / – / 0.218 | |
| | 2 | 0.182 ± 0.011 / 0.073 / 0.226 | **0.217 ± 0.009** / 0.087 / 0.269 | |
| | 4 | 0.148 ± 0.011 / 0.043 / 0.242 | **0.212 ± 0.013** / 0.082 / 0.325 | |
| | 8 | 0.154 ± 0.016 / 0.031 / 0.277 | **0.218 ± 0.010** / 0.079 / 0.368 | |

Bold marks the better of mean and triplet when $\lvert t \rvert > 2.1$.

**Welch $t$ of triplet against mean:**

| Dataset | $P = 1$ | $P = 2$ | $P = 4$ | $P = 8$ | Cross MRR, $P = 2 / 4 / 8$ |
|---|---|---|---|---|---|
| CITESEER | $-0.003$ ($t = -0.6$) | $+0.106$ ($11.9$) | $+0.150$ ($21.7$) | $+0.175$ ($34.2$) | $t = 24.4 / 55.0 / 52.7$ |
| AstroPh | $-0.144$ ($-22.6$) | $-0.003$ ($-0.3$) | $+0.033$ ($2.1$) | $+0.114$ ($13.4$) | $t = 12.3 / 8.1 / 52.6$ |
| Enron | $-0.143$ ($-19.7$) | $-0.074$ ($-6.2$) | $-0.074$ ($-12.9$) | $-0.026$ ($-2.3$) | $t = 1.0 / 16.3 / 10.0$ |
| AS-Oregon | $-0.036$ ($-6.2$) | $+0.034$ ($7.7$) | $+0.065$ ($11.9$) | $+0.063$ ($10.4$) | $t = 3.1 / 14.1 / 25.2$ |

**Findings.**

1. **Without alignment, cut edges are ranked near chance and MRR falls with $P$.** The cross MRR of mean is $1.1\times$ chance on CITESEER at $P = 4$ and $8$, and $2.2$ to $5.4\times$ chance elsewhere, falling as $P$ grows on every dataset. This is what [Theorem 2](../../../doc/theory/partition-alignment.md#theorem-2-a-random-isometry-is-as-far-as-a-random-direction) predicts for independently trained partitions. Overall MRR falls with $P$ on every dataset. From $P = 1$ to $P = 8$: CITESEER $0.263 \to 0.106$, AstroPh $0.683 \to 0.381$, Enron $0.517 \to 0.323$, AS-Oregon $0.250 \to 0.154$.
2. **Triplet alignment is nearly flat in $P$.** From $P = 2$ to $P = 8$ its MRR moves by at most $0.025$ on every dataset (CITESEER $0.273 \to 0.282$, AS-Oregon $0.217 \to 0.218$). Its cross MRR is $13$ to $14\times$ chance on CITESEER and $6$ to $12\times$ on the other datasets, and it beats mean on cross MRR in every partitioned group except Enron $P = 2$ ($t = 1.0$).
3. **Overall, triplet beats no alignment on CITESEER and AS-Oregon for every $P \geq 2$, and on AstroPh from $P = 4$**, with the margin growing with $P$. On Enron it loses at every $P$. There the within MRR drop ($-0.04$ to $-0.10$) outweighs the cross gain, because $80$ to $87\%$ of Enron's test edges are within.
4. **Procrustes (CITESEER only) sits between mean and triplet**: $+0.04$ to $+0.11$ over mean, $-0.065$ under triplet at $P = 2, 4, 8$ ($\lvert t \rvert \geq 8.2$). Its cross MRR, $0.069$ to $0.139$, is half that of triplet. An orthogonal map can rotate a partition but cannot fix a partition whose geometry differs by more than an isometry (Theorem 1).
5. **Contrastive alignment (AstroPh only) is at $0.21$ to $0.24$ for every $P$, including $P = 1$.** It is far below mean and triplet everywhere ($t \geq 17.6$ against triplet). At $P = 1$ there is nothing to align, so the loss of $0.444$ against mean is damage done by the aligner itself. This is the collapse of [Theorem 6](../../../doc/theory/partition-alignment.md#theorem-6-collapse-and-frustration-of-the-contrastive-loss). These runs kept $20000$ links, so they are not the same configuration as the contrastive AstroPh sections above.

## The cost of the aligner at $P = 1$

At $P = 1$ every edge is within the single partition, and by [Theorem 4](../../../doc/theory/partition-alignment.md#theorem-4-the-alignment-signal-lives-in-cross-partition-pairs) there is no alignment signal. A learned aligner can only distort the node2vec geometry there. This gives a clean control.

**Definitions.** Let $M(a, P)$ be the mean MRR of aggregator $a$ with $P$ partitions. Take mean at $P = 1$ (one node2vec model on the whole graph) as the reference. The gap of aggregator $a$ at $P$ splits as
$$ \underbrace{M(\text{mean}, 1) - M(a, P)}_{\text{gap}} = \underbrace{M(\text{mean}, 1) - M(a, 1)}_{\text{aligner cost}} + \underbrace{M(a, 1) - M(a, P)}_{\text{partition loss}}. $$
This is an identity: the middle term $M(a, 1)$ is added and subtracted.

| Dataset | $M(\text{mean}, 1)$ | Triplet: aligner cost | Triplet: partition loss at $P = 4$ | Gap at $P = 4$ | Share of the gap that is aligner cost |
|---|---|---|---|---|---|
| CITESEER | 0.263 | 0.003 ($t = 0.6$) | $-0.013$ | $-0.009$ | – (no gap) |
| AS-Oregon | 0.250 | 0.036 | 0.002 | 0.038 | $95\%$ |
| AstroPh | 0.683 | 0.144 | 0.062 | 0.206 | $70\%$ |
| Enron | 0.517 | 0.143 | 0.098 | 0.241 | $59\%$ |
| AstroPh, contrastive | 0.683 | 0.444 | $0.003$ | 0.446 | $99.4\%$ |

- **On AstroPh, Enron and AS-Oregon, most of what triplet alignment loses against the unpartitioned model is lost already at $P = 1$**, where nothing needs aligning. Partitioning itself costs triplet only $0.002$ to $0.098$.
- **On CITESEER the aligner costs nothing**, and triplet at $P = 4$ and $8$ is even slightly above the unpartitioned mean ($0.272$ and $0.282$ against $0.263$; $t = 1.6$ and $3.7$).
- **Two explanations fit, and the runs cannot separate them.**
  - (a) *The aligner moves within pairs.* The within triplets of AstroPh have a floor of active triplets ([Theorem 18](2026-11-07-cross-weighted-triplet-astroph.md#theorem-18-a-triplet-with-a-correlated-negative)); those of CITESEER are satisfied from the start (Theorem 17). So the aligner moves AstroPh embeddings even at $P = 1$, and leaves CITESEER's alone.
  - (b) *The unit output scale removes the norm.* Under the triplet aggregator the decoder sees unit vectors. A node2vec norm can carry information, such as node frequency in the walks, which grows with degree. AstroPh, Enron and AS-Oregon have heavy-tailed degrees; CITESEER's degrees are small.
- The contrastive aggregator keeps the replica norm, and still costs $0.444$ at $P = 1$. So (b) is not needed to explain a large aligner cost, but it may still add to the triplet's.

## Decoder operating point

| Aggregator | Specificity at the threshold, range over datasets and $P$ | Decoder train loss, range |
|---|---|---|
| mean | 0.734 to 0.984 (falls with $P$) | 0.027 to 0.282 |
| Procrustes | 0.925 to 0.959 | 0.133 to 0.215 |
| triplet | 0.440 to 0.660 | 0.385 to 0.530 |
| contrastive | 0.509 to 0.537 | 0.475 to 0.510 |
| `bilinear` (CITESEER) | 0.511 to 0.550 | 0.223 to 0.356 |

Specificity is computed as in the sections above from the full-model precision and recall, $1 - \text{FP}/\lvert \text{negatives} \rvert$ with $\text{FP} = \text{TP}/\text{precision} - \text{TP}$.

- The low specificity and high decoder loss of the two recent triplet runs ([CITESEER](2026-11-07-cross-weighted-triplet-citeseer.md#cross-weighted-triplet-loss-on-citeseer-analysis), [AstroPh](2026-11-07-uniform-triplet-astroph.md#uniform-triplet-loss-on-astroph-analysis)) are typical of the triplet aggregator: all $16$ triplet groups lie in $[0.440, 0.660]$.
- **The low specificity is not caused by the unit output scale alone.** The test3 contrastive aggregator keeps the replica norm and has the same operating point ($0.51$). The recent contrastive runs, with all links, had $0.89$ to $0.90$. Low specificity goes with a strongly transformed or collapsed embedding.
- The threshold metrics do not track ranking. On CITESEER at $P = 1$, mean has specificity $0.955$ and triplet $0.458$, with equal MRR ($0.263$ against $0.259$). The HeaRT ranking stays the fair comparison.

## Graph reconstruction

At $P = 4$, mean over 10 runs. $S$ is the within-partition pair F1, "macro F1" the global node-averaged F1, and "cut TP" the reconstructed true cut edges.

| Dataset | mean: $S$ / macro F1 / cut TP | triplet: $S$ / macro F1 / cut TP | other |
|---|---|---|---|
| CITESEER | 0.260 / 0.390 / 0 | **0.608 / 0.683** / 4.6 | Procrustes 0.271 / 0.400 / 0 |
| AstroPh | 0.224 / 0.427 / 0 | **0.472 / 0.570 / 1979** | contrastive 0.315 / 0.465 / 839 |
| Enron | 0.042 / 0.135 / 0 | 0.070 / 0.188 / 0 | |
| AS-Oregon | **0.166 / 0.290** / 0 | 0.128 / 0.152 / 0 | |

- **Triplet improves reconstruction on CITESEER, AstroPh and Enron, but not on AS-Oregon.**
- **At $P = 1$ the unit output scale already helps CITESEER's reconstruction a lot**: macro F1 $0.667$ (triplet) against $0.243$ (mean), with no alignment involved. Reconstruction ranks pairs by Euclidean distance, and raw node2vec vectors of different norms are not comparable that way ([Theorem 7](../../../doc/theory/partition-alignment.md#theorem-7-why-normalisation-is-needed)). On AstroPh it goes the other way ($0.615$ against $0.703$). The matched runs of [Aligner start, training and output scale on AstroPh at $P = 1$ analysis](2026-11-08-aligner-start-astroph-p1.md#aligner-start-training-and-output-scale-on-astroph-at-p--1-analysis) confirm this direction: normalising AstroPh's node2vec vectors, with no training, lowers macro F1 from $0.704$ to $0.655$.
- **Only the learned aligners reconstruct cut edges.** The test3 AstroPh triplet runs recovered $1979 \pm 313$ cut edges at $P = 4$ with uniform weighting, against $0$ in the recent uniform run and $1335$ in the recent weighted one. Their within F1 ($0.472$) is also far above the recent runs ($0.273$ uniform, $0.319$ weighted). The test3 aligner saw only $20000$ links, so it may have moved the within geometry less. Which links those were depends on the edge order of the dataset, and the cut share among them is unknown.

## The unlabelled `bilinear` group

The `CITESEER-*-bilinear` runs reach MRR $0.080 \pm 0.010$ at $P = 1$, against $0.263$ for mean. At $P = 1$ any working pipeline should match the unpartitioned model, so this group used a different, damaging aggregation step.

- Its aggregation took $6.2$ s at $P = 1$, the same as triplet ($6.2$ s) and $370\times$ mean ($0.017$ s). So a learned aligner was trained.
- Its specificity ($0.55$) and its cut reconstruction ($10$ to $16$ cut TP) are those of an alignment aggregator.
- It ran from the code before commit `86f1dc1`, with the aggregator name not recorded (the persisted parameters omit the default aggregator). It is most likely an early version of the contrastive aligner, which has no other CITESEER group in test3. This is an inference from the timing and the metrics. The group is excluded from the comparisons.

## Comparison with the recent single runs

The recent runs are single runs; the test3 groups give a distribution. $z$ is the recent value minus the test3 mean, divided by the test3 standard deviation.

| Configuration, $P = 4$ | test3: MRR all / cross / within | Recent run | $z$ (all / cross / within) |
|---|---|---|---|
| CITESEER triplet, uniform | 0.272 ± 0.018 / 0.179 ± 0.009 / 0.418 ± 0.026 | 0.212 / 0.107 / 0.380 | $-3.4$ / $-8.2$ / $-1.5$ |
| AstroPh triplet, uniform | 0.477 ± 0.042 / 0.129 ± 0.034 / 0.535 ± 0.022 | 0.556 / 0.246 / 0.581 | $+1.9$ / $+3.4$ / $+2.1$ |
| AstroPh contrastive, uniform | 0.237 ± 0.011 / 0.108 ± 0.010 / 0.259 ± 0.019 | 0.343 / 0.155 / 0.333 | $+9.8$ / $+4.5$ / $+3.8$ |
| AstroPh, no alignment (mean) | 0.444 ± 0.027 / 0.040 ± 0.005 / 0.503 ± 0.021 | – | – |
| CITESEER, no alignment (mean) | 0.122 ± 0.013 / 0.015 ± 0.004 / 0.303 ± 0.032 | – | – |

- **The "earlier $P = 4$ baseline" used in the sections above is confirmed as no alignment.** Its values, $0.452 \pm 0.022$ (AstroPh) and $0.138 \pm 0.007$ (CITESEER), agree with test3's mean groups ($0.444 \pm 0.027$, $0.122 \pm 0.013$) within about one standard deviation.
- **On AstroPh the recent runs are better than test3**: contrastive by $+9.8$ and triplet by $+1.9$ standard deviations. The main difference in configuration is the link memory: test3 kept $11.2\%$ of the links, the recent runs all of them. The recent uniform triplet reaches $81\%$ of the unpartitioned MRR ($0.556 / 0.683$), against $70\%$ for test3 triplet.
- **On CITESEER the recent uniform triplet run is worse than every test3 triplet run** ($-3.4$ sd overall, $-8.2$ sd cross). The configuration looks identical: $20000 > 4083$ links, uniform weighting, the same aligner settings and the same node2vec parameters. Two facts show the runs did not execute the same way:
  - The aggregation step took $10.5 \pm 5.2$ s in test3 at CITESEER $P = 4$, and $1.26$ s in the recent run. On AstroPh it took $16.6$ s in test3 with $20000$ links, and $5.6$ s recently with $178299$ links.
  - Training the node2vec partitions took the same time in both ($29.7$ against $23.5$ s on CITESEER, $222$ against $225$ s on AstroPh), so machine load does not explain this.
- The likely causes are uncommitted code or environment overrides during the test3 runs, for example more aligner steps. Neither can be recovered from the `.out` files, which contain no aligner log lines. **Until this is reconciled, the recent CITESEER runs should be compared only with each other, not with test3.**

## Reading

1. **With 10 runs per group, triplet alignment is the only learned alignment that reliably helps**: on CITESEER and AS-Oregon at every $P \geq 2$, on AstroPh from $P = 4$ on, and on cross-partition edges almost everywhere. Its MRR barely depends on $P$, while no alignment degrades steadily.
2. **The largest remaining cost is not alignment but the aligner itself.** At $P = 1$, triplet already loses $0.14$ MRR on AstroPh and Enron, which is $60$ to $70\%$ of its gap to the unpartitioned model at $P = 4$. Contrastive loses $0.44$.
3. **The decoder operating point (specificity $\approx 0.5$) belongs to the learned aligners**, not to the unit output scale alone.
4. **Enron is the counterexample**: there, no alignment beats triplet at every $P$, because the within-partition loss dominates its mostly-within test set.
5. **The test3 and recent CITESEER runs disagree**, beyond what the documented changes can explain. Reproducibility of the aligner runs needs to be checked before further single-run comparisons.

