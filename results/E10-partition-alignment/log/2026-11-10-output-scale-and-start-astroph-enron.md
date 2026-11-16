# Output scale and aligner start on AstroPh and Enron analysis

Source: $24$ runs of `vv test` in `VertexVoyage/logs/`, one file per configuration, named `<dataset>-<output scale>-<P>-<start>.log`, run between 2026-11-09 15:56 and 2026-11-10 10:25. They cross the output scale with the aligner start, at $P \in \{1, 4\}$, on AstroPh and Enron. They also replicate R5 of [Aligner training on AstroPh at $P = 4$ analysis](2026-11-08-aligner-training-astroph-p4.md#aligner-training-on-astroph-at-p--4-analysis). In addition, two 0-step reference runs (identity start, unit scale, no training steps) were reported without logs: AstroPh $0.66$ MRR and Enron $0.44$ MRR. Their $P$ is not stated. They are read here as $P = 1$, because the AstroPh value agrees with R3 at $P = 1$ ($0.670$) and not with R6 at $P = 4$ ($0.537$), and both runs share the node2vec models with these. Each configuration is a **single run**.

## What was run

Let $z_{u,p}$ be the node2vec vector of node $u$ in partition $p$, and let $W(u, p) = N(z_{u,p} + \Delta_p(z_{u,p}))$ with $\Delta_p(z) = B \, \text{ReLU}(A_p z + a_p) + b$ be the residual aligner of [Approach](../README.md#approach), where $N(x) = x / \lVert x \rVert$ and the [ReLU](../../../doc/glossary.md#relu-function) acts coordinatewise. The aggregated vector of $u$ is

| Output scale | Aggregated vector of $u$ |
|---|---|
| `unit` | $W(u, p(u))$, of norm $1$ |
| `replica_norm` | $\lVert z_{u,p(u)} \rVert \, W(u, p(u))$, the norm of its replica |

The three starts set the parameters before the first step:

| Start | $A_p$, $a_p$ | $B$, $b$ | $W$ at step 0 |
|---|---|---|---|
| identity | random | $0$, $0$ | $N(z)$ |
| random | random | random | $N(z + \Delta_p(z))$ with a random $\Delta_p$ |
| Procrustes | $\begin{bmatrix} Q_p - I \\ -(Q_p - I) \end{bmatrix}$, $0$ | $\begin{bmatrix} I & -I \end{bmatrix}$, $0$ | $Q_p z$ ([Theorem 3](../../../doc/theory/partition-alignment.md#theorem-3-the-residual-block-can-represent-any-linear-map)) |

Here $Q_p$ is the [orthogonal Procrustes](../../../doc/glossary.md#orthogonal-procrustes-problem) rotation fitted for partition $p$, with $Q_0 = I$ for the reference partition $0$.

**Shared setup.** Everything else is as in the previous sections: triplet loss, margin $0.5$, uniform link weighting, $100$ steps, learning rate $0.01$, hidden width $h = 2d = 100$, link memory $10^6$, LPA seed $42$, bilinear decoder, HeaRT with $500$ negatives. Within a (dataset, $P$) cell, all six runs share the aligner seed and the node2vec models (the partition F1 values agree to all printed digits).

| Dataset | $P$ | Within edges per partition | Cut share $\delta$ of training edges | Test edges, cut | Cut share $s$ of test edges |
|---|---|---|---|---|---|
| AstroPh | 1 | $178299$ | $0$ | $19811$, $0$ | $0$ |
| AstroPh | 4 | $129242, 10964, 10600, 11853$ | $0.0877$ | $19811$, $2124$ | $0.1072$ |
| Enron | 1 | $165448$ | $0$ | $18383$, $0$ | $0$ |
| Enron | 4 | $116494, 17723, 8007, 9894$ | $0.0806$ | $18383$, $3022$ | $0.1644$ |

"Recomposed" MRR is $s \cdot \text{cross} + (1 - s) \cdot \text{within}$, as before.

## None of the eight Procrustes runs is a full Procrustes start

The logs show that the Procrustes start (two output scales × two $P$ × two datasets) was a bias-only aligner in four runs and the identity start in two, and worked only partly in the remaining two.

| Dataset, $P$ | Log line | What the aligner was |
|---|---|---|
| AstroPh, 1 and 4 (4 runs) | `Rotations fitted on 52 anchors, mean anchor cosine 1.000000` | all $Q_p = I$; only $b$ trains |
| Enron, 1 (2 runs) | `No anchors for Procrustes, aligner initialised as identity` | the identity start, a second identity run |
| Enron, 4 (2 runs) | `Rotations fitted on 13330 anchors, mean anchor cosine 0.552346` | real rotations; partition 0 keeps only $b$ |

**Why AstroPh gets $52$ anchors with cosine $1$.** `anchor_links` first keeps the links $((p, u), (q, v))$ with $u = v$, meant to be two replicas of one node, and falls back to the cut links only when there are none. It does not require $p \neq q$. A self-loop $(u, u)$ of the training graph passes the test, and joins a row to itself, so its cosine is $1$. $13330 = 0.0806 \cdot 165448$ is exactly Enron's number of cut training links, so Enron has no self-loops and got the intended anchors. On AstroPh, the self-loops lie inside a partition. For $p \neq 0$ there are then no anchors from $p$ into a fitted partition, and every $Q_p$ stays $I$.

**Why $Q_p = I$ freezes the residual of partition $p$.**

*Observation.* Assume $Q_p = I$, so that $A_p = 0$ and $a_p = 0$, and assume, as PyTorch does, that the derivative of ReLU at $0$ is taken to be $0$. Then $A_p$, $a_p$ and the columns of $B$ never change for the nodes of $p$, and the aligner on partition $p$ stays $W(u, p) = N(z_{u,p} + b)$ for the whole training.

*Derivation.*
1. For every $u$ in $p$, the hidden pre-activation is $A_p z_{u,p} + a_p = 0$, so the hidden activation is $\text{ReLU}(0) = 0$ and $\Delta_p(z_{u,p}) = B \cdot 0 + b = b$.
2. The gradient of the loss $L$ with respect to $A_p$ is $\big((B^\top \partial L / \partial \Delta) \odot \text{ReLU}'(0)\big) z^\top$, summed over the nodes of $p$. With $\text{ReLU}'(0) = 0$ it is $0$; likewise for $a_p$.
3. The contribution of the nodes of $p$ to the gradient of $B$ is $(\partial L / \partial \Delta) \cdot 0^\top = 0$. Other partitions can change $B$, but by step 1 the nodes of $p$ never see $B$.
4. So after any number of steps $A_p = 0$ and $a_p = 0$ still hold, and step 1 applies again. Only the shared output bias $b$ moves the vectors of $p$. ∎

*Examples.* On AstroPh every $Q_p = I$, so the whole aligner is $N(z + b)$, a **bias-only aligner**. The loss shows it: at $P = 4$ it falls from $0.0914$ to $0.0829$ by epoch 10 and then stays at $0.0825$, against $0.0381$ for the identity start. On Enron at $P = 4$, partition 0 ($116494$ of $152118$ within edges) is frozen to $N(z + b)$, and only partitions 1 to 3 train; the loss ends at $0.0765$, against $0.0388$.

So below, the AstroPh "Procrustes" runs are read as a bias-only aligner, Enron $P = 1$ as a second identity run, and only Enron $P = 4$ as a (partly frozen) Procrustes start.

## Noise

- **HeaRT scores $1000$ sampled positives** (`ps=1000` in `heart_benchmark`). The [standard deviation](../../../doc/glossary.md#standard-deviation) of a reciprocal rank is about $0.4$, so one MRR has a standard error of about $0.013$, and the difference of two runs about $0.018$.
- **At $P = 1$ the within scope is a second evaluation** of the same model on a fresh sample. It moves AstroPh by $0.000$ to $0.015$ and Enron `unit` by $0.004$ to $0.020$. On Enron `replica_norm` the first evaluation is higher in all three runs, by $0.032$, $0.036$ and $0.052$. This is $1.7$ to $2.8$ standard errors of a difference, always in the same direction, and is not explained. At $P = 1$ the tables below use the mean of both evaluations.
- **The replicate of R5** (AstroPh, `unit`, $P = 4$, identity) gives MRR $0.555$ / cross $0.254$ / within $0.575$, against R5's $0.554$ / $0.251$ / $0.573$.
- **The two Enron $P = 1$ identity runs share the seed and the start, and still differ.** Their final losses are $0.02648$ and $0.02787$ (`unit`), so the aligner training is not reproducible bit for bit. Their MRRs differ by $0.002$ (`unit`) and $0.027$ (`replica_norm`). They are used as two replicates.

## Link prediction

HeaRT MRR. $P = 1$: mean of the two evaluations. $P = 4$: all (logged) / recomposed / cross / within. "id" is the identity start, "rand" the random start, "bias" the bias-only aligner, "Procr." the Enron Procrustes start, "id 2" the second identity run.

**AstroPh.**

| | `unit` id | `unit` rand | `unit` bias | `replica_norm` id | `replica_norm` rand | `replica_norm` bias |
|---|---|---|---|---|---|---|
| $P = 1$ | 0.618 | 0.616 | **0.648** | 0.515 | 0.513 | 0.471 |
| $P = 4$, all / recomposed | 0.555 / 0.540 | **0.566** / **0.545** | 0.562 / 0.539 | 0.486 / 0.454 | 0.484 / 0.472 | 0.410 / 0.389 |
| $P = 4$, cross / within | 0.254 / 0.575 | **0.262** / 0.578 | 0.013 / **0.602** | 0.207 / 0.483 | 0.215 / 0.502 | 0.018 / 0.434 |

References on AstroPh: mean $0.687$ at $P = 1$ (R4), $0.474$ / $0.476$ at $P = 4$ (R7, cross $0.047$, within $0.527$). 0 steps, `unit`: $0.670$ (R3) and $0.66$ (new) at $P = 1$, $0.537$ / $0.530$ at $P = 4$ (R6, cross $0.023$, within $0.591$).

**Enron.**

| | `unit` id | `unit` rand | `unit` id 2 / Procr. | `replica_norm` id | `replica_norm` rand | `replica_norm` id 2 / Procr. |
|---|---|---|---|---|---|---|
| $P = 1$ | 0.385 | 0.382 | **0.387** | 0.345 | 0.340 | 0.318 |
| $P = 4$, all / recomposed | 0.368 / **0.361** | **0.371** / 0.358 | 0.354 / 0.357 | 0.292 / 0.288 | 0.290 / 0.291 | 0.306 / 0.309 |
| $P = 4$, cross / within | 0.079 / **0.416** | 0.068 / **0.416** | 0.100 / 0.407 | 0.078 / 0.330 | 0.068 / 0.335 | **0.144** / 0.342 |

References on Enron: test3 mean $0.517 \pm 0.016$ at $P = 1$ and $0.350 \pm 0.009$ at $P = 4$ (cross $0.044$, within $0.410$, recomposed $0.350$); test3 triplet $0.374$ and $0.276$. 0 steps, `unit`: $0.44$ at $P = 1$ (new); not run at $P = 4$.

## Output scale: `unit` against `replica_norm`

Difference `unit` − `replica_norm` in matched pairs (same dataset, $P$ and start).

| Dataset, $P$ | id: all / within | rand: all / within | third start: all / within | Reconstruction macro F1, three starts |
|---|---|---|---|---|
| AstroPh, 1 | $+0.103$ / – | $+0.103$ / – | $+0.177$ / – (bias) | $-0.083$, $-0.089$, $-0.049$ |
| AstroPh, 4 | $+0.070$ / $+0.091$ | $+0.082$ / $+0.076$ | $+0.152$ / $+0.169$ (bias) | $-0.012$, $-0.004$, $-0.019$ |
| Enron, 1 | $+0.040$ / – | $+0.043$ / – | $+0.069$ / – (id 2) | $+0.007$, $+0.013$, $+0.011$ |
| Enron, 4 | $+0.077$ / $+0.087$ | $+0.081$ / $+0.081$ | $+0.048$ / $+0.065$ (Procr.) | $+0.015$, $+0.054$, $+0.046$ |

- **`unit` wins all $12$ matched pairs on MRR**, by $0.040$ to $0.177$, that is $2$ to $10$ standard errors of a difference. With the identity or random start the gain is $0.04$ to $0.10$.
- **The gain is on within edges.** At $P = 4$, cross MRR moves by $+0.047$ on AstroPh and by at most $0.001$ on Enron (identity and random starts), while within MRR moves by $+0.076$ to $+0.091$.
- **This refutes the expectation of [the $P = 1$ next runs](2026-11-08-aligner-start-astroph-p1.md#aligner-start-training-and-output-scale-on-astroph-at-p--1-analysis)**, that `replica_norm` could recover up to the $0.017$ of the normalisation. At $P = 1$, giving the trained directions back their node2vec norms costs $0.10$ on AstroPh. The raw vectors $z$ rank better than $N(z)$ there (R4 against R3), but trained directions with the old norms rank much worse.
  - *Hypothesis, not tested.* The norm of $z_{u,p}$ belongs to the direction of $z_{u,p}$. After training, $W(u, p)$ points elsewhere, and multiplying it by $\lVert z_{u,p} \rVert$ scales the logits of the bilinear decoder by a factor that no longer matches the direction.
- **The decoder overfits the `replica_norm` vectors on Enron**: train / validation loss $0.383$ / $0.447$ at $P = 1$ (identity), against $0.408$ / $0.410$ for `unit`. On AstroPh the gaps are $0.009$ and $0.002$.
- **Reconstruction prefers `replica_norm` on AstroPh at $P = 1$** ($0.674$ against $0.591$ macro F1, identity), as the norm result of the $P = 1$ section predicted. At $P = 4$ the difference is $0.004$ to $0.019$, and on Enron `unit` is better in all six pairs.

## Start: identity, random and Procrustes

- **Identity and random end at the same MRR.** The differences (identity − random) are $+0.002$ (AstroPh $P = 1$), $-0.011$ (AstroPh $P = 4$), $+0.002$ (Enron $P = 1$) and $-0.003$ (Enron $P = 4$) with `unit`. All are below one standard error of a difference. This extends the $P = 1$ AstroPh result to $P = 4$ and to Enron.
- **The random start begins higher in loss and ends at the same loss**: AstroPh $P = 4$ $0.1038 \to 0.0381$ against $0.0914 \to 0.0381$; Enron $P = 4$ $0.1437 \to 0.0396$ against $0.1313 \to 0.0388$.
- **The bias-only aligner (AstroPh) is the closest trained aggregator to the 0-step one.** At $P = 1$ it is at $0.648$, between full training ($0.618$) and 0 steps ($0.66$ to $0.670$). At $P = 4$ it ranks within edges best of all aggregators ($0.602$, against $0.591$ for 0 steps), but cut edges at chance ($0.013$; chance is $0.0136$). A shared shift $b$ cannot align partitions.
- **Real Procrustes rotations (Enron $P = 4$) trade within for cross ranking.** With `unit`, cross MRR rises to $0.100$ (identity $0.079$) and within falls to $0.407$ (identity $0.416$); recomposed, $0.357$ against $0.361$. With `replica_norm`, which keeps norms as a rotation does, cross reaches $0.144$, the best Enron cut-edge ranking in this set, and the run is the best `replica_norm` run. But the start begins at a higher loss than the identity ($0.1728$ against $0.1313$), and partition 0 never trains. This run does not test Procrustes as a start; it tests a broken version of it.

## Training against 0 steps

With $M$ the MRR, unit scale and identity start:

| Dataset, $P$ | Mean | 0 steps | 100 steps | Normalisation, 0 steps − mean | Training, 100 − 0 steps |
|---|---|---|---|---|---|
| AstroPh, 1 | **0.687** | 0.670, 0.66 | 0.618 | $-0.017$ to $-0.027$ | $-0.042$ to $-0.052$ |
| AstroPh, 4 (recomposed) | 0.476 | 0.530 | **0.540** (rand $0.545$) | $+0.054$ | $+0.010$ (rand $+0.015$) |
| Enron, 1 | **0.517** (test3) | 0.44 | 0.385 | $-0.077$ | $-0.055$ |
| Enron, 4 (recomposed) | 0.350 (test3) | not run | **0.361** | – | – |

- **At $P = 1$ every trained aggregator is worse than 0 steps on both datasets**, and 0 steps is worse than mean. On Enron both costs are larger than on AstroPh: the normalisation costs $0.077$ (against $0.017$ to $0.027$) and the training $0.055$ (against $0.042$ to $0.052$). The Enron mean value is test3's, which AstroPh's R4 reproduced; it has not been rerun on Enron with the current code.
- **On Enron at $P = 4$, triplet with the current code now beats test3 mean**: $0.368$ against $0.350 \pm 0.009$ (logged), $0.361$ against $0.350$ (recomposed). Within $0.416$ against $0.410$; cross $0.079$ against $0.044$. In test3, triplet lost to mean at every $P$ on Enron ($0.276$ at $P = 4$). The test3 runs kept $20000$ links ($12.1\%$ of Enron's edges); these keep all of them. The gain over mean ($+0.011$ recomposed) is less than one standard error, so the current reading is that triplet and mean are equal on Enron at $P = 4$, not that triplet loses.
- **Without the Enron 0-step run at $P = 4$ it is unknown whether the training helps on Enron at all.** On AstroPh at $P = 4$, the normalisation gives $0.054$ and the training $0.010$ to $0.015$.

## Decoder and reconstruction

- **Thresholded metrics do not separate the configurations.** Precision is $0.67$ to $0.70$ and recall $0.92$ to $0.99$ in all $24$ runs. Every trained aggregator has the low-specificity operating point of the earlier triplet runs.
- **Reconstruction (global macro F1).**

| | `unit` id | `unit` rand | `unit` third | `replica_norm` id | `replica_norm` rand | `replica_norm` third |
|---|---|---|---|---|---|---|
| AstroPh, 1 | 0.591 | 0.587 | 0.660 (bias) | 0.674 | 0.676 | **0.709** (bias) |
| AstroPh, 4 | 0.490 | 0.495 | 0.283 (bias) | **0.502** | 0.499 | 0.303 (bias) |
| Enron, 1 | 0.115 | 0.111 | **0.116** (id 2) | 0.108 | 0.098 | 0.105 (id 2) |
| Enron, 4 | 0.193 | **0.205** | 0.195 (Procr.) | 0.178 | 0.151 | 0.148 (Procr.) |

  - At $P = 1$ on AstroPh, less training reconstructs better (bias-only $0.709$, against $0.674$ trained), as with link prediction.
  - At $P = 4$ on AstroPh, the bias-only aligner reconstructs as badly as 0 steps (R6, $0.319$), so the reconstruction gain of R5 needs the trained $A_p$ and $B$, not a shift.
  - On Enron, reconstruction is low everywhere ($0.10$ to $0.21$), and $P = 4$ is better than $P = 1$. The node2vec F1 of the whole Enron graph is only $0.094$, against $0.22$ to $0.34$ per partition at $P = 4$.

## Reading

1. **The output scale is settled: `unit`.** It wins all $12$ matched MRR comparisons on both datasets and both $P$, by $0.04$ to $0.18$. `replica_norm` helps only AstroPh reconstruction at $P = 1$.
2. **The start does not matter between identity and random**, at $P = 1$ and $4$, on both datasets. Identity is preferable because its step 0 is the 0-step baseline $N(z)$.
3. **The Procrustes start is broken in its current form.** Self-loops are taken as replica anchors, and a rotation equal to $I$ freezes its partition's residual through $\text{ReLU}'(0) = 0$. The AstroPh runs measure a bias-only aligner, not Procrustes.
4. **The training costs within ranking whenever it moves the vectors, and pays only through cut edges.** At $P = 1$ it costs $0.04$ to $0.06$ on both datasets. At $P = 4$ it gains $0.010$ to $0.015$ on AstroPh. On Enron the 0-step run that decides it is missing.
5. **With the current code, triplet is at least as good as mean on Enron at $P = 4$**, which overturns the test3 reading for Enron, but only to equality within noise.
6. **The aligner training is not reproducible bit for bit at a fixed seed.** Two Enron identity runs with the same seed end at different losses.

## Recommended defaults for the next experiments

| Setting | Value | Environment variable | Reason |
|---|---|---|---|
| Aggregator | triplet | – | the only aligner that ranks cut edges well (test3, R5) |
| Output scale | `unit` | `VERTEX_VOYAGE_ALIGNMENT_OUTPUT_SCALE=unit` | wins all $12$ matched pairs |
| Start | `identity` | `VERTEX_VOYAGE_ALIGNMENT_INITIALISATION=identity` | equal to random in MRR; step 0 is the 0-step baseline |
| Link weighting | `uniform` | `VERTEX_VOYAGE_ALIGNMENT_LINK_WEIGHTING=uniform` | better overall and within MRR on CITESEER and AstroPh ([Effect of the weighting on the triplet loss, both datasets](2026-11-07-uniform-triplet-astroph.md#effect-of-the-weighting-on-the-triplet-loss-both-datasets)) |
| Steps, margin, learning rate | $100$, $0.5$, $0.01$ | unchanged | the reference of every run so far; to be varied, not yet replaced |
| Controls in every comparison | mean and 0 steps, same seed | `VERTEX_VOYAGE_ALIGNMENT_TRAIN_STEPS=0` for 0 steps | most of the gain over mean can be the normalisation |

The code defaults differ in two settings: `alignment_initialisation` defaults to `random` and `alignment_link_weighting` to `inverse_cross_fraction`. So the next runs should set both explicitly. Procrustes should not be used as a start until it is fixed (next runs, item 3).

## Next runs

1. **Enron at $P = 4$ with 0 steps** (`unit`, identity), and Enron mean at $P = 1$ and $P = 4$ with the current code. These complete the Enron row of the decomposition and show whether the training helps on Enron at all.
2. **A step sweep at $P = 4$**: $0$, $10$, $25$, $50$, $100$ steps on AstroPh and Enron, with the defaults above. At $P = 1$ the training only costs, at $P = 4$ it gains little in total. The sweep shows whether the cut-edge gain comes before the within loss.
3. **Fix the Procrustes start, then rerun it.**
   - In `anchor_links`, keep only links with $p \neq q$ as replica anchors, so that self-loops are not anchors.
   - Give the reference partition, and any partition with $Q_p = I$, a residual that can still train: start $A_p$ and $a_p$ at small random values ($\varepsilon$-scale) instead of exactly $0$, so that the hidden units are not all at the kink of ReLU.
   - Rerun at $P = 4$ on AstroPh and Enron, `unit` and `replica_norm`. On Enron the broken version already gave the best cut-edge ranking with `replica_norm`.
4. **The partition-mean norm scale** of [Mean against triplet on AstroPh at $P = 4$ analysis](2026-11-08-mean-vs-triplet-astroph-p4.md#mean-against-triplet-on-astroph-at-p--4-analysis), $z_{u,p} / \overline{\lVert z \rVert}_p$, now also on Enron, where the normalisation costs $0.077$ at $P = 1$, four times AstroPh's.
5. **Reproducibility.** Find why the same seed and start give different losses (unseeded negative sampling, or nondeterministic threaded kernels), then repeat the defaults and both controls with $5$ seeds on AstroPh and Enron at $P = 4$. Every difference above between identity and random, and between trained and 0 steps at $P = 4$, is one to two standard errors.
