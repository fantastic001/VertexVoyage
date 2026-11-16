# Aligner start, training and output scale on AstroPh at $P = 1$ analysis

Source: four runs of `vv test` on AstroPh with $P = 1$, logged in `VertexVoyage/vv.log`:

| Run | Aggregator | Time (2026-11-07/08) |
|---|---|---|
| R1 | triplet, identity start, 100 steps | 21:49 to 22:15 |
| R2 | triplet, random start, 100 steps | 22:18 to 22:44 |
| R3 | triplet, identity start, 0 steps | 23:20 to 23:46 |
| R4 | mean | 00:02 to 00:27 |

These runs test the second half of item 1 of [Future work](../README.md#future-work), starting the residual at zero. They also separate explanations (a) and (b) of [The cost of the aligner at $P = 1$](2026-11-07-test3-ten-run-experiments.md#the-cost-of-the-aligner-at-p--1). Each configuration is a **single run**.

## Starting parameters

The aligner is that of [Aligner, term by term](../../../doc/theory/partition-alignment.md#aligner-term-by-term), $W(u, p) = N\big(z_{u,p} + \Delta_p(z_{u,p})\big)$ with $\Delta_p(z) = B \, \text{ReLU}(A_p z + a_p) + b$. The setting `alignment_initialisation` selects the starting parameters.

- **Random** (`random`): $A_p, a_p, B, b$ keep PyTorch's default random initialisation of a [fully connected layer](../../../doc/glossary.md#fully-connected-layer). Then $\Delta_p \neq 0$ in general, and the aligner starts as a random perturbation of the identity.
- **Identity** (`identity`): $B = 0$ and $b = 0$, with $A_p, a_p$ random. Then $\Delta_p(z) = 0 \cdot \text{ReLU}(A_p z + a_p) + 0 = 0$ for every $z$ and $p$, so
$$ W(u, p) = N(z_{u,p}) = \frac{z_{u,p}}{\lVert z_{u,p} \rVert}. $$
  The aligner starts as unit-normalised node2vec. The gradient with respect to $B$ is $\text{ReLU}(A_p z + a_p)$ times the output gradient, which is nonzero in general. So training is not stuck at the start, even though the residual is zero.

**Example.** $d = 2$, $z = (3, 4)$, $B = 0$, $b = 0$. Whatever $A_p$ and $a_p$ are, $\Delta_p(z) = (0, 0)$ and $W(u, p) = (3, 4)/5 = (0.6, 0.8)$.

**The three aggregated embeddings.** Let $z_u$ be the node2vec vector of $u$ (at $P = 1$ every node has one replica, so the index $p$ is dropped).

| Run | Aggregated vector of $u$ |
|---|---|
| R4, mean | $z_u$: at $P = 1$ the mean of one replica is the replica itself |
| R3, identity, 0 steps | $N(z_u) = z_u / \lVert z_u \rVert$ |
| R1, R2, 100 steps | $W(u, 1)$ after 100 triplet steps |

So R4 → R3 changes only the norm, and R3 → R1 changes only the direction, by training.

## Setup

| Quantity | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| Shared random state, LPA seed | 42, 42 | 42, 42 | 42, 42 | 42, 42 |
| node2vec F1 (one partition, $p = 2$, $q = 0.25$, $d = 50$) | $0.70393$ | $0.70393$ | $0.70393$ | $0.70393$ |
| Aligner seed | 931325380 | 931325380 | 931325380 | – |
| `initialisation` | `identity` | `random` | `identity` | – |
| `train_steps` | 100 | 100 | 0 | – |
| Log line | `Aligner initialised as the identity` | `Aligner keeps its random initialisation` | `Aligner initialised as the identity`, `loss not computed` | no aligner lines |
| Links / pairs / replica rows | $178299$ / $356598$ / $18655$ | same | same | – |
| Aggregation time | $5.05$ s | $5.00$ s | $1.95$ s | $0.11$ s |

The other aligner settings, from the configuration line, are the same in R1 to R3:
- triplet loss with margin $0.5$, uniform weighting, `effective_output_scale = unit`, cross-partition fraction $0$;
- `link_memory` $= 10^6$, `learning_rate` $= 0.01$, `negative_pool` $= 256$, `hidden_width_factor` $= 2$;
- no validation split, torch 2.8.0 with 8 threads.

The node2vec embedding is identical in all four runs (F1 equal to all printed digits). R4 logs no aligner configuration, and its reconstruction macro F1 ($0.703932$) equals the node2vec F1 to all printed digits, as it must when the aggregate is $z_u$ itself. **Each pair of runs differs in one thing only**: R1/R2 in the start, R1/R3 in the training, R3/R4 in the norm. These are the first matched runs in this page.

## Training dynamics

Each epoch is one optimiser step over all $356598$ pairs, and the logged loss is computed before the step. So the epoch-1 loss is the loss at the starting parameters. R3 takes no step and logs no loss.

| Epoch | 1 | 2 | 5 | 10 | 15 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| $L$, R1 (identity) | 0.0556 | 0.0420 | 0.0336 | 0.0303 | 0.0291 | 0.0284 | 0.0271 | 0.0263 | 0.0263 |
| $L$, R2 (random) | 0.0791 | 0.0486 | 0.0372 | 0.0325 | 0.0304 | 0.0293 | 0.0278 | 0.0271 | 0.0271 |
| $\lVert g \rVert$, R1 | 0.0844 | 0.0424 | 0.0156 | 0.0087 | – | 0.0034 | 0.0020 | 0.0019 | 0.0048 |
| $\lVert g \rVert$, R2 | 0.1299 | 0.0655 | 0.0401 | 0.0180 | – | 0.0053 | 0.0024 | 0.0016 | 0.0024 |

- **Unit-normalised node2vec is not a minimum of the triplet loss on AstroPh.** At the identity start, which is the embedding of R3, $L = 0.0556$. On CITESEER the same start gave $0.0104$. This agrees with the floor of active within triplets of [Theorem 18](2026-11-07-cross-weighted-triplet-astroph.md#theorem-18-a-triplet-with-a-correlated-negative). Training halves the loss ($-53\%$, to $0.0263$), so **the aligner moves the embedding even when it starts as the identity.**
- **The random start begins $42\%$ higher** ($0.0791$): the random residual worsens the geometry before training.
- **The random start falls fastest in the first step and then stays behind.** From epoch 2 on it is higher by $0.0007$ to $0.0066$, a gap that shrinks with the epochs. The random run reaches the identity run's epoch-50 loss ($0.0271$) at epoch 75. At epoch 100 the difference is $0.0008$ ($3\%$ of $L$).
- No vanishing-gradient warning fired (threshold $10^{-7}$).

## Downstream metrics

**HeaRT link prediction** ($19811$ positive test edges). The test3 columns are the 10-run AstroPh $P = 1$ groups of [Earlier 10-run experiments (test3) analysis](2026-11-07-test3-ten-run-experiments.md#earlier-10-run-experiments-test3-analysis). $z$ is (value − test3 mean) / test3 [standard deviation](../../../doc/glossary.md#standard-deviation). At $P = 1$ the within scope is the whole test set, so the second evaluation scores the same model on the same positives with fresh negatives.

| Quantity | R4 mean | R3 identity, 0 steps | R1 identity, 100 steps | R2 random, 100 steps | test3 triplet | test3 mean |
|---|---|---|---|---|---|---|
| MRR | **0.6870** | 0.6699 | 0.6184 | 0.6113 | $0.539 \pm 0.017$ | $0.683 \pm 0.011$ |
| MRR, second evaluation | **0.6852** | 0.6667 | 0.6189 | 0.6210 | – | – |
| Hits@1 / @3 / @10 | 0.561 / 0.781 / 0.912 | 0.546 / 0.762 / 0.873 | 0.498 / 0.699 / 0.837 | 0.490 / 0.688 / 0.832 | – | – |
| Mean rank | 7.54 | 8.28 | 9.82 | 10.16 | – | – |
| $z$ against test3 mean | $+0.4$ | $-1.2$ | $-5.8$ | $-6.5$ | – | – |

- **R4 reproduces test3 mean** ($z = +0.4$). The current code does not change the unpartitioned reference, and the test3 value used in the sections above was valid.
- **The evaluation noise is $0.0005$ to $0.0097$ MRR.** That is how much the second evaluation moves each model: $0.0018$ (R4), $0.0032$ (R3), $0.0005$ (R1), $0.0097$ (R2).
- **The difference between the starts, $0.0071$ MRR, is within this noise.** The second evaluation even reverses the order of R1 and R2 ($0.6189 < 0.6210$).
- **R1 and R2 are far above test3 triplet** ($+4.3$ to $+4.7$ sd). Both use the current code with all links; test3 used $20000$ links ($11.2\%$) and older code. This is the same improvement as in [Comparison with the recent single runs](2026-11-07-test3-ten-run-experiments.md#comparison-with-the-recent-single-runs), and the initialisation plays no part in it.

**Decomposition of the aligner cost.** With $M_{\text{R}k}$ the MRR of run R$k$, the aligner cost of the identity-start triplet aggregator splits, by adding and subtracting $M_{\text{R}3}$, as
$$ \underbrace{M_{\text{R}4} - M_{\text{R}1}}_{\text{aligner cost}} = \underbrace{M_{\text{R}4} - M_{\text{R}3}}_{\text{unit normalisation}} + \underbrace{M_{\text{R}3} - M_{\text{R}1}}_{\text{triplet training}}. $$

| Term | First evaluation | Share | Second evaluation | Share |
|---|---|---|---|---|
| Unit normalisation, $M_{\text{R}4} - M_{\text{R}3}$ | $0.0171$ | $25\%$ | $0.0185$ | $28\%$ |
| Triplet training, $M_{\text{R}3} - M_{\text{R}1}$ | $0.0515$ | $75\%$ | $0.0478$ | $72\%$ |
| Aligner cost, $M_{\text{R}4} - M_{\text{R}1}$ | $0.0686$ | | $0.0663$ | |
| Aligner cost with random start, $M_{\text{R}4} - M_{\text{R}2}$ | $0.0757$ | | $0.0642$ | |

- **Triplet training causes about three quarters of the cost** ($0.048$ to $0.052$ MRR). This is $5$ to $100\times$ the evaluation noise of one model.
- **Unit normalisation causes the other quarter** ($0.017$ to $0.019$). It is the same in both evaluations and above the noise of R3 and R4 ($0.002$ to $0.003$), so it is likely real, but it rests on one pair of runs.
- **Example.** Out of $0.687$ MRR, normalising the node2vec vectors leaves $0.670$, and 100 triplet steps then leave $0.618$.
- This settles the two explanations of [The cost of the aligner at $P = 1$](2026-11-07-test3-ten-run-experiments.md#the-cost-of-the-aligner-at-p--1) for AstroPh: both act, and (a), the aligner moving within pairs, is the larger one.

**Thresholded metrics and decoder.** Specificity is $1 - \text{FP}/\lvert \text{negatives} \rvert$ with $\text{FP} = \text{TP}/\text{precision} - \text{TP}$, as in [Decoder operating point](2026-11-07-test3-ten-run-experiments.md#decoder-operating-point). The test set has as many negatives as positives, so it equals $1 - R\,(1/\text{precision} - 1)$, where $R$ is the recall. The accuracy $(R + \text{specificity})/2$ matches the log in every column.

| Quantity | R4 mean | R3 identity, 0 steps | R1 identity, 100 steps | R2 random, 100 steps | test3 triplet | test3 mean |
|---|---|---|---|---|---|---|
| Precision / recall | 0.9844 / 0.9395 | 0.9647 / 0.9518 | 0.6742 / 0.9932 | 0.6751 / 0.9930 | 0.7071 / 0.9777 (run 0) | 0.9809 / 0.9380 (run 0) |
| Specificity | **0.985** | 0.965 | 0.520 | 0.522 | 0.595 (run 0) | 0.982 (run 0) |
| Accuracy | **0.9623** | 0.9585 | 0.7566 | 0.7576 | 0.7864 (run 0) | 0.9599 (run 0) |
| Decoder train / validation loss | **0.0316** / 0.0356 | 0.1268 / 0.1261 | 0.4067 / 0.4081 | 0.4069 / 0.4086 | 0.381 to 0.392 (10 runs) | 0.031 to 0.033 (10 runs) |

- **Unit vectors alone keep the decoder at a good operating point.** On $N(z_u)$ the specificity is $0.965$ and the accuracy $0.959$, against $0.985$ and $0.962$ on $z_u$. The decoder loss rises $4\times$ ($0.032 \to 0.127$), but the threshold metrics barely move.
- **Triplet training moves the decoder to the triplet operating point.** After 100 steps the specificity is $0.52$ and the decoder loss $0.41$, whichever start is used. This is what all $16$ triplet groups of test3 show ($0.440$ to $0.660$).
- **This refutes the unit-output-scale hypothesis** of the earlier sections, that a bilinear logit $x^\top W y$ of unit vectors cannot separate the classes sharply. With unit vectors and no training, it separates them almost as well as with raw node2vec vectors. The earlier sections are corrected accordingly.

**Graph reconstruction** ($k = 178299$ closest pairs).

| Quantity | R4 mean | R3 identity, 0 steps | R1 identity, 100 steps | R2 random, 100 steps | test3 triplet | test3 mean |
|---|---|---|---|---|---|---|
| Macro precision / recall / F1 | 0.661 / 0.752 / **0.704** | 0.583 / 0.746 / 0.655 | 0.638 / 0.557 / 0.595 | 0.641 / 0.546 / 0.590 | – / – / $0.615 \pm 0.002$ | – / – / $0.703 \pm 0.001$ |
| S (within F1, micro over pairs) | **0.585** | 0.515 | 0.348 | 0.332 | – | – |

- **For reconstruction, the norm matters almost as much as the training.** Macro F1 falls by $0.049$ with normalisation (R4 → R3) and by $0.060$ with training (R3 → R1). S falls by $0.070$ and $0.167$.
- The two steps fail differently:
  - normalisation lowers the **precision** ($0.661 \to 0.583$) at the same recall: the closest pairs of the unit vectors contain more non-edges;
  - training lowers the **recall** ($0.746 \to 0.557$): true neighbours leave the closest pairs.
- This is the effect predicted by explanation (b). A node2vec norm grows with the node's frequency in the walks, hence with its degree. Removing it changes which pairs are closest. Reconstruction depends on that more than link prediction does, because link prediction trains a decoder on top of the embedding.
- R1 reconstructs slightly better than R2 (macro F1 $+0.005$, S $+0.016$). With one run per start, this is not separable from noise.
- **R1 and R2 reconstruct worse than test3 triplet** (macro F1 $0.590$ to $0.595$ against $0.615 \pm 0.002$), while their MRR is far better. More links make the aligner better at ranking held-out edges and worse at reproducing the training graph's nearest neighbours. This is an observation from these runs; nothing in this page explains it.

## Reading

1. **The start does not matter.** Starting from the identity does not remove the cost ($0.069$ against $0.076$ MRR, within noise). A random residual at the start is not why triplet loses to mean at $P = 1$.
2. **On AstroPh at $P = 1$, the triplet training causes three quarters of the aligner cost and the unit normalisation one quarter.** Unit-normalised node2vec has triplet loss $0.056$, so the triplet objective pulls the embedding away from node2vec's geometry, and that costs $0.05$ MRR.
3. **The low specificity of the triplet aggregator comes from training, not from unit vectors.** Unit vectors without training keep specificity $0.965$.
4. **The norm matters more for reconstruction than for link prediction**: it costs $0.049$ macro F1 but only $0.017$ MRR.
5. **Identity start is a safe default.** It starts $30\%$ lower in loss, never ends worse than the random start, and keeps a matched run interpretable: at epoch 1 the aggregator is exactly $N(z_u)$.
6. **The current code is much better than test3 on AstroPh at $P = 1$** ($0.61$ to $0.62$ against $0.54$). The aligner cost is now $0.064$ to $0.076$ instead of $0.144$.

## Next runs

1. **Keep the training close to the start.** At $P = 1$ the best aligner is no training at all, but at $P > 1$ the aligner has to move cut pairs. Candidates, each run first at $P = 1$ against R1 and R3:
   - an anchor term on within pairs, $\lambda \lVert W(u, p) - N(z_{u,p}) \rVert^2$, added to the triplet loss;
   - a smaller margin ($0.1$ to $0.2$), which makes fewer within triplets active ([Theorem 17](2026-11-07-cross-weighted-triplet-citeseer.md#theorem-17-a-triplet-with-an-orthogonal-negative-is-a-cosine-threshold));
   - fewer steps (for example $10$ or $25$).
2. **Identity start at $P = 4$, with 0 against 100 steps.** This measures how much of triplet's gain over mean at $P = 4$ is alignment, and how much survives with less training.
   - *Status:* done, see [Aligner training on AstroPh at $P = 4$ analysis](2026-11-08-aligner-training-astroph-p4.md#aligner-training-on-astroph-at-p--4-analysis). The training raises cross MRR from $0.023$ to $0.251$ and lowers within MRR by $0.018$.
3. **`VERTEX_VOYAGE_ALIGNMENT_OUTPUT_SCALE=replica_norm`** with identity start. It can recover at most the $0.017$ MRR of the normalisation, but up to $0.049$ reconstruction macro F1.
4. **Repeat R1, R3 and R4 with 5 seeds**, to put error bars on the decomposition.

