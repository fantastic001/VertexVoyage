# E10 — Aligning partition embeddings (residual aligner)

| | |
|---|---|
| **Research question** | [RQ4](../README.md#research-questions) |
| **Status** | 🔵 ongoing |
| **Period** | 2026-10-13 → 2026-11-10 |
| **Pipeline** | S, static (LPA, $RF = 1$, node2vec per partition), bilinear decoder, HeaRT evaluation with 500 negatives |
| **Datasets** | CITESEER, AstroPh, AS-Oregon, Enron |
| **Headline** | A residual aligner trained with the **triplet** loss removes almost all partition dependence of link prediction. It recovers cross-partition ranking to 6–14× chance and beats no alignment on CITESEER, AS-Oregon and AstroPh ($P \ge 4$). The contrastive loss collapses, as Theorem 6 predicts. |
| **Theory** | Theorems 1–18 in [doc/theory/partition-alignment.md](../../doc/theory/partition-alignment.md) and the [log](#analysis-log), all proved in [`PartitionAlignment.lean`](../../proofs/VertexVoyageProofs/PartitionAlignment.lean) |
| **Defaults in code** | Verified 2026-11-16: `alignment_initialisation=identity` and `alignment_link_weighting=uniform` are now the code defaults, matching the recommendation below. |

# Problem statement 

Given [graph](../../doc/glossary.md#graph) $G = (V, E)$ and its [partitioned subgraphs](../../doc/glossary.md#graph-partitioning) $\{G_1, \dots, G_P\}$, where $G_i = (V_i, E_i)$ for $i = 1, \dots, P$, the individual embeddings obtained from using [node2vec](../../doc/glossary.md#node2vec) or [dynnode2vec](../../doc/glossary.md#dynnode2vec) are misaligned across partitions. The goal is to align these embeddings to a common space.


# Foundations

- [graph](../../doc/glossary.md#graph)
- [graph partitioning](../../doc/glossary.md#graph-partitioning)
- [node2vec](../../doc/glossary.md#node2vec)
- [dynnode2vec](../../doc/glossary.md#dynnode2vec)
- [metric space](../../doc/glossary.md#metric-space) - embeddings individually form metric space. 
- [isometry](../../doc/glossary.md#isometry) - a distance-preserving map; per-partition embeddings are determined only up to one.
- [orthogonal Procrustes problem](../../doc/glossary.md#orthogonal-procrustes-problem) - the best linear isometry between two point sets.
- [stochastic block model](../../doc/glossary.md#stochastic-block-model) - the synthetic graph family used in the check below.

General forms of the analysis tools used below:

- [frustration](../../doc/glossary.md#frustration) - Theorem 6 (b) and its consequence for whole graphs.
- [gradient conflict](../../doc/glossary.md#gradient-conflict) - Theorems 11 and 16, the cosine and cancellation ratio of term gradients.
- [representation collapse](../../doc/glossary.md#representation-collapse) - Theorems 6 (a), 7 and 9 (b).
- [stratified metric decomposition](../../doc/glossary.md#stratified-metric-decomposition) - Theorems 10, 13, 14 and 15.
- [ranking metrics](../../doc/glossary.md#ranking-metrics) - MRR, Hits@$k$ and their chance levels.
- [model evaluation](../../doc/glossary.md#model-evaluation) - specificity and the decoder operating point.


# Related work 

- [Contrastive loss](../../doc/glossary.md#contrastive-loss)
- [Triplet loss](../../doc/glossary.md#triplet-loss)

# Approach 

We train transformation model $W(f_{\pi(x)}(x), p) = x + \Delta(x, p)$, where $\Delta(x, p)$ is a learned offset that aligns the embedding $x$ from partition $p$ to the common space.

$$ \Delta(x, p) = \text{MLP}_2(\text{ReLU}(\text{MLP}_{1, \pi(x)}(f_{\pi(x)}(x)))) $$

where $\pi(x)$ maps the embedding $x$ to its corresponding partition index and $f_{\pi(x)}(x)$ retrieves the embedding of $x$ from its partition.

We evaluate two possible approaches:

- Minimize contrastive loss between aligned embeddings.
- Minimize triplet loss between aligned embeddings.



**Contrastive Loss:** Minimize 

$$ L = \sum_{uv \in E} \|W(f_{\pi(x_i)}(x_i)) - W(f_{\pi(x_j)}(x_j))\|^2 + \sum_{uv \notin E} \max(0, m - \|W(f_{\pi(x_i)}(x_i)) - W(f_{\pi(x_j)}(x_j))\|)^2 $$

**Triplet Loss:** Minimize


$$
\sum_{u \in V} \sum_{v \in S_u} \sum_{w \notin S_u} \max \left( d(W(f_{\pi(u)}(u)), W(f_{\pi(v)}(v))) - d(W(f_{\pi(u)}(u)), W(f_{\pi(w)}(w))) + \alpha, 0 \right)
$$

where $S_u$ is the set of sampled neighbors of node $u$ and $\alpha$ is the margin parameter.


# Conclusion

## Take-home

Independently trained node2vec embeddings of LPA partitions rank cross-partition edges near chance, and their link prediction degrades steadily with the number of partitions. A learned residual aligner trained with the **triplet** loss removes almost all of this partition dependence. Over 10 runs it beats no alignment on CITESEER, AS-Oregon and (from $P = 4$) AstroPh, recovering cross-partition ranking to $6$ to $14\times$ chance. The **contrastive** loss does not: it collapses and frustrates as Theorem 6 predicts, and it damages the embedding even when there is nothing to align. Weighting cut links by $1/\delta$ is not a general fix. It is the only setting that reconstructs cut edges and lifts cross-partition ranking on AstroPh, where $\delta$ is small. Everywhere it costs within-partition quality, and on CITESEER it buys nothing. The most important open problem is the aligner's own cost: on degree-heterogeneous graphs (AstroPh, Enron) the triplet aggregator loses $0.14$ MRR already at $P = 1$, which is most of its remaining gap to the unpartitioned model. All comparisons of the new configurations rest on single runs, and the recent CITESEER runs do not reproduce test3. So the next step is matched, seeded repetition, not more single runs.

## Future work

1. **Reduce the aligner cost at $P = 1$.**
   - Run the triplet aggregator with `alignment_output_scale=replica_norm`, and the contrastive one with `unit`, at $P = 1$ and $P = 4$. This separates the output scale from the learned transform (explanations (a) and (b) above).
   - Add a geometry-preserving term on within pairs, for example $\lVert W(u, p) - z_{u,p} \rVert^2$, or start the residual at zero so that the aligner starts as the identity.
   - Make $P = 1$ a standard control for every new aggregator.
   - *Status:* the identity start and the 0-step control are done on AstroPh at $P = 1$, in [Aligner start, training and output scale on AstroPh at $P = 1$ analysis](log/2026-11-08-aligner-start-astroph-p1.md#aligner-start-training-and-output-scale-on-astroph-at-p--1-analysis). The start does not matter; the triplet training causes three quarters of the cost.
2. **Reproducibility.**
   - Find out why the CITESEER triplet runs of test3 and of today differ. Compare aligner timings and environment variables, and log the full aligner configuration at the start of each run.
   - Fix the LPA seed and repeat every recent configuration at least $5$ to $10$ times, so that the weighting and loss comparisons of this page have error bars on matched partitions.
3. **Weighting between the extremes.** Use a cut share between $\delta$ (uniform) and $1/(2 - \delta)$ (inverse fraction), for example weights $\delta^{-\beta}$ with $0 < \beta < 1$. On AstroPh the two extremes trade within against cross ranking.
4. **Measure the quantities the theorems are about.**
   - Log per-scope terms under uniform weighting.
   - Log the cosine of cut pairs, within pairs and sampled negatives ($\kappa$).
   - This would test Theorems 17 and 18 directly, instead of inferring them from hinge means.
5. **Negative sampling.** Theorem 18 shows that close negatives keep within triplets active whatever the alignment. Compare uniform negatives with negatives drawn from outside the anchor's partition, or with a curriculum of hard negatives.
6. **Reconstruction without size bias.** Use a per-partition top-$k$ or a balanced partitioner, since the global top-$k$ bias against the giant AstroPh partition limits reconstruction in every run.
7. **Coverage.**
   - Rerun Enron and AS-Oregon, and $P = 8$, with the current code (all links), since test3 used only $20000$ links there.
   - Extend to $RF > 1$, where replica links give a direct alignment signal (Theorem 9).
   - Use the validation split (`alignment_validation=true`) for early stopping.
8. **Combine Procrustes and triplet.** Procrustes alone recovers half of triplet's cross MRR on CITESEER and keeps norms. Use it as the initialisation of the residual aligner, so that the triplet loss only has to learn the non-isometric part.


# Current recommended configuration

From the latest analysis,
[2026-11-10 output scale and start on AstroPh and Enron](log/2026-11-10-output-scale-and-start-astroph-enron.md#recommended-defaults-for-the-next-experiments):

| Setting | Value | Environment variable | Code default (2026-11-16) |
|---|---|---|---|
| Aggregator | `triplet` | `--aggregator triplet` | `triplet` |
| Output scale | `unit` | `VERTEX_VOYAGE_ALIGNMENT_OUTPUT_SCALE=unit` | `auto` (`unit` for triplet) |
| Start | `identity` | `VERTEX_VOYAGE_ALIGNMENT_INITIALISATION=identity` | `identity` |
| Link weighting | `uniform` | `VERTEX_VOYAGE_ALIGNMENT_LINK_WEIGHTING=uniform` | `uniform` |
| Steps, margin, learning rate | 100, 0.5, 0.01 | `VERTEX_VOYAGE_ALIGNMENT_TRAIN_STEPS`, … | 100, 0.5, 0.01 |
| Controls in every comparison | `mean` and 0 steps, same seed | `VERTEX_VOYAGE_ALIGNMENT_TRAIN_STEPS=0` | — |

The Procrustes start must not be used until it is fixed: self-loops were
taken as replica anchors, and an identity rotation froze the residual. Commit
`5152253` addresses the identity-rotation case; a re-run is pending.

# Open issues

1. **Aligner cost at $P = 1$.** On degree-heterogeneous graphs (AstroPh,
   Enron) the triplet aggregator loses 0.14 MRR already at $P = 1$, most of
   its remaining gap to the unpartitioned model. About three quarters of
   that cost comes from the training itself.
2. **Reproducibility.** Aligner training is not bit-reproducible at a fixed
   seed: two Enron identity runs with the same seed ended at different
   losses. The recent CITESEER runs do not reproduce the 10-run "test3"
   results. Most comparisons after test3 rest on single runs.
3. **Coverage.** Enron and AS-Oregon need re-runs with the current code (test3
   used 20,000 links there), as do $P = 8$ and $RF > 1$.

The full list of next runs is under [Future work](#future-work) and in
each log entry. Seeded repetition is the next step ([ROADMAP](../../ROADMAP.md)).

# Analysis log

Each entry records one analysis of a set of runs, in chronological order.
Theorems 11–18 were derived in these entries.

| Date | Entry | Finding |
|---|---|---|
| 2026-10-13 | [Synthetic check on an SBM](log/2026-10-13-synthetic-check.md) | On an SBM, both losses reach the oracle on cut edges (cross AUC 0.81 vs 0.80 at leak 0.15); a sanity check only |
| 2026-11-07 | [Contrastive loss, CITESEER](log/2026-11-07-contrastive-citeseer.md) | Collapse and gradient conflict (Theorems 11–13) |
| 2026-11-07 | [Contrastive loss, AstroPh](log/2026-11-07-contrastive-astroph.md) | Recall splits by scope (Theorem 14) |
| 2026-11-07 | [Cross-weighted contrastive, AstroPh](log/2026-11-07-cross-weighted-contrastive-astroph.md) | Weighting by $1/\delta$ shifts the frustration but does not remove it (Theorems 15–16) |
| 2026-11-07 | [Cross-weighted contrastive, CITESEER](log/2026-11-07-cross-weighted-contrastive-citeseer.md) | No improvement on CITESEER |
| 2026-11-07 | [Cross-weighted triplet, CITESEER](log/2026-11-07-cross-weighted-triplet-citeseer.md) | Triplet with orthogonal negative = cosine threshold (Theorem 17) |
| 2026-11-07 | [Cross-weighted triplet, AstroPh](log/2026-11-07-cross-weighted-triplet-astroph.md) | Correlated negatives keep within triplets active (Theorem 18) |
| 2026-11-07 | [Uniform triplet, CITESEER](log/2026-11-07-uniform-triplet-citeseer.md) | Uniform weighting is better overall |
| 2026-11-07 | [Uniform triplet, AstroPh](log/2026-11-07-uniform-triplet-astroph.md) | Same conclusion on AstroPh |
| 2026-11-07 | [Earlier 10-run experiments (test3)](log/2026-11-07-test3-ten-run-experiments.md) | Triplet beats no alignment over 10 runs; Procrustes recovers half of triplet's cross MRR |
| 2026-11-08 | [Aligner start, AstroPh $P = 1$](log/2026-11-08-aligner-start-astroph-p1.md) | The start does not matter; training causes most of the $P = 1$ cost |
| 2026-11-08 | [Aligner training, AstroPh $P = 4$](log/2026-11-08-aligner-training-astroph-p4.md) | Training gains only through cut edges |
| 2026-11-08 | [Mean vs triplet, AstroPh $P = 4$](log/2026-11-08-mean-vs-triplet-astroph-p4.md) | Much of the gain over mean is the normalisation |
| 2026-11-10 | [Output scale and start, AstroPh and Enron](log/2026-11-10-output-scale-and-start-astroph-enron.md) | `unit` output scale wins all 12 matched pairs; Procrustes start broken |
