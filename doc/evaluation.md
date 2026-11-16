# Evaluation metrics and statistical protocol

All tables in [`results/`](../results/README.md) use the definitions below.
Notation is that of [methods.md](methods.md): $G = (V, E)$ is the graph seen
so far, $\pi$ the partition map, $P$ the number of partitions, and
$z_v \in \mathbb{R}^d$ the final (aggregated) embedding of $v$.

## 1. Graph reconstruction F1 ("RESTORE F1")

**Reconstructed graph.** Let $k = |E|$. The reconstructed edge set
$\hat E$ consists of the $k$ vertex pairs $\{u, v\}$ with the smallest
Euclidean distance $\lVert z_u - z_v \rVert_2$
(`reconstruct` in [`vertex_voyage/reconstruction.py`](../vertex_voyage/reconstruction.py),
computed by `get_reconstructed_edges` in
[`vertex_voyage_native/src/lib.rs`](../vertex_voyage_native/src/lib.rs)).
Write $N(v)$ for the neighbours of $v$ in $G$ and $\hat N(v)$ for its
neighbours in $\hat E$.

**Score.** `get_f1_score` averages per-vertex ratios over all vertices
(vertices with an empty denominator contribute $0$):

$$
\mathrm{Prec} = \frac{1}{|V|}\sum_{v \in V} \frac{|N(v) \cap \hat N(v)|}{|\hat N(v)|},
\qquad
\mathrm{Rec} = \frac{1}{|V|}\sum_{v \in V} \frac{|N(v) \cap \hat N(v)|}{|N(v)|},
\qquad
F_1 = \frac{2\,\mathrm{Prec}\,\mathrm{Rec}}{\mathrm{Prec} + \mathrm{Rec}}.
$$

This is a vertex-averaged score, not the edge-level
$2TP/(2TP + FP + FN)$. [E08](../results/E08-f1-decomposition/README.md)
analyses the difference and decomposes the score into within-partition and
cross-partition parts.

**Example.** A path $a - b - c$ with $\hat E = \{ab, ac\}$:
$N(a) = \{b\}, \hat N(a) = \{b, c\}$; $N(b) = \{a, c\}, \hat N(b) = \{a\}$;
$N(c) = \{b\}, \hat N(c) = \{a\}$. Then
$\mathrm{Prec} = (1/2 + 1 + 0)/3 = 1/2$,
$\mathrm{Rec} = (1 + 1/2 + 0)/3 = 1/2$, and $F_1 = 1/2$.

**Streaming runs.** $F_1$ is recorded after every buffer. The headline number
is the $F_1$ after the last buffer (`iteration_f1s_*.pkl`, last element).

**Scoped variants.** `get_macro_f1_report` and
`get_pair_probability_f1_report` split pairs into *within* ($\pi(u) = \pi(v)$)
and *cross* ($\pi(u) \ne \pi(v)$) and report precision, recall and $F_1$ per
scope. Their quantities ($S$, $\alpha$, $\delta$) are defined in E08.

## 2. Partition quality

**Edge-cut ratio.**
$\delta = |\{\{u,v\} \in E : \pi(u) \ne \pi(v)\}| / |E|$.
For uniformly random assignment, every edge is cut with probability
$1 - 1/P$, so $\mathbb{E}[\delta] = (P-1)/P$ (linearity of expectation; see
Stanton and Kliot, KDD 2012).

**Balance.** For partition sizes $s_1, \dots, s_K$ with mean
$\bar s = \frac{1}{K}\sum_i s_i > 0$, the code
(`get_partition_average_balance` in
[`vertex_voyage/partitioning.py`](../vertex_voyage/partitioning.py)) computes

$$
\beta = \frac{1}{K}\sum_{i=1}^{K}\frac{|s_i - \bar s|}{\bar s},
$$

and the result tables report $B = 1 + \beta$.

> **Theorem (range of the balance metric).** Let $K \ge 1$ and
> $s_1, \dots, s_K \ge 0$ with $\sum_i s_i > 0$. Then
> $0 \le \beta \le 2 - \tfrac{2}{K}$, so $1 \le B \le 3 - \tfrac{2}{K}$.
> The upper bound is attained when all vertices are in one partition.
>
> *Lean:* `balance_nonneg`, `balance_le`, `balance_of_single_partition` in
> [`proofs/VertexVoyageProofs/PartitionBalance.lean`](../proofs/VertexVoyageProofs/PartitionBalance.lean).
>
> *Proof.* Let $S = \sum_i s_i$, so $\bar s = S/K$ and
> $\beta = \frac{1}{S}\sum_i |s_i - \bar s|$. Non-negativity is immediate.
> For any reals $x$ and $m$, $|x - m| = x + m - 2\min(x, m)$. Summing over
> $i$ gives
> $\sum_i |s_i - \bar s| = S + K\bar s - 2\sum_i \min(s_i, \bar s) = 2S - 2\sum_i\min(s_i, \bar s)$.
> There are two cases. If some $s_j \ge \bar s$, its term alone gives
> $\sum_i \min(s_i, \bar s) \ge \min(s_j, \bar s) = \bar s$, since all
> terms are non-negative. Otherwise every term equals $s_i$, and the sum is
> $S \ge \bar s$. In both cases $\sum_i \min(s_i, \bar s) \ge \bar s$, so
> $\beta \le (2S - 2\bar s)/S = 2 - 2/K$. For $s = (c, 0, \dots, 0)$, the
> deviations are $c - c/K$ once and $c/K$ for the other $K-1$ partitions.
> They sum to $2c - 2c/K$, so $\beta = 2 - 2/K$. $\square$
>
> *Examples.* $K = 2$: $B \in [1, 2]$. $K = 4$: $B \in [1, 2.5]$. These are
> exactly the values in the $\mu = 0$ rows of
> [E04](../results/E04-balance-penalty/README.md), where the partitioner
> puts every vertex in one partition.

Seminar 2 ([E00](../results/E00-legacy-distributed/README.md)) reports
$\beta$, not $B$.

**Repartitioning rate.** The number of vertices whose partition changes
while a buffer is processed, divided by the number of vertices seen so far.

## 3. Link prediction

Protocol ([E09](../results/E09-link-prediction/README.md)):
1. Remove 20% of the edges as the test set.
2. Embed the remaining graph.
3. Train a decoder on training edges against sampled negatives.
4. Rank each test edge against 500 negatives (the HeaRT protocol).

Decoders: hadamard $\sigma(w^\top(z_u \odot z_v))$, bilinear
$\sigma(z_u^\top W z_v)$, and a random forest on $[z_u; z_v]$.

Metrics:
- $\mathrm{MRR} = \frac{1}{|T|}\sum_{e \in T} 1/\mathrm{rank}(e)$.
- Hits@k, the fraction of test edges ranked $\le k$.
- Both are also reported per scope (within and cross).

Ties between a positive and negatives must be broken explicitly. E09 shows
that RDF results are sensitive to this. The reference protocol of the
dynnode2vec paper is summarised in
[dynnode2vec-reference-protocol.md](dynnode2vec-reference-protocol.md).

## 4. Statistical protocol

- **Seeds.** Each configuration has $n = 10$ runs with seeds $42..51$
  ([guide/seeding.md](guide/seeding.md)). Tables give
  $\bar F \pm \sigma$, with $\sigma$ the population standard deviation, and
  $n$.
- **Two-group comparisons:** Welch's t-test (mean shift), Mann–Whitney U
  (distribution shift, no normality assumption), and Cohen's $d$ (effect
  size). A difference is called *significant* if $p < 0.05$ for Welch's
  test, and the result is reported together with $d$. With $n = 10$ per
  group, the smallest attainable Mann–Whitney $p$ is about
  $1.8 \cdot 10^{-4}$.
- **Multiple comparisons.** Analyses up to 2026-11 do not correct for
  multiple comparisons. New analyses should use a Bonferroni correction, or
  Friedman/ANOVA with a post-hoc test across all $P$ at once
  ([ROADMAP.md](../ROADMAP.md)).

Definitions of all tests: [glossary.md](glossary.md#evaluation-and-statistics).
