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

### Prequential link prediction (pipeline T)

`vv temporal_test --link-prediction-model hadamard` (or `bilinear`) runs a
[prequential](glossary.md#prequential-evaluation) (test-then-train)
evaluation, following Gama et al. 2013. Each buffer is first used to test the
decoder and then to train it. The code is in
[`vertex_voyage/tasks/prequential_link_prediction.py`](../vertex_voyage/tasks/prequential_link_prediction.py).

**Setting.**
- The event stream is cut into buffers $B_0, B_1, \dots, B_{T-1}$, indexed
  from $0$ as in the code. Each buffer is a set of undirected edges
  $\{u, v\}$.
- $E_t = \bigcup_{i \le t} B_i$ and $V_t$ are the edges and vertices after
  buffer $t$, with $E_{-1} = V_{-1} = \emptyset$.
- $z^{(t)} : V_t \to \mathbb{R}^d$ is the aggregated embedding after buffer
  $t$, i.e. after the models, the partitioner and the aggregator have all
  been updated with $B_t$.
- $\pi_t(v)$ is the set of partitions that hold $v$ after buffer $t$.

**Assumptions.**
- A1. Buffers are processed in stream order. The `--ordering` flag sets
  that order.
- A2. The setting is transductive: only the vertices of $V_{t-1}$ have a
  vector in $z^{(t-1)}$.
- A3. A pair that is not in $E_t$ is a valid negative at $t$, even if it
  becomes an edge later (Poursafaei et al. 2022).
- A4. The stream only adds edges. A buffer that removes an edge raises an
  error.

**Edge classes.** Every edge $\{u, v\}$ of $B_t$ falls in the first class
that matches:
1. self-loop: $u = v$;
2. new vertex: $u \notin V_{t-1}$ or $v \notin V_{t-1}$;
3. repeated: $\{u, v\} \in E_{t-1}$, or it already occurred in $B_t$;
4. new edge: every other edge.

The positives $P_t$ are the new edges. A positive is *within* if
$\pi_{t-1}(u) \cap \pi_{t-1}(v) \ne \emptyset$ and *cross* otherwise, the
same rule as in the static pipeline. The other three classes are counted in
every buffer report but are not scored.

**Negatives.** Let
$S_t = \{\{u, v\} : u, v \in V_{t-1},\ u \ne v,\ \{u, v\} \notin E_t\}$ be
the complement of the snapshot among the known vertices. The negatives
$N_t \subseteq S_t$ are $\min(r\,|P_t|, |S_t|)$ pairs drawn from $S_t$
uniformly without replacement, where $r$ is `temporal_lp_negative_ratio`
(default 1). The sampler draws uniform vertex pairs and rejects edges,
self-pairs and repeats. If `temporal_lp_negative_attempts` (default 100)
draws per negative are not enough, it enumerates $S_t$, which only happens
for small, dense graphs.

**Decoder.** $f_\theta(z_u, z_v) \in [0, 1]$, either `hadamard`,
$\sigma(w^\top(z_u \odot z_v))$, or `bilinear`, $\sigma(z_u^\top W z_v)$.
- One decoder is created per iteration and is never reinitialised (warm
  start).
- Each refit runs `temporal_lp_epochs` (default 10) epochs of SGD on the
  pairs of one buffer, without a validation split; the next buffer is the
  held-out test.
- The random forest cannot continue training, so it is rejected before the
  run starts.

**Steps for buffer $t$:**
1. Classify $B_t$ against $E_{t-1}$ and $V_{t-1}$, and split $P_t$ by
   $\pi_{t-1}$.
2. Set $E_t = E_{t-1} \cup B_t$.
3. Sample $N_t \subseteq S_t$.
4. Test: if the decoder has been trained at least once, score
   $f_\theta(z^{(t-1)}_u, z^{(t-1)}_v)$ on $P_t \cup N_t$.
5. Train: continue training $\theta$ on $(z^{(t-1)}, P_t, N_t)$.
6. Freeze $z^{(t)}$ and $\pi_t$ for buffer $t + 1$.

Steps 4 and 5 are skipped when $P_t$ or $N_t$ is empty. Buffer 0 therefore
only freezes, because every edge in it has a new vertex. The first buffer
with new edges only trains, and testing starts at the buffer after it.

**Metrics.** Metrics are reported per buffer, and pooled over all tested
pairs of an iteration. Each is computed for all positives, for within
positives and for cross positives, each time against all negatives:
- precision, recall, $F_1$ and accuracy at threshold $0.5$, the same rule as
  `evaluate_predictions`;
- [AUC and average precision](glossary.md#auc-and-average-precision). Both
  are NaN when one of the two classes is missing.

The raw labels, probabilities and scopes are stored, so other pairings, such
as within positives against within negatives, can be recomputed later. A run
stores `lp_reports_<iteration>` and `lp_pooled_<iteration>`, and logs the
mean and standard deviation of the pooled scores over iterations.

**Lemma L1 (order).** Training at the end of buffer $t$, after the test,
gives the same decoder as training at the start of buffer $t + 1$, before
$z^{(t-1)}$ is replaced. Both read $(z^{(t-1)}, P_t, N_t)$ and the same
decoder state, nothing changes them in between, and the decoder has its own
random state. This is about program order, so it is checked by the test
`test_each_buffer_is_tested_before_it_trains` rather than in Lean.

**Lemma L2 (no leakage).** Under A4, for every $t$:
$$P_t \cap \Big(E_{t-1} \cup \bigcup_{i < t} P_i\Big) = \emptyset.$$
*Proof.* Let $e \in P_t$. By definition, $e \in B_t$ and $e \notin E_{t-1}$.
Suppose $e \in P_i$ for some $i < t$. Then $e \in B_i$, and $B_i \subseteq
E_{t-1}$ because $E_{t-1}$ is the union of $B_0, \dots, B_{t-1}$. This
contradicts $e \notin E_{t-1}$. $\blacksquare$

*Consequence.* The embedding $z^{(t-1)}$ was trained on $E_{t-1}$ and the
decoder on $P_0, \dots, P_{t-1}$, so neither has seen a positive it is
tested on. This relies on freezing $z^{(t-1)}$ before $B_t$ reaches the
partitioner and the aggregator. An embedding computed from copies of the
models after $B_t$ would still go through an aligner trained on $B_t$.

*Remark.* A pair of $P_t$ can be a negative of $N_{t-1}$, because it was not
an edge at $t - 1$. That is information from time $t - 1$, not leakage.

Lean: `PrequentialLinkPrediction.no_leakage`, and
`positives_pairwise_disjoint` for $P_t \cap P_i = \emptyset$ when $i < t$.

**Lemma L3 (valid negatives).** If $N_t \subseteq S_t$, then
$N_t \cap E_t = \emptyset$ and $N_t \cap P_t = \emptyset$, and every pair of
$N_t$ joins two distinct vertices of $V_{t-1}$.
*Proof.* Every pair of $S_t$ lies outside $E_t$, has distinct endpoints and
joins vertices of $V_{t-1}$, by definition. Since $P_t \subseteq B_t
\subseteq E_t$, a pair outside $E_t$ is not in $P_t$. $\blacksquare$
Lean: `PrequentialLinkPrediction.negatives_valid` and
`stream_negatives_disjoint_positives`.

**Lemma L4 (chance level).** Let $s$ be a score that is independent of the
label. Its expected AUC is
$\Pr[s^+ > s^-] + \tfrac12 \Pr[s^+ = s^-] = \tfrac12$ (Hanley and McNeil
1982). With $r = 1$, its expected accuracy at any threshold is $\tfrac12$.
This lemma is not formalised.

**Example.** Let $E_0 = \{01, 12, 23\}$, so $V_0 = \{0, 1, 2, 3\}$, and let
$B_1 = (0,2), (3,4), (1,2), (2,2)$.
- $02$ is a new edge, so $P_1 = \{02\}$.
- $34$ has the new vertex $4$.
- $12$ is repeated.
- $22$ is a self-loop.

Then $E_1 = E_0 \cup \{02, 34\}$, $S_1 = \{03, 13\}$, and $N_1$ is one of
these two pairs. This is the unit test `test_worked_example`.

**Corner cases.**
- *Empty complement.* If $S_t$ is smaller than requested, all of it is
  used and a warning is logged.
- *Retraining from scratch.* While a partition model has fewer vertices
  than `retrain_threshold` (by default 10% of the dataset), DynNode2Vec
  retrains it from scratch. $z^{(t-2)}$ and $z^{(t-1)}$ then live in
  unrelated coordinates, so the warm-started decoder transfers badly early in
  the stream. This is why the per-buffer series is stored, not only the
  pooled score.
- *Checkpoints.* `link_prediction_model` is a run parameter only when it is
  set. A checkpoint created with a different setting fails the metadata
  check instead of being reused.

**Reproducibility.**
- The negatives come from a `np.random.Generator` seeded with the iteration
  seed.
- The decoder is created and trained inside `torch.random.fork_rng`, with
  one seed per buffer.
- So turning link prediction on does not change the embedding: a seeded run
  gives the same reconstruction F1 series with and without it (test
  `test_same_scores_and_embedding_with_and_without_link_prediction`).
- Without this isolation, the decoder shifts the draws of the triplet
  aligner and the F1 changes.

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
