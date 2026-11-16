# E08 — Why reconstruction F1 can rise with the number of partitions

| | |
|---|---|
| **Research question** | [RQ1](../README.md#research-questions) |
| **Status** | ✅ done: theory proved in Lean; Erdős–Rényi sweep finished with a negative verdict |
| **Period** | 2026-09-28 → 2026-10-28 |
| **Pipeline** | S, static (`vv test`, LPA), 10 runs per cell |
| **Datasets** | CITESEER, DBLP, AstroPh, Enron, Zachary, Erdős–Rényi $G(n, p)$ |
| **Headline** | $\mathbb{E}[F_1] = S\,(1 - (1 - \alpha)\delta)$. In every run $\Pr[\text{reconstructed} \mid \text{cross edge}] = 0$, so recall is capped at $1 - \delta$ and any F1 change with $P$ comes from within-partition reconstruction. The hypothesis that low average degree explains CITESEER's rise is **not confirmed**: the gain is a chance-floor effect, not a skill gain. |
| **Proofs** | [`PartitioningF1.lean`](../../proofs/VertexVoyageProofs/PartitioningF1.lean), [`PartitioningF1NodeAveraged.lean`](../../proofs/VertexVoyageProofs/PartitioningF1NodeAveraged.lean), [`PartitioningF1ChanceFloor.lean`](../../proofs/VertexVoyageProofs/PartitioningF1ChanceFloor.lean), `ErdosRenyi*.lean` |
| **Open issues** | Ten follow-up experiments are listed in [Additional experiments](#additional-experiments), most importantly a fixed-$k$ partitioner and an SBM sweep. |

**Question.** [E03](../E03-partition-scaling/README.md) shows reconstruction F1
*rising* with $P$ on sparse graphs such as CITESEER, although partitions are
embedded independently. Is that a real quality gain? How does F1 decompose
into within-partition and cross-partition parts, and does graph density
explain which way it moves?

# Setup 

Let $F_1$ denote the reconstruction F1 score for a given partitioning setup, treated as a per-edge score averaged over a uniformly random edge $(u,v)$. Since it is random variable, we express expectation as:

Let $\pi(x)$ denote the partition of vertex x. Both conditional expectations below require $0 < p(\pi(u) = \pi(v)) < 1$.

The decomposition assumes $F_1$ is an average of per-edge scores. The F1 score itself, $2TP/(2TP + FP + FN)$, is a nonlinear ratio of edge counts, so for the global F1 the decomposition holds only approximately.

$$ \mathbb{E}[F_1] = \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] p(\pi(u) = \pi(v)) + \mathbb{E}[F_1 \mid \pi(u) \neq \pi(v)] p(\pi(u) \neq \pi(v)) = \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] p(\pi(u) = \pi(v)) + \mathbb{E}[F_1 \mid \pi(u) \neq \pi(v)] (1 - p(\pi(u) = \pi(v))) $$

Now, lets say that $\mathbb{E}[F_1 \mid \pi(u) \neq \pi(v)] = \alpha \mathbb{E}[F_1 \mid \pi(u) = \pi(v)]$, where $\alpha \geq 0$ is a scaling factor representing the relative impact of cross-partition edges on the reconstruction F1 score. Since $F_1 \geq 0$ and $\mathbb{E}[F_1 \mid \pi(u) = \pi(v)] > 0$, $\alpha \geq 0$ is forced, but $\alpha \leq 1$ is not: the range $[0,1]$ would exclude the case $\alpha > 1$ discussed below.

Then, our expected reconstruction F1 score can be expressed as:

$$ \mathbb{E}[F_1] = \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] p(\pi(u) = \pi(v)) + \alpha \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] (1 - p(\pi(u) = \pi(v))) $$

Also, $1 - p(\pi(u) = \pi(v)) = p(\pi(u) \neq \pi(v))$ is actually the probability of an edge being a cross-partition edge - edge cut fraction so $p(\pi(u) \neq \pi(v)) = \frac{|\Delta|}{|E|} = \delta$ where $|\Delta|$ is the number of cross-partition edges and $|E|$ is the total number of edges.

Substituting this into our expression for the expected reconstruction F1 score, we get:

$$ \mathbb{E}[F_1] = \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] (1 - \delta) + \alpha \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] \delta $$

Function $\mathbb{E}[F_1]$ is thus a convex combination of the expected reconstruction F1 score for within-partition edges (weight $1-\delta$) and for cross-partition edges (weight $\delta$), the latter being $\alpha \, \mathbb{E}[F_1 \mid \pi(u) = \pi(v)]$. Writing $S = \mathbb{E}[F_1 \mid \pi(u) = \pi(v)]$ it simplifies to

$$ \mathbb{E}[F_1] = S \, (1 - (1 - \alpha) \delta) $$

so $\mathbb{E}[F_1]$ is affine in $\delta$ with slope $-(1-\alpha) S$. For $S > 0$ it strictly decreases in $\delta$ iff $\alpha < 1$, is constant iff $\alpha = 1$, and strictly increases iff $\alpha > 1$. For $\delta \in [0,1]$ it lies between $S \min(1,\alpha)$ and $S \max(1,\alpha)$.

When $\alpha = 0$, cross-partition edges do not contribute to the reconstruction F1 score, and the expected reconstruction F1 score is solely determined by within-partition edges:

$$ \mathbb{E}[F_1] = \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] (1 - \delta) $$

When $\alpha = 1$, cross-partition edges contribute equally to the reconstruction F1 score as within-partition edges, and the expected reconstruction F1 score is independent of the edge cut fraction $\delta$:

$$ \mathbb{E}[F_1] = \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] (1 - \delta) + \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] \delta = \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] $$

In other experiments, we also observed that the value of $\alpha$ can be higher than 1, indicating that cross-partition edges can sometimes have a disproportionately large impact on the reconstruction F1 score compared to within-partition edges. In that case cutting more edges raises the expected score.

The derivation above is formalized in [`proofs/VertexVoyageProofs/PartitioningF1.lean`](../../proofs/VertexVoyageProofs/PartitioningF1.lean) (law of total expectation, the closed form, the three cases of $\alpha$, monotonicity in $\delta$, bounds, $\alpha \geq 0$, and $\delta = |\Delta|/|E|$ for a uniformly random edge).

In this experiment, we vary number of partitions and datasets and measure $\mathbb{E}[F_1 \mid \pi(u) = \pi(v)]$, $\mathbb{E}[F_1 \mid \pi(u) \neq \pi(v)]$, and $\delta$ to understand their impact on the expected reconstruction F1 score.

We then analyze how changes in the number of partitions and the resulting edge cut fraction $\delta$ affect the expected reconstruction F1 score, considering different values of the scaling factor $\alpha$.

# Node-level expectation 

We have expectation of F1 score, now taking $u$ to be a vertex drawn uniformly at random from $V$ (rather than an edge endpoint), so that $p(\pi(u) = i) = |S_i|/|V|$:

$$ \mathbb{E}[F_1] = \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] (\alpha + (1 - \alpha) \sum_{i=1}^k \frac{|S_i|}{|V|} p(\pi(v) = i \mid \pi(u) = i)) $$

This is exactly the earlier closed form $\mathbb{E}[F_1] = S(1-(1-\alpha)\delta)$ with $S = \mathbb{E}[F_1\mid\pi(u)=\pi(v)]$, once $\delta = 1 - p(\pi(u)=\pi(v))$ is expanded by the law of total probability over the $k$ partitions, $p(\pi(u)=\pi(v)) = \sum_{i=1}^k \frac{|S_i|}{|V|} p(\pi(v)=i\mid\pi(u)=i)$ — this uses only that $u$'s marginal distribution is uniform over $V$; nothing is assumed about $v$'s distribution beyond the conditional $p(\pi(v)=i\mid\pi(u)=i)$. Formalized as `prob_samePart_eq_sum_partShare_mul_condProb` and `expectedScore_eq_alpha_form` in [`proofs/VertexVoyageProofs/PartitioningF1NodeAveraged.lean`](../../proofs/VertexVoyageProofs/PartitioningF1NodeAveraged.lean).

So this expectation depends on the following factors:

- The scaling factor $\alpha$ for cross-partition edges
- The distribution of nodes across partitions, represented by $\frac{|S_i|}{|V|}$ for each partition $i$
- The conditional probability $p(\pi(v) = i \mid \pi(u) = i)$ for each partition $i$ - representing the likelihood that a node $v$ is in partition $i$ given that node $u$ is in the same partition.
- The number of partitions $k$, which affects the summation and the distribution of nodes across partitions.
- The overall structure of the network and how nodes are connected, which can influence the conditional probabilities and the impact of cross-partition edges.
- The edge cut fraction $\delta$, which represents the proportion of edges that are cut due to partitioning and affects the contribution of cross-partition edges to the expected F1 score.

**Special cases:**

- When $k=1$, there is only one partition, so all edges are within-partition edges, and the expected F1 score depends solely on $\mathbb{E}[F_1 \mid \pi(u) = \pi(v)]$.
- When $k \to \infty$, the number of partitions becomes very large, and most edges are likely to be cross-partition edges, making the expected F1 score more dependent on the scaling factor $\alpha$ and the edge cut fraction $\delta$.

## Decomposing F1 score expectation 

Conditioning on $\pi(u) = \pi(v)$ changes which partition $u$ is likely to be in, so by Bayes' rule the correct mixing weight for decomposing the within-partition expectation is the *conditional* probability $p(\pi(u) = i \mid \pi(u) = \pi(v))$, not the bare, unconditional share $|S_i|/|V| = p(\pi(u) = i)$:

$$ \mathbb{E}[F_1 \mid \pi(u) = \pi(v)] = \sum_{i=1}^k p(\pi(u) = i \mid \pi(u) = \pi(v)) \, \mathbb{E}[F_1 \mid \pi(u) = \pi(v) = i] $$

This reduces to a $|S_i|/|V|$-weighted sum only when $p(\pi(u) = \pi(v) \mid \pi(u) = i)$ is the same for every partition $i$ — a homogeneity assumption (every partition has the same internal "cohesion"), not a general identity. Formalized as `condMean_eq_sum_condProb_mul_condMean` (the decomposition) and `condProb_partition_eq_share_iff` (the iff-characterization of when the $|S_i|/|V|$ shortcut is valid) in [`proofs/VertexVoyageProofs/PartitioningF1NodeAveraged.lean`](../../proofs/VertexVoyageProofs/PartitioningF1NodeAveraged.lean).

Within a fixed partition $i$, writing $A = \{uv \in E\}$ (true edge), $B = \{uv \in \bar{E}\}$ (reconstructed edge), $TP = p(A \cap B \mid \pi(u)=\pi(v)=i)$, $FP = p(B\mid\pi(u)=\pi(v)=i) - TP$, $FN = p(A\mid\pi(u)=\pi(v)=i) - TP$: since $2TP+FP+FN=p(A\mid\cdot)+p(B\mid\cdot)$ is a pure algebraic identity in $TP,p(A\mid\cdot),p(B\mid\cdot)$, it holds *conditionally* on $\pi(u)=\pi(v)=i$ exactly as it does unconditionally — every probability below must be conditioned on that same event, not just some of them:

$$ \mathbb{E}[F_1 \mid \pi(u)=\pi(v)=i] = \frac{2\,p(uv\in E \land uv\in\bar E \mid \pi(u)=\pi(v)=i)}{p(uv\in E\mid\pi(u)=\pi(v)=i) + p(uv\in \bar E\mid\pi(u)=\pi(v)=i)} $$

the harmonic mean of precision $p(uv\in E\mid uv\in\bar E,\pi(u)=\pi(v)=i)$ and recall $p(uv\in\bar E\mid uv\in E,\pi(u)=\pi(v)=i)$. Formalized as `condProb_two_TP_add_FP_add_FN_eq` in `PartitioningF1NodeAveraged.lean`.

**These two exact facts do not combine into a single closed form for $\mathbb{E}[F_1\mid\pi(u)=\pi(v)]$.** $TP$, $p(A\mid\pi(u)=\pi(v))$ and $p(B\mid\pi(u)=\pi(v))$ each decompose *exactly* as the $p(\pi(u)=i\mid\pi(u)=\pi(v))$-weighted sum of their per-partition values above (three applications of the decomposition theorem, to the indicators of $A$, $B$, and $A\cap B$) — but $F_1$ is their *ratio*, and a ratio of weighted sums is not the weighted sum of the ratios (the same nonlinearity already flagged in the Setup section for the edge-level decomposition). Concretely: with equal weights $\tfrac12,\tfrac12$ and partitions realizing $(TP,p(A\mid\cdot),p(B\mid\cdot)) = (\tfrac12,\tfrac12,\tfrac12)$ (perfect reconstruction, $F_1=1$) and $(\tfrac1{20},\tfrac1{10},\tfrac1{10})$ ($F_1=\tfrac12$), the exact mixture gives $\mathbb{E}[F_1\mid\pi(u)=\pi(v)] = \tfrac{11}{12}$, not the naively mixed $\sum_i p(\pi(u)=i\mid\pi(u)=\pi(v))\,\mathbb{E}[F_1\mid\pi(u)=\pi(v)=i] = \tfrac34$. Formalized as `F1_does_not_distribute_over_mixture` in `PartitioningF1NodeAveraged.lean`.

Where E is set of original edges and $\bar{E}$ is the set of reconstructed edges. Here, $uv \in E$ indicates that the edge between nodes $u$ and $v$ exists in the original graph, and $uv \in \bar{E}$ indicates that the edge between nodes $u$ and $v$ exists in the reconstructed graph.

## Computing $\mathbb{E}[F_1 \mid \pi(u)=\pi(v)]$ exactly

There is still an exact way to get $\mathbb{E}[F_1\mid\pi(u)=\pi(v)]$, without going through the (invalid) per-partition average. Applying the harmonic-mean identity directly to the event $\pi(u)=\pi(v)$ itself, with no reference to individual partitions at all:

$$ \mathbb{E}[F_1\mid\pi(u)=\pi(v)] = \frac{2\,p(uv\in E \land uv\in\bar E\mid\pi(u)=\pi(v))}{p(uv\in E\mid\pi(u)=\pi(v)) + p(uv\in\bar E\mid\pi(u)=\pi(v))} $$

which is just `condProb_two_TP_add_FP_add_FN_eq` instantiated at $R := \{\pi(u)=\pi(v)\}$ instead of $R := \{\pi(u)=\pi(v)=i\}$.

If a per-partition sum is wanted anyway — e.g. because $TP_i, p(A\mid i), p(B\mid i)$ are computed locally per partition in practice — the exact way to combine them is **micro-averaging**: pool the weighted numerators and the weighted denominators separately, and divide only once, at the end:

$$ \mathbb{E}[F_1\mid\pi(u)=\pi(v)] = \frac{2\sum_{i=1}^k p(\pi(u)=i\mid\pi(u)=\pi(v))\,TP_i}{\sum_{i=1}^k p(\pi(u)=i\mid\pi(u)=\pi(v))\,(p(A\mid i)+p(B\mid i))} $$

This is exact because $TP$, $p(A\mid\pi(u)=\pi(v))$, and $p(B\mid\pi(u)=\pi(v))$ *individually* decompose linearly (that's the correct decomposition already established above) — only *their ratio* fails to decompose. **Macro-averaging** — dividing within each partition first, i.e. $\sum_i p(\pi(u)=i\mid\pi(u)=\pi(v))\cdot F_1^{(i)}$ — is the operation `F1_does_not_distribute_over_mixture` shows is not exact. Formalized as `F1_micro_average` (built from `condProb_eq_sum_condProb_mul_condProb`, the indicator-based specialization of the decomposition theorem) in `PartitioningF1NodeAveraged.lean`.

# Macro-averaging F1 score 

Every $F_1$ so far — $S$, the cross-partition mean, the micro-average across partitions — is built the same way: pool the raw counts (or conditional probabilities) *first*, then take exactly one ratio. The codebase's other F1 estimator, `get_f1_score` in `vertex_voyage/reconstruction.py` (the "Global scores" log line, computed over the whole graph with no partition scoping at all), does the opposite: it divides *per vertex* first, averages the resulting ratios *second*, and only *then* takes one harmonic mean. This section makes that computation precise in the note's own notation and shows it does not compute $S$, $\mathbb{E}[F_1]$, or micro-average — it is a fourth, distinct quantity.

For a vertex $u$, let $A(u) = \{v \in V : uv \in E\}$ and $B(u) = \{v \in V : uv \in \bar E\}$ be $u$'s true and reconstructed neighbor sets. Writing $u$ for the event "the first coordinate is $u$" (as in the "Node-level expectation" section, $u$ drawn uniformly from $V$), define the per-vertex conditional probabilities

$$ \mathrm{recall}(u) = p(uv \in \bar E \mid uv \in E, u), \qquad \mathrm{precision}(u) = p(uv \in E \mid uv \in \bar E, u) $$

— the same precision/recall pair from the "Decomposing" section (there conditioned on $\pi(u)=\pi(v)=i$), now conditioned on the single vertex $u$ instead of a partition. By convention $\mathrm{recall}(u) = 0$ when $A(u) = \emptyset$; $\mathrm{precision}(u)$ is left *undefined at $u$* (excluded from the average below, not set to $0$) when $B(u) = \emptyset$.

Macro-averaging takes the plain arithmetic mean of each of these across vertices — **not** the conditional mean $\mathbb{E}[\,\cdot\mid \pi(u)=\pi(v)=i\,]$ from the earlier sections, which would weight by how often the conditioning event occurs. Recall is averaged over every vertex (undefined vertices already contribute $0$); precision is averaged only over vertices where it is defined:

$$ \mathrm{Recall}^{\mathrm{macro}} = \frac{1}{|V|}\sum_{u\in V} \mathrm{recall}(u), \qquad \mathrm{Precision}^{\mathrm{macro}} = \frac{1}{|\{u : B(u)\neq\emptyset\}|}\sum_{u : B(u)\neq\emptyset} \mathrm{precision}(u) $$

and the macro-averaged global score is the harmonic mean of these two *already-averaged* numbers, taken once, at the very end:

$$ F_1^{\mathrm{macro}} = \frac{2\,\mathrm{Precision}^{\mathrm{macro}}\,\mathrm{Recall}^{\mathrm{macro}}}{\mathrm{Precision}^{\mathrm{macro}} + \mathrm{Recall}^{\mathrm{macro}}} $$

(taken to be $0$ when $\mathrm{Precision}^{\mathrm{macro}} + \mathrm{Recall}^{\mathrm{macro}} = 0$, matching `get_f1_score`'s own convention).

**$F_1^{\mathrm{macro}} \neq S$, even at $k=1$.** With a single partition, $\pi(u)=\pi(v)$ holds for every pair, so $S = \mathbb{E}[F_1\mid\pi(u)=\pi(v)]$ collapses to the plain micro/pooled global score $2\,p(A\cap B)/(p(A)+p(B))$ — dividing once, after pooling every vertex's true positives, true edges, and reconstructed edges together. $F_1^{\mathrm{macro}}$ divides once *per vertex*, before any pooling. These are different orderings of division and averaging, and — since $F_1$ is a nonlinear (ratio) function of counts, the same obstruction flagged throughout this page — they need not agree. Concretely, two vertices with $(TP,A,B) = (2,2,2)$ (perfect: $\mathrm{recall}=\mathrm{precision}=1$) and $(1,10,1)$ ($\mathrm{recall}=\tfrac1{10}$, $\mathrm{precision}=1$) give $\mathrm{Recall}^{\mathrm{macro}} = \tfrac{11}{20}$, $\mathrm{Precision}^{\mathrm{macro}}=1$, so $F_1^{\mathrm{macro}} = \tfrac{22}{31} \approx 0.710$, against the pooled $2\cdot 3/(12+3) = \tfrac25 = 0.4$. Formalized as `macroF1_ne_microF1` in `PartitioningF1NodeAveraged.lean`. Unlike `F1_does_not_distribute_over_mixture`, which needs at least two *partitions* to exhibit the mediant gap, this gap needs only two *vertices* and no partitioning at all: macro-averaging never pools raw counts across vertices in the first place, so there is no partition structure left for it to be exact about.

This is not a corner case: in a real run at $k=1$ (so $\delta=0$ and $S$ is exactly the pooled global score), `get_f1_score` reported $F_1^{\mathrm{macro}} = 0.369763$ while the pair-probability report's $S$ (pooled, as derived above) reported $0.138007$ on the same reconstruction — a real instance of the same gap, not a rounding artifact. On a graph with a heavy-tailed degree distribution, macro-averaging gives every low-degree vertex the same weight as every hub, so a handful of lucky low-degree vertices can pull $F_1^{\mathrm{macro}}$ well above the hub-dominated pooled $S$. `get_f1_score`'s "Global scores" line and the pair-probability report's $S$/$\alpha$/$\delta$ are therefore two distinct, both legitimate, statistics of the same reconstruction, not two measurements of the same quantity — they should never be expected to agree, with or without partitioning.

## Decomposition

$$ \mathrm{recall}(u) = p(uv \in \bar E \mid uv \in E, u, \pi(u) = \pi(v)) p(\pi(u) = \pi(v) \mid uv \in E, u) + p(uv \in \bar E \mid uv \in E, u, \pi(u) \neq \pi(v)) p(\pi(u) \neq \pi(v) \mid uv \in E, u) $$

$$\mathrm{precision}(u) = p(uv \in E \mid uv \in \bar E, u, \pi(u) = \pi(v)) p(\pi(u) = \pi(v) \mid uv \in \bar E, u) + p(uv \in E \mid uv \in \bar E, u, \pi(u) \neq \pi(v)) p(\pi(u) \neq \pi(v) \mid uv \in \bar E, u) $$

where $\bar E$ denotes the set reconstructed edges.

In our experiments, we report:

- $p(uv \in \bar E \mid uv \in E, u, \pi(u) = \pi(v))$ - probability of correctly reconstructing an edge within the same partition
- $p(uv \in \bar E \mid uv \in E, u, \pi(u) \neq \pi(v))$ - probability of correctly reconstructing an edge across different partitions
- $p(uv \in E \mid uv \in \bar E, u, \pi(u) = \pi(v))$ - probability of reconstructed edge being correct within the same partition
- $p(uv \in E \mid uv \in \bar E, u, \pi(u) \neq \pi(v))$ - probability of reconstructed edge being correct across different partitions
- $p(\pi(u) = \pi(v) \mid uv \in E, u)$ - probability that an edge is within the same partition given it exists
- $p(\pi(u) \neq \pi(v) \mid uv \in E, u)$ - probability that an edge is across different partitions given it exists  - this is called [Edge cut fraction](../../doc/glossary.md#edge-cut)
- $p(\pi(u) \neq \pi(v) \mid uv \in \bar E, u)$ - probability that a non-edge is across different partitions given it is reconstructed
- $p(\pi(u) = \pi(v) \mid uv \in \bar E, u)$ - probability that a non-edge is within the same partition given it is reconstructed
- $p(\pi(u) \neq \pi(v) \mid uv \in \bar E, u)$ - probability that a non-edge is across different partitions given it is reconstructed
- $\{u : B(u)\neq\emptyset\}|$ - number of vertices with at least one reconstructed edge

# Why macro F1 can rise or fall with the number of partitions

Sweeping the partition count $k\in\{1,2,4,8\}$ on CITESEER and AstroPh (`note.md`) gives opposite trends in the macro-averaged global $F_1$ (`get_f1_score`, the "Global scores" line — not $S$, which is the micro/pair-probability quantity from the previous sections):

| dataset | $k=1$ | $k=2$ | $k=4$ | $k=8$ |
|---|---|---|---|---|
| CITESEER $F_1^{\mathrm{macro}}$ | 0.3860 | 0.4478 | 0.4742 | 0.5450 |
| AstroPh $F_1^{\mathrm{macro}}$ | 0.7232 | 0.6864 | 0.5395 | 0.6803 |

CITESEER rises monotonically; AstroPh falls net (with a dip at $k=4$). Hypothesis: this tracks the two datasets' density. On the exact files this codebase loads (`data/citeseer.txt`, `data/AstroPh.txt`): CITESEER has $N=3264$, $E=4536$, density $2E/(N(N-1))\approx 0.00085$, average degree $2E/N\approx 2.78$; AstroPh has $N=18771$, $E=198050$, density $\approx 0.00112$, average degree $\approx 21.10$. The two densities themselves differ by only $\sim\!1.3\times$ — at this graph size, density in the $N^2$-normalized sense is a weak signal either way. **Average degree is the sharper statistic**: CITESEER's is barely above $2$ (most vertices have only one or two true neighbors — closer to a forest than a well-connected graph), AstroPh's is $\sim\!7.6\times$ higher. The explanation below is really about *average degree* (equivalently, how signal-starved a vertex's local neighborhood is), for which the note's "density" hypothesis is a reasonable proxy, not about the $N^2$-normalized density figure on its own.

## The mechanism: recall factors exactly when no cross edge is ever reconstructed

Across every run in the sweep, $p(uv\in\bar E\mid uv\in E,u,\pi(u)\neq\pi(v)) = 0$ *exactly*, with zero variance across all ten trials, for both datasets, at every $k$: no cross-partition edge is ever reconstructed. This isn't a rare edge case to caveat around — it's the universal case in this data, and it collapses the "Decomposing" section's law-of-total-probability split for $\mathrm{recall}(u)$ down to a single term. Writing $p(u) := p(\pi(u)=\pi(v)\mid uv\in E,u)$ (the fraction of $u$'s true neighbors kept within its own partition — its *retention share*) and $Q(u) := p(uv\in\bar E\mid uv\in E,u,\pi(u)=\pi(v))$ (its within-partition recall):

$$ \mathrm{recall}(u) = p(u)\,Q(u) + \underbrace{(1-p(u))\cdot 0}_{=\,0} = p(u)\,Q(u) $$

exactly — no approximation, for every $u$ with this data's empirical zero cross-recall. Increasing $k$ always shrinks $p(u)$ on average (finer partitions cut more of each vertex's edges), so $\mathrm{recall}(u)$ can only rise despite that if $Q(u)$ — reconstruction quality on the edges that *do* survive — rises enough to compensate. This is exactly the algebra `product_le_iff_ratio_le` in `PartitioningF1NodeAveraged.lean` makes precise:

$$ p\,Q \le p'\,Q' \iff \frac{p}{p'} \le \frac{Q'}{Q} $$

for $p,p'$ (retention share before/after) and $Q,Q'$ (within-recall before/after) all positive — recall does not drop from one $k$ to the next iff $Q$'s relative *gain* is at least as large as $p$'s relative *loss*. `probMean_mono` and its normalized form `condMean_mono` in the same file lift this from one vertex to the macro average: if the per-vertex product does not decrease pointwise across every vertex, neither does its mean — turning the single-vertex criterion above into a criterion for $F_1^{\mathrm{macro}}$ itself (via $F_1^{\mathrm{macro}}$'s own dependence on macro-recall).

Since $p(u)$ isn't directly reported (only the edge-averaged $\delta$ is), define the aggregate proxy $\hat p(k) := \mathrm{recall}(k)/Q(k)$ from the two macro-averaged quantities the codebase *does* report (`MacroF1Report.global_f1.recall` and `.recall_by_scope.within`). This is not exactly the macro average of $p(u)$ — mean-of-products $\ne$ product-of-means, the same caveat as everywhere else in this page — but it's a reasonable summary, and the real numbers make the mechanism unambiguous:

| dataset | $\hat p(1)$ | $\hat p(2)$ | $\hat p(4)$ | $\hat p(8)$ | $Q(1)$ | $Q(2)$ | $Q(4)$ | $Q(8)$ |
|---|---|---|---|---|---|---|---|---|
| CITESEER | 1.000 | 0.914 | 0.869 | 0.848 | 0.2815 | 0.3890 | 0.4678 | 0.6099 |
| AstroPh | 1.000 | 0.910 | 0.858 | 0.887 | 0.7683 | 0.7587 | 0.4804 | 0.7491 |

The retention-share proxy $\hat p$ ends up in the *same* $0.85$–$0.89$ range for both datasets at $k=8$ — partitioning cuts a comparable share of each dataset's edges. The divergence is almost entirely in $Q$: CITESEER's within-partition recall more than doubles ($0.2815\to0.6099$, $+117\%$), while AstroPh's is essentially flat-to-declining ($0.7683\to0.7491$, $-2.5\%$, dipping to $0.48$ at $k=4$ along the way). Applying the criterion at $k=1$ vs. $k=8$:

$$ \text{CITESEER: } \frac{\hat p(1)}{\hat p(8)} = 1.179 \;<\; \frac{Q(8)}{Q(1)} = 2.166 \quad\Rightarrow\quad \text{recall (and }F_1\text{) rises} $$
$$ \text{AstroPh: } \frac{\hat p(1)}{\hat p(8)} = 1.128 \;>\; \frac{Q(8)}{Q(1)} = 0.975 \quad\Rightarrow\quad \text{recall (and }F_1\text{) falls} $$

matching the observed $\mathrm{recall}(1)=0.2815\to\mathrm{recall}(8)=0.5172$ (up) for CITESEER and $\mathrm{recall}(1)=0.7683\to\mathrm{recall}(8)=0.6644$ (down) for AstroPh exactly. (Precision moves far less across $k$ for both datasets — every reconstructed edge is within-partition here too, so $\mathrm{precision}(u)=p(uv\in E\mid uv\in\bar E,u,\pi(u)=\pi(v))$ exactly, and this stays in a comparatively narrow band, $0.57$–$0.79$ across both datasets and all $k$ — so recall, not precision, is what drives the divergence.)

## A pool-size effect that is provable and density-independent — but not the whole story

Part of $Q$'s rise for smaller partitions is structural, not about embedding skill at all: `reconstruct()` builds edges by nearest-neighbor search among only the vertices *in that partition*, so shrinking a partition shrinks the candidate pool a true neighbor has to be found among. Even a hypothetical *no-skill* reconstruction — picking $M$ candidates uniformly at random out of $N$ available — recovers a fixed true neighbor with probability exactly $M/N$: the count of $M$-subsets of an $N$-set containing a fixed element is $\binom{N-1}{M-1}$ (drop the fixed element, choose the rest from what remains), against $\binom{N}{M}$ subsets overall, and cross-multiplying gives $N\binom{N-1}{M-1}=\binom{N}{M}M$ exactly — `count_subsets_containing_fixed_element_eq` in `PartitioningF1NodeAveraged.lean` (a direct restatement of Mathlib's `Nat.add_one_mul_choose_eq`). Shrinking the pool $N$ for a fixed per-vertex budget $M$ mechanically raises this baseline, for *any* graph, dense or sparse.

But this alone cannot be the (main) explanation: it predicts *some* recall improvement from partitioning regardless of density, yet AstroPh's $Q$ does not rise — it is flat-to-declining. So the pool-size tailwind is real but is being outweighed, for AstroPh, by something pulling in the other direction; for CITESEER, whatever that same something is either doesn't act, or is itself working with the tailwind rather than against it.

## The density (average-degree) hypothesis for $Q$'s divergence

The "something" is the quality of the embedding the reconstruction step is built on — this part is a domain hypothesis about the node2vec-style embedding + $k$-NN reconstruction pipeline, not a theorem: random-walk embeddings need enough *local* connectivity per vertex to produce a meaningful co-occurrence signal. On CITESEER, average degree $\approx 2.78$ means most vertices have only one or two true neighbors to begin with; training one embedding over the *entire* $3264$-vertex graph spreads a already-thin signal even thinner, and $Q(1)=0.28$ reflects that. Partitioning into smaller, community-coherent pieces lets the model represent each vertex against a far smaller, more relevant "world" — concentrating what little signal there is — and $Q$ climbs sharply as $k$ grows. On AstroPh, average degree $\approx 21.1$ means the *global* embedding already has abundant local signal per vertex ($Q(1)=0.77$, already close to the range CITESEER only reaches at $k=8$); there is little room left for partitioning to help, so only the downside — edges permanently lost to cross-partition cuts, since $p(uv\in\bar E\mid uv\in E,u,\pi(u)\ne\pi(v))=0$ always — remains, and $Q$ (and hence recall and $F_1^{\mathrm{macro}}$) trends down.

This is consistent with, but weaker than, a proof: it explains why *this* sparse graph and *this* dense graph diverge the way they do, using the exact mechanism (`product_le_iff_ratio_le`) proved above, but it does not establish that low average degree *always* produces a $Q$-improving partitioning, only that it is the most plausible reading of the one contrast actually measured here. A direct test would sweep partition count on graphs of matched size but varying average degree and check whether $Q(k)/Q(1)$ tracks $1/\bar d$ as this hypothesis predicts.

# Results 

Summary for all groups:
|    | dataset   |   partitions | precision                   | recall                      | f1                          | p reconstructed given within and true   | sigma of p reconstructed given within and true   | p reconstructed given cross and true   | sigma of p reconstructed given cross and true   | p true given within and reconstructed   | sigma of p true given within and reconstructed   | p true given cross and reconstructed   | sigma of p true given cross and reconstructed   | S                           | cross_S                     | alpha                       | delta                       |
|---:|:----------|-------------:|:----------------------------|:----------------------------|:----------------------------|:----------------------------------------|:-------------------------------------------------|:---------------------------------------|:------------------------------------------------|:----------------------------------------|:-------------------------------------------------|:---------------------------------------|:------------------------------------------------|:----------------------------|:----------------------------|:----------------------------|:----------------------------|
|  0 | AstroPh   |            1 | 0.683080 +- 0.000616 (N=10) | 0.768257 +- 0.000485 (N=10) | 0.723169 +- 0.000317 (N=10) | 0.768257 +- 0.000485 (N=10)             | 0.276115 +- 0.000387 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.683080 +- 0.000616 (N=10)             | 0.293956 +- 0.000367 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.612356 +- 0.000581 (N=10) | nan +- nan (N=0)            | nan +- nan (N=0)            | 0.000000 +- 0.000000 (N=10) |
|  1 | AstroPh   |            2 | 0.682830 +- 0.016054 (N=10) | 0.690525 +- 0.010125 (N=10) | 0.686417 +- 0.004895 (N=10) | 0.758741 +- 0.031312 (N=10)             | 0.301966 +- 0.023760 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.682830 +- 0.016054 (N=10)             | 0.313585 +- 0.002390 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.552445 +- 0.032960 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.083820 +- 0.045983 (N=10) |
|  2 | AstroPh   |            4 | 0.787728 +- 0.003128 (N=10) | 0.412146 +- 0.046430 (N=10) | 0.539535 +- 0.040628 (N=10) | 0.480388 +- 0.048970 (N=10)             | 0.420558 +- 0.009266 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.787728 +- 0.003128 (N=10)             | 0.429673 +- 0.023280 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.269380 +- 0.036836 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.082729 +- 0.003163 (N=10) |
|  3 | AstroPh   |            8 | 0.696997 +- 0.004664 (N=10) | 0.664405 +- 0.007037 (N=10) | 0.680263 +- 0.002141 (N=10) | 0.749098 +- 0.008095 (N=10)             | 0.314624 +- 0.004961 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.696997 +- 0.004664 (N=10)             | 0.320792 +- 0.002091 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.532374 +- 0.009594 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.088214 +- 0.002187 (N=10) |
|  4 | CITESEER  |            1 | 0.613962 +- 0.012323 (N=10) | 0.281512 +- 0.010791 (N=10) | 0.385983 +- 0.012352 (N=10) | 0.281512 +- 0.010791 (N=10)             | 0.432320 +- 0.004614 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.613962 +- 0.012323 (N=10)             | 0.405783 +- 0.005235 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.146539 +- 0.006186 (N=10) | nan +- nan (N=0)            | nan +- nan (N=0)            | 0.000000 +- 0.000000 (N=10) |
|  5 | CITESEER  |            2 | 0.606150 +- 0.016847 (N=10) | 0.355415 +- 0.027344 (N=10) | 0.447794 +- 0.025842 (N=10) | 0.389024 +- 0.030180 (N=10)             | 0.463761 +- 0.005242 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.606150 +- 0.016847 (N=10)             | 0.447432 +- 0.005080 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.216800 +- 0.020791 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.149713 +- 0.004559 (N=10) |
|  6 | CITESEER  |            4 | 0.568823 +- 0.010974 (N=10) | 0.406606 +- 0.010450 (N=10) | 0.474213 +- 0.010734 (N=10) | 0.467760 +- 0.013411 (N=10)             | 0.480513 +- 0.002216 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.568823 +- 0.010974 (N=10)             | 0.445689 +- 0.002815 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.257104 +- 0.007497 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.213448 +- 0.009574 (N=10) |
|  7 | CITESEER  |            8 | 0.576117 +- 0.008231 (N=10) | 0.517181 +- 0.012206 (N=10) | 0.545035 +- 0.010111 (N=10) | 0.609888 +- 0.014663 (N=10)             | 0.466402 +- 0.003471 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.576117 +- 0.008231 (N=10)             | 0.433411 +- 0.001094 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.353538 +- 0.011018 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.254916 +- 0.015318 (N=10) |
|  8 | DBLP      |            1 | 0.654751 +- 0.002504 (N=10) | 0.562123 +- 0.005835 (N=10) | 0.604884 +- 0.002672 (N=10) | 0.562123 +- 0.005835 (N=10)             | 0.398135 +- 0.001648 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.654751 +- 0.002504 (N=10)             | 0.391876 +- 0.002019 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.369541 +- 0.006756 (N=10) | nan +- nan (N=0)            | nan +- nan (N=0)            | 0.000000 +- 0.000000 (N=10) |
|  9 | DBLP      |            2 | 0.642803 +- 0.004523 (N=10) | 0.604436 +- 0.006201 (N=10) | 0.622989 +- 0.002325 (N=10) | 0.697553 +- 0.006276 (N=10)             | 0.361213 +- 0.003512 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.642803 +- 0.004523 (N=10)             | 0.365811 +- 0.002220 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.426433 +- 0.007167 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.192369 +- 0.006208 (N=10) |
| 10 | DBLP      |            4 | 0.694940 +- 0.002699 (N=10) | 0.436889 +- 0.017845 (N=10) | 0.536291 +- 0.013555 (N=10) | 0.545342 +- 0.023618 (N=10)             | 0.439383 +- 0.005847 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.694940 +- 0.002699 (N=10)             | 0.434006 +- 0.007927 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.210983 +- 0.015264 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.280617 +- 0.011052 (N=10) |
| 11 | DBLP      |            8 | 0.687816 +- 0.004215 (N=10) | 0.516492 +- 0.013676 (N=10) | 0.589821 +- 0.007783 (N=10) | 0.680204 +- 0.018187 (N=10)             | 0.401116 +- 0.010245 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.687816 +- 0.004215 (N=10)             | 0.400791 +- 0.008357 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.277992 +- 0.017229 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.330970 +- 0.007484 (N=10) |
| 12 | Enron     |            1 | 0.613861 +- 0.003172 (N=4)  | 0.119311 +- 0.000864 (N=4)  | 0.199790 +- 0.001357 (N=4)  | 0.119311 +- 0.000864 (N=4)              | 0.308221 +- 0.000889 (N=4)                       | 0.000000 +- 0.000000 (N=4)             | 0.000000 +- 0.000000 (N=4)                      | 0.613861 +- 0.003172 (N=4)              | 0.309225 +- 0.002094 (N=4)                       | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=4)                      | 0.028575 +- 0.000181 (N=4)  | nan +- nan (N=0)            | nan +- nan (N=0)            | 0.000000 +- 0.000000 (N=4)  |
| 13 | zachary   |            1 | 0.767222 +- 0.031900 (N=10) | 0.788599 +- 0.030783 (N=10) | 0.777689 +- 0.030487 (N=10) | 0.788599 +- 0.030783 (N=10)             | 0.223937 +- 0.017693 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.767222 +- 0.031900 (N=10)             | 0.276049 +- 0.015333 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.711539 +- 0.039411 (N=10) | nan +- nan (N=0)            | nan +- nan (N=0)            | 0.000000 +- 0.000000 (N=10) |
| 14 | zachary   |            2 | 0.710855 +- 0.023281 (N=10) | 0.691644 +- 0.031418 (N=10) | 0.700786 +- 0.023196 (N=10) | 0.766882 +- 0.046032 (N=10)             | 0.247769 +- 0.024372 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.710855 +- 0.023281 (N=10)             | 0.316583 +- 0.022540 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.646350 +- 0.036421 (N=10) | 0.000000 +- 0.000000 (N=9)  | 0.000000 +- 0.000000 (N=9)  | 0.126923 +- 0.061069 (N=10) |
| 15 | zachary   |            4 | 0.665751 +- 0.030691 (N=10) | 0.649618 +- 0.034002 (N=10) | 0.656803 +- 0.024125 (N=10) | 0.805159 +- 0.065797 (N=10)             | 0.234894 +- 0.052023 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.665751 +- 0.030691 (N=10)             | 0.329858 +- 0.025642 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.627577 +- 0.039456 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.000000 +- 0.000000 (N=10) | 0.232051 +- 0.062400 (N=10) |
| 16 | zachary   |            8 | 0.706640 +- 0.043209 (N=10) | 0.674936 +- 0.059689 (N=10) | 0.690225 +- 0.051255 (N=10) | 0.797052 +- 0.041266 (N=10)             | 0.239276 +- 0.029335 (N=10)                      | 0.000000 +- 0.000000 (N=10)            | 0.000000 +- 0.000000 (N=10)                     | 0.706640 +- 0.043209 (N=10)             | 0.319209 +- 0.020592 (N=10)                      | nan +- nan (N=0)                       | 0.000000 +- 0.000000 (N=10)                     | 0.652136 +- 0.043004 (N=10) | 0.000000 +- 0.000000 (N=9)  | 0.000000 +- 0.000000 (N=9)  | 0.179487 +- 0.086953 (N=10) |


# Experiment ideas

- Sweep partition count $k$ on graphs of matched size but varying average degree $\bar d$ and measure $Q(k)/Q(1)$ to test the density hypothesis. Example graphs include [Erdős–Rényi](../../doc/glossary.md#erdos-renyi-random-graph) random graphs.
- Compare the results across different graph densities to see if the low average degree consistently produces a $Q$-improving partitioning.
- Investigate the effect of partitioning on recall and $F_1^{\mathrm{macro}}$ for both sparse and dense graphs to understand the trade-offs involved.
# The Erdős–Rényi sweep: does $Q(1)/Q(8)$ depend on average degree?

The experiment proposed in the previous section was run on [Erdős–Rényi](../../doc/glossary.md#erdos-renyi-random-graph) graphs; the aggregated results are in `VertexVoyage/er_tests/note.md` and the per-trial logs in `VertexVoyage/er_tests/<k>-<N>-<100p>/<trial>.out`. **The answer is no: the sweep does not confirm that $Q(1)/Q(8)$ depends on average degree, and the data available actively argues against the average-degree reading.** This section states what was measured, which parts of the sweep carry information, what the informative parts say, and what would have to be run to decide the question.

## The design as run

Each cell of the sweep is a triple $(k, N, p)$: requested partition count $k \in \{1,2,4,8\}$, vertex count $N \in \{200, 500, 1000, 5000\}$, and edge probability $p$, encoded in the directory name as $100p$, so $p$ ranges over $\{0.01,\dots,0.05\} \cup \{0.10,\dots,0.50\}$. For $G(N,p)$ the expected average degree is

$$ \bar d = (N-1)p \approx Np, \qquad \text{expected density } \rho = p $$

so the sweep varies $\bar d$ from $2.38$ to $2500$ and $\rho$ from $0.0099$ to $0.50$. Trial counts are 10 per cell except at $N = 5000$, where they are 5 or 6. Every cell reuses **one** realized graph across all its trials — the logged edge count is identical in every trial of every cell, and identical across $k$ within a cell. The reported $\pm$ figures are therefore embedding-seed dispersion at fixed graph, not graph-to-graph dispersion; comparisons across $k$ within a cell are paired, which is a strength, but no cell carries an estimate of sampling variation in $G(N,p)$ itself. Embedding hyperparameters were selected per run by the same grid search in every cell, and in every log inspected the selected point was identical ($p_{n2v} = q_{n2v} = 0.25$, $\dim = 100$), so embedding capacity is a constant of the sweep while $N$ varies by $25\times$.

## Only 17 of the 90 multi-partition cells partitioned anything

`label_propagation_partitioner` in `vertex_voyage/partitioning.py` does not cut a graph into $k$ pieces. It calls `asyn_lpa_communities` to find communities, then [bin-packs](../../doc/glossary.md#bin-packing) the communities it found into $k$ bins with `to_constant_bin_number`. If [label propagation](../../doc/glossary.md#label-propagation) returns a *single* community — the expected outcome on a graph with no [community](../../doc/glossary.md#community-detection) structure — there is one item to pack, so one bin is non-empty and $k-1$ bins are empty, whatever $k$ was requested. The logs show exactly this, e.g. `8-500-10/0.out`: `Total number of partitions: 8`, then `Partition size: 500` followed by seven occurrences of `Skipping empty partition`, and `delta: 0.000000`.

Such a cell is formally indistinguishable from $k=1$. Writing $\pi$ for the partition map, if $\pi$ is constant then no pair is a cross pair, so the [edge cut fraction](../../doc/glossary.md#edge-cut) vanishes and conditioning on $\pi(u) = \pi(v)$ is conditioning on a certain event:

$$ (\forall v)(\pi(v) = c) \;\Longrightarrow\; \delta = p(\pi(u) \neq \pi(v)) = 0 \;\text{ and }\; \mathbb{E}[f \mid \pi(u) = \pi(v)] = \mathbb{E}[f] $$

for every statistic $f$. Formalized as `cut_fraction_eq_zero_of_constant_partition` and `condMean_within_eq_total_of_constant_partition` in [`proofs/VertexVoyageProofs/PartitioningF1ChanceFloor.lean`](../../proofs/VertexVoyageProofs/PartitioningF1ChanceFloor.lean). A collapsed cell therefore measures the *unpartitioned* pipeline: it is not a datum about partitioning at all.

Counting cells with $\delta > 0.01$: **73 of the 90 requested $k > 1$ cells collapsed**; 17 realized a cut, and only 9 of those realized the full requested $k$ in all trials. The surviving cells are exactly the sparsest ones:

| $N$ | $p$ | $\bar d$ | realized $k$ | mean non-empty blocks | $\delta(k{=}8)$ |
|---|---|---|---|---|---|
| 200 | 0.01 | 2.38 | 2, 4, 8 | 2.0, 4.0, 8.0 | 0.387 |
| 200 | 0.02 | 4.31 | 2, 4, 8 | 2.0, 4.0, 8.0 | 0.558 |
| 200 | 0.03 | 6.06 | 2, 4, 8 (partial) | 1.7, 2.2, 5.2 | 0.375 |
| 200 | 0.04 | 8.56 | 8 only, marginal | 1.1 | 0.028 |
| 500 | 0.01 | 4.97 | 2, 4, 8 | 2.0, 4.0, 8.0 | 0.617 |
| 500 | 0.02 | 10.06 | 8 only, marginal | 1.1 | 0.028 |
| 1000 | 0.01 | 9.93 | 2, 4, 8 (partial) | 1.6, 2.8, 5.2 | 0.440 |

Every cell with $p \geq 0.03$ at $N \geq 500$, and every cell at $N = 5000$, collapsed completely. **In particular the whole dense half of the sweep ($p \in \{0.10,\dots,0.50\}$, $\bar d$ from 49 to 2500) contains no partitioning measurement whatsoever.** The flatness of $Q$ in $k$ reported there — e.g. $Q = 0.3014, 0.3036, 0.3060, 0.3017$ at $N=500$, $p=0.1$ for $k=1,2,4,8$ — is not evidence that partitioning fails to change $Q$ at high average degree. It is four repetitions of the same unpartitioned experiment with different random seeds, and the spread across them ($\pm 0.002$) is a seed-noise estimate.

## The chance floor: a global pair budget makes density the no-skill baseline

`reconstruct(k, embedding, nodes)` in `vertex_voyage/reconstruction.py` emits the $k$ globally closest embedding pairs (the native kernel `get_reconstructed_edges` keeps one size-$k$ heap over all $\binom{n}{2}$ pairs), and `train_partition` passes $k := $ `pg.number_of_edges()`. So the reconstruction operates under a *global pair budget equal to the true edge count*, which is why $p(uv \in \bar E) = p(uv \in E)$ holds exactly, not approximately, in every log line of the sweep.

**Statement.** Let $P$ be the number of candidate pairs and $m \leq P$ the budget. If the reconstructed set is a uniformly random $m$-subset of the candidate pairs, then for any fixed pair $e$,

$$ p(e \in \bar E) = \frac{\binom{P-1}{m-1}}{\binom{P}{m}} = \frac{m}{P} $$

**Assumptions.** $0 < m \leq P$; the $m$-subset is uniform on all $\binom{P}{m}$ subsets; no assumption on $e$, so the baseline is the same for every pair regardless of the degrees of its endpoints. **Consequence.** With $P = \binom{n}{2}$ and $m = |E|$, the no-skill recall, precision and $F_1$ all equal the density $\rho = |E|/\binom{n}{2}$ — for $G(n,p)$, in expectation, $\rho = p$. Formalized as `fixed_element_subset_fraction` and `chanceFloor_eq_density` in `PartitioningF1ChanceFloor.lean`. **Example.** At $N=500$, $p=0.5$ the measured $Q(1) = 0.5258$ against a floor of $0.4987$: the embedding beats chance by $2.7$ percentage points, a lift of $1.05\times$.

This changes the reading of the headline $Q$-versus-$p$ curve. Raw $Q$ at fixed $N$ is U-shaped in $p$, with a minimum near $p \approx 0.1$–$0.2$ and a rise afterwards — at $N=1000$: $0.990, 0.943, 0.791, 0.598, 0.478, 0.294, 0.299, 0.362, 0.439, 0.526$ for $p = 0.01, \dots, 0.50$. The rising branch is **entirely the floor rising**. Measured against the floor, reconstruction skill decays monotonically over the whole range:

| $\rho = p$ | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|---|---|---|---|---|---|
| $Q(1)$, $N{=}1000$ | 0.990 | 0.943 | 0.791 | 0.598 | 0.478 | 0.294 | 0.299 | 0.362 | 0.439 | 0.526 |
| lift $Q(1)/\rho$ | 99.6 | 46.7 | 26.2 | 14.8 | 9.7 | 2.9 | 1.5 | 1.2 | 1.1 | 1.1 |
| excess $Q(1) - \rho$ | 0.981 | 0.923 | 0.761 | 0.558 | 0.428 | 0.194 | 0.099 | 0.061 | 0.039 | 0.025 |

At $N = 5000$ with $p \geq 0.2$ the lift is $1.02$–$1.14$: the pipeline is within a few percent of chance, and $Q$ there is reporting the budget, not the embedding.

## $Q(1)$ is governed by density, not by average degree

For $G(N,p)$ the two candidate explanatory statistics $\bar d = Np$ and $\rho = p$ are independent once $N$ varies, so the sweep can separate them. Binning the $k=1$ cells by each statistic within a $10\%$ tolerance and measuring the spread of $Q(1)$ inside each bin:

| matched statistic | bins | mean within-bin spread of $Q(1)$ |
|---|---|---|
| average degree $\bar d$ | 6 | 0.240 |
| density $\rho$ | 10 | 0.075 |

Density collapses the data about three times better, and the individual bins are decisive rather than marginal:

- **Matched $\bar d \approx 10$:** $Q(1) = 0.578$ at $(N{=}200, p{=}0.05)$, $0.895$ at $(500, 0.02)$, $0.990$ at $(1000, 0.01)$ — a spread of $0.41$ at constant average degree.
- **Matched $\bar d \approx 500$:** $Q(1) = 0.526$ at $(1000, 0.50)$ against $0.153$ at $(5000, 0.10)$ — a factor $3.4$ at constant average degree.
- **Matched $\rho = 0.03$:** $Q(1) = 0.774, 0.772, 0.791$ at $N = 200, 500, 1000$, i.e. $\bar d = 6.1, 15.5, 30.2$ — a $5\times$ change in average degree moves $Q(1)$ by $0.019$.
- **Matched $\rho = 0.40$:** $Q(1) = 0.444, 0.439, 0.413$ at $N = 500, 1000, 5000$, i.e. $\bar d = 199, 399, 1999$.

The mechanism in the previous section — random-walk embeddings need local connectivity per vertex, so low $\bar d$ starves the signal and depresses $Q(1)$ — predicts that low average degree depresses $Q(1)$. In this sweep **the direction is reversed**: the lowest-$\bar d$ cells have the *highest* $Q(1)$ in the entire dataset ($\bar d = 2.38 \Rightarrow Q(1) = 0.808$; $\bar d = 9.93 \Rightarrow Q(1) = 0.990$), and the highest-$\bar d$ cells have the lowest ($\bar d = 500 \Rightarrow Q(1) = 0.153$). The chance-floor identity explains why: the reconstruction task is "name $|E|$ pairs out of $\binom{n}{2}$", and its difficulty is set by how many candidate pairs compete per true edge, which is $1/\rho$, not by $\bar d$.

The density law is not universal either. It is an *Erdős–Rényi* regularity, and it does not extrapolate to the two real graphs that motivated the hypothesis. As loaded by this codebase (`data/citeseer.txt` and `data/AstroPh.npz`, the latter being what `experiments/datasets.py` actually registers for AstroPh — the `.txt` entry is commented out):

| dataset | $N$ | $E$ | $\rho$ | $\bar d$ | degree CV | avg. clustering | $Q(1)$ | lift $Q(1)/\rho$ |
|---|---|---|---|---|---|---|---|---|
| CITESEER | 3264 | 4536 | 0.000852 | 2.78 | 1.22 | 0.145 | 0.2815 | 330 |
| AstroPh | 18772 | 198050 | 0.001124 | 21.10 | 1.45 | 0.631 | 0.7683 | 684 |
| $G(1000, 0.01)$ | 1000 | 4966 | 0.00994 | 9.93 | — | 0.0096 | 0.9904 | 99.6 |

Both real graphs are an order of magnitude *sparser* than the sweep's sparsest cell, so the sweep never samples their density regime, and at $\rho \approx 0.001$ the Erdős–Rényi trend would put $Q(1)$ near its ceiling for both. Observed values are $0.28$ and $0.77$. Density does not explain the real-graph contrast (the two differ by only $1.3\times$ in $\rho$, as the previous section already noted), average degree does not explain the Erdős–Rényi contrast, and neither statistic explains both families. What the two real graphs *do* differ in sharply is clustering ($0.145$ vs $0.631$) — a quantity that is $\approx p$ by construction in $G(n,p)$ and therefore held at essentially zero throughout the sparse half of the sweep.

## The ratio $Q(8)/Q(1)$ in the realized cuts is a saturation artifact

In the 5 cells where a cut was realized across all of $k = 1,2,4,8$:

| $N$ | $p$ | $\bar d$ | $Q(1)$ | $Q(8)$ | $Q(8)/Q(1)$ | ceiling $1/Q(1)$ | fraction of ceiling | $1/\bar d$ |
|---|---|---|---|---|---|---|---|---|
| 200 | 0.01 | 2.38 | 0.8076 | 0.9712 | 1.203 | 1.238 | 0.971 | 0.421 |
| 200 | 0.02 | 4.31 | 0.7915 | 0.9931 | 1.255 | 1.263 | 0.993 | 0.232 |
| 200 | 0.03 | 6.06 | 0.7745 | 0.9078 | 1.172 | 1.291 | 0.908 | 0.165 |
| 500 | 0.01 | 4.97 | 0.8971 | 0.9997 | 1.114 | 1.115 | 1.000 | 0.201 |
| 1000 | 0.01 | 9.93 | 0.9904 | 0.9962 | 1.006 | 1.010 | 0.996 | 0.101 |

Since $Q(8)$ is a probability, $Q(8) \leq 1$, so

$$ \frac{Q(8)}{Q(1)} \leq \frac{1}{Q(1)}, \qquad \text{with equality iff } Q(8) = 1 $$

(`ratio_le_inv_of_le_one` and `ratio_eq_inv_iff` in `PartitioningF1ChanceFloor.lean`; assumptions $Q(1) > 0$ and $Q(8) \leq 1$). In every cell $Q(8)$ is between $0.91$ and $1.00$, so the observed ratio attains $91\%$–$100\%$ of its ceiling. **Once $Q(8)$ saturates, $Q(8)/Q(1)$ is a function of $Q(1)$ alone and carries no information about what partitioning did.** The correlations confirm which variable the ratio is tracking: across these five cells, $\mathrm{corr}(Q(8)/Q(1),\, 1/Q(1)) = +0.914$ against $\mathrm{corr}(Q(8)/Q(1),\, 1/\bar d) = +0.621$.

The matched-size comparison the previous section asked for is available in one family, $N = 200$ with $\bar d \in \{2.38, 4.31, 6.06\}$, and it gives $Q(8)/Q(1) = 1.203, 1.255, 1.172$ — non-monotone in $\bar d$, inside a $\pm 4\%$ band, and bounded above by the ceilings $1.238, 1.263, 1.291$. **At matched graph size, varying average degree by $2.5\times$ produces no detectable change in $Q(8)/Q(1)$.** The prediction under test, $Q(k)/Q(1) \sim 1/\bar d$, would have required this ratio to fall by $2.5\times$ across that family.

## The raw $Q$ gain is a chance-floor gain, not a skill gain

Each raw score ratio factors into a skill ratio and a floor ratio: writing $L = Q/\rho_{\text{within}}$ for the lift over the within-scope chance floor,

$$ \frac{Q'}{Q} = \frac{L'}{L} \cdot \frac{\rho'_{\text{within}}}{\rho_{\text{within}}} $$

for $Q \neq 0$ and both floors non-zero (`score_ratio_factors`). The logs report $\rho_{\text{within}} = p(uv \in E \mid \pi(u) = \pi(v))$ directly as `p(true edge)` on the `Within-partition pair terms` line, so both factors are measurable:

| $N$ | $p$ | $\rho_{\text{within}}(1)$ | $\rho_{\text{within}}(8)$ | floor ratio | $Q(8)/Q(1)$ | skill ratio $L(8)/L(1)$ |
|---|---|---|---|---|---|---|
| 200 | 0.01 | 0.0136 | 0.0692 | 5.10 | 1.203 | 0.236 |
| 200 | 0.02 | 0.0217 | 0.0790 | 3.65 | 1.255 | 0.344 |
| 200 | 0.03 | 0.0305 | 0.0691 | 2.27 | 1.172 | 0.517 |
| 500 | 0.01 | 0.0100 | 0.0312 | 3.11 | 1.114 | 0.358 |
| 1000 | 0.01 | 0.0099 | 0.0168 | 1.69 | 1.006 | 0.597 |

**In every realized cut, reconstruction skill falls — by factors of $1.7$ to $4.2$ — while raw $Q$ rises.** The raw rise is the floor overtaking it. So the sweep does not exhibit a $Q$-improving partitioning in the sense the hypothesis needs; it exhibits a $Q$-inflating one.

This also corrects the scope of the pool-size argument of the earlier section. That argument assumes a **fixed per-vertex budget $M$** among a shrinking pool $N$, giving a baseline $M/N$ that rises as blocks shrink. This pipeline's budget is not fixed: it is the block's own edge count, so the baseline is the block's density, and the question becomes whether partitioning raises within-block density. For an Erdős–Rényi graph and a cut chosen independently of the edges, it does not:

**Statement.** For $G(n,p)$ and any vertex block $A$ with $|A| \geq 2$ fixed before the edges are drawn, the expected density of the subgraph induced on $A$ is $p$, independent of $|A|$ and of which vertices $A$ contains. Hence all blocks of all partitions share one chance floor, equal to the whole graph's. Formalized as `integral_within_density_eq` and `integral_within_density_eq_of_two_blocks` in [`proofs/VertexVoyageProofs/ErdosRenyiChanceFloor.lean`](../../proofs/VertexVoyageProofs/ErdosRenyiChanceFloor.lean), building on `integral_within_count` of `ErdosRenyiPartitions.lean`. **Example.** A balanced 8-way split of $G(500, 0.01)$ has expected within-block density $0.01$, the same as the whole graph.

Label propagation's cut is *not* independent of the edges, and the measured floors above show by how much it is not: at $(500, 0.01)$, $k=8$ it leaves $\rho_{\text{within}} = 0.0312$, a $3.1\times$ inflation over $p$. The same fact shows up in the cut fraction. For a structure-blind cut, the expected cut fraction is purely combinatorial and free of $p$:

$$ \mathbb{E}[\text{cut edges}] \big/ \mathbb{E}[\text{all edges}] = \frac{2|A| \, |A^{c}|}{n(n-1)} $$

for a bipartition (`expected_cut_fraction_eq`), which for a balanced $k$-way split of $n$ vertices gives $1 - \sum_i \binom{n_i}{2} / \binom{n}{2}$, i.e. $0.876$–$0.880$ at $k=8$ for $n \in \{200, 500, 1000\}$. Measured $\delta(8)$ is $0.375$–$0.617$ — well below the blind prediction. Label propagation is therefore extracting *something* from these graphs, namely the sparse fragmented periphery of a near-threshold sample (at $N=200$, $p=0.01$ the graph has 6 connected components; at $N=500$, $p=0.01$ each of the 8 blocks of 62 vertices splits into 5–11 components). But that something is a fluctuation of the sample, not structure of the model, and the price is the floor inflation above.

## In every realized cut, macro $F_1$ falls with $k$

Cross-partition recall is exactly zero in every multi-partition run of this sweep as well (`cross=0.000000` on every `Macro recall by scope` line), so $\mathrm{recall} = \bar p \, Q$ holds exactly with $\bar p$ the retention share — e.g. $0.430 \times 0.9997 = 0.4298$ at $(N{=}500, p{=}0.01, k{=}8)$, matching the reported recall — and the elasticity criterion `product_le_iff_ratio_le` applies. Retention loss dominates the $Q$ gain in every cell:

| $N$ | $p$ | $\bar p(1)/\bar p(8)$ | $Q(8)/Q(1)$ | $F_1^{\mathrm{macro}}(1)$ | $F_1^{\mathrm{macro}}(8)$ |
|---|---|---|---|---|---|
| 200 | 0.01 | 1.399 | 1.203 | 0.7610 | 0.6852 |
| 200 | 0.02 | 1.980 | 1.255 | 0.7622 | 0.5215 |
| 200 | 0.03 | 1.563 | 1.172 | 0.7707 | 0.5638 |
| 500 | 0.01 | 2.326 | 1.114 | 0.8789 | 0.4463 |
| 1000 | 0.01 | 1.770 | 1.006 | 0.9884 | 0.5692 |

Since $\bar p(1)/\bar p(8) > Q(8)/Q(1)$ throughout, recall and $F_1^{\mathrm{macro}}$ decrease, in agreement with the measured values. **The sweep contains no Erdős–Rényi analogue of the CITESEER rise.** This is consistent with the Erdős–Rényi model having no community structure to cut along: $\delta$ is large ($0.38$–$0.62$ at $k = 8$, against $0.255$ for CITESEER and $0.088$ for AstroPh), so a lot of signal is destroyed for a $Q$ gain that is capped by saturation and is in any case a floor effect.

## Side observation: the degree–recall correlation changes sign with density

The reported `degree_recall_correlation` is strongly negative in sparse cells and positive in moderately dense ones, with a clean crossing: at $N = 500$ it runs $-0.686, -0.667, -0.590, -0.282, -0.012$ for $p = 0.01, \dots, 0.05$ and $+0.501, +0.489, +0.432, +0.278, +0.233$ for $p = 0.10, \dots, 0.50$; at $N = 1000$ it crosses between $p = 0.02$ and $p = 0.03$. Under a global pair budget the *chance* recall of a vertex is degree-independent, so the sign and its flip are properties of the embedding, not of the budget. At $N = 5000$, however, the correlation is $\approx 0$ throughout ($+0.004$ to $-0.079$), which does not fit either branch. This is recorded as an observation, not explained; it is a candidate for its own experiment rather than a supporting datum for the density hypothesis.

## Verdict

1. **Not confirmed.** $Q(1)/Q(8)$ cannot be shown to depend on average degree from this sweep. The one matched-size family available shows no dependence ($\pm 4\%$ over a $2.5\times$ range of $\bar d$), and the variation that does exist across cells tracks $1/Q(1)$ ($r = +0.914$), i.e. the saturation ceiling, better than $1/\bar d$ ($r = +0.621$).
2. **Mostly unmeasured.** $73$ of $90$ multi-partition cells collapsed to a single community, including every cell with $\bar d > 20$. The hypothesis is a claim about how partitioning gain varies with $\bar d$, and the sweep contains no partitioning measurement at moderate or high $\bar d$.
3. **Contradicted in direction, where measured.** Within Erdős–Rényi graphs, $Q(1)$ is controlled by density, not average degree (within-bin spread $0.075$ vs $0.240$), and the lowest-$\bar d$ graphs are the *easiest* to reconstruct, not the hardest — the opposite of the mechanism proposed for CITESEER.
4. **The gain that exists is not a quality gain.** Skill over the chance floor falls by $1.7$–$4.2\times$ in every realized cut; the raw $Q$ rise comes from label propagation inflating within-block density by $1.7$–$5.1\times$.
5. **Erdős–Rényi is the wrong family for this question.** The model has no community structure, so the partitioner either finds one community (and measures nothing) or cuts along sample noise (destroying $38$–$62\%$ of edges). The two real graphs differ by $1.3\times$ in density and $4.4\times$ in clustering; a family whose clustering is pinned to $p$ cannot reproduce that contrast.

## Additional experiments

Ordered by how much each would narrow the question.

1. **Match the real graphs exactly, one statistic at a time.** Run $k = 1$ on $G(3264, 0.000852)$, which matches CITESEER in $N$, $\rho$ *and* $\bar d = 2.78$ simultaneously, and on $G(18772, 0.001124)$, matching AstroPh ($\bar d = 21.1$). If $Q(1)$ comes out near the Erdős–Rényi trend ($\approx 1$) in both while the real graphs give $0.28$ and $0.77$, then neither density nor average degree explains the real-graph contrast and the cause is structural — clustering and degree heterogeneity being the available candidates. This is the cheapest decisive run and it closes the density gap: the current sweep's sparsest cell is $12\times$ denser than CITESEER.
2. **Use a partitioner that realizes the requested $k$.** The sweep's main failure is instrumental, not statistical. Add a fixed-$k$ partitioner — a balanced random cut as a structure-blind control, plus a spectral or modularity-based $k$-way cut — behind the existing partitioner interface, and re-run the whole grid. The balanced random cut is the right control precisely because `integral_within_density_eq` pins its chance floor to $p$, so any $Q$ change it produces is a genuine embedding effect.
3. **Report lift, not raw $Q$.** Log $\rho_{\text{within}}$ per block (it is already computed) and report $L = Q/\rho_{\text{within}}$ alongside $Q$ everywhere. Every cross-$k$ and cross-dataset comparison in this page's results tables is otherwise confounded by a moving floor.
4. **Avoid the ceiling by design.** Choose cells with $Q(1) \in [0.3, 0.7]$ so the ratio $Q(k)/Q(1)$ has headroom to move. From the $k=1$ table those are $p \approx 0.04$–$0.05$ at $N \in \{200, 500, 1000\}$ and $p \approx 0.10$–$0.30$ at $N = 5000$ — but all of these need experiment 2 first, since label propagation collapses on them.
5. **Separate density from average degree by design, not by accident.** A two-factor grid with $\bar d \in \{5, 10, 20, 40\}$ crossed with $\rho \in \{0.001, 0.005, 0.02\}$, taking $N = \bar d/\rho$, measures the two effects independently. Matched-$\bar d$ triples such as $(N, p) = (200, 0.05), (500, 0.02), (1000, 0.01)$ already exist in the sweep and already disagree by $0.41$ in $Q(1)$; a designed grid would turn that into an estimate rather than an anecdote.
6. **Test on a model that has communities.** Repeat the sweep on the [stochastic block model](../../doc/glossary.md#stochastic-block-model) with in-block probability $p_{\text{in}}$ and cross probability $p_{\text{out}}$, sweeping the ratio $p_{\text{in}}/p_{\text{out}}$ at fixed overall density. This is the only way to get a $\delta$ comparable to the real graphs' $0.09$–$0.26$ and the only family in which "partitioning concentrates signal" is even a meaningful statement.
7. **Vary degree heterogeneity and clustering at fixed $\bar d$ and $\rho$.** The [Barabási–Albert model](../../doc/glossary.md#barabasi-albert-model) and the [configuration model](../../doc/glossary.md#configuration-model) with prescribed degree sequences isolate heterogeneity (CITESEER's degree CV $1.22$ vs AstroPh's $1.45$); a Watts–Strogatz-style rewiring sweep isolates clustering ($0.145$ vs $0.631$). Since $F_1^{\mathrm{macro}}$ is macro-averaged, heterogeneity is a first-order suspect in its own right, as the "Macro-averaging" section's hub/low-degree weighting argument already implies.
8. **Hold the per-vertex budget fixed in a variant run.** Add a reconstruction mode that emits $\deg(u)$ candidates per vertex instead of $|E|$ globally. That is the premise under which the pool-size theorem `count_subsets_containing_fixed_element_eq` predicts a gain, so the two modes together test whether the predicted tailwind is real and how much of the observed $Q$ behaviour it accounts for.
9. **Resample the graph per trial.** Each cell currently reuses one graph for all trials, so the error bars exclude sampling variation in $G(N,p)$. This matters most in the near-threshold cells ($\bar d \approx 2$–$5$), where component structure, and hence what label propagation finds, varies substantially between samples.
10. **Scale embedding capacity with $N$.** Embedding dimension was $100$ in every run while $N$ varied by $25\times$, which is a plausible contributor to the residual $N$-dependence at fixed density (at $p = 0.1$: $Q(1) = 0.301, 0.294, 0.153$ for $N = 500, 1000, 5000$). Sweeping $\dim$ with $N$ separates task difficulty from model capacity.
