# Glossary

Short definitions of the terms used across [`results/`](../results/README.md)
and [`doc/`](README.md), each with a standard reference. Notation follows
[methods.md](methods.md): $G = (V, E)$ is an undirected graph, $P$ the number
of partitions, $\pi(v)$ the partition of vertex $v$, $z_v \in \mathbb{R}^d$ an
embedding.

## Graphs and partitioning

<a id="graph"></a>
### Graph
A pair $G = (V, E)$ with a finite vertex set $V$ and an edge set
$E \subseteq \{\{u, v\} : u, v \in V, u \neq v\}$. The density is
$\rho = |E| / \binom{|V|}{2}$ and the average degree is
$\bar d = 2|E| / |V|$.
Reference: R. Diestel, *Graph Theory*, 5th ed., Springer, 2017, §1.1.

<a id="subgraph"></a>
### Subgraph
$G' = (V', E')$ with $V' \subseteq V$ and $E' \subseteq E$. The subgraph
*induced* by $V'$ keeps every edge of $E$ with both endpoints in $V'$; the
partition subgraphs $G_i$ in this project are induced subgraphs.
Reference: Diestel 2017, §1.1.

<a id="graph-partitioning"></a>
### Graph partitioning
A map $\pi : V \to \{1, \dots, P\}$ (with replication factor $RF > 1$, a map
to subsets of size at most $RF$). The goal is to minimise the
[edge cut](#edge-cut) while keeping partitions balanced; the balanced
problem is NP-hard.
Reference: A. Buluç et al., "Recent Advances in Graph Partitioning",
*Algorithm Engineering*, LNCS 9220, 2016.

<a id="edge-cut"></a>
### Edge cut
The set $\Delta = \{\{u, v\} \in E : \pi(u) \neq \pi(v)\}$. The project
reports the edge-cut ratio $\delta = |\Delta| / |E|$. For $P$ partitions
assigned uniformly at random, $\mathbb{E}[\delta] = (P - 1)/P$.
Reference: Buluç et al. 2016.

<a id="streaming-graph-partitioning"></a>
### Streaming graph partitioning
Partitioning in which vertices or edges arrive one at a time (or in buffers)
and are assigned using only what has been seen so far. LDG and FENNEL are
the standard one-pass heuristics.
References: I. Stanton, G. Kliot, "Streaming Graph Partitioning for Large
Distributed Graphs", KDD 2012 (LDG); C. Tsourakakis et al., "FENNEL:
Streaming Graph Partitioning for Massive Scale Graphs", WSDM 2014.

<a id="bin-packing"></a>
### Bin packing
Packing items of given sizes into the fewest bins of fixed capacity. The
project uses it to place partitions on workers.
Reference: E. G. Coffman et al., "Bin Packing Approximation Algorithms:
Survey and Classification", *Handbook of Combinatorial Optimization*, 2013.

<a id="community-detection"></a>
### Community detection
Finding vertex groups that are densely connected internally and sparsely
connected to the rest of the graph.
Reference: S. Fortunato, "Community detection in graphs", *Physics Reports*
486, 2010.

<a id="label-propagation"></a>
### Label propagation (LPA)
Each vertex repeatedly adopts the most frequent label among its neighbours
until the labels stabilise. Used here as the static partitioner of the
`vv test` pipeline.
Reference: U. N. Raghavan, R. Albert, S. Kumara, "Near linear time algorithm
to detect community structures in large-scale networks", *Phys. Rev. E* 76,
2007.

<a id="balanced-label-propagation"></a>
### Balanced label propagation
Label propagation with a constraint or penalty on community size, so that it
yields $P$ partitions of similar size.
Reference: J. Ugander, L. Backstrom, "Balanced label propagation for
partitioning massive graphs", WSDM 2013.

## Random graph models

<a id="erdos-renyi-random-graph"></a>
### Erdős–Rényi random graph $G(n, p)$
$n$ vertices; each of the $\binom{n}{2}$ edges is present independently with
probability $p$. The expected density is $p$ and the expected average degree
is $(n - 1)p$. It has no community structure. Formalised in
[`proofs/VertexVoyageProofs/ErdosRenyi.lean`](../proofs/VertexVoyageProofs/ErdosRenyi.lean).
Reference: P. Erdős, A. Rényi, "On random graphs I", *Publ. Math. Debrecen*
6, 1959.

<a id="stochastic-block-model"></a>
### Stochastic block model (SBM)
Vertices have block labels; an edge between blocks $a$ and $b$ is present
independently with probability $B_{ab}$. The planted-partition case uses
$p_{\text{in}}$ inside blocks and $p_{\text{out}}$ across blocks.
Reference: E. Abbe, "Community detection and stochastic block models: recent
developments", *JMLR* 18, 2018.

<a id="barabasi-albert-model"></a>
### Barabási–Albert model
Growth with preferential attachment, which produces a power-law degree
distribution.
Reference: A.-L. Barabási, R. Albert, "Emergence of scaling in random
networks", *Science* 286, 1999.

<a id="configuration-model"></a>
### Configuration model
A uniformly random graph (or multigraph) with a prescribed degree sequence.
Reference: M. E. J. Newman, *Networks*, 2nd ed., OUP, 2018, §12.

<a id="random-walk-on-graph"></a>
### Random walk on a graph
A Markov chain on $V$ that moves to a uniformly chosen neighbour,
$\Pr[v \mid u] = 1/\deg(u)$ for $\{u, v\} \in E$.
Reference: L. Lovász, "Random walks on graphs: a survey", *Combinatorics,
Paul Erdős is Eighty* 2, 1993.

## Embedding

<a id="word2vec"></a>
### word2vec (skip-gram with negative sampling)
Learns vectors so that tokens appearing in the same context window have a
large dot product, contrasted against randomly sampled negative tokens.
Reference: T. Mikolov et al., "Distributed Representations of Words and
Phrases and their Compositionality", NeurIPS 2013.

<a id="node2vec"></a>
### node2vec
Runs word2vec on second-order biased random walks. The return parameter $p$
and the in-out parameter $q$ interpolate between BFS-like and DFS-like walks;
$p = q = 1$ is an unbiased walk (DeepWalk).
Reference: A. Grover, J. Leskovec, "node2vec: Scalable Feature Learning for
Networks", KDD 2016.

<a id="dynnode2vec"></a>
### dynnode2vec
Updates node2vec on a dynamic graph: walks are regenerated only for vertices
affected by a snapshot change, and the previous skip-gram model is
fine-tuned instead of retrained from scratch.
Reference: S. Mahdavi, S. Khoshraftar, A. An, "dynnode2vec: Scalable Dynamic
Network Embedding", IEEE Big Data 2018. Evaluation protocol:
[dynnode2vec-reference-protocol.md](dynnode2vec-reference-protocol.md).

## Linear algebra and geometry

<a id="metric-space"></a>
### Metric space
A set with a distance $d$ that is non-negative, symmetric, zero exactly on
equal points, and satisfies the triangle inequality.
Reference: W. Rudin, *Principles of Mathematical Analysis*, 3rd ed., 1976,
Def. 2.15.

<a id="normed-space"></a>
### Normed space
A vector space with a norm $\lVert\cdot\rVert$: $\lVert x \rVert = 0$ iff
$x = 0$, $\lVert \lambda x \rVert = |\lambda| \lVert x \rVert$, and
$\lVert x + y \rVert \le \lVert x \rVert + \lVert y \rVert$. It is a metric
space with $d(x, y) = \lVert x - y \rVert$.
Reference: E. Kreyszig, *Introductory Functional Analysis with
Applications*, 1978, §2.2.

<a id="inner-product-space"></a>
### Inner product space
A vector space with a symmetric, bilinear, positive-definite
$\langle\cdot,\cdot\rangle$. It induces the norm
$\lVert x \rVert = \sqrt{\langle x, x \rangle}$. Cosine similarity is
$\langle x, y \rangle / (\lVert x \rVert \lVert y \rVert)$.
Reference: Kreyszig 1978, §3.1.

<a id="isometry"></a>
### Isometry
A map $f$ with $\lVert f(x) - f(y) \rVert = \lVert x - y \rVert$. In
$\mathbb{R}^d$, the linear isometries are the orthogonal matrices $O(d)$.
Reference: Kreyszig 1978, §1.6.

<a id="general-linear-group"></a>
### General linear group $GL(d)$
The group of invertible $d \times d$ matrices.
Reference: M. Artin, *Algebra*, 2nd ed., 2011, §2.1.

<a id="rotation-group"></a>
### Rotation group $SO(d)$
The orthogonal matrices with determinant $1$. $O(d) = SO(d) \cup$
reflections.
Reference: Artin 2011, §5.1.

<a id="singular-value-decomposition"></a>
### Singular value decomposition (SVD)
$M = U \Sigma V^\top$ with orthogonal $U, V$ and a non-negative diagonal
$\Sigma$.
Reference: G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed.,
2013, §2.4.

<a id="orthogonal-procrustes-problem"></a>
### Orthogonal Procrustes problem
$\min_{Q \in O(d)} \lVert X Q - Y \rVert_F$. The solution is $Q = U V^\top$,
where $X^\top Y = U \Sigma V^\top$ is an SVD.
Reference: P. H. Schönemann, "A generalized solution of the orthogonal
Procrustes problem", *Psychometrika* 31, 1966.

## Machine learning

<a id="fully-connected-layer"></a>
### Fully connected layer
An affine map $x \mapsto A x + a$.
Reference: I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT
Press, 2016, §6.

<a id="relu-function"></a>
### ReLU
$\mathrm{ReLU}(x) = \max(0, x)$, applied elementwise. Its derivative at $0$
is taken to be $0$ in PyTorch.
Reference: Goodfellow et al. 2016, §6.3.1.

<a id="sigmoid-function"></a>
### Sigmoid
$\sigma(x) = 1 / (1 + e^{-x})$.
Reference: Goodfellow et al. 2016, §3.10.

<a id="stochastic-gradient-descent"></a>
### Stochastic gradient descent (SGD)
$\theta \leftarrow \theta - \eta \nabla_\theta L_B(\theta)$ on mini-batches
$B$.
Reference: L. Bottou, "Large-Scale Machine Learning with Stochastic Gradient
Descent", COMPSTAT 2010.

<a id="adam-optimization"></a>
### Adam
SGD with bias-corrected running estimates of the first and second moments of
the gradient.
Reference: D. P. Kingma, J. Ba, "Adam: A Method for Stochastic
Optimization", ICLR 2015.

<a id="vanishing-gradient-problem"></a>
### Vanishing gradient
The gradient norm falls to near zero, so the parameters stop changing.
Reference: Goodfellow et al. 2016, §8.2.5.

<a id="validation-loss"></a>
### Validation loss
The loss on held-out data that is not used for gradient steps, used for
early stopping and model selection.
Reference: Goodfellow et al. 2016, §5.3, §7.8.

<a id="contrastive-loss"></a>
### Contrastive loss
$\sum_{\text{pos}} d^2 + \sum_{\text{neg}} \max(0, m - d)^2$: positive pairs
are pulled together and negative pairs pushed beyond margin $m$.
Reference: R. Hadsell, S. Chopra, Y. LeCun, "Dimensionality Reduction by
Learning an Invariant Mapping", CVPR 2006.

<a id="triplet-loss"></a>
### Triplet loss
$\max(0, d(a, p) - d(a, n) + \alpha)$ for an anchor $a$, a positive $p$ and a
negative $n$.
Reference: F. Schroff, D. Kalenichenko, J. Philbin, "FaceNet", CVPR 2015.

<a id="representation-collapse"></a>
### Representation collapse
All embeddings converge to (nearly) the same point, which makes a contrastive
objective trivial.
Reference: L. Jing et al., "Understanding Dimensional Collapse in
Contrastive Self-supervised Learning", ICLR 2022.

<a id="gradient-conflict"></a>
### Gradient conflict
Gradients of different loss terms point in opposing directions (negative
cosine), so their sum is much smaller than its parts.
Reference: T. Yu et al., "Gradient Surgery for Multi-Task Learning",
NeurIPS 2020.

<a id="frustration"></a>
### Frustration
A state in which no configuration satisfies all pairwise constraints at
once, so some loss terms stay active at every optimum. The term comes from
frustrated spin systems.
Reference: G. Toulouse, "Theory of the frustration effect in spin glasses",
*Commun. Phys.* 2, 1977.

## Evaluation and statistics

<a id="model-evaluation"></a>
### Model evaluation
Precision $= TP/(TP + FP)$, recall $= TP/(TP + FN)$,
$F_1 = 2TP / (2TP + FP + FN)$. The project's reconstruction metrics are
defined in [evaluation.md](evaluation.md).
Reference: C. D. Manning, P. Raghavan, H. Schütze, *Introduction to
Information Retrieval*, CUP, 2008, §8.3.

<a id="ranking-metrics"></a>
### Ranking metrics (MRR, Hits@k)
For each positive test edge, rank it among $K$ negatives. MRR is the mean of
$1/\text{rank}$ and Hits@k is the fraction ranked $\le k$. Ties must be
broken explicitly (optimistic or pessimistic).
Reference: J. Li et al., "Evaluating Graph Neural Networks for Link
Prediction: Current Pitfalls and New Benchmarking" (HeaRT), NeurIPS 2023.

<a id="auc-and-average-precision"></a>
### AUC and average precision
Let $s^+$ and $s^-$ be the scores of a random positive and a random
negative. Then
$\mathrm{AUC} = \Pr[s^+ > s^-] + \tfrac12 \Pr[s^+ = s^-]$, which is
$\tfrac12$ for a score that is independent of the label.

Sort the pairs by decreasing score, and let $P_n$ and $R_n$ be the precision
and recall at the $n$-th threshold. Then
$\mathrm{AP} = \sum_n (R_n - R_{n-1})\, P_n$, as computed by scikit-learn.
When no scores are tied, this is the mean precision at the rank of each
positive. Both metrics need at least one positive and one negative.

References: J. A. Hanley, B. J. McNeil, "The meaning and use of the area
under a receiver operating characteristic (ROC) curve", *Radiology* 143(1),
1982; Manning et al. 2008, §8.4.

<a id="prequential-evaluation"></a>
### Prequential evaluation
A way of evaluating a stream learner in which every example is first used
to test the current model and then to train it ("test-then-train"). The
model scored at time $t$ was trained only on data from before $t$. The
streaming link prediction follows this protocol, see
[evaluation.md](evaluation.md#prequential-link-prediction-pipeline-t).
Reference: J. Gama, R. Sebastião, P. P. Rodrigues, "On evaluating stream
learning algorithms", *Machine Learning* 90(3), 2013.

<a id="stratified-metric-decomposition"></a>
### Stratified metric decomposition
Writing a metric as a weighted sum over strata, here within-partition and
cross-partition edges, via the law of total expectation:
$\mathbb{E}[F] = (1 - \delta)\,\mathbb{E}[F \mid \text{within}] + \delta\,
\mathbb{E}[F \mid \text{cross}]$. Formalised in
[`proofs/VertexVoyageProofs/PartitioningF1.lean`](../proofs/VertexVoyageProofs/PartitioningF1.lean).
Reference: P. Billingsley, *Probability and Measure*, 3rd ed., 1995, §34.

<a id="standard-deviation"></a>
### Standard deviation
Population: $\sigma = \sqrt{\frac{1}{n}\sum (x_i - \bar x)^2}$ (`np.std`).
Sample: divide by $n - 1$ (`ddof=1`). Tables state which one they use.
Reference: Billingsley 1995, §21.

<a id="t-test"></a>
### Welch's t-test
$t = (\bar x_a - \bar x_b) / \sqrt{s_a^2/n_a + s_b^2/n_b}$. Tests equal
means without assuming equal variances.
Reference: B. L. Welch, "The generalization of 'Student's' problem when
several different population variances are involved", *Biometrika* 34, 1947.

<a id="mann-whitney-test"></a>
### Mann–Whitney U test
A rank-based test of whether one sample tends to be larger than the other.
It makes no normality assumption.
Reference: H. B. Mann, D. R. Whitney, *Ann. Math. Statist.* 18, 1947.

<a id="cohens-d"></a>
### Cohen's d
$d = (\bar x_b - \bar x_a) / s_p$ with
$s_p = \sqrt{(s_a^2 + s_b^2)/2}$. Rough guide: 0.2 is small, 0.5 medium and
0.8 large.
Reference: J. Cohen, *Statistical Power Analysis for the Behavioral
Sciences*, 2nd ed., 1988.

<a id="bonferroni-correction"></a>
### Bonferroni correction
With $m$ tests, reject at level $\alpha/m$, which controls the family-wise
error rate at $\alpha$.
Reference: O. J. Dunn, "Multiple comparisons among means", *JASA* 56, 1961.

<a id="anova"></a>
### ANOVA
Tests equality of $k \ge 2$ group means by comparing between-group to
within-group variance (F-test).
Reference: R. A. Fisher, *Statistical Methods for Research Workers*, 1925.

<a id="tukey-hsd"></a>
### Tukey's HSD
A post-hoc test of all pairwise mean differences after ANOVA, controlling
the family-wise error rate.
Reference: J. W. Tukey, "Comparing individual means in the analysis of
variance", *Biometrics* 5, 1949.

<a id="friedman-and-wilcoxon-test"></a>
### Friedman and Wilcoxon signed-rank tests
Non-parametric tests for matched samples: Friedman for $k$ conditions,
Wilcoxon for 2.
References: M. Friedman, *JASA* 32, 1937; F. Wilcoxon, *Biometrics
Bulletin* 1, 1945.
