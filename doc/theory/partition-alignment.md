# Partition alignment: formal restatement and theorems

This page holds the theory behind the residual aligner of [E10](../../results/E10-partition-alignment/README.md). Every theorem is proved in Lean without `sorry` (see [proofs/](../../proofs/README.md)). Theorems 11–18 were derived while analysing individual runs and are stated in the E10 log entries listed in [the theory index](README.md).

# Formal restatement

Standing assumption for the whole analysis: **every embedding is unit-normalised**, both the per-partition inputs and the aligned outputs.

All theorems below are proved in [`proofs/VertexVoyageProofs/PartitionAlignment.lean`](../../proofs/VertexVoyageProofs/PartitionAlignment.lean) (no `sorry`); the Lean name is given after each statement.

## Partitions with replication

- $V_1, \dots, V_P \subseteq V$ cover $V$. With replication factor $RF = 1$ they are disjoint; with $RF > 1$ they overlap.
- $\Pi(u) = \{p : u \in V_p\}$ is the set of partitions that hold node $u$, with $1 \leq |\Pi(u)| \leq RF$. The $\pi(u)$ of the sections above is one chosen element of $\Pi(u)$; it is a function of $u$ alone only when $RF = 1$.
- A **replica** is a pair $(u, p)$ with $p \in \Pi(u)$. The set of replicas is $R = \{(u, p) : u \in V, p \in \Pi(u)\}$ and $|R| = \sum_p |V_p| \geq |V|$.
- $f_p : V_p \to S^{d-1}$ is the node2vec / dynnode2vec embedding trained on $G_p$ alone, and $z_{u,p} = f_p(u)$ with $\|z_{u,p}\| = 1$.

The embedding is a function of the replica, not of the node: for $p \neq q$ in $\Pi(u)$ the vectors $z_{u,p}$ and $z_{u,q}$ come from different [subgraphs](../glossary.md#subgraph) and different training runs, so in general $z_{u,p} \neq z_{u,q}$. In the experiment below their cosine before alignment is $0.0 \pm 0.06$.

**Example:** $V = \{1, 2, 3, 4\}$, $V_1 = \{1, 2, 3\}$, $V_2 = \{3, 4\}$. Then $\Pi(3) = \{1, 2\}$, $R = \{(1,1), (2,1), (3,1), (3,2), (4,2)\}$, and node $3$ has two unrelated vectors $z_{3,1}$ and $z_{3,2}$.

## Edge classes

- An edge $uv$ is **covered** when $\Pi(u) \cap \Pi(v) \neq \emptyset$: some partition holds both endpoints and trained on that edge.
- An edge is **cut** otherwise; no embedding ever saw it. $E_{\text{cut}}$ is the [edge cut](../glossary.md#edge-cut) and $\delta = |E_{\text{cut}}| / |E|$ is the cut fraction.

Replication exists to shrink $\delta$: in the experiment, replicating $30\%$ of the nodes lowers $\delta$ from $0.30$ to $0.14$.

## Aligner, term by term

$$ W(u, p) = N\big(z_{u,p} + \Delta_p(z_{u,p})\big), \qquad \Delta_p(z) = B \, \text{ReLU}(A_p z + a_p) + b, \qquad N(x) = \frac{x}{\|x\|} $$

| Term | Shape | Role |
|---|---|---|
| $z_{u,p}$ | $\mathbb{R}^d$, unit | Skip connection: with $\Delta_p = 0$ the aligner is the identity. |
| $A_p, a_p$ | $h \times d$, $h$ | First [layer](../glossary.md#fully-connected-layer), one per partition. The only place where the partition index enters. |
| [ReLU](../glossary.md#relu-function) | $\mathbb{R}^h \to \mathbb{R}^h$ | Makes $\Delta_p$ piecewise linear instead of linear. |
| $B, b$ | $d \times h$, $d$ | Second layer, shared by all partitions: it maps every partition's hidden code into the same output coordinates. |
| $N$ | $\mathbb{R}^d \setminus \{0\} \to S^{d-1}$ | Unit normalisation of the output. Undefined when $z + \Delta_p(z) = 0$. |

The model has $P h (d + 1) + d (h + 1)$ parameters. Both arguments of $W$ are needed: $u$ selects the node and $p$ selects which replica of it is transformed.

## Pair sets

The losses are sums over pairs of replicas.

| Set | Definition | Meaning |
|---|---|---|
| $E^+_{\text{intra}}$ | $((u,p),(v,p))$ with $uv \in E$ and $p \in \Pi(u) \cap \Pi(v)$ | An edge seen inside partition $p$. |
| $E^+_{\text{cut}}$ | $((u,p),(v,q))$ with $uv \in E_{\text{cut}}$, $p \in \Pi(u)$, $q \in \Pi(v)$ | A cut edge, between replicas in different partitions. |
| $E^+_{\text{rep}}$ | $((u,p),(u,q))$ with $p < q$ in $\Pi(u)$ | Two replicas of the same node. Empty when $RF = 1$. |
| $E^-$ | sampled pairs of replicas that are not adjacent | Negatives. The full set has $\Theta(R^2)$ elements, so it is sampled. |

$E^+_{\text{rep}}$ is not in the losses written above. It is the natural extra term once the embedding depends on the partition: two replicas of one node should coincide in the common space.

## Losses, term by term

Write $D(r, s) = \|W(r) - W(s)\|$ for replicas $r, s$. On the unit sphere (Theorem 5)

$$ D(r, s)^2 = 2 - 2 \cos(r, s), \qquad 0 \leq D(r, s) \leq 2 $$

so distance and cosine carry the same information.

**Contrastive loss:**

$$ L_m = \sum_{(r,s) \in E^+} D(r,s)^2 + \sum_{(r,s) \in E^-} \max(0, m - D(r,s))^2 $$

- Attraction term $D^2 = 2 - 2\cos$: zero only when the two replicas coincide, at most $4$ (antipodal). Its gradient pulls each positive pair together, linearly in the cosine.
- Repulsion term $\max(0, m - D)^2$: zero when the pair is at distance at least $m$. It is active exactly when $\cos > 1 - m^2/2$.
- Margin $m \in (0, 2]$: $m = 1$ pushes negatives until $\cos \leq 0.5$ (angle $60°$), $m = \sqrt{2}$ until $\cos \leq 0$ (orthogonal), $m = 2$ until antipodal. For $m > 2$ every negative pays at least $(m - 2)^2$ whatever the embedding.

**Triplet loss:**

$$ L_\alpha = \sum_{(r, s) \in E^+} \sum_{t \in N(r)} \max\big(D(r, s) - D(r, t) + \alpha, 0\big) $$

- $r$ is the anchor replica, $s$ a positive (the $S_u$ of the note: a sampled neighbour, or another replica of the same node), $t \in N(r)$ a sampled negative.
- The term is zero when the negative is farther from the anchor than the positive by at least $\alpha$. It constrains the **order** of distances, not their values: a positive pair may stay far apart as long as negatives are farther.
- Margin $\alpha \in (0, 2]$: since $D \leq 2$, zero loss on any triplet forces $\alpha \leq 2$.

## Final embedding of a node

After alignment node $u$ still has $|\Pi(u)|$ vectors. The merged embedding is their normalised sum

$$ w_u = N\Big(\sum_{p \in \Pi(u)} W(u, p)\Big) $$

which is the unit vector closest to all replicas (Theorem 9).

# Intuition check

| Intuition in the note | Verdict |
|---|---|
| Per-partition embeddings are misaligned | Correct and maximal: before alignment two embeddings of the same node have expected cosine $0$ (Theorem 2). Measured: $-0.04$ to $0.01$ between replicas. |
| $f$ must be indexed by the partition | Correct. With $RF > 1$ the embedding is a function of the replica $(u, p)$. This also gives exact anchors, the pairs $E^+_{\text{rep}}$, which the losses above do not use yet. |
| The misalignment is an offset, $z + \Delta$ | Wrong prior. With unit normalisation the only linear maps that keep unit vectors unit are isometries (Theorem 1), and a random isometry moves $z$ as far as a random direction (Theorem 2). The residual block can still represent a rotation (Theorem 3), it only starts far from it. |
| Contrastive / triplet loss on aligned embeddings measures alignment | Only the cross-partition pairs do, $E^+_{\text{cut}}$ and $E^+_{\text{rep}}$ (Theorem 4). Nothing ties $W(u,p)$ to $z_{u,p}$, so with an MLP this is re-embedding supervised by edges. |
| Neighbours at distance $0$, non-neighbours at distance $\geq m$ is reachable | No. Every open wedge costs at least $m^2/3$ (Theorem 6). |
| Margins are meaningful | Yes, because of unit normalisation, and only for $m, \alpha \leq 2$ (Theorem 5). Without normalisation they are not (Theorem 7). |
| Alignment improves downstream scores | For $RF = 1$ the gain is at most $\delta (1 - C)$ and needs a structured cut (Theorem 10). For $RF > 1$ alignment is needed even for covered edges, because merging unaligned replicas damages them. |

# Theorems

Each theorem lists its assumptions, the statement, a step-by-step proof that follows the Lean proof, the consequence for this page, and a numeric example. $\langle \cdot, \cdot \rangle$ is the Euclidean inner product and $\|\cdot\|$ its norm.

## Theorem 1 (when two runs differ by a linear map, and what normalisation does to it)

**What is and is not claimed.** The sentence "two partitions embed the same nodes differently, up to a linear map $T$" is not a theorem and is false in general. What can be proved is narrower:

- (a) the skip-gram score has a $GL(d)$ symmetry, so a linear change of basis is invisible to the loss;
- (b) a linear map that keeps unit vectors unit preserves every norm;
- (c) if two runs produce exactly the same score for every (node, context) pair and all factors have rank $d$, then the runs differ by exactly one invertible linear map $A$;
- (d) after unit normalisation that relation is still linear only if $A$ is a positive multiple of an orthogonal map.

The hypothesis of (c) does not hold for two partitions, see "Where the hypothesis of (c) fails" below. The linear map is therefore an approximation, and its error is measured at the end of this theorem.

**Notation:** $n$ nodes, $m$ contexts, dimension $d$. $W \in \mathbb{R}^{n \times d}$ has the node vector $w_i^T$ as row $i$, $C \in \mathbb{R}^{m \times d}$ has the context vector $c_j^T$ as row $j$, both from [word2vec](../glossary.md#word2vec). The score matrix is $W C^T \in \mathbb{R}^{n \times m}$, with entries $(W C^T)_{ij} = \langle w_i, c_j \rangle$. A second run gives $W', C'$.

### How $W$, $C$ and the scores are computed

$W$ and $C$ are not given by a formula. They are the state of an iterative optimiser at the moment it stops, started from random values. This is the skip-gram model with negative sampling of [word2vec](../glossary.md#word2vec), run on random walks as in [node2vec](../glossary.md#node2vec).

**Inputs:** the graph $G_p$ of one partition, with nodes numbered $1, \dots, n$; dimension $d$; walks per node $r$; walk length $L$; window $c$; number of negative samples $K$; learning rate $\eta > 0$; a random seed. In the script $d = 32$, $r = 10$, $L = 40$, $c = 5$, $K = 5$, with 5 passes over the walks. Every node is also a context, so $m = n$.

**Step 1, walks.** From every node start $r$ [random walks](../glossary.md#random-walk-on-graph) $(v_1, \dots, v_L)$ on $G_p$: $v_{t+1}$ is a uniformly chosen neighbour of $v_t$ (node2vec with $p = q = 1$). The walks never leave $G_p$.

**Step 2, training pairs and counts.** Every position $t$ of every walk and every offset $s$ with $1 \leq |s| \leq c$ gives one pair (node $v_t$, context $v_{t+s}$). Count them:

$$ N_{ij} = \#\{\text{pairs with node } i \text{ and context } j\}, \qquad N_i = \sum_{j} N_{ij}, \qquad N'_j = \sum_{i} N_{ij}, \qquad N = \sum_{i,j} N_{ij} $$

Example: the walk $(1, 2, 3)$ with $c = 1$ gives the pairs $(1,2), (2,1), (2,3), (3,2)$, so $N_{12} = N_{21} = N_{23} = N_{32} = 1$, $N_{13} = 0$, $N_2 = 2$ and $N = 4$.

**Step 3, initialisation.** $C = 0$. Every entry of $W$ is drawn independently and uniformly from $[-1/d, 1/d]$ using the seed (gensim's choice). This is the first place where two runs differ.

**Step 4, one update per pair.** For a pair $(i, j)$ draw $K$ negative contexts $j_1, \dots, j_K$ independently from the noise distribution

$$ P_n(j) = \frac{(N'_j)^{3/4}}{\sum_{j'} (N'_{j'})^{3/4}} $$

Give the true context the label $y = 1$ and each negative the label $y = 0$. With the [sigmoid](../glossary.md#sigmoid-function) $\sigma(x) = 1 / (1 + e^{-x})$, for each of the $K + 1$ contexts $j'$ compute the error

$$ e_{j'} = y_{j'} - \sigma(\langle w_i, c_{j'} \rangle) $$

and then update, using the old values on the right-hand side:

$$ c_{j'} \leftarrow c_{j'} + \eta \, e_{j'} \, w_i \qquad\qquad w_i \leftarrow w_i + \eta \sum_{j'} e_{j'} \, c_{j'} $$

This is one step of [stochastic gradient](../glossary.md#stochastic-gradient-descent) ascent on the per-pair objective

$$ \log \sigma(\langle w_i, c_j \rangle) + \sum_{k=1}^{K} \log \sigma(-\langle w_i, c_{j_k} \rangle) $$

because $\nabla_w \log \sigma(\langle w, c \rangle) = (1 - \sigma(\langle w, c \rangle)) \, c$ and $\nabla_w \log \sigma(-\langle w, c \rangle) = -\sigma(\langle w, c \rangle) \, c$, which are $e \, c$ for $y = 1$ and $y = 0$. The order of the pairs and the negative samples are the second place where two runs differ.

Example of one update: $d = 1$, $\eta = 0.1$, $w_i = 1$, $c_j = 0$, one negative with $c_{j_1} = 0$. Both scores are $0$ and $\sigma(0) = 0.5$, so $e_j = 0.5$ and $e_{j_1} = -0.5$. Then $c_j \leftarrow 0.05$, $c_{j_1} \leftarrow -0.05$, and $w_i \leftarrow 1 + 0.1 (0.5 \cdot 0 - 0.5 \cdot 0) = 1$. The true context moves towards $w_i$ and the negative moves away.

**Step 5, output.** After all passes, row $i$ of $W$ is $w_i^T$ and row $j$ of $C$ is $c_j^T$ (gensim: `model.wv` and `model.syn1neg`). The score of node $i$ and context $j$ is

$$ S_{ij} = (W C^T)_{ij} = \langle w_i, c_j \rangle = \sum_{k=1}^{d} W_{ik} C_{jk} $$

and $\sigma(S_{ij})$ is the model's probability that $(i, j)$ is a real pair and not a negative sample. Only $W$ is kept as the embedding: $z_{u,p}$ is row $u$ of $W$ divided by its norm. $C$ is discarded, which is why the aligner cannot simply read $A$ off part (c).

gensim adds four details that do not change the picture: the window is drawn uniformly from $\{1, \dots, c\}$ at every position, frequent nodes are subsampled, $\eta$ decays linearly from $0.025$ to $0.0001$, and the noise distribution uses the number of times node $j$ appears in the walks in place of $N'_j$ (the two differ only at walk ends).

**What the scores converge to.** Averaged over the negative samples, the updates of step 4 ascend

$$ \mathcal{L}(W, C) = \sum_{i=1}^{n} \sum_{j=1}^{n} \Big[ N_{ij} \log \sigma(S_{ij}) + K N_i P_n(j) \log \sigma(-S_{ij}) \Big] $$

The term for the pair $(i, j)$ depends on $W, C$ only through the score $S_{ij}$. It has the form $a \log p + b \log(1 - p)$ with $a = N_{ij}$, $b = K N_i P_n(j)$ and $p = \sigma(S_{ij})$.

**Lemma (the best score of one pair).**

*Assumptions:* $a > 0$, $b > 0$, $0 < p < 1$.

*Statement:*

$$ a \log p + b \log(1 - p) \;\leq\; a \log \frac{a}{a + b} + b \log \frac{b}{a + b}, \qquad \sigma\!\left(\log \frac{a}{b}\right) = \frac{a}{a + b}, \qquad 1 - \sigma(x) = \sigma(-x) $$

So the term is largest when $\sigma(S_{ij}) = a / (a + b)$, that is when $S_{ij} = \log(a / b)$.

*Proof:*

1. Put $p^* = a / (a + b)$, so $1 - p^* = b / (a + b)$; both are positive.
2. For every $t > 0$, $\log t \leq t - 1$. Apply it to $t = p / p^*$ and to $t = (1 - p)/(1 - p^*)$: $\log p - \log p^* \leq p / p^* - 1$ and $\log(1 - p) - \log(1 - p^*) \leq (1 - p)/(1 - p^*) - 1$.
3. Multiply the first inequality by $a > 0$ and the second by $b > 0$. Since $a / p^* = a + b$ and $b / (1 - p^*) = a + b$, the right-hand sides are $(a + b) p - a$ and $(a + b)(1 - p) - b$.
4. Add them. The right-hand sides sum to $(a + b)(p + 1 - p) - a - b = 0$, so $a \log p + b \log(1 - p) \leq a \log p^* + b \log(1 - p^*)$.
5. $e^{-\log(a/b)} = b / a$, so $\sigma(\log(a/b)) = 1 / (1 + b/a) = a / (a + b)$.
6. $1 - 1/(1 + e^{-x}) = e^{-x} / (1 + e^{-x}) = 1 / (e^{x} + 1) = \sigma(-x)$. $\blacksquare$

Lean: `pair_objective_le_at_ratio`, `sigmoid_at_log_ratio`, `one_sub_sigmoid`.

*Consequence:* if every score could be chosen freely, the optimum would be

$$ S^*_{ij} = \log \frac{N_{ij}}{K \, N_i \, P_n(j)} $$

which for $P_n(j) = N'_j / N$ is the pointwise mutual information $\log \frac{N_{ij} N}{N_i N'_j}$ minus $\log K$ (Levy and Goldberg, 2014; cited, only the lemma above is formalized). The scores cannot be chosen freely for two reasons: $S = W C^T$ has rank at most $d$ while $S^*$ in general has rank up to $n$, and for pairs with $N_{ij} = 0$ the lemma does not apply ($a = 0$) and the term only pushes $S_{ij}$ towards $-\infty$. So $W$ and $C$ are a rank-$d$ compromise found by a random search, and $S^*$ depends on the counts $N_{ij}$, which depend on the partition.

*Example:* $N_{ij} = 30$, $N_i = 400$, $K = 5$, $P_n(j) = 0.002$. Then $a = 30$, $b = 5 \cdot 400 \cdot 0.002 = 4$, $S^*_{ij} = \log 7.5 \approx 2.01$ and $\sigma(S^*_{ij}) = 30/34 \approx 0.88$. If another partition sees the pair only $N_{ij} = 10$ times with the same $N_i$ and $P_n(j)$, then $S^*_{ij} = \log 2.5 \approx 0.92$: a different target score for the same pair of nodes.

### Part (a): the symmetry of the score

**Assumptions:** $A \in \mathbb{R}^{d \times d}$ is invertible, an element of the [general linear group](../glossary.md#general-linear-group), and $w, c \in \mathbb{R}^d$.

**Statement:**

$$ \langle A w, A^{-T} c \rangle = \langle w, c \rangle $$

**Proof:**

1. Write the inner product as a matrix product: $\langle A w, A^{-T} c \rangle = (A^{-T} c)^T (A w)$.
2. Transpose the first factor: $(A^{-T} c)^T = c^T A^{-1}$.
3. Regroup: $c^T A^{-1} A w = c^T (A^{-1} A) w$.
4. $A^{-1} A = I$ because $A$ is invertible, so the expression is $c^T w = \langle w, c \rangle$. $\blacksquare$

Lean: `sgns_score_gl_invariant`.

**Consequence:** if $(W, C)$ minimises the skip-gram loss, so does $(W A^T, C A^{-1})$ for every invertible $A$. This says a linear ambiguity exists. It does not say it is the only difference between two runs; that is part (c).

**Example:** $d = 2$, $A = \operatorname{diag}(2, \tfrac12)$, $w = c = (1, 1)$. Then $A w = (2, \tfrac12)$, $A^{-T} c = (\tfrac12, 2)$ and $\langle A w, A^{-T} c \rangle = 1 + 1 = 2 = \langle w, c \rangle$.

### Part (b): linear maps that keep the unit sphere

**Assumptions:** $Q : \mathbb{R}^d \to \mathbb{R}^d$ is linear and $\|Q x\| = 1$ for every $x$ with $\|x\| = 1$.

**Statement:**

$$ \|Q x\| = \|x\| \quad \text{for every } x \in \mathbb{R}^d $$

**Proof:**

1. If $x = 0$, then $Q x = 0$ by linearity and both sides are $0$.
2. If $x \neq 0$, put $e = x / \|x\|$. Then $\|e\| = 1$ and $x = \|x\| e$.
3. By linearity $Q x = \|x\| \, Q e$, so $\|Q x\| = \|x\| \, \|Q e\|$.
4. $\|Q e\| = 1$ by the assumption, hence $\|Q x\| = \|x\|$. $\blacksquare$

Lean: `linear_preserves_norm_of_preserves_sphere`.

**Consequence:** such a $Q$ is a linear [isometry](../glossary.md#isometry), an element of $O(d)$, see [rotation group](../glossary.md#rotation-group).

**Example:** $A = \operatorname{diag}(2, \tfrac12)$ is excluded, because $\|A (1, 0)\| = 2 \neq 1$. The rotation by $90°$, $Q(x_1, x_2) = (-x_2, x_1)$, is allowed: $\|Q x\|^2 = x_2^2 + x_1^2 = \|x\|^2$.

### Part (c): equal scores force a linear relation

**Assumptions:**

- $W, W' \in \mathbb{R}^{n \times d}$ and $C, C' \in \mathbb{R}^{m \times d}$.
- Equal scores: $W C^T = W' C'^T$, that is $\langle w_i, c_j \rangle = \langle w'_i, c'_j \rangle$ for every node $i$ and context $j$.
- Rank $d$: the matrices $W^T W$, $C^T C$ and $C'^T C'$ in $\mathbb{R}^{d \times d}$ are invertible (the $d$ columns of $W$, of $C$ and of $C'$ are linearly independent).

**Statement:** there is an invertible $A \in \mathbb{R}^{d \times d}$ with

$$ W' = W A \qquad \text{and} \qquad C' = C A^{-T} $$

Row by row, $w'_i = A^T w_i$ and $c'_j = A^{-1} c_j$: the same matrix for every node.

**Proof:**

1. Cancellation lemma: if $W X = W Y$ then $X = Y$. Multiply on the left by $W^T$ to get $(W^T W) X = (W^T W) Y$, then by $(W^T W)^{-1}$, which exists by the rank assumption.
2. Define two candidate matrices, $A = C^T C' (C'^T C')^{-1}$ and $B = C'^T C (C^T C)^{-1}$. Both inverses exist by the rank assumption.
3. Compute $W A = (W C^T) C' (C'^T C')^{-1}$. Replace $W C^T$ by $W' C'^T$ (equal scores): $W A = W' (C'^T C') (C'^T C')^{-1} = W'$. This is the first claim.
4. Symmetrically $W' B = (W' C'^T) C (C^T C)^{-1} = (W C^T) C (C^T C)^{-1} = W (C^T C)(C^T C)^{-1} = W$.
5. Combine steps 3 and 4: $W (A B) = (W A) B = W' B = W = W I$. By the cancellation lemma $A B = I$.
6. $A$ and $B$ are square, so $A B = I$ gives $B A = I$. Hence $A$ is invertible and $A^{-1} = B$.
7. From equal scores and step 3, $W C^T = W' C'^T = W (A C'^T)$. By the cancellation lemma $C^T = A C'^T$.
8. Multiply step 7 on the left by $B$: $B C^T = B A C'^T = C'^T$. Transpose: $C' = C B^T = C A^{-T}$. This is the second claim. $\blacksquare$

Lean: `mul_left_cancel_of_gram_unit`, `factorisation_unique_up_to_gl`.

**Consequence:** under these assumptions, and only under them, "the two runs differ by a linear map" is exact, and the map is $A = C^T C' (C'^T C')^{-1}$. $A$ is a general invertible matrix, not an orthogonal one.

**Example:** $n = m = 2$, $d = 1$. Run 1: $W = (1, 2)^T$, $C = (1, 1)^T$. Run 2: $W' = (2, 4)^T$, $C' = (\tfrac12, \tfrac12)^T$. Both give the score matrix with rows $(1, 1)$ and $(2, 2)$. The formula gives $A = C^T C' / (C'^T C') = 1 / \tfrac12 = 2$, and indeed $W A = (2, 4)^T = W'$ and $C A^{-T} = (\tfrac12, \tfrac12)^T = C'$.

### Part (d): after normalisation the relation is linear only for a scaled rotation

Normalisation replaces $w_i$ by $w_i / \|w_i\|$ and $w'_i = A^T w_i$ by $A^T w_i / \|A^T w_i\|$. So unit vectors of run 1 are sent to unit vectors of run 2 by $x \mapsto f x / \|f x\|$ with $f = A^T$. Part (d) asks when this map is linear.

**Assumptions:**

- $f, g : \mathbb{R}^d \to \mathbb{R}^d$ are linear.
- $f x \neq 0$ for every $x$ with $\|x\| = 1$.
- $g$ agrees with the normalised $f$ on the whole unit sphere: $g x = f x / \|f x\|$ for every $x$ with $\|x\| = 1$.

**Statement:** there is a constant $c > 0$ with

$$ f x = c \, g x \quad \text{and} \quad \|g x\| = \|x\| \quad \text{for every } x \in \mathbb{R}^d $$

So $f$ is $c$ times an orthogonal map. Contrapositive: if $A$ is not a positive multiple of an orthogonal matrix, no linear map equals $x \mapsto A^T x / \|A^T x\|$ on the unit sphere.

**Proof:**

1. For unit $x$, $\|g x\| = \|f x\| / \|f x\| = 1$. By part (b), $\|g x\| = \|x\|$ for every $x$. In particular $g x = 0$ forces $x = 0$.
2. For $x \neq 0$ define the stretch $\lambda(x) = \|f x\| / \|x\|$. Apply the third assumption to the unit vector $e = x / \|x\|$: $g e = f e / \|f e\|$. Since $f e = f x / \|x\|$ and $g e = g x / \|x\|$, this rearranges to $f x = \lambda(x) \, g x$.
3. The stretch ignores scaling: for $t \neq 0$, $\lambda(t x) = |t| \|f x\| / (|t| \|x\|) = \lambda(x)$.
4. Let $x, y$ be linearly independent. Then $x + y \neq 0$, and step 2 applied to $x$, $y$ and $x + y$ gives two expressions for $f(x + y)$: $\lambda(x) g x + \lambda(y) g y$ and $\lambda(x + y)(g x + g y)$.
5. Subtract them and use linearity of $g$: $g\big( (\lambda(x) - \lambda(x+y)) x + (\lambda(y) - \lambda(x+y)) y \big) = 0$.
6. By step 1 the argument of $g$ is $0$. Since $x, y$ are linearly independent, both coefficients are $0$, so $\lambda(x) = \lambda(x + y) = \lambda(y)$.
7. Let $x, y$ be non-zero and linearly dependent. Then $y = t x$ with $t \neq 0$ and $\lambda(y) = \lambda(x)$ by step 3.
8. Steps 6 and 7 show that $\lambda$ takes one value $c$ on all non-zero vectors, and $c > 0$ because $f x \neq 0$. Step 2 becomes $f x = c \, g x$; for $x = 0$ both sides are $0$. $\blacksquare$

Lean: `normalised_linear_is_linear_only_if_conformal`.

**Consequence:** even when part (c) applies, normalised embeddings of two runs are related by the non-linear map $z \mapsto A^T z / \|A^T z\|$, unless $A$ happens to be a scaled rotation. Nothing in word2vec forces that. A rotation fitted by [orthogonal Procrustes](../glossary.md#orthogonal-procrustes-problem) then leaves a residual, and a non-linear aligner has something to gain. The statement is about all unit vectors; for finitely many nodes it gives no bound on the residual, which has to be measured.

**Example:** $d = 2$, $f = \operatorname{diag}(2, \tfrac12)$. The stretch is not constant: $\lambda(1, 0) = 2$ and $\lambda(0, 1) = \tfrac12$. Normalised, $f$ fixes $(1, 0)$ and $(0, 1)$, so a linear $g$ agreeing with it must be the identity. But $x = (1, 1)/\sqrt2 \approx (0.707, 0.707)$ is sent to $(2, \tfrac12)/\sqrt{4.25} \approx (0.970, 0.243)$, not to itself. For $f = 3 Q$ with $Q$ a rotation, $f x / \|f x\| = Q x$ is linear and $c = 3$.

### Where the hypothesis of (c) fails

- **Different partitions see different graphs.** Random walks in partition $p$ only use edges inside $p$, so the co-occurrence counts of a replicated node differ between partitions, and the two losses have different minimisers. Example: node $u$ has neighbours $\{a, b\}$ in partition $p$ and $\{a, c\}$ in partition $q$. In $p$ the pair $(u, c)$ never co-occurs and is only a negative sample; in $q$ it is a positive pair. The scores $\langle w_u, c_c \rangle$ differ, so $W C^T \neq W' C'^T$.
- **The dimension is too small for an exact fit.** With $d$ far below the rank of the co-occurrence statistics (here $d = 32$, $n = 1600$) the loss is a weighted low-rank problem, and two runs on the same graph need not reach the same score matrix.
- **Training is finite.** Stochastic gradient descent with a few epochs stops before any minimum.
- **Near-equal scores are not enough.** From step 3, $W' - W A = (W' C'^T - W C^T) C' (C'^T C')^{-1}$ exactly, so a small score difference is multiplied by $C' (C'^T C')^{-1}$, which is large when $C'$ is ill-conditioned.

**Measured** (same generator as the [synthetic check](../../results/E10-partition-alignment/log/2026-10-13-synthetic-check.md#synthetic-check), $d = 32$, mean of 3 seeds, squared error relative to the target):

| Compared runs | Score matrices differ by | Best linear map, raw vectors | Best linear map, unit vectors | Best rotation, unit vectors |
|---|---|---|---|---|
| Same graph, two seeds | $0.065$ | $0.254$ | $0.253$ | $0.357$ |
| Two overlapping partitions, shared nodes | $0.074$ | $0.286$ | $0.285$ | $0.410$ |

The second row embeds two induced subgraphs (communities 1 to 6 and 3 to 8) and compares the 800 shared nodes. A relative error of $1$ means no better than predicting $0$, and no alignment gives $1.83$ (Theorem 2). So a linear map explains about $71$ to $75\%$ of the squared norm and a rotation about $59$ to $64\%$; neither is exact, on the same graph or across partitions. The fit is in-sample, so these errors are lower bounds on what a linear map leaves.

## Theorem 2 (a random isometry is as far as a random direction)

**Assumptions:**

- $Q : \mathbb{R}^d \to \mathbb{R}^d$ is a linear [isometry](../glossary.md#isometry), $\|Q x\| = \|x\|$, and $z \in \mathbb{R}^d$.
- For part (b): $Q$ is random on a probability space $(\Omega, \mu)$, there is a measure-preserving bijection $\sigma : \Omega \to \Omega$ with $Q_{\sigma(\omega)} z = -Q_\omega z$ (the law of $Q$ is symmetric under $Q \mapsto -Q$), and $\omega \mapsto \langle Q_\omega z, z \rangle$ is integrable.

**Statement:**

$$ \text{(a)} \quad \|Q z - z\|^2 = 2\|z\|^2 - 2 \langle Q z, z \rangle \qquad\qquad \text{(b)} \quad \mathbb{E}\|Q z - z\|^2 = 2 \|z\|^2 $$

**Proof of (a):**

1. Expand the square: $\|Q z - z\|^2 = \|Q z\|^2 - 2 \langle Q z, z \rangle + \|z\|^2$.
2. $\|Q z\| = \|z\|$ because $Q$ is an isometry, so the right side is $2 \|z\|^2 - 2 \langle Q z, z \rangle$. $\blacksquare$

**Proof of (b):**

1. Let $g(\omega) = \langle Q_\omega z, z \rangle$. Because $\sigma$ preserves $\mu$, the change of variables gives $\int g(\sigma(\omega)) \, d\mu = \int g(\omega) \, d\mu$.
2. By the symmetry assumption $g(\sigma(\omega)) = \langle -Q_\omega z, z \rangle = -g(\omega)$, so the left side is $-\int g \, d\mu$.
3. Hence $\int g \, d\mu = -\int g \, d\mu$, which forces $\mathbb{E}\langle Q z, z \rangle = 0$.
4. Take the expectation of (a): $\mathbb{E}\|Q z - z\|^2 = 2\|z\|^2 - 2 \cdot 0$; the constant integrates to itself because $\mu$ is a probability measure. $\blacksquare$

Lean: `norm_isometry_sub_sq`, `integral_inner_eq_zero_of_symmetric`, `expected_sq_displacement_of_symmetric`. The Haar measure on $O(d)$ satisfies the symmetry assumption with $\sigma(Q) = -Q$, because $-I \in O(d)$; this step is not formalized.

**Consequence:** for unit $z$ the expected squared displacement is $2$, equivalently the expected cosine between $z$ and $Q z$ is $0$. The misalignment is not a small offset, so the identity is a poor starting point for $W$.

**Example:** $d = 2$, $Q$ the rotation by angle $\theta$, $\|z\| = 1$. Then $\|Q z - z\|^2 = 2 - 2 \cos\theta$: $0$ at $\theta = 0$, $2$ at $90°$, $4$ at $180°$, and mean $2$ for $\theta$ uniform. Measured: relative squared error $1.83 \pm 0.02$ between two seeds on the same graph, cosine $\approx 0$ between unaligned replicas.

## Theorem 3 (the residual block can represent any linear map)

**Assumptions:** $Q \in \mathbb{R}^{d \times d}$ is any matrix, $z \in \mathbb{R}^d$, and $\text{ReLU}(t) = \max(t, 0)$ acts coordinate-wise.

**Statement:**

$$ z + \text{ReLU}((Q - I) z) - \text{ReLU}(-(Q - I) z) = Q z $$

**Proof:**

1. Scalar identity $\text{ReLU}(t) - \text{ReLU}(-t) = t$. If $t \geq 0$: $\text{ReLU}(t) = t$ and $\text{ReLU}(-t) = 0$, difference $t$. If $t < 0$: $\text{ReLU}(t) = 0$ and $\text{ReLU}(-t) = -t$, difference $0 - (-t) = t$.
2. Apply it in every coordinate $i$ with $t = ((Q - I) z)_i$: the two ReLU terms sum to $(Q - I) z$.
3. $z + (Q - I) z = z + Q z - z = Q z$. $\blacksquare$

Lean: `relu_sub_relu_neg`, `residual_relu_represents_linear`.

**Consequence:** with hidden width $h \geq 2d$, the parameters $A_p = \begin{bmatrix} Q_p - I \\ -(Q_p - I) \end{bmatrix}$, $a_p = 0$, $B = \begin{bmatrix} I & -I \end{bmatrix}$, $b = 0$ give $W(u, p) = Q_p z_{u,p}$, already of unit norm when $Q_p$ is orthogonal. So the architecture of this page contains every per-partition rotation, and the shared second layer is not an obstacle. For $h < 2d$ this construction is not available.

**Example:** $d = 1$, $Q = -1$ (a flip), $z = 1$. Then $(Q - I) z = -2$, $\text{ReLU}(-2) = 0$, $\text{ReLU}(2) = 2$, and $1 + 0 - 2 = -1 = Q z$. The offset $\Delta = -2$ is twice the norm of $z$: representable, but not small.

## Theorem 4 (the alignment signal lives in cross-partition pairs)

**Assumptions:**

- The replicas $R$ are embedded by $W : R \to X$ into a [metric space](../glossary.md#metric-space) $(X, \text{dist})$, and $D(r, s) = \text{dist}(W(r), W(s))$.
- $T, T_1, \dots, T_P : X \to X$ are [isometries](../glossary.md#isometry): $\text{dist}(T x, T y) = \text{dist}(x, y)$. In $\mathbb{R}^d$ these are exactly the maps $x \mapsto Q x + b$ with $Q^T Q = I$.
- The pair sets $E^+$, $E^-$ and the triplet set are finite.
- For part (b): every pair (every triplet) of the loss consists of replicas of one partition.
- For part (c): $E^+ = E^+_{\text{intra}} \cup E^+_{\text{cross}}$ and $E^- = E^-_{\text{intra}} \cup E^-_{\text{cross}}$ are disjoint unions.

**Statement:** for the contrastive loss $L_m$ and the triplet loss $L_\alpha$,

$$ \text{(a)} \quad L(T \circ W) = L(W) \qquad \text{(b)} \quad L\big((u, p) \mapsto T_p(W(u, p))\big) = L(W) \qquad \text{(c)} \quad L_m = L_{m, \text{intra}} + L_{m, \text{cross}} $$

**Proof:**

1. Every term of either loss is a function of distances $D(r, s)$ only: $D^2$, $\max(0, m - D)^2$, or $\max(D(r,s) - D(r,t) + \alpha, 0)$.
2. (a) $\text{dist}(T W(r), T W(s)) = \text{dist}(W(r), W(s))$ because $T$ is an isometry. Every term is unchanged, so the sum is unchanged.
3. (b) Take a pair $r = (u, p)$, $s = (v, p)$ in the same partition $p$. Both are moved by the same $T_p$, so $\text{dist}(T_p W(r), T_p W(s)) = D(r, s)$ as in step 2. By assumption every pair is of this kind, so every term is unchanged.
4. (c) A sum over a disjoint union is the sum of the two sums; apply this to the positive and to the negative sum and regroup. $\blacksquare$

Lean: `contrastiveLoss_comp_isometry`, `tripletLoss_comp_isometry`, `contrastiveLoss_partitionwise_isometry`, `tripletLoss_partitionwise_isometry`, `contrastiveLoss_union`.

**Consequence:** (a) the common space is defined up to one global isometry, so fix partition $1$ as the reference. (b) and (c): $L_{\text{intra}}$ takes the same value however the partitions are rotated against each other, so it contributes nothing to alignment. The only informative terms are over $E^+_{\text{cut}}$, $E^+_{\text{rep}}$ and cross-partition negatives. With $RF = 1$ these are $\delta |E|$ noisy pairs (neighbours); with $RF > 1$ they include exact pairs (the same node).

**Example:** $P = 2$, $d = 2$, $z_{u,1} = (1, 0)$, $z_{v,1} = (0, 1)$, $z_{x,2} = (1, 0)$, with $uv$ an intra edge and $ux$ a cut edge. Rotate partition $1$ by $90°$: $z_{u,1} \mapsto (0, 1)$, $z_{v,1} \mapsto (-1, 0)$. The intra distance stays $\sqrt{2}$, while the cut distance changes from $0$ to $\sqrt{2}$. Only the cut term noticed the rotation.

## Theorem 5 (geometry of the unit sphere)

**Assumptions:** $x, y, a, p, q \in \mathbb{R}^d$ with $\|x\| = \|y\| = \|a\| = \|q\| = 1$, and $m, \alpha \in \mathbb{R}$.

**Statement:**

- (a) $\|x - y\|^2 = 2 - 2 \langle x, y \rangle$.
- (b) $\|x - y\| \leq 2$.
- (c) If $m > 0$: $\|x - y\| < m \iff \langle x, y \rangle > 1 - \tfrac{m^2}{2}$.
- (d) If $m \geq 2$: $\max(0, m - \|x - y\|)^2 \geq (m - 2)^2$.
- (e) If $\max(\|a - p\| - \|a - q\| + \alpha, 0) = 0$, then $\alpha \leq 2$.

**Proof:**

1. (a) Expand: $\|x - y\|^2 = \|x\|^2 - 2\langle x, y \rangle + \|y\|^2 = 1 - 2 \langle x, y \rangle + 1$.
2. (b) Triangle inequality: $\|x - y\| \leq \|x\| + \|y\| = 2$.
3. (c) Both $\|x - y\|$ and $m$ are non-negative, so $\|x - y\| < m \iff \|x - y\|^2 < m^2$. By (a) this is $2 - 2\langle x, y \rangle < m^2$, which rearranges to $\langle x, y \rangle > 1 - m^2/2$.
4. (d) By (b), $m - \|x - y\| \geq m - 2 \geq 0$, so the maximum equals $m - \|x - y\|$. Squaring is increasing on non-negative numbers, so $(m - \|x - y\|)^2 \geq (m - 2)^2$.
5. (e) A maximum equal to $0$ means $\|a - p\| - \|a - q\| + \alpha \leq 0$, so $\alpha \leq \|a - q\| - \|a - p\|$. By (b) $\|a - q\| \leq 2$, and $\|a - p\| \geq 0$, hence $\alpha \leq 2$. $\blacksquare$

Lean: `norm_sub_sq_of_unit`, `norm_sub_le_two_of_unit`, `hinge_active_iff_inner`, `hinge_ge_of_unit_of_large_margin`, `triplet_margin_le_two_of_zero_loss`.

**Consequence:** distance and cosine are interchangeable, both margins live in $(0, 2]$, and each margin reads as a cosine threshold. Loss values are comparable across runs and datasets, which is the point of normalising.

**Example:** $x = (1, 0)$, $y = (0, 1)$: $\langle x, y \rangle = 0$ and $\|x - y\|^2 = 2$. With $m = 1$ the threshold is $1 - \tfrac12 = 0.5$; since $0 \not> 0.5$ the repulsion term is inactive, consistent with $\sqrt{2} \geq 1$. With $m = 3$ every negative pair pays at least $(3 - 2)^2 = 1$, even an antipodal one.

## Theorem 6 (collapse and frustration of the contrastive loss)

**Assumptions:**

- Replicas are embedded in a metric space, $m \geq 0$, and $E^+$, $E^-$ are finite.
- For part (b): there are replicas $r, s, t$ with $r \neq s$, $(r, s) \in E^+$, $(s, t) \in E^+$ and $(r, t) \in E^-$ (an open wedge: two edges whose far ends are not adjacent).

**Statement:**

- (a) If all replicas are mapped to one point, $L_m = |E^-| \, m^2$; in particular $L_m = 0$ when $E^- = \emptyset$.
- (b) For every embedding,

$$ L_m \geq D(r,s)^2 + D(s,t)^2 + \max(0, m - D(r,t))^2 \geq \frac{m^2}{3} $$

**Proof of (a):** every distance is $0$, so each positive term is $0^2 = 0$ and each negative term is $\max(0, m - 0)^2 = m^2$ (using $m \geq 0$). There are $|E^-|$ negative terms. $\blacksquare$

**Proof of (b):**

1. All terms of $L_m$ are squares, hence non-negative. The positive sum is therefore at least the sum of its two terms for $(r,s)$ and $(s,t)$, which are distinct pairs because $r \neq s$; the negative sum is at least its single term for $(r,t)$. This is the first inequality.
2. Put $a = D(r,s)$, $b = D(s,t)$, $c = D(r,t)$. All are $\geq 0$, and $c \leq a + b$ by the triangle inequality.
3. $a^2 + b^2 \geq \tfrac12 (a + b)^2$, because the difference is $\tfrac12 (a - b)^2 \geq 0$.
4. Case $c \geq m$: the hinge is $0$. From $a + b \geq c \geq m \geq 0$ we get $(a + b)^2 \geq m^2$, so by step 3 $a^2 + b^2 \geq m^2/2 \geq m^2/3$.
5. Case $c < m$: the hinge is $(m - c)^2$. From $a + b \geq c \geq 0$ we get $(a + b)^2 \geq c^2$, so $a^2 + b^2 \geq c^2/2$. Then

$$ \frac{c^2}{2} + (m - c)^2 - \frac{m^2}{3} = \frac32 c^2 - 2 m c + \frac23 m^2 = \frac32 \Big(c - \frac{2m}{3}\Big)^2 \geq 0 $$

6. In both cases the three terms sum to at least $m^2/3$. $\blacksquare$

Lean: `contrastiveLoss_const`, `contrastiveLoss_const_of_no_negatives`, `wedge_lower_bound`, `dist_wedge_lower_bound`, `contrastiveLoss_ge_of_open_wedge`, `wedge_lower_bound_attained`.

In $\mathbb{R}^d$ the bound is attained by three collinear points at spacing $m/3$. On the sphere three distinct points are never collinear, so the bound is strict there; this remark is not formalized.

**Consequence:** zero loss would need every connected component of the graph to be a clique, which no real graph is. The minimiser is a compromise that pulls neighbourhoods into blobs, a community-style geometry and not the one node2vec produces. By (a) the attraction term alone is minimised by collapse; only the negatives prevent it, so their sampling rate is a real hyperparameter.

**Example:** $m = 1$ and a path $r - s - t$ with $rt$ a non-edge. Points on a line at $0, \tfrac13, \tfrac23$ give $\tfrac19 + \tfrac19 + (1 - \tfrac23)^2 = \tfrac13$, the bound. Collapsing all three gives $0 + 0 + 1 = 1$. Spreading them at $0, \tfrac12, 1$ gives $\tfrac14 + \tfrac14 + 0 = \tfrac12$. No placement reaches $0$.

## Theorem 7 (why normalisation is needed)

**Assumptions:** $c > 0$ (a scale factor applied to an unnormalised embedding), and $m, \alpha, D, a, b \in \mathbb{R}$, where $a$ is the anchor-positive and $b$ the anchor-negative distance.

**Statement:**

$$ \text{(a)} \quad \max(0, m - c D) = c \max\big(0, \tfrac{m}{c} - D\big) \qquad \text{(b)} \quad \max(c a - c b + \alpha, 0) = c \max\big(a - b + \tfrac{\alpha}{c}, 0\big) $$

(c) If $a < b$ and $c \geq \dfrac{\alpha}{b - a}$, then $\max(c a - c b + \alpha, 0) = 0$.

**Proof:**

1. (a), (b) For $c \geq 0$, $c \max(x, y) = \max(c x, c y)$. Apply it with $c \cdot 0 = 0$ and $c (\tfrac{m}{c} - D) = m - c D$, respectively $c (a - b + \tfrac{\alpha}{c}) = c a - c b + \alpha$; the division is valid because $c \neq 0$.
2. (c) $b - a > 0$, so multiplying $c \geq \alpha / (b - a)$ by $b - a$ gives $c (b - a) \geq \alpha$, that is $c a - c b + \alpha \leq 0$. The maximum with $0$ is then $0$. $\blacksquare$

Lean: `contrastive_hinge_scale`, `triplet_hinge_scale`, `triplet_hinge_vanishes_of_large_scale`.

**Consequence:** scaling an unnormalised embedding by $c$ is the same as dividing the margin by $c$. An unnormalised $z + \Delta_p(z)$ could inflate the space until every correctly ordered triplet has zero loss, so $\alpha$ would select a scale and not a geometry. The output normalisation $N$ removes this degree of freedom; this theorem is the reason for the standing assumption.

**Example:** $a = 1.0$, $b = 1.1$, $\alpha = 0.5$: the triplet term is $1.0 - 1.1 + 0.5 = 0.4$. Scale by $c = 5 = \alpha / (b - a)$: $5.0 - 5.5 + 0.5 = 0$. Nothing about the ordering improved, yet the loss vanished.

## Theorem 8 (orthogonal alignment is a maximum-cosine problem)

**Assumptions:**

- $Q : \mathbb{R}^d \to \mathbb{R}^d$ is a linear isometry (the restricted aligner $W(u, p) = Q_p z_{u,p}$).
- $(a_i, b_i)$, $i = 1, \dots, n$, are cross-partition pairs of unit vectors: $a_i$ in the partition being rotated, $b_i$ in the reference.
- For part (c): $Q_1, Q_2$ are linear and the set of anchors spans $\mathbb{R}^d$.

**Statement:**

- (a) $\displaystyle \sum_{i=1}^n \|Q a_i - b_i\|^2 = 2n - 2 \sum_{i=1}^n \langle Q a_i, b_i \rangle$.
- (b) If $Q a_i = b_i$ for all $i$, then $\langle b_i, b_j \rangle = \langle a_i, a_j \rangle$ for all $i, j$.
- (c) If $Q_1 a = Q_2 a$ for every anchor $a$, then $Q_1 = Q_2$.

**Proof:**

1. (a) $\|Q a_i\| = \|a_i\| = 1$ because $Q$ is an isometry, so $Q a_i$ and $b_i$ are unit vectors and Theorem 5(a) gives $\|Q a_i - b_i\|^2 = 2 - 2 \langle Q a_i, b_i \rangle$. Sum over $i$: the constants add to $2n$.
2. (b) Substitute $b_i = Q a_i$ and $b_j = Q a_j$: $\langle b_i, b_j \rangle = \langle Q a_i, Q a_j \rangle$. A linear [isometry](../glossary.md#isometry) preserves inner products, so this equals $\langle a_i, a_j \rangle$.
3. (c) Every $x$ is a linear combination $\sum_k \lambda_k a_k$ of anchors. By linearity $Q_1 x = \sum_k \lambda_k Q_1 a_k = \sum_k \lambda_k Q_2 a_k = Q_2 x$. $\blacksquare$

Lean: `procrustes_objective_of_unit` (general form `procrustes_objective`), `gram_eq_of_exact_alignment`, `linear_alignment_unique_of_spanning_anchors`.

**Consequence:** by (a), minimising the attraction term over rotations is maximising the mean cosine of the pairs, which is the [orthogonal Procrustes problem](../glossary.md#orthogonal-procrustes-problem): $Q^* = U V^T$ from the [SVD](../glossary.md#singular-value-decomposition) $M = \sum_i a_i b_i^T = U \Sigma V^T$, with best mean cosine $\operatorname{tr}(\Sigma)/n$. No negatives are needed because an isometry cannot collapse. Take the pairs from $E^+_{\text{rep}}$ (the same node, $a_i = z_{u,p}$, $b_i = z_{u,q}$) when $RF > 1$, or from $E^+_{\text{cut}}$ (neighbours) when $RF = 1$. By (c), $d$ replicated nodes in general position fix the rotation uniquely. By (b), a perfect fit needs the two partitions to agree on all angles between anchors; $\operatorname{tr}(\Sigma)/n$ measures how far they are from that, and the experiment gives $0.70$ for replica pairs.

**Example:** $d = 2$, $a_1 = (1, 0)$, $a_2 = (0, 1)$, $b_1 = (0, 1)$, $b_2 = (-1, 0)$. The $90°$ rotation maps $a_i$ to $b_i$, so the residual is $0$ and the mean cosine is $1$; the Gram condition holds, $\langle a_1, a_2 \rangle = 0 = \langle b_1, b_2 \rangle$. Change $b_2$ to $(0, 1)$: now $\langle b_1, b_2 \rangle = 1 \neq 0$, so by (b) no rotation fits exactly.

## Theorem 9 (merging replicas)

**Assumptions:** $w_1, \dots, w_k \in \mathbb{R}^d$ are the aligned unit vectors of the $k = |\Pi(u)|$ replicas of one node, $S = \sum_{i=1}^k w_i$, and $c$ is any unit vector (a candidate merged embedding). For the equality case, $S \neq 0$.

**Statement:**

$$ \text{(a)} \quad \sum_{i=1}^k \|w_i - c\|^2 = 2k - 2 \langle S, c \rangle \geq 2k - 2\|S\|, \ \text{ with equality at } c = \frac{S}{\|S\|} $$

$$ \text{(b)} \quad \sum_{i=1}^k \sum_{j=1}^k \|w_i - w_j\|^2 = 2k^2 - 2\|S\|^2 $$

**Proof:**

1. (a) By Theorem 5(a) each term is $2 - 2 \langle w_i, c \rangle$. Summing, and using linearity of the inner product in its first argument, gives $2k - 2 \langle S, c \rangle$.
2. Cauchy-Schwarz: $\langle S, c \rangle \leq \|S\| \, \|c\| = \|S\|$, which is the inequality.
3. At $c = S / \|S\|$ (a unit vector because $S \neq 0$): $\langle S, c \rangle = \|S\|^2 / \|S\| = \|S\|$, so equality holds.
4. (b) By Theorem 5(a) each term is $2 - 2 \langle w_i, w_j \rangle$. There are $k^2$ terms, and $\sum_i \sum_j \langle w_i, w_j \rangle = \langle \sum_i w_i, \sum_j w_j \rangle = \|S\|^2$ by bilinearity. $\blacksquare$

Lean: `replica_spread_of_unit`, `replica_spread_ge`, `replica_spread_at_normalised_sum`, `replica_pairwise_spread_of_unit`.

**Consequence:** by (a) the normalised sum $w_u = S / \|S\|$ is the best single unit embedding of a replicated node. By (b) the replica term of the loss, $\sum_{E^+_{\text{rep}}} D^2$, is small exactly when $\|S\|$ is large, so training with that term directly reduces the error of the merge. Without alignment the replicas are near-orthogonal, $\|S\| \approx \sqrt{k}$, and the merged vector averages unrelated directions.

**Example:** $k = 2$. Aligned replicas with cosine $0.9$: $\|S\|^2 = 2 + 2 \cdot 0.9 = 3.8$, $\|S\| \approx 1.95$, residual $4 - 3.90 = 0.10$; check of (b): $2 \|w_1 - w_2\|^2 = 2 (2 - 1.8) = 0.4 = 8 - 7.6$. Unaligned replicas with cosine $0$: $\|S\| = \sqrt{2}$, residual $4 - 2.83 = 1.17$, and the merged vector has cosine only $1/\sqrt{2} \approx 0.71$ with each replica.

## Theorem 10 (headroom)

**Assumptions:**

- An edge is drawn at random; $\text{Cut}$ is the event that it is a cut edge, with $0 < \delta = \mathbb{P}(\text{Cut}) < 1$.
- $F$ and $F'$ are integrable per-edge scores before and after alignment (for example the indicator that the edge is reconstructed).
- $S = \mathbb{E}[F \mid \neg\text{Cut}]$, $C = \mathbb{E}[F \mid \text{Cut}]$, and likewise $S'$, $C'$ for $F'$, as in [partitioning impact on F1 score](../../results/E08-f1-decomposition/README.md).
- The aligner leaves the covered mean unchanged, $S' = S$, and $C' \leq 1$.

**Statement:**

$$ \mathbb{E}[F'] - \mathbb{E}[F] \leq \delta (1 - C) $$

**Proof:**

1. Law of total expectation: $\mathbb{E}[F] = S (1 - \delta) + C \delta$ and $\mathbb{E}[F'] = S' (1 - \delta) + C' \delta$.
2. Substitute $S' = S$ and subtract: $\mathbb{E}[F'] - \mathbb{E}[F] = \delta (C' - C)$.
3. $\delta \geq 0$ and $C' \leq 1$ give $\delta (C' - C) \leq \delta (1 - C)$. $\blacksquare$

Lean: `alignment_gain_le`, which reuses `expected_eq_within_cross` from `PartitioningF1.lean`.

**Consequence:** for $RF = 1$ a per-partition isometry leaves every within-partition distance unchanged (Theorem 4), so $S' = S$ holds and the gain is capped by $\delta (1 - C)$: measure $\delta$ and $C$ before investing in an aligner. For $RF > 1$ the assumption $S' = S$ fails, because $S$ is computed from merged vectors $w_u$ that depend on the alignment (Theorem 9); there alignment also moves $S$, and the cap does not apply.

**Example:** $\delta = 0.1$, $C = 0.5$: the gain is at most $0.1 \cdot 0.5 = 0.05$, however good the aligner. In the experiment with $RF = 2$, alignment raises the covered-edge AUC from $0.78$ to $0.85$, an effect outside this bound.

