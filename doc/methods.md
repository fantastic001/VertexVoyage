# Methods

VertexVoyage embeds a large, possibly growing graph by splitting it into $P$
partitions, embedding every partition independently, and merging the
partition embeddings into one embedding per vertex. This page defines the two
pipelines the experiments use. Metrics are defined in
[evaluation.md](evaluation.md), and terms in [glossary.md](glossary.md).

**Notation.**
- $G = (V, E)$: an undirected graph.
- $\pi : V \to 2^{\{1..P\}}$, $1 \le |\pi(v)| \le RF$: the partition map;
  $RF$ is the replication factor.
- $G_i = G[\{v : i \in \pi(v)\}]$: the subgraph induced by partition $i$.
- $z_{v,i} \in \mathbb{R}^d$: the embedding of $v$ learned in partition $i$.
- $z_v$: the merged embedding.

## Pipeline S — static (`vv test`)

1. Load $G$ (registered in [`experiments/datasets.py`](../experiments/datasets.py)).
2. Partition it with **label propagation** (`--use-lpa`) or, by default,
   with the **modified LFM** community partitioner
   (`partition_graph(..., use_modified_lfm=True)`).
3. Run node2vec (or DistGER, `--algorithm distger`) on every $G_i$.
4. Merge with the aggregator (`--aggregator`).
5. Evaluate reconstruction F1 and, with `--link-prediction-model`, link
   prediction.

Used by: [E00](../results/E00-legacy-distributed/README.md),
[E01](../results/E01-static-node2vec/README.md),
[E08](../results/E08-f1-decomposition/README.md),
[E09](../results/E09-link-prediction/README.md),
[E10](../results/E10-partition-alignment/README.md).

## Pipeline T — streaming (`vv temporal_test`)

1. **Edge ordering.** Edges are turned into a stream of events, in dataset
   order or by `--ordering bfs|dfs|random`.
2. **Buffering.** The stream is cut into buffers of `--buffer-size` events.
3. **Partitioning.** The streaming neighbour partitioner (below) assigns new
   and affected vertices.
4. **Embedding.** Every partition updates its model with dynnode2vec. Walks
   are generated only from affected vertices, and the skip-gram model is
   fine-tuned for `--epochs`.
5. **Aggregation.** The partition embeddings are merged (below).
6. **Evaluation.** Reconstruction F1, edge cut, balance and repartitioning
   are recorded after every buffer.

Used by: [E02](../results/E02-hyperparameters/README.md)–[E07](../results/E07-buffer-size/README.md)
and [E11](../results/E11-ablation/README.md).

The full pipeline (the `baseline` variant of the ablation study) is

```
vv temporal_test --name D --partitions P --replication-factor RF --seed S \
  --long-run --use-dataset-params --ordering bfs --track-seen \
  --buffer-size 1000 --partitioner-name neighbors.all --mu 1 --alpha 1 \
  --aggregator triplet --iterations 1 --skip-full-graph-baseline
```

The hyperparameters were selected in [E02](../results/E02-hyperparameters/README.md).
`--track-seen` has no effect when `--ordering` is set
([guide/cli.md](guide/cli.md#additional-temporal_test-flags)).

<a id="streaming-neighbour-partitioner"></a>
## Streaming neighbour partitioner (`neighbors.all`)

The partition of a vertex is the partition holding most of its neighbours in
the sampled events. The partitioner samples events according to a
distribution $\Pr(\text{event} \mid \text{vertex})$ and scores each
partition $Q$ for vertex $v$ by

$$
S(Q, v) = N(Q, v) - \mu \cdot \max\Big(0,\; |Q| - \alpha (1 + \epsilon) \frac{1}{P} \sum_{Q'} |Q'|\Big),
$$

where:
- $N(Q, v)$ is the number of neighbours of $v$ in $Q$ among the sampled
  events;
- $\mu$ is the capacity-penalty coefficient;
- $\alpha$ weights the average partition size;
- $\epsilon$ is the imbalance tolerance.

The vertex is assigned to the top $RF$ partitions, and ties are broken at
random. With `--decay` $\lambda > 0$, each neighbour seen $t$ steps ago
counts $e^{-\lambda t}$ instead of $1$:

$$
S(Q, v) = \sum_{u \in N(Q, v)} e^{-\lambda\, t(u)} - \mu \cdot \max\Big(0,\; |Q| - \alpha (1 + \epsilon) \frac{1}{P} \sum_{Q'} |Q'|\Big).
$$

With $\mu = 0$ there is no capacity penalty, and in every run the
partitioner places all vertices in one partition
([E04](../results/E04-balance-penalty/README.md)). The `random` partitioner
assigns vertices uniformly at random and is the structure-blind control.

**Related work.** LDG (Stanton and Kliot, KDD 2012) uses the score
$|N(Q, v)| \cdot (1 - |Q|/C)$ and FENNEL (Tsourakakis et al., WSDM 2014)
uses $|N(Q, v)| - \alpha\gamma|Q|^{\gamma-1}$. Both are multiplicative or
convex size penalties applied to the same neighbour count. A direct
comparison is planned ([ROADMAP.md](../ROADMAP.md)).

## Aggregators (`--aggregator`)

| Name | Merged embedding $z_v$ | Defined in |
|---|---|---|
| `mean` | $\frac{1}{\lvert\pi(v)\rvert}\sum_{i \in \pi(v)} z_{v,i}$ | `aggregation/mean.py` |
| `random_partition` | the embeddings of one randomly chosen partition | `aggregation/random_selection.py` |
| `random_replica` | $z_{v,i}$ for a random $i \in \pi(v)$ | `aggregation/random_selection.py` |
| `procrustes` | per-partition orthogonal map fitted on replica anchors | `aggregation/procrustes.py` |
| `contrastive`, `triplet` | learned residual aligner, below | `aggregation/alignment.py` |

**Residual aligner.** The aligner computes
$W(u, p) = N\big(z_{u,p} + \Delta_p(z_{u,p})\big)$, where
$\Delta_p(z) = B\,\mathrm{ReLU}(A_p z + a_p) + b$ and $N$ normalises to unit
length. It is trained after every buffer for `alignment_train_steps` steps
with a triplet or contrastive loss on graph edges. The theory (Theorems
1–18, proved in Lean) is in
[theory/partition-alignment.md](theory/partition-alignment.md), and the
experiments are in [E10](../results/E10-partition-alignment/README.md).

Defaults, all overridable through `VERTEX_VOYAGE_ALIGNMENT_*`:

| Setting | Default | Source |
|---|---|---|
| initialisation | `identity` (zero residual) | `aggregation/alignment.py` |
| link weighting | `uniform` | `aggregation/alignment.py` |
| output scale | `auto` (aggregator's choice; `unit` for triplet) | `aggregation/replica_alignment.py` |
| steps, learning rate, margin | 100, 0.01, 0.5 | `aggregation/alignment.py` |

## Pipeline-component switches used by the ablation study

Walk strategy (`VERTEX_VOYAGE_WALK_STRATEGY`), word2vec training
(`VERTEX_VOYAGE_WORD2VEC_TRAINING`), partition training
(`VERTEX_VOYAGE_PARTITION_TRAINING`) and the retrain threshold
(`VERTEX_VOYAGE_RETRAIN_THRESHOLD`) are listed with their variants in
[`experiments/ablation/variants.py`](../experiments/ablation/variants.py)
and [E11](../results/E11-ablation/README.md).
