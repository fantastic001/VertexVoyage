# E11 — Ablation study of the streaming pipeline

| | |
|---|---|
| **Research question** | [RQ5](../README.md#research-questions) |
| **Status** | 🔵 running: exploratory results recorded; the 19-variant grid × 10 seeds is set up (runner in [`experiments/ablation/`](../../experiments/ablation/README.md)) and its tables are being filled |
| **Period** | 2026-07-14 → ongoing |
| **Pipeline** | T, streaming, full pipeline as in [methods](../../doc/methods.md#pipeline-t--streaming-vv-temporal_test) |
| **Datasets** | CITESEER, AstroPh, DBLP, Enron; $P \in \{1,2,4\}$, $RF \in \{1,3\}$ |
| **Headline (exploratory)** | The structural partitioner is essential (random partitioning collapses F1, more so as $P$ grows). Removing cross-partition edges alone does not reproduce the full pipeline. Dividing epochs by $P$ is roughly neutral on CITESEER and hurts AstroPh at high $P$. |
| **Open issues** | `random_edge_order` keeps `--track-seen`, which has no effect once `--ordering` is set. Exploratory random-partitioning F1 conflicts with [E03](../E03-partition-scaling/README.md#random-partitioner-baseline). |

**Question.** Which part of the pipeline is responsible for the reconstruction
quality seen in [partition-count scaling](../E03-partition-scaling/README.md)? Each
variant re-runs that experiment with **one** component replaced or removed
and compares the RESTORE F1 (graph-reconstruction F1) against the full
pipeline.

# Pipeline 

For temporal pipeline, the following components are considered: 

| Component | Description | Regulated by |
|---|---|---|
| [Edge ordering](#edge-ordering) | The order in which edges are processed in the temporal graph | `--ordering`, `--track-seen` |
| [Stream buffering](#stream-buffering) | Creates stream of buffers from stream of edges | `--buffer-size` |
| [Buffer partitioning](#buffer-partitioning) | Divides the stream of buffers into partitions for processing | `--partitioner-name`, `--partitions`, `--replication-factor`, `--mu` |
| [Partition random walk generation](#partition-random-walk-generation) | Generates random walks within each partition for training | `--default-p`, `--default-q`, `--use-dataset-params`, `--long-run`, `--n-walks`, `--walk-size`, `--window-size`, `VERTEX_VOYAGE_WALK_STRATEGY` |
| [Training word2vec model](#training-word2vec-model) | Trains the word2vec model on the generated random walks | `--algorithm`, `--epochs`, `--dim` |
| [Aggregation](#aggregation) | Combines the embeddings from different partitions into a final representation | `--aggregator`, `VERTEX_VOYAGE_ALIGNMENT_*` |

Every variant runs `vv temporal_test` (`TestCustomCLICommandExecutor.temporal_test` in `vertex_voyage/plugins/test_commands.py`) with one setting changed from the full pipeline. A setting is either a CLI flag (an argument of `temporal_test`, with `_` written as `-`; boolean flags are switched on by passing them and off by leaving them out) or an environment variable `VERTEX_VOYAGE_X`, where `x` is a key read by a `get_config_*` function. Every setting listed below exists in the code. The walk and word2vec variants only have an effect with `--algorithm dynnode2vec`, because `dummy` returns zero embeddings and never runs walks or word2vec.

# Ablation Variants

## Edge ordering

### Randomized edge order

**Setting.** Set `--ordering` to `random` (`RandomOrdering` shuffles the event list uniformly at random before buffering).

We report mean +- standard deviation.

RF=1

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |

RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

### No edge ordering

**Setting.** Set `--ordering` to `""` (the default) and leave out `--track-seen`. Events are then processed in the order they appear in the dataset file.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

## Stream buffering

### No stream buffering - only one big buffer 

**Setting.** Set `--buffer-size` to $\lvert E \rvert$, the number of events in the dataset, so `buffered` yields a single buffer: CITESEER `4536`, AstroPh `198110`, DBLP `52867`, Enron `367662`.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

### Buffer size is 1 

**Setting.** Set `--buffer-size` to `1`.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

## Buffer partitioning

### Random partitioning 

**Setting.** Set `--partitioner-name` to `random` (`RandomPartitioner.uniform`).

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |


RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |


**Conclusion:**

### No partitioning - all nodes in a single partition

**Setting.** Set `--partitions` to `1`. The $P$ columns then all coincide with the $P = 1$ run.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

### Single partition - with cut edges removed 

**Setting.** Set `VERTEX_VOYAGE_PARTITION_TRAINING` to `single_model_without_cut_edges` (default `per_partition`): partition with `--partitioner-name neighbors.all` and `--partitions P`, then train one model on the union of the within-partition edges.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

## Partition random walk generation 

### Unbiased random walk - p = q = 1 

**Setting.** Set `--default-p` to `1` and `--default-q` to `1`, and leave out `--use-dataset-params` (it overrides $p$, $q$ and `dim` from `dataset_params`). Set `--dim` to the dataset's value instead: CITESEER `50`, AstroPh `50`, DBLP `25`, Enron `128`.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

### No walk at all - just random sequence of vertices 

**Setting.** Set `VERTEX_VOYAGE_WALK_STRATEGY` to `random_vertex_sequence` (default `node2vec`): each walk starts at its start vertex and continues with `walk_size - 1` vertices drawn uniformly at random from the partition, ignoring edges.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

### P(x | y) = 0 if x != y - walk with repeated start vertex

**Setting.** Set `VERTEX_VOYAGE_WALK_STRATEGY` to `repeat_start_vertex` (default `node2vec`): each walk is its start vertex repeated `walk_size` times.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

### Just one walk 

**Setting.** Set `--n-walks` to `1`. `--long-run` sets all three walk parameters at once (on: `n_walks=10`, `walk_size=80`, `window_size=10`; off: `n_walks=1`, `walk_size=10`, `window_size=3`), and `--n-walks`, `--walk-size` and `--window-size` each override one of them; `0`, the default of all three, keeps the `--long-run` value. So the single-walk variant is `--long-run --n-walks 1`: one walk per vertex, with the walk and window sizes of the full pipeline.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

## Training word2vec model 

### random weights only 

**Setting.** Set `VERTEX_VOYAGE_WORD2VEC_TRAINING` to `random_init_only` (default `train`): build the word2vec vocabulary but skip `train`. `--epochs 0` does not work, because gensim rejects a zero epoch count.
We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

### Only one epoch 

**Setting.** Set `--epochs` to `1`.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**


### retrain word2vec every time 

**Setting.** Set `VERTEX_VOYAGE_RETRAIN_THRESHOLD` to `100` (default `0.1`). The value is a fraction $f$ of $\lvert V \rvert$, and `retrain_threshold` becomes $\lfloor f \cdot \lvert V \rvert \rfloor$. The default $f = 0.1$ is `DEFAULT_RETRAIN_THRESHOLD_FRACTION` in `retrain_threshold_of`. `DynNode2Vec.update` retrains a partition from scratch while `len(G.nodes()) < retrain_threshold`. A partition never holds more than $\lvert V \rvert$ vertices, and $\lvert V \rvert < \lfloor 100 \cdot \lvert V \rvert \rfloor$, so every update retrains from scratch for every $P$.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

## Aggregation 

### Select random partition for final embedding 

**Setting.** Set `--aggregator` to `random_partition`: one partition is drawn uniformly at random on the first buffer and its model embeds every vertex for the rest of the run.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**

### Select random partition embedding 

**Setting.** Set `--aggregator` to `random_replica`: every time a vertex is embedded, one of its replicas is drawn uniformly at random and its model embeds the vertex.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**



### Aggregate with average 

**Setting.** Set `--aggregator` to `mean`. At $P = 1$, `mean` is always used, since `VERTEX_VOYAGE_ALIGNMENT_MINIMUM_PARTITIONS` defaults to `2`.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**



### Aggregate with triplet without training 

**Setting.** Set `--aggregator` to `triplet` and `VERTEX_VOYAGE_ALIGNMENT_TRAIN_STEPS` to `0`. With the default `VERTEX_VOYAGE_ALIGNMENT_INITIALISATION`, the untrained aligner stays at its initial parameters.

We report mean +- standard deviation.

| Dataset  | P=1 | P=2 | P=4 | 
|---|---|---|---|
| CITESEER |     |     |     |
| AstroPh  |     |     |     |
| DBLP     |     |     |     |
| Enron    |     |     |     |



RF=3 


| Dataset  | P=4 | 
|---|---|
| CITESEER |     |
| AstroPh  |     |

**Conclusion:**


# How it was run

The variants are run by the ablation runner,
[`experiments/ablation/`](../../experiments/ablation/README.md) (its `README.md` describes the quick local test), and
submitted to SLURM with `scripts/slurm/submit_ablation.sh`.

**Full pipeline.** Let $D$ be a dataset, $P$ the number of partitions, RF the
replication factor and $s$ a seed. The full pipeline (variant `baseline`) is
the base command of [partition-count scaling](../E03-partition-scaling/README.md)
with BFS edge ordering:

```
vv temporal_test --name D --partitions P --replication-factor RF --seed s
  --long-run --use-dataset-params --ordering bfs --track-seen
  --buffer-size 1000 --partitioner-name neighbors.all --mu 1 --alpha 1
  --aggregator triplet --iterations 1 --skip-full-graph-baseline
```

Every variant changes exactly the setting named in its section and keeps all
other flags. For example, *Randomized edge order* replaces `--ordering bfs`
by `--ordering random` and keeps `--track-seen`, while *No edge ordering*
removes both `--ordering` and `--track-seen`. The full-graph Node2Vec
baseline is skipped in every run.

**Grid.** Each variant runs on the cells $(D, P, \mathrm{RF})$:

- RF $= 1$, $P \in \{1, 2, 4\}$, $D \in$ {CITESEER, AstroPh, DBLP, Enron};
- RF $= 3$, $P = 4$, $D \in$ {CITESEER, AstroPh}; for *Random partitioning*
  $D \in$ {CITESEER, AstroPh, DBLP, Enron}.

**Seeds.** Each cell is run $n = 10$ times as independent SLURM jobs with
`--iterations 1` and seeds $s \in \{42, 43, \dots, 51\}$. Let
$F_1, \dots, F_n$ be the final RESTORE F1 scores of these runs. A table entry
is $\bar F \pm \sigma$ with $\bar F = \frac{1}{n} \sum_{i=1}^{n} F_i$ and
$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (F_i - \bar F)^2}$. For example,
the runner reports $F = (0.3055, 0.3158)$ as $0.311 \pm 0.005$ $(n = 2/2)$.

**Commands.** Variants are submitted one at a time:

```
scripts/slurm/submit_ablation.sh list
scripts/slurm/submit_ablation.sh submit <variant> [dataset ...]
scripts/slurm/submit_ablation.sh status <variant>
scripts/slurm/submit_ablation.sh collect <variant>
```

`collect` prints the RF $= 1$ and RF $= 3$ tables of a variant in the layout
of this page. The variant names are listed below.

| Section | Variant |
|---|---|
| Full pipeline | `baseline` |
| Randomized edge order | `random_edge_order` |
| No edge ordering | `no_edge_order` |
| No stream buffering | `single_buffer` |
| Buffer size is 1 | `unit_buffer` |
| Random partitioning | `random_partitioning` |
| No partitioning | `no_partitioning` |
| Single partition with cut edges removed | `single_model_without_cut_edges` |
| Unbiased random walk | `unbiased_walk` |
| Random sequence of vertices | `random_vertex_sequence` |
| Walk with repeated start vertex | `repeat_start_vertex` |
| Just one walk | `one_walk` |
| Random weights only | `random_init_only` |
| Only one epoch | `one_epoch` |
| Retrain word2vec every time | `retrain_every_update` |
| Select random partition for final embedding | `random_partition_aggregator` |
| Select random partition embedding | `random_replica_aggregator` |
| Aggregate with average | `mean_aggregator` |
| Aggregate with triplet without training | `untrained_triplet_aggregator` |

The earlier exploratory runs reported under [Results](#results) used the
following variants:

| Variant | Change vs. full pipeline |
|---|---|
| Full pipeline | baseline (structural partitioning + `neighbors.all` walks) |
| Removing real partitioning | keep the partition induced subgraphs but drop the cross-partition ("noise") edges only; no structural partitioner |
| Random partitioning | assign vertices to partitions uniformly at random |
| Reducing epochs $P\times$ | full pipeline but each partition trained for `epochs/P` |
| dynnode2vec on epochs$/P$ | streaming dynnode2vec update with the reduced epoch budget |

# Results

## Full pipeline (baseline)

CITESEER: P1 0.476 · P2 0.458 · P4 0.481 · P8 0.548
AstroPh:  P1 0.577 · P2 0.547 · P4 0.510 · P8 0.459

see [partition-count scaling](../E03-partition-scaling/README.md) for the source
mix-up this replaces

## Removing real partitioning

CITESEER: P2 0.408 · P4 0.396 · P8 0.420 · P16 0.412
AstroPh:  P1 0.704 · P2 0.683 · P4 0.688 · P8 0.688

CITESEER: dropping the cross-partition edges does **not** reproduce the
full-pipeline embeddings — F1 stays flat around 0.41 instead of climbing to
0.54. AstroPh: reconstruction is unaffected (~0.69 throughout).

## Random partitioning

CITESEER: P2 0.298 · P4 0.266 · P8 0.173
AstroPh:  P2 0.502 · P4 0.095 · P8 0.083

Random partitioning **collapses** reconstruction F1, and worse as $P$ grows —
the structural partitioner is doing real work.

## Reducing epochs $P$ times

CITESEER: P2 0.409 · P4 0.424 · P8 0.551
AstroPh:  P2 0.660 · P4 0.485 · P8 0.440

## dynnode2vec on epochs$/P$

CITESEER: P2 0.483 · P4 0.469 · P8 0.510
AstroPh:  P2 0.557 · P4 0.489 · P8 0.416

# Conclusion

- The **structural partitioner is essential**: replacing it with random
  assignment destroys reconstruction quality, increasingly so with $P$.
- Merely deleting cross-partition edges is not equivalent to the full
  partitioning pipeline (clear on CITESEER), so the gain is not just "noise
  removal".
- Splitting the epoch budget by $P$ keeps CITESEER roughly on par with the
  baseline but hurts AstroPh at high $P$ — the per-partition training budget
  matters more for the denser graph.
