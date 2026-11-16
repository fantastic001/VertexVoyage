# Experiment commands (CLI)

 

To run experiment for partitioning and embedding on a specific dataset, run:

    vv test --name Cit-HepPh --partitions 1  --break-early  --use-dataset-params --long-run  --epochs 10

For temporal experiments, run:

    vv temporal_test --name CITESEER  --long-run --iterations 10  --use-dataset-params  --buffer-size 50 --partitions 5 --replication-factor 2 --partitioner neighbors.all  --mu 0.5

## Test command flags

 Runs a test of the embedding quality for a given dataset and partitioning parameters. It loads the dataset, partitions it using the specified method, computes embeddings for each partition using the specified algorithm, and then evaluates the quality of the embeddings by reconstructing the graph and computing the F1 score against the original graph. It also computes a global embedding by averaging the partition embeddings and evaluates its F1 score as well.

Parameters:

| Parameter | Description |
|-----------|-------------|
| `--name` |  The name of the dataset to use. |
| `--partitions` | The number of partitions to create. |
| `--alpha` | The alpha parameter for the partitioning algorithm (if applicable). |
| `--threshold` | The threshold parameter for the partitioning algorithm (if applicable). |
| `--break-early` | If True, breaks after the first combination of p and q parameters is tested. |
| `--skip-global` | If True, skips the global F1 score computation. |
| `--dim` | The dimensionality of the embeddings. |
| `--default-p` | If > 0, uses this value for p in the embedding algorithm instead of testing multiple values. |
| `--default-q` | If > 0, uses this value for q in the embedding algorithm instead of testing multiple values. |
| `--epochs` | The number of epochs to train the embedding model for. |
| `--long-run` | If True, uses more walks and larger walk sizes for the embedding algorithm, which may lead to better embeddings but takes longer to compute. |
| `--use-dataset-params` | If True, overrides the parameters with dataset-specific parameters from the dataset_params dictionary if they are available. |
| `--use-lpa` | If True, uses label propagation for partitioning instead of the default partitioning algorithm. |
| `--algorithm` | The embedding algorithm to use (e.g., "node2vec", "distger", "dynnode2vec"). |

## Temporal test command flags

Runs a temporal test of the embedding quality for a given dataset and partitioning parameters. It loads the dataset as a stream of events, partitions it using the specified method, computes embeddings for each partition using the specified algorithm, and then evaluates the quality of the embeddings by reconstructing the graph and computing the F1 score against the original graph after each buffer of events is processed. It also computes a global embedding by averaging the partition embeddings and evaluates its F1 score as well.

Parameters:

| Parameter | Description |
|-----------|-------------|
| `--name` | The name of the dataset to use. |
| `--partitions` | The number of partitions to create. |
| `--partitioner-name` | The name of the partitioning algorithm to use (e.g., "random", "random.degree", "neighbors.all", "neighbors.degree"). |
| `--dim` | The dimensionality of the embeddings. |
| `--default-p` | If > 0, uses this value for p in the embedding algorithm instead of testing multiple values. |
| `--default-q` | If > 0, uses this value for q in the embedding algorithm instead of testing multiple values. |
| `--epochs` | The number of epochs to train the embedding model for after each buffer of events. |
| `--long-run` | If True, uses more walks and larger walk sizes for the embedding algorithm, which may lead to better embeddings but takes longer to compute. |
| `--use-dataset-params` | If True, overrides the parameters with dataset-specific parameters from the dataset_params dictionary if they are available. |
| `--use-lpa` | If True, uses label propagation for partitioning instead of the default partitioning algorithm (not implemented in this method). |
| `--algorithm` | The embedding algorithm to use (e.g., "dynnode2vec"). |
| `--track-seen` | If True, processes events in an order that prioritizes events connected to already seen nodes, which simulates a more realistic temporal scenario where new events are more likely to involve nodes that have already been observed. |
| `--iterations` | The number of times to repeat the entire process for averaging results. |
| `--limit` | If > 0, limits the number of events to process from the dataset for quicker testing. |
| `--buffer-size` | The number of events to process in each buffer before updating the embeddings and evaluating the F1 score. |


### Additional `temporal_test` flags

These flags were added after the table above was written. The source of
truth is the signature of `TestCustomCLICommandExecutor.temporal_test` in
[`vertex_voyage/plugins/test_commands.py`](../../vertex_voyage/plugins/test_commands.py).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ordering` | `""` | Edge order of the stream: `bfs`, `dfs` or `random`. Empty keeps the dataset order, reshuffled by `--track-seen` if that is set. When `--ordering` is given, `--track-seen` has no effect. |
| `--track-seen` | off | Without `--ordering`: shuffle events, then prefer events that touch an already seen vertex. |
| `--buffer-size` | 100 | Events per buffer. The partitioner and the embedding are updated once per buffer. |
| `--replication-factor` | 1 | Maximum number of partitions a vertex is assigned to. |
| `--mu`, `--alpha`, `--epsilon`, `--decay` | 0, 1, 0.1, 0 | Parameters of the neighbour partitioner, see [methods.md](../methods.md#streaming-neighbour-partitioner). |
| `--aggregator` | `triplet` | How partition embeddings are merged: `mean`, `triplet`, `contrastive`, `procrustes`, `random_partition`, `random_replica`. |
| `--n-walks`, `--walk-size`, `--window-size` | 0 (use defaults) | Random-walk and word2vec context settings. |
| `--iterations` | 1 | Repetitions of the whole run. The ablation runner uses 1 and varies `--seed` instead. |
| `--seed` | `DEFAULT_SEED` | Master seed, see [seeding.md](seeding.md). |
| `--limit` | -1 | Process only the first N events (quick tests). |
| `--checkpoint` | `""` | Directory of the `PersistedRun` checkpoint. A run resumes from it. |
| `--skip-full-graph-baseline` | off | Do not train the full-graph node2vec baseline. |
| `--algorithm` | `dynnode2vec` | `dynnode2vec`, or `dummy`, which skips embedding and only measures partitioning (balance, edge cut, repartitioning). |

Environment variables `VERTEX_VOYAGE_<KEY>` override every `get_config_*`
setting, for example `VERTEX_VOYAGE_WALK_STRATEGY` or
`VERTEX_VOYAGE_ALIGNMENT_TRAIN_STEPS`. See [cluster.md](cluster.md#configuration)
for how configuration is resolved.
