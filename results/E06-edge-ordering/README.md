# E06 — Edge-ordering impact

| | |
|---|---|
| **Research question** | [RQ2](../README.md#research-questions) |
| **Status** | 🟡 partial: CITESEER only; AstroPh cells empty |
| **Period** | 2026-09 → 2026-10 |
| **Pipeline** | T, streaming, `--ordering bfs|dfs|random` |
| **Datasets** | CITESEER |
| **Headline** | No ordering dominates: random is best at $P = 1$ (0.58 vs 0.48 for BFS) and worst at $P = 4$ (0.36 vs 0.44). BFS was adopted for the ablation baseline. |
| **Open issues** | `--track-seen` is ignored whenever `--ordering` is set (`_order_temporal_events` in `vertex_voyage/plugins/test_commands.py`), so the "baseline without ordering" column (track-seen) and the BFS columns use different stream semantics. The ablation variants `random_edge_order` and `no_edge_order` ([E11](../E11-ablation/README.md#edge-ordering)) supersede this experiment. |

**Question:** Does the order in which edges are streamed into dynnode2vec affect the reconstruction F1 score?

# Setup 

```sh 
vv temporal_test \
 --name CITESEER \
 --long-run \
 --iterations 5  \
 --use-dataset-params \
 --buffer-size 1000 \
 --replication-factor 1 \
 --partitioner neighbors.all  \
 --mu 1 \
 --algorithm dynnode2vec \
 --epochs 10  \
 --skip-full-graph-baseline \
 --partitions 4 \
 --ordering bfs 
```

# Results 

For P=1

| Dataset | BFS | DFS | Random | Baseline without ordering
| --- | --- | --- | --- | --- |
| CITESEER    | 0.48 +- 0.0064    | 0.52 +- 0.0141    |0.58 +- 0.0349     | 0.476    |
| AstroPh    |     |     |     | 0.577  |

For P=2

| Dataset | BFS | DFS | Random | Baseline without ordering |
| --- | --- | --- | --- | --- |
| CITESEER    | 0.33 +- 0.0135    |     |     | 0.458    |
| AstroPh    |     |     |     | 0.547    |

For P=4

| Dataset | BFS | DFS | Random | Baseline without ordering |
| --- | --- | --- | --- | --- |
| CITESEER    |0.44 +- 0.0019     | 0.45 +- 0.0041    | 0.36 +- 0.0079    | 0.481    |
| AstroPh    |     |     |     | 0.510   |


For P=8

| Dataset | BFS | DFS | Random | Baseline without ordering |
| --- | --- | --- | --- | --- |
| CITESEER    |     |     |     | 0.548   |
| AstroPh    |     |     |     | 0.459   |

Baseline results took from [partition-count scaling](../E03-partition-scaling/README.md) experiment.

# Time profile

| Dataset | BFS | DFS | Random |
| --- | --- | --- | --- |
| CITESEER    | 0.1s    | 0.26s    | 0.02s    |
| AstroPh    |     |     |     |


Time is reported as total time over all 5 iterations.