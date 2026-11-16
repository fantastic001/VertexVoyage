# Experiment registry

Every experiment has a stable ID, a page in this directory, and a status
header with the same fields: research question, status, period, pipeline,
datasets, headline, open issues. Pipelines S (static) and T (streaming) are
defined in [doc/methods.md](../doc/methods.md), and metrics in
[doc/evaluation.md](../doc/evaluation.md).

**Status legend:** ✅ done · 🔵 running or ongoing · 🟡 partial · 🔁 re-run
planned · ⚪ superseded

## Research questions

| RQ | Question | Experiments | Current answer |
|---|---|---|---|
| **RQ1** | Does community-aware partitioning preserve reconstruction quality, and why does F1 move with $P$? | E00, E01, E03, E08 | Structural partitioning does not collapse F1, but it never reconstructs cross-partition edges. F1 changes with $P$ come from within-partition recall, capped at $1 - \delta$. |
| **RQ2** | What are the trade-offs of the streaming partitioner (balance penalty $\mu$, buffer size, ordering, repartitioning)? | E02, E04, E05, E06, E07 | $\mu = 1$ and buffer 1000 are the chosen operating point. $\mu = 0$ degenerates to one partition. Repartitioning grows with $P$, $\mu$ and $RF$. |
| **RQ3** | Are partitioned embeddings useful for link prediction? | E09 | Only with a robust decoder (random forest). Neural decoders lose 39–60% MRR from $P = 1$ to 8, because partition spaces are unaligned. |
| **RQ4** | Can independently trained partition spaces be aligned? | E10 | Yes, with a triplet-trained residual aligner (6–14× chance on cross-partition ranking). It costs MRR at $P = 1$ on heterogeneous graphs. |
| **RQ5** | Which pipeline components cause the quality? | E11 | Exploratory: the structural partitioner is essential, and dropping cut edges is not equivalent to it. The seeded 19-variant grid is running. |
| **RQ6** | How does the partitioner compare with LDG and FENNEL? | planned | — |
| **RQ7** | Do the findings transfer to other embedders (e.g. GraphSAGE)? | planned | — |

## Experiments

| ID | Experiment | RQ | Pipeline | Status | Last update | One-line result |
|---|---|---|---|---|---|---|
| [E00](E00-legacy-distributed/README.md) | Static distributed node2vec with LFM / LPA / DistGER (Seminars 1–2) | 1 | S | ⚪ | 2025-11 | LPA beats modified LFM in F1 and edge cut |
| [E01](E01-static-node2vec/README.md) | Static node2vec baseline and sweep | 1 | S | 🟡 | 2026-09 | Baseline F1 20–70% depending on the dataset; sweep not collected |
| [E02](E02-hyperparameters/README.md) | Hyperparameter tuning | 2 | T | ✅ | 2026-09 | Buffer 1000, $\mu = 1$, $\alpha = 1$, $\epsilon = 0.1$ |
| [E03](E03-partition-scaling/README.md) | Partition-count scaling | 1 | T | ✅ 🔁 | 2026-11 | No uniform degradation; dataset-dependent trend |
| [E04](E04-balance-penalty/README.md) | Balance penalty $\mu$ | 2 | T | ✅ 🔁 | 2026-09 | $\mu = 1$: $B \le 1.3$ at $P = 2$ on 4 of 5 datasets |
| [E05](E05-repartitioning/README.md) | Repartitioning volume | 2 | T | ✅ 🔁 | 2026-09 | Grows with $P$, $\mu$ and $RF$ |
| [E06](E06-edge-ordering/README.md) | Edge ordering | 2 | T | 🟡 | 2026-10 | No ordering dominates; superseded by E11 variants |
| [E07](E07-buffer-size/README.md) | Buffer size | 2 | T | 🟡 | 2026-09 | Buffer 100 fails (≈1% F1); 1000 is the trade-off |
| [E08](E08-f1-decomposition/README.md) | F1 decomposition under partitioning | 1 | S | ✅ | 2026-10 | Cross edges never reconstructed; density hypothesis not confirmed |
| [E09](E09-link-prediction/README.md) | Link prediction (480 runs) | 3 | S | ✅ | 2026-10 | Random forest > bilinear > hadamard; LPA > LFM |
| [E10](E10-partition-alignment/README.md) | Partition alignment | 4 | S | 🔵 | 2026-11 | Triplet aligner works; contrastive collapses |
| [E11](E11-ablation/README.md) | Ablation study (19 variants) | 5 | T | 🔵 | 2026-11 | Partitioner essential (exploratory) |

## Adding an experiment

1. Pick the next free ID, `E12`, and create `results/E12-<slug>/README.md`.
2. Start it with the status header, copied from any experiment page, then
   the sections **Question**, **How it was run** (exact command, commit,
   seeds), **Results**, **Conclusion**.
3. Put figures in `results/E12-<slug>/figures/` and raw score tables
   (CSV) next to the README.
4. Add a row above, link it from the matching RQ, and add an entry to
   [JOURNAL.md](../JOURNAL.md).
