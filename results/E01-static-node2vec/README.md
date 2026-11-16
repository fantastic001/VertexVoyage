# E01 — Static node2vec baseline and sweep

| | |
|---|---|
| **Research question** | [RQ1](../README.md#research-questions) — baseline for every partitioned result |
| **Status** | 🟡 partial: baseline table recorded; the $P$ / $RF$ / epoch sweep is not collected |
| **Period** | 2026-05 → 2026-09 |
| **Pipeline** | S, static (`vv test`), see [methods](../../doc/methods.md#pipeline-s--static-vv-test) |
| **Datasets** | CITESEER, Cit-HepPh, Cit-HepTh, AS-Oregon, AstroPh, DBLP, Enron |
| **Headline** | Full-graph node2vec reconstruction F1: AstroPh 70.4%, DBLP 60.3%, CITESEER 34.8%, AS-Oregon 31.3%, Enron 20.4% |
| **Open issues** | The settings (epochs, number of runs, seeds) behind the baseline table are not recorded. Other $P = 1$ numbers for CITESEER differ: 38.6% ([E08](../E08-f1-decomposition/README.md#results), 10 runs), 47.6% ([E03](../E03-partition-scaling/README.md), streaming), 58% ([E02](../E02-hyperparameters/README.md), single run). Re-run with the seed protocol. |

**Question.** For static (non-streaming) [node2vec](../../doc/glossary.md#node2vec),
what precision / recall / F1 does graph reconstruction reach as a function of
dataset, epoch count and partition number — i.e. the static counterpart of
[partition-count scaling](../E03-partition-scaling/README.md), used to pick the
per-dataset epoch budget and replication factor.

# Full-graph node2vec baseline

| Dataset  | Average F1 score |
|----------|------------------|
| CITESEER | 34.82%           |
| DBLP     | 60.3%            |
| AstroPh  | 70.41%           |
| AS-Oregon| 31.26%           |
| Enron | 20.42%           |


Recorded in the project README before 2026-11. The number of runs,
the epochs and the seeds were not recorded (see Open issues above).

# How it was run

```bash
vv test --name <DATASET> --partitions <P> --long-run --epochs <E> --use-dataset-params
```

Sweep: datasets CITESEER / Cit-HepPh / Cit-HepTh / AS-Oregon / AstroPh,
epochs $\in \{1, 5, 10, 50, 100\}$, $P \in \{1, 2, 4, 8, 16\}$, and for each
$P$ the replication factor $RF \in \{1, 2, 4, 8, 16\}$ up to $P$.

# Results

Data collection in progress — table below to be filled from the run outputs.

| Dataset | Epochs | Partition number | P | R | F1 |
| --- | --- | --- | --- | --- | --- |
| CITESEER | 1 | 1 | | | | |
| CITESEER | 1 | 2 | | | |
| CITESEER | 1 | 4 | | | |
| CITESEER | 1 | 8 | | | |
| CITESEER | 1 | 16 | | | |
| CITESEER | 5 | 1 | | | |
| CITESEER | 5 | 2 | | | |
| CITESEER | 5 | 4 | | | |
| CITESEER | 5 | 8 | | | |
| CITESEER | 5 | 16 | | | |
| CITESEER | 10 | 1 | | | |
| CITESEER | 10 | 2 | | | |
| CITESEER | 10 | 4 | | | |
| CITESEER | 10 | 8 | | | |
| CITESEER | 10 | 16 | | | |
| CITESEER | 50 | 1 | | | |
| CITESEER | 50 | 2 | | | |
| CITESEER | 50 | 4 | | | |
| CITESEER | 50 | 8 | | | |
| CITESEER | 50 | 16 | | | |
| CITESEER | 100 | 1 | | | |
| CITESEER | 100 | 2 | | | |
| CITESEER | 100 | 4 | | | |
| CITESEER | 100 | 8 | | | |
| CITESEER | 100 | 16 | | | |
| Cit-HepPh | 1 | 1 | | | | |
| Cit-HepPh | 1 | 2 | | | |
| Cit-HepPh | 1 | 4 | | | |
| Cit-HepPh | 1 | 8 | | | |
| Cit-HepPh | 1 | 16 | | | |
| Cit-HepPh | 5 | 1 | | | |
| Cit-HepPh | 5 | 2 | | | |
| Cit-HepPh | 5 | 4 | | | |
| Cit-HepPh | 5 | 8 | | | |
| Cit-HepPh | 5 | 16 | | | |
| Cit-HepPh | 10 | 1 | | | |
| Cit-HepPh | 10 | 2 | | | |
| Cit-HepPh | 10 | 4 | | | |
| Cit-HepPh | 10 | 8 | | | |
| Cit-HepPh | 10 | 16 | | | |
| Cit-HepPh | 50 | 1 | | | |
| Cit-HepPh | 50 | 2 | | | |
| Cit-HepPh | 50 | 4 | | | |
| Cit-HepPh | 50 | 8 | | | |
| Cit-HepPh | 50 | 16 | | | |
| Cit-HepPh | 100 | 1 | | | |
| Cit-HepPh | 100 | 2 | | | |
| Cit-HepPh | 100 | 4 | | | |
| Cit-HepPh | 100 | 8 | | | |
| Cit-HepPh | 100 | 16 | | | |
| Cit-HepTh | 1 | 1 | | | | |
| Cit-HepTh | 1 | 2 | | | |
| Cit-HepTh | 1 | 4 | | | |
| Cit-HepTh | 1 | 8 | | | |
| Cit-HepTh | 1 | 16 | | | |
| Cit-HepTh | 5 | 1 | | | |
| Cit-HepTh | 5 | 2 | | | |
| Cit-HepTh | 5 | 4 | | | |
| Cit-HepTh | 5 | 8 | | | |
| Cit-HepTh | 5 | 16 | | | |
| Cit-HepTh | 10 | 1 | | | |
| Cit-HepTh | 10 | 2 | | | |
| Cit-HepTh | 10 | 4 | | | |
| Cit-HepTh | 10 | 8 | | | |
| Cit-HepTh | 10 | 16 | | | |
| Cit-HepTh | 50 | 1 | | | |
| Cit-HepTh | 50 | 2 | | | |
| Cit-HepTh | 50 | 4 | | | |
| Cit-HepTh | 50 | 8 | | | |
| Cit-HepTh | 50 | 16 | | | |
| Cit-HepTh | 100 | 1 | | | |
| Cit-HepTh | 100 | 2 | | | |
| Cit-HepTh | 100 | 4 | | | |
| Cit-HepTh | 100 | 8 | | | |
| Cit-HepTh | 100 | 16 | | | |
| AS - Oregon | 1 | 1 | | | | |
| AS - Oregon | 1 | 2 | | | |
| AS - Oregon | 1 | 4 | | | |
| AS - Oregon | 1 | 8 | | | |
| AS - Oregon | 1 | 16 | | | | |
| AS - Oregon | 5 | 1 | | | |
| AS - Oregon | 5 | 2 | | | |
| AS - Oregon | 5 | 4 | | | |
| AS - Oregon | 5 | 8 | | | |
| AS - Oregon | 5 | 16 | | | |
| AS - Oregon | 10 | 1 | | | |
| AS - Oregon | 10 | 2 | | | |
| AS - Oregon | 10 | 4 | | | |
| AS - Oregon | 10 | 8 | | | |
| AS - Oregon | 10 | 16 | | | |
| AS - Oregon | 50 | 1 | | | |
| AS - Oregon | 50 | 2 | | | |
| AS - Oregon | 50 | 4 | | | |
| AS - Oregon | 50 | 8 | | | |
| AS - Oregon | 50 | 16 | | | |
| AS - Oregon | 100 | 1 | | | |
| AS - Oregon | 100 | 2 | | | |
| AS - Oregon | 100 | 4 | | | |
| AS - Oregon | 100 | 8 | | | |
| AS - Oregon | 100 | 16 | | | |
| AstroPh | 1 | 1 | | | | |
| AstroPh | 1 | 2 | | | |
| AstroPh | 1 | 4 | | | |
| AstroPh | 1 | 8 | | | |
| AstroPh | 1 | 16 | | | |
| AstroPh | 5 | 1 | | | |
| AstroPh | 5 | 2 | | | |
| AstroPh | 5 | 4 | | | |
| AstroPh | 5 | 8 | | | |
| AstroPh | 5 | 16 | | | |
| AstroPh | 10 | 1 | | | |
| AstroPh | 10 | 2 | | | |
| AstroPh | 10 | 4 | | | |
| AstroPh | 10 | 8 | | | |
| AstroPh | 10 | 16 | | | |
| AstroPh | 50 | 1 | | | |
| AstroPh | 50 | 2 | | | |
| AstroPh | 50 | 4 | | | |
| AstroPh | 50 | 8 | | | |
| AstroPh | 50 | 16 | | | |
| AstroPh | 100 | 1 | | | |
| AstroPh | 100 | 2 | | | |
| AstroPh | 100 | 4 | | | |
| AstroPh | 100 | 8 | | | |
| AstroPh | 100 | 16 | | | |