# Progress journal

Dated log of the project, newest first. Each entry says what was done, what
was learned and where the evidence is. Entries up to 2026-11-16 were
reconstructed from the git history of this repository and of the author's
working notes, which were migrated here on that date.

Add a new entry at the top for every result, decision or change of
direction.

## 2026-11

- **2026-11-17** — Gap analysis against a standard research workflow
  (hypotheses, baselines, instrument, design, sample size, analysis,
  ablation, reproducibility).
  - The [roadmap](ROADMAP.md) is reorganised into M0–M12. A
    pre-registered analysis plan (M0) now comes before any confirmatory
    run.
  - Main gaps found:
    - no hypotheses with statistics and thresholds;
    - the closest prior work (Local2Global, DeHIN) is not cited;
    - no datasets with real timestamps;
    - no graph above 37k vertices;
    - no runtime or memory measurements.
  - A 3-paper publication plan is proposed.
- **2026-11-16** — The repository becomes the single source of truth.
  - All working notes moved into [`results/`](results/README.md) (E00–E11),
    [`doc/`](doc/README.md) and [`proofs/`](proofs/README.md).
  - Dataset statistics re-measured from the data files: the CITESEER and
    AS-Oregon counts were wrong, and DBLP is a citation graph
    ([datasets](doc/datasets.md)).
  - Balance range $1 \le B \le 3 - 2/K$ proved in Lean.
  - Found that `--track-seen` is ignored when `--ordering` is set (E06,
    E11).
- **2026-11-16** — Ablation runner with SLURM support and 19 variants
  ([E11](results/E11-ablation/README.md),
  [runner](experiments/ablation/README.md)).
- **2026-11-10** — Alignment: the `unit` output scale wins all 12 matched
  comparisons; the Procrustes start is broken
  ([log](results/E10-partition-alignment/log/2026-11-10-output-scale-and-start-astroph-enron.md)).
  Code defaults changed to `identity` start and `uniform` weighting.
- **2026-11-08** — Alignment: the start does not matter; training causes
  most of the $P = 1$ cost
  ([log](results/E10-partition-alignment/log/2026-11-08-aligner-start-astroph-p1.md)).
- **2026-11-07** — Alignment: nine analyses of contrastive vs triplet losses
  and link weightings. Triplet wins and contrastive collapses
  (Theorems 11–18) ([E10](results/E10-partition-alignment/README.md#analysis-log)).

## 2026-10

- **2026-10-28** — F1 decomposition finished: the Erdős–Rényi sweep does
  not confirm the density hypothesis
  ([E08](results/E08-f1-decomposition/README.md#verdict)).
- **2026-10-13** — Alignment plan, Theorems 1–10 proved in Lean, and a
  synthetic SBM check
  ([E10](results/E10-partition-alignment/README.md),
  [theory](doc/theory/partition-alignment.md)). Aggregation interface added
  to the code.
- **2026-10-12** — Link-prediction study finished (480 runs): random forest >
  bilinear > hadamard ([E09](results/E09-link-prediction/README.md)).
- **2026-10-05** — Edge-ordering results on CITESEER
  ([E06](results/E06-edge-ordering/README.md)).

## 2026-09

- **2026-09-28** — Started the F1 decomposition: cross-partition edges are
  never reconstructed ([E08](results/E08-f1-decomposition/README.md)).
- **2026-09-07** — Significance testing (Welch, Mann–Whitney, Cohen's $d$)
  added to partition-count scaling
  ([E03](results/E03-partition-scaling/README.md#significance-tests)).
- **2026-09-06** — Random-partitioner baseline
  ([E03](results/E03-partition-scaling/README.md#random-partitioner-baseline)).
- **2026-09-04** — Buffer-size timing plots
  ([E07](results/E07-buffer-size/README.md)).

## 2026-08

- **2026-08-24** — $\mu$ sensitivity analysis
  ([E04](results/E04-balance-penalty/README.md)).
- **2026-08-11** — Temporal evolution plots of F1, balance, edge cut and
  repartitioning ([E03](results/E03-partition-scaling/README.md#temporal-evolution-plots)).
- **2026-08-02** — Full-graph node2vec baseline
  ([E01](results/E01-static-node2vec/README.md)).

## 2026-05 → 2026-07

- **2026-07-14** — First exploratory ablation runs
  ([E11](results/E11-ablation/README.md#results)).
- **2026-06-23** — Repartitioning metrics
  ([E05](results/E05-repartitioning/README.md)).
- **2026-05-26** — Static link-prediction evaluation started
  ([E09](results/E09-link-prediction/README.md)).
- **2026-05-22** — Streaming pipeline results: dynnode2vec with buffered
  events and $P \in \{1,2,4,8\}$
  ([E03](results/E03-partition-scaling/README.md)).
- **2026-05-02** — gensim-based dynnode2vec updates; hyperparameter tuning
  of the streaming pipeline ([E02](results/E02-hyperparameters/README.md)).

## 2025

- **2025-11-23** — Seminar 2 paper: DistGER with label propagation
  ([E00](results/E00-legacy-distributed/README.md)).
- **2025-11-12** — Seminar 1 paper: LFM vs LPA vs random partitioning
  (E00).
- **2025-10** — Comparative analysis by number of partitions, presentation
  ([`doc/pp-presentation`](doc/pp-presentation/presentation.tex)).
- **2025-03** — Event-sequence abstractions for temporal graphs, the basis of
  the streaming pipeline.

## 2023-11 → 2024-12

- **2024-07 → 2024-11** — Distributed system: ZooKeeper coordination,
  XML-RPC API, modified LFM partitioner, partitioned node2vec and
  reconstruction tests ([cluster guide](doc/guide/cluster.md)).
- **2024-01-19** — Thesis proposal finalised
  ([proposal](doc/proposal.tex)).
- **2023-11** — Literature review on distributed node2vec and graph
  embedding.
