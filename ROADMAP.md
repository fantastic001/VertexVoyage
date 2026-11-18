# Roadmap

The research continues as PhD work; the master's thesis is
[`doc/Rad.odt`](doc/Rad.odt). The goal is a series of peer-reviewed journal
papers that form the basis of the PhD dissertation. The core of the work is
community-aware streaming partitioning for distributed graph embedding,
where every partition is embedded independently. The plan has four parts:

1. Fix the hypotheses and the analysis plan, and build the instrument,
   including the downstream evaluation (link prediction and node
   classification), before any confirmatory run.
2. Bring in the related work and its baselines (LDG, FENNEL, HDRF, METIS,
   Local2Global); see [doc/references.md](doc/references.md).
3. Re-run and extend the experiments under that plan: seeded core results,
   partitioner comparison, ablation, downstream tasks, scalability, other
   embedders.
4. Analyse the results and write the papers.

No defence date is fixed yet, so milestones are ordered, not dated.

**Ordering rule.** Finish these before the confirmatory runs of M4–M8:
- M0;
- the instrument fixes M1.1–M1.4;
- the downstream evaluation M7.0–M7.2, M7.4 and M7.5, because M0.2 makes
  temporal link-prediction MRR a primary metric.

A run started earlier, or a hypothesis added after the data are in, is
reported as exploratory. The E11 runs may continue while M0 is written, but
the E11 analysis plan (M0.7) has to be committed before `collect` fills the
tables.

Status legend: ⬜ not started · 🔵 in progress · ✅ done

## Publication plan

Proposed series of three papers for the PhD dissertation, to be confirmed
([open decisions](#open-decisions)).

| Paper | Contribution | RQs | Candidate venues |
|---|---|---|---|
| 1 | (a) Evaluation pitfall: with independently embedded partitions, recall is capped at $1 - \delta$, and F1 rising with $P$ is a chance-floor effect, proved in Lean ([E08](results/E08-f1-decomposition/README.md)). (b) Experimental study of streaming partitioners when partitions are embedded without synchronisation, with the ablation. | RQ1, RQ2, RQ5, RQ6 | TMLR; Data Mining and Knowledge Discovery or Machine Learning (also via the ECML PKDD journal track); Applied Network Science. VLDB Journal, IEEE TPDS or JPDC only with a real distributed run at scale. |
| 2 | Triplet-trained residual aligner for non-overlapping partitions ($RF = 1$, uses cut edges instead of anchors), with the contrastive-collapse theory ([E10](results/E10-partition-alignment/README.md)) | RQ3 (static link prediction), RQ4 | Machine Learning; Knowledge and Information Systems |
| 3 | Downstream evaluation of partitioned streaming embeddings: prequential link prediction and node classification on real temporal graphs, across embedders (DeepWalk, ProNE or NetMF, GraphSAGE) | RQ3 (downstream tasks), RQ7 | TMLR; Data Mining and Knowledge Discovery; Applied Network Science |

## Open decisions

| Decision | Options | Affects |
|---|---|---|
| Venue tier | depends on any required journal category | M5, M7, M12 |
| Systems claim | a real distributed run on the ZooKeeper cluster ([guide](doc/guide/cluster.md)), or a partition-parallel framing with simulated makespan and memory | M1.6, M7.3 |
| Compute budget | SLURM CPU-hours available; the E11 grid alone is about 19 variants × 14 cells × 10 seeds ≈ 2,700 runs, dominated by the ≈2 h AstroPh runs | M0.6, M4–M8 |

# Part I — Before the confirmatory runs

## M0 — Analysis plan

| | ID | Task | Notes |
|---|---|---|---|
| ⬜ | M0.1 | Write `doc/analysis-plan.md` with numbered hypotheses H1…Hn. Give each a statistic, a threshold, a test and its hypothesis family, and commit it before the runs (the commit timestamp is the pre-registration; optionally also on OSF). | Example H1: at $P = 4$, `neighbors.all` has a higher lift over the chance floor than `random` on at least 4 of 5 datasets (Wilcoxon signed-rank over datasets, Holm, $\alpha = 0.05$). |
| ⬜ | M0.2 | Choose the primary metrics: lift over the chance floor ($Q / \rho_{\text{within}}$) with within/cross scoped F1, and temporal link-prediction MRR (M7.1). Raw F1 becomes secondary. | Raw F1 is confounded by the edge cut ([E08](results/E08-f1-decomposition/README.md)). |
| ⬜ | M0.3 | Define "preserves quality" as a non-inferiority margin, e.g. lift at $P$ ≥ 0.95 × lift at $P = 1$, tested with TOST. | A non-significant difference is not evidence of preservation. |
| ⬜ | M0.4 | Fix one canonical full pipeline for all RQs. | E03 used the default aggregator; the E11 baseline uses `--ordering bfs` and `triplet`. Depends on M1.1. |
| ⬜ | M0.5 | Fix the datasets, cells, budgets and timeouts before the runs. | Datasets added later are reported as exploratory. |
| ⬜ | M0.6 | Pilot with 3 seeds per cell to estimate $\sigma$; choose $n$ per dataset for a target 95% CI half-width $h$, $n \approx (z\sigma/h)^2$. | CITESEER, $P = 1$, $\sigma \approx 1.6$ pp: $n = 10$ gives ±1.1 pp; ±0.5 pp needs $n \approx 39$. AstroPh ($\sigma \approx 0.1$–$0.5$ pp) is precise enough at $n = 10$. |
| ⬜ | M0.7 | Write the E11 analysis plan: metrics, comparisons against the baseline, correction. | Before `collect` ([E11](results/E11-ablation/README.md)). |
| ⬜ | M0.8 | Fix the downstream protocol parameters (M7): split fractions, number and kind of negatives per positive, tie rule, the rank given to a test edge whose vertex has no embedding, the prequential evaluation interval, node-classification train ratios and split seeds. | The static code holds out 10% of the edges, while [evaluation.md](doc/evaluation.md#3-link-prediction) says 20% (M7.0d). |

## M1 — Instrument

| | ID | Task | Notes |
|---|---|---|---|
| ⬜ | M1.1 | Decide and fix the `--track-seen` semantics: it is ignored once `--ordering` is set, so either drop it from the baseline or make it combine with an ordering. | [cli guide](doc/guide/cli.md#additional-temporal_test-flags) |
| ⬜ | M1.2 | Find the source of non-determinism in aligner training. | [E10](results/E10-partition-alignment/README.md) |
| ⬜ | M1.3 | Report the sample standard deviation (`ddof=1`) and 95% CIs. | `experiments/ablation/collect.py` and `scripts/analyze.py` use the population `np.std`. |
| ⬜ | M1.4 | Store provenance (commit, date, seeds, environment, full configuration) and the per-seed raw values in every `scores.csv`; generate the results tables from them, starting with E11. | |
| ⬜ | M1.5 | Measure test coverage and add tests for edge cut, balance, MRR tie-breaking, the scoped metrics and the new partitioners; target ≥ 80% branch coverage on metrics and partitioners. | |
| ⬜ | M1.6 | Measure cost: per-partition time per buffer (`TimeMetric("model_update_part")`), simulated makespan $\sum_b \max_p t_{b,p}$, peak memory per partition, and replica and repartitioning volume as communication proxies. | Partitions are trained one after another in one process (`PerPartitionTraining.update_models` in [`partition_training.py`](vertex_voyage/partition_training.py)). |
| ⬜ | M1.7 | Profile and speed up the streaming pipeline so that graphs with ≥ 1M edges are feasible. | AstroPh takes ≈2 h per run at $P = 1$ ([E02](results/E02-hyperparameters/README.md)). `_sort_temporal_events` removes events with `list.pop(0)`, which is quadratic in the stream length. |
| ⬜ | M1.8 | Reproducibility package: a LICENSE, a pinned lockfile, an Apptainer container for SLURM, `scripts/download_data.sh` from public sources with checksums, `CITATION.cff`. | 1 of 18 entries in `requirements.txt` is pinned; Python 3.9 is end-of-life. |
| ⬜ | M1.9 | Replace notebook plotting with a script that regenerates every figure in `results/*/figures/`. | |

## M2 — Related work and baselines

| | ID | Task | Notes |
|---|---|---|---|
| ⬜ | M2.1 | Write `doc/related-work.md` as a table: method, setting, synchronisation, partitioner, datasets, metric, limitation. | Coverage: [doc/references.md](doc/references.md). |
| ⬜ | M2.2 | Write a one-sentence gap statement and check it against M2.1. | Kochsiek and Gemulla found random partitioning competitive *with* parameter synchronisation, while E03 finds that it collapses F1 *without*. The gap statement has to explain this. |
| ⬜ | M2.3 | Implement LDG (Stanton and Kliot, KDD 2012) and FENNEL (Tsourakakis et al., WSDM 2014) behind the existing partitioner interface. | [methods](doc/methods.md#streaming-neighbour-partitioner) |
| ⬜ | M2.4 | Implement HDRF (Petroni et al., CIKM 2015), a vertex-cut partitioner and the natural competitor at $RF > 1$. | |
| ⬜ | M2.5 | Offline METIS (via pymetis) as the edge-cut upper bound. | |
| ⬜ | M2.6 | Validate every reimplementation against a published edge-cut number on the published graph. | |
| ⬜ | M2.7 | Ceiling and floor baselines: full-graph node2vec tuned at the same epoch budget, with recorded settings; random embeddings; the chance floor of E08. One $P = 1$ reference per pipeline. | [E01](results/E01-static-node2vec/README.md) |
| ⬜ | M2.8 | Give every partitioner and embedder the same tuning budget, and tune on datasets that are not reported. | E02 tuned on CITESEER, which is in the headline table. |
| ⬜ | M2.9 | Reproduce a published node2vec / dynnode2vec number with the published protocol. | [reference protocol](doc/dynnode2vec-reference-protocol.md) |

The works to cover in M2.1 are listed by topic, with BibTeX keys, in
[doc/references.md](doc/references.md). The closest prior work is in its
[alignment section](doc/references.md#independent-partition-or-patch-embedding-with-alignment):
Local2Global, DDHH and DeHIN.

## M3 — Data

| | ID | Task | Notes |
|---|---|---|---|
| ⬜ | M3.1 | Add datasets with real timestamps: SNAP sx-mathoverflow, sx-superuser, wiki-talk-temporal and the registered StackOverflow, or TGB datasets; check sizes and licences. | No dataset used so far has a real time order. Already on disk with Unix timestamps: `data/reddit.txt` (286,561 events, Reddit hyperlinks) and `data/stackoverflow.txt`. |
| ⬜ | M3.2 | Add a large labelled graph, e.g. ogbn-arxiv (≈169k vertices, 1.2M edges, publication years, labels). | Largest streaming graph so far: Enron, 36,692 vertices. |
| ⬜ | M3.3 | Synthetic SBM and LFR sweeps over the mixing parameter, with Erdős–Rényi as the null graph. | Answers *when* partitioning preserves quality. |
| ⬜ | M3.4 | Confirm the DBLP source. | [datasets](doc/datasets.md) |
| ⬜ | M3.5 | Record source, licence and preprocessing of every dataset in [datasets](doc/datasets.md). | |
| ⬜ | M3.6 | Labelled datasets for node classification: DBLP (labels and attributes already in `data/dblp.npz`); CITESEER rebuilt from Planetoid ([open decision](#open-decisions)); ogbn-arxiv (M3.2); Coauthor CS / Physics and Amazon Computers / Photo (Shchur et al. 2018). | DBLP has 4 classes (7,920 / 5,645 / 2,169 / 1,982) and 1,639 attributes. The `label` array of `AstroPh.npz` holds 18,772 distinct values, i.e. node ids, not classes. The current CITESEER file has no labels and renumbered ids. |

# Part II — Experiments

## M4 — Reproducible core results (RQ1, RQ2)

Re-run every pre-protocol result with the seed protocol of
[doc/guide/seeding.md](doc/guide/seeding.md): seeds from 42, $n$ per cell
from M0.6 (at least 10), $\bar F \pm s$ with a 95% CI.

| | ID | Task | Replaces |
|---|---|---|---|
| ⬜ | M4.1 | Partition-count scaling, $P \in \{1,2,4,8\}$, 5 datasets, F1 and partition metrics | [E03](results/E03-partition-scaling/README.md) tables |
| ⬜ | M4.2 | $\mu$ sweep with `--algorithm dummy` (balance and edge cut do not depend on the embedding), plus F1 at $\mu \in \{0.5, 1, 1.5\}$ | [E04](results/E04-balance-penalty/README.md) |
| ⬜ | M4.3 | Repartitioning volume with `--algorithm dummy`, $RF \in \{1,2,4\}$ on all datasets | [E05](results/E05-repartitioning/README.md) |
| ⬜ | M4.4 | Buffer-size sweep at $P \in \{1, 4\}$ | [E07](results/E07-buffer-size/README.md) |
| ⬜ | M4.5 | Resolve the random-partitioning conflict (E03 ≈0.03% vs E11 exploratory 8–50%) through the seeded `random_partitioning` variant | E03, E11 |

## M5 — Partitioner comparison (RQ6)

| | ID | Task |
|---|---|---|
| ⬜ | M5.1 | Grid: partitioner (`neighbors.all`, LDG, FENNEL, HDRF, random, METIS) × dataset × $P$ × seed. Metrics: $\delta$, $B$, repartitioning, makespan, memory, lift, temporal-LP MRR, node-classification F1. |
| ⬜ | M5.2 | Performance profiles (Dolan–Moré) of the partitioners over all instances. |
| ⬜ | M5.3 | Explain the random-partitioning result against Kochsiek and Gemulla: with vs without synchronisation. |

## M6 — Ablation study (RQ5)

| | ID | Task |
|---|---|---|
| 🔵 | M6.1 | Run all 19 variants of [E11](results/E11-ablation/README.md) on SLURM and fill the tables with `collect` (after M0.7 and M1.1). |
| ⬜ | M6.2 | Interaction rows for components suspected to substitute for each other, e.g. `random_partitioning` with `mean_aggregator`. |
| ⬜ | M6.3 | "Baseline version" rows: LDG / FENNEL as the partitioner, Procrustes / Local2Global as the aggregator. |
| ⬜ | M6.4 | Compute-matched row: epochs divided by $P$. |
| ⬜ | M6.5 | Report lift over the chance floor and within/cross scoped F1 next to raw F1. |
| ⬜ | M6.6 | $RF = 3$ cells on all four datasets. |
| ⬜ | M6.7 | Summary table: variant × dataset, Δ vs baseline with CI, Holm-corrected. |

## M7 — Downstream tasks and scalability (RQ3)

Current state:
- Link prediction exists only in the static pipeline S (`vv test`,
  [E09](results/E09-link-prediction/README.md)).
- `vv temporal_test` has no downstream evaluation.
- There is no node-classification code.

M7.0–M7.2, M7.4 and M7.5 build the evaluation instrument and come before
the confirmatory runs (ordering rule). The runs themselves are part of
M4–M6 and M8. The protocol parameters are fixed in M0.8, the metric
theorems are M9.9–M9.10, and the references are in
[doc/references.md](doc/references.md#link-prediction-and-node-classification-protocols).

### M7.0 — Repair the static link prediction (pipeline S)

| | ID | Task | Notes |
|---|---|---|---|
| ⬜ | M7.0a | Rename `heart_benchmark`, which ranks against uniformly random tail corruptions, not HeaRT's heuristic hard negatives. Correct the "HeaRT protocol" claim in [evaluation.md](doc/evaluation.md#3-link-prediction), E09 and E10. | Li et al. 2023 |
| ⬜ | M7.0b | Tie-averaged ranks (a `TieBreaking` enum, default `AVERAGE`, the OGB convention); persist raw scores, not only ranks. | `Ranks.add_scores` counts ties against the positive. A 20-tree random forest outputs probabilities in steps of 0.05, so ties are frequent; E09 found its MRR "depends on tie handling". |
| ⬜ | M7.0c | Filtered, seeded negatives: exclude every known edge (train, validation, test) and self-pairs; sample from a seeded `np.random.Generator`. | Ranking negatives can be self-pairs; training negatives can be held-out test edges. |
| ⬜ | M7.0d | Edge splitter with a validation split that keeps every vertex at training degree ≥ 1; settle the test fraction (M0.8). | `LINK_PREDICTION_TEST_FRACTION` is 0.1; [evaluation.md](doc/evaluation.md#3-link-prediction) says 20%. |
| ⬜ | M7.0e | Batch scoring for all decoders. | `predict_links` calls `predict_proba` once per pair. |
| ⬜ | M7.0f | A second negative set: HeaRT-style hard negatives, following the HeaRT repository. | |
| ⬜ | M7.0g | Threshold-free metrics AUC and AP, for comparison with the node2vec paper. | |
| ⬜ | M7.0h | Report chance levels and lift for MRR and Hits@$k$ (M9.9). | With 500 negatives: chance MRR $= H_{501}/501 \approx 0.0136$, chance Hits@10 $= 10/501 \approx 0.020$. |

### M7.1 — Streaming link prediction (pipeline T)

| | ID | Task | Notes |
|---|---|---|---|
| ⬜ | M7.1a | A `DownstreamTask` plug-in interface with lifecycle hooks (`on_stream_start`, `before_buffer`, `after_buffer`, `report`), a registry, and a `--downstream-tasks` flag on `vv test` and `vv temporal_test`. Constants come from `VERTEX_VOYAGE_LP_*` and `VERTEX_VOYAGE_NC_*`. | Replaces the ad-hoc `EM` wrapper in `test_commands.py`. |
| ⬜ | M7.1b | Prequential (test-then-train) evaluation: score the edges of buffer $b+1$ with the embeddings after buffer $b$, every $k$ buffers. | Gama et al. 2013 |
| ⬜ | M7.1c | Negatives per source node: half historical (seen earlier, absent now) and half random, filtered. | Poursafaei et al. 2022; TGB |
| ⬜ | M7.1d | Scopes `WITHIN`, `CROSS`, `NEW_VERTEX` and `REPEATED_EDGE`, with the coverage of vertices that have no embedding. | node2vec cannot embed a vertex it has not seen. |
| ⬜ | M7.1e | Baselines: EdgeBank (memorisation), the static full-graph ceiling, and the streaming pipeline at $P = 1$. | EdgeBank exposes easy negatives. |
| ⬜ | M7.1f | A chronological 70/15/15 split as a second protocol, for comparison with TGB. | |
| ⬜ | M7.1g | A `timestamp` ordering in `EDGE_ORDERINGS`, and the real temporal streams of M3.1. | Ordering `""` is file order. |
| ⬜ | M7.1h | A decoder retraining policy (an enum: every $k$ buffers, or once at the end), with its cost measured. | |

### M7.2 — Node classification

| | ID | Task | Notes |
|---|---|---|---|
| ⬜ | M7.2a | A `dataset_labels` registry next to `datasets` in `experiments/datasets.py`, filled from M3.6. | |
| ⬜ | M7.2b | Protocol: frozen embeddings; one-vs-rest logistic regression with the regularisation strength chosen by inner cross-validation; stratified train ratios {0.1, 0.5, 0.9} × split seeds; official splits where they exist (Planetoid, ogbn-arxiv); micro-F1, macro-F1 and accuracy. | DeepWalk and node2vec protocol; Shchur et al. 2018 on split variance |
| ⬜ | M7.2c | Baselines: majority class, label spreading on the graph, features only (DBLP attributes), full-graph node2vec. | DBLP majority class: accuracy 44.7%, macro-F1 ≈ 15.4% (M9.10). |
| ⬜ | M7.2d | Scopes: boundary vertices (with a cut edge) vs interior vertices; replicated vs single-copy vertices. | |
| ⬜ | M7.2e | Streaming: evaluate at checkpoints and at the end of the stream; vertices not yet seen count as coverage loss. | |
| ⬜ | M7.2f | Reproduce DeepWalk on CiteSeer with the Planetoid split. | 43.2% (Kipf and Welling 2017, Table 2) |

### M7.3–M7.5

| | ID | Task |
|---|---|---|
| ⬜ | M7.3 | Scalability: runtime against $\lvert E \rvert$ on a log-log plot against the predicted slope (M9.6), and makespan and memory against $P$. |
| ⬜ | M7.4 | Integration: task reports go into `PersistedRun` and the per-seed `scores.csv`, which `collect` reads. Add an experiment page E12 (downstream evaluation). Rewrite §3 and add a node-classification section in [evaluation.md](doc/evaluation.md), with glossary entries. Extend RQ3 in the README and the [registry](results/README.md) from link prediction to downstream tasks. |
| ⬜ | M7.5 | Tests, ≥ 80% branch coverage: tie modes; chance levels by simulation against M9.9; no leakage in the splitters; negatives contain no known edge and no self-pair; test-then-train order; node-classification metrics against scikit-learn; EdgeBank on a toy stream. |

## M8 — Other embedding methods (RQ7)

| | ID | Task |
|---|---|---|
| ⬜ | M8.1 | DeepWalk ($p = q = 1$, already the `unbiased_walk` variant) and DistGER, for continuity with [E00](results/E00-legacy-distributed/README.md). |
| ⬜ | M8.2 | One embedder that is not walk-based and needs no features, e.g. ProNE or NetMF. |
| ⬜ | M8.3 | GraphSAGE (Hamilton, Ying, Leskovec, NeurIPS 2017) per partition, with the same aggregators, only on graphs with node features (CITESEER rebuilt from Planetoid, DBLP, ogbn-arxiv); AstroPh, Enron and AS-Oregon have none. |
| ⬜ | M8.4 | Check whether the E08 decomposition and the E10 alignment results transfer. |

## M9 — Theory

| | ID | Task |
|---|---|---|
| ⬜ | M9.1 | Fixed-$k$ partitioner and SBM sweep for the E08 density question ([E08 additional experiments](results/E08-f1-decomposition/README.md#additional-experiments)). |
| ⬜ | M9.2 | Report lift over the chance floor ($Q / \rho_{\text{within}}$) alongside raw F1 everywhere. |
| ⬜ | M9.3 | Prove $\mathbb{E}[\delta] = (P-1)/P$ for uniformly random assignment in Lean, and any property of lift that M0.2 relies on (e.g. lift is 1 at chance). |
| ⬜ | M9.4 | Mark every formal statement as proved, cited or estimated; the E08 decomposition of the global F1 holds only approximately. |
| ⬜ | M9.5 | Formal problem statement of the stream model: what the partitioner sees and how much state it keeps. |
| ⬜ | M9.6 | Time and memory complexity of `neighbors.all` per event, the predicted slope for M7.3. |
| ⬜ | M9.7 | Check that Theorems 1–18 are stated for the losses as implemented (sampled negatives, normalisation). |
| ⬜ | M9.8 | Cite the NP-hardness of balanced $k$-partitioning (Andreev and Räcke, 2006). |
| ⬜ | M9.9 | Lean `RankingMetrics.lean`, for M7.0h and M7.1. (R1) If the rank $R$ of a positive among $K$ negatives is uniform on $\{1, \dots, K+1\}$, then $\mathbb{E}[1/R] = H_{K+1}/(K+1)$ and $\Pr[R \le k] = k/(K+1)$. (R2) With $g$ negatives scored above the positive and $t$ tied with it, the mean position under uniformly random tie-breaking is $1 + g + t/2$, the tie-averaged rank. (R3) By Jensen's inequality, $1/(1 + g + t/2) \le \mathbb{E}[1/R]$, so tie-averaged MRR is a lower bound on the expected MRR under random tie-breaking. Example: scores 0.6 for the positive and (0.9, 0.6, 0.6, 0.1) for the negatives give $g = 1$, $t = 2$, and ranks 4 (pessimistic, the current code), 2 (optimistic) and 3 (average). |
| ⬜ | M9.10 | Lean, for M7.2. (N1) For single-label multiclass classification, micro-F1 equals accuracy. (N2) The majority-class classifier has accuracy $p$ and macro-F1 $2p/((1+p)C)$, where $p$ is the majority share and $C$ the number of classes, taking F1 = 0 for classes never predicted. Example: DBLP, $p = 7920/17716 \approx 0.447$, $C = 4$: accuracy 44.7%, macro-F1 ≈ 15.4%. |

# Part III — Alignment (paper 2)

## M10 — Alignment (RQ3, RQ4)

| | ID | Task |
|---|---|---|
| ⬜ | M10.1 | Seeded repetition ($n$ from M0.6) of the recommended configuration and both controls (mean, 0 steps) on AstroPh and Enron at $P = 4$. |
| ⬜ | M10.2 | Fix and re-run the Procrustes start; try Procrustes as initialisation of the residual aligner. |
| ⬜ | M10.3 | Reduce the aligner cost at $P = 1$ (geometry-preserving term, step sweep 0–100). |
| ⬜ | M10.4 | Extend to $RF > 1$, $P = 8$, Enron and AS-Oregon with all links. |
| ⬜ | M10.5 | Baselines: Local2Global at $RF > 1$, where the replicas provide its overlapping patches, and DeHIN-style Procrustes on anchors. |
| ⬜ | M10.6 | Bring the aligner into the streaming pipeline and the ablation (`untrained_triplet_aggregator`, `mean_aggregator` variants). |

Details: [E10 future work](results/E10-partition-alignment/README.md#future-work).
The non-determinism of aligner training is M1.2.

# Part IV — Analysis and publication

## M11 — Statistical analysis

| | ID | Task |
|---|---|---|
| ⬜ | M11.1 | Run the tests fixed in M0.1: Friedman with Holm-corrected post-hoc tests and Wilcoxon signed-rank over datasets, TOST for the non-inferiority claims, bootstrap CIs, effect sizes, assumption checks. |
| ⬜ | M11.2 | Report the size of every hypothesis family and the resampled unit; flag deviations from the plan as exploratory. |
| ⬜ | M11.3 | Report failures and timeouts per method separately. |

## M12 — Write-up and submission

| | ID | Task |
|---|---|---|
| ⬜ | M12.1 | Paper 1 draft: formal problem statement, one-figure pipeline, notation table, related-work comparison, limitations. |
| ⬜ | M12.2 | Paper 2 draft (after M10). |
| ⬜ | M12.3 | Paper 3 draft (after M7 and M8). |
| ⬜ | M12.4 | Master's thesis ([`doc/Rad.odt`](doc/Rad.odt)): record its status and dates in the README timeline. |
| ⬜ | M12.5 | PhD dissertation built on Papers 1–3. |
| ⬜ | M12.6 | Adversarial self-review of each paper against the review criteria (understandable, scalable, generalisable, reproducible, honestly evaluated, statistically sound), then supervisor review. |
| ⬜ | M12.7 | Clean-room reproduction of the headline numbers by someone else, from the README alone. |
| ⬜ | M12.8 | For each paper: archive code and scores with a Zenodo DOI, post an arXiv preprint, submit. |

## Repository tooling

| | Task |
|---|---|
| ⬜ | Deduplicate the presentation assets in `doc/Seminar1`, `doc/Seminar2`, `doc/presentation` and `doc/pp-presentation` |
| ⬜ | Present the project in the README as PhD research that continues the master's thesis: supervisor, timeline dates (master's thesis, start of the PhD), and an `@phdthesis` entry next to `@mastersthesis`. Needs these details from the author. |
