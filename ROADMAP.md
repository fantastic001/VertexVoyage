# Roadmap

The goal is a publishable comparison of community-aware streaming
partitioning for distributed graph embedding. The plan has three parts:
finish the ablation study, compare against standard streaming partitioners
(LDG, FENNEL), and check that the findings transfer to other embedding
methods (e.g. GraphSAGE). The defence date is not fixed yet, so milestones
are ordered, not dated.

Status legend: ⬜ not started · 🔵 in progress · ✅ done

## M1 — Reproducible core results

Re-run every pre-protocol result with the seed protocol of
[doc/guide/seeding.md](doc/guide/seeding.md): 10 seeds (42–51) per cell,
$\bar F \pm \sigma$, `n=k/10`.

| | Task | Replaces |
|---|---|---|
| ⬜ | Partition-count scaling, $P \in \{1,2,4,8\}$, 5 datasets, F1 and partition metrics | [E03](results/E03-partition-scaling/README.md) tables |
| ⬜ | $\mu$ sweep with `--algorithm dummy` (balance and edge cut do not depend on the embedding), plus F1 at $\mu \in \{0.5, 1, 1.5\}$ | [E04](results/E04-balance-penalty/README.md) |
| ⬜ | Repartitioning volume with `--algorithm dummy`, $RF \in \{1,2,4\}$ on all datasets | [E05](results/E05-repartitioning/README.md) |
| ⬜ | Buffer-size sweep at $P \in \{1, 4\}$ | [E07](results/E07-buffer-size/README.md) |
| ⬜ | Static node2vec baseline with recorded settings; one $P = 1$ reference per pipeline | [E01](results/E01-static-node2vec/README.md) |
| ⬜ | Resolve the random-partitioning conflict (E03 ≈0.03% vs E11 exploratory 8–50%) through the seeded `random_partitioning` variant | E03, E11 |
| ⬜ | Significance analysis with multiple-comparison correction (Friedman + post-hoc, or Bonferroni) | E03 |

## M2 — Ablation study

| | Task |
|---|---|
| 🔵 | Run all 19 variants of [E11](results/E11-ablation/README.md) on SLURM and fill the tables with `collect` |
| ⬜ | Decide the `--track-seen` semantics: it is ignored once `--ordering` is set, so either drop it from the baseline or make it combine with an ordering |
| ⬜ | Summary table: variant × dataset, ΔF1 vs baseline, with significance |

## M3 — Alignment

| | Task |
|---|---|
| ⬜ | Seeded repetition (5–10 seeds) of the recommended configuration and both controls (mean, 0 steps) on AstroPh and Enron at $P = 4$ |
| ⬜ | Find the source of non-determinism in aligner training |
| ⬜ | Fix and re-run the Procrustes start; try Procrustes as initialisation of the residual aligner |
| ⬜ | Reduce the aligner cost at $P = 1$ (geometry-preserving term, step sweep 0–100) |
| ⬜ | Extend to $RF > 1$, $P = 8$, Enron and AS-Oregon with all links |
| ⬜ | Bring the aligner into the streaming pipeline and the ablation (`untrained_triplet_aggregator`, `mean_aggregator` variants) |

Details: [E10 future work](results/E10-partition-alignment/README.md#future-work).

## M4 — Comparison with standard streaming partitioners (RQ6)

| | Task |
|---|---|
| ⬜ | Implement LDG (Stanton and Kliot, KDD 2012) and FENNEL (Tsourakakis et al., WSDM 2014) behind the existing partitioner interface |
| ⬜ | Same grid as M1: F1, edge cut, balance, repartitioning, runtime |
| ⬜ | Optional: offline METIS as a quality upper bound for edge cut |

## M5 — Other embedding methods (RQ7)

| | Task |
|---|---|
| ⬜ | GraphSAGE (Hamilton, Ying, Leskovec, NeurIPS 2017) per partition, with the same aggregators |
| ⬜ | Optionally DeepWalk ($p = q = 1$, already the `unbiased_walk` variant) and DistGER, for continuity with [E00](results/E00-legacy-distributed/README.md) |
| ⬜ | Check whether the E08 decomposition and the E10 alignment results transfer |

## M6 — Theory follow-ups

| | Task |
|---|---|
| ⬜ | Fixed-$k$ partitioner and SBM sweep for the E08 density question ([E08 additional experiments](results/E08-f1-decomposition/README.md#additional-experiments)) |
| ⬜ | Report lift over the chance floor ($Q / \rho_{\text{within}}$) alongside raw F1 everywhere |

## M7 — Write-up

| | Task |
|---|---|
| ⬜ | Paper draft: RQ1, RQ5, RQ6 as core; RQ4 as method contribution |
| ⬜ | Thesis ([`doc/Rad.odt`](doc/Rad.odt)) |
| ⬜ | Replace notebook plotting with a script that regenerates every figure in `results/*/figures/` |

## Repository tooling

| | Task |
|---|---|
| ⬜ | Generate results tables from `scores.csv` with provenance (commit, date, seeds), starting with E11 |
| ⬜ | Deduplicate the presentation assets in `doc/Seminar1`, `doc/Seminar2`, `doc/presentation` and `doc/pp-presentation` |
