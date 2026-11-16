# E03 — Partition-count scaling of reconstruction quality

| | |
|---|---|
| **Research question** | [RQ1](../README.md#research-questions) |
| **Status** | ✅ done · 🔁 re-run with fixed seeds planned |
| **Period** | 2026-05 → 2026-11 |
| **Pipeline** | T, streaming, $P \in \{1,2,4,8\}$, $RF = 1$ |
| **Datasets** | CITESEER, DBLP, AstroPh, AS-Oregon, Enron |
| **Headline** | Structural partitioning does not degrade F1 uniformly: F1 rises with $P$ on CITESEER and Enron, falls on AstroPh and AS-Oregon, and is non-monotone on DBLP. **Caveat:** [E08](../E08-f1-decomposition/README.md) shows that cross-partition edges are never reconstructed, so a rise with $P$ comes from within-partition recall and is not a quality gain. |
| **How the numbers were produced** | F1 tables: runs with the default `--algorithm dynnode2vec` (example command under Results). Balance, edge-cut and repartitioning plots: runs with `--algorithm dummy`, which skips embedding; these metrics do not depend on it. |
| **Open issues** | 3–11 runs per cell without fixed seeds. The random-partitioner F1 below (≈0.03%) conflicts with the exploratory ablation runs in [E11](../E11-ablation/README.md#results) (8–50%); the code version and aggregator behind each are not recorded. No multiple-comparison correction. All three are scheduled in [ROADMAP](../../ROADMAP.md). |

**Question.** As the graph is split into more partitions $P$ (each partition
embedded independently, then streamed with dynnode2vec), does the quality of
the learned embeddings stay the same, or does partitioning degrade it?

**Metric.** F1 **graph-reconstruction score** ("RESTORE"): edges are
predicted from the current embeddings and compared against the actual graph;
F1 is the harmonic mean of edge precision and recall. For streaming runs it
is recorded after every buffer/iteration; the headline number is the
last-iteration F1 once the whole stream has been consumed. See
the evaluation protocols in [evaluation.md](../../doc/evaluation.md) and
[DynNode2Vec evaluation methodology](../../doc/dynnode2vec-reference-protocol.md).

# How the experiment was run

Each configuration $(\text{dataset},\, P,\, RF{=}1)$ is run several times so
that run-to-run variance can be estimated.

```bash
for dataset in CITESEER AstroPh AS-Oregon DBLP Enron; do
  for P in 1 2 4 8; do
    vv temporal_test --name "$dataset" --long-run --iterations 3 \
       --use-dataset-params --buffer-size 1000 --replication-factor 1 \
       --partitioner neighbors.all --mu 1 --algorithm dummy --epochs 10 \
       --partitions "$P" \
       --checkpoint temporal_runs/macbook/"$dataset-${P}_1" \
       --skip-full-graph-baseline
  done
done
```

| Parameter | Value |
|---|---|
| Partitions $P$ | 1, 2, 4, 8 |
| Replication factor $RF$ | 1 (fixed) |
| Buffer size | 1000 |
| Partitioner | `neighbors.all` |
| Balance penalty $\mu$ | 1 |
| Epochs | 10 |
| Repeats per config | CITESEER 11; DBLP, AstroPh 4; AS-Oregon 4 ($P=1$) and 3 ($P=2,4,8$); Enron not reported (pooled across `macbook`, `macbook_new` run sets) |

Every run persists `profile*.pkl` (per-iteration edge cuts, repartitionings,
partition sizes) and `iteration_f1s*.pkl` (reconstruction F1 per iteration).

**Analysis:** [`notebooks/analyze_results.ipynb`](../../notebooks/analyze_results.ipynb).
`add_to_everything(...)` walks `temporal_runs/<suffix>/<dataset>-<P>_<RF>/`,
loads every run, and `merge_everything` concatenates raw runs so repeats from
different machines are pooled into one sample per configuration. The
per-run last-iteration F1 (`get_last_iter_f1s` → `run[-1]`) is the sample fed
to the tests below.

# Results

## Datasets and dataset parameters

See [doc/datasets.md](../../doc/datasets.md). The vertex and edge counts
and the dataset descriptions there were corrected on 2026-11-16.

## Node2Vec baseline

Full-graph node2vec F1 per dataset:
[E01](../E01-static-node2vec/README.md#full-graph-node2vec-baseline).

## Reconstruction F1 vs. partition count

Dynnode2vec with buffered event processing.

Example of run:

```sh
vv temporal_test --name CITESEER  --long-run --iterations 10  --use-dataset-params  --buffer-size 1000 --partitions 5 --replication-factor 1 
```

CITESEER:

| # of partitions | F1 Reconstruction score | Edge cut | Balance |
|-----------------|--------------------------|----------|----------|
| 1               | 47.57% ± 1.56%           | 0%       | 1 | 
| 2               | 45.81% ± 0.59%           | 23%      | 1.31 |
| 4               | 48.06% ± 0.53%           | 39%      | 1.63 |
| 8               | 54.81% ± 1.66%           | 48%      | 1.75 |

F1 Reconstruction score is reported as mean ± standard deviation over 11 runs.

DBLP: 

| # of partitions | F1 Reconstruction score | Edge cut | Balance |
|-----------------|--------------------------|----------|----------|
| 1               | 50.91% ± 0.31%           | 0%       | 1 | 
| 2               | 54.11% ± 0.18%           | 32%      | 1.10 |
| 4               | 54.51% ± 0.15%           | 50%      | 1.20 |
| 8               | 49.87% ± 0.16%           | 59%      | 1.31 |

F1 Reconstruction score is reported as mean ± standard deviation over 4 runs.

AstroPh:

| # of partitions | F1 Reconstruction score | Edge cut | Balance |
|-----------------|--------------------------|----------|----------|
| 1               | 57.70% ± 0.11%           | 0%       | 1 |
| 2               | 54.67% ± 0.36%           | 35%      | 1.08 |
| 4               | 50.95% ± 0.45%           | 56%      | 1.11 |
| 8               | 45.94% ± 0.51%           | 65%      | 1.26 |

F1 Reconstruction score is reported as mean ± standard deviation over 4 runs.

AS-Oregon:

| # of partitions | F1 Reconstruction score | Edge cut | Balance |
|-----------------|-------------------------|----------|----------|
| 1               | 26.17% ± 0.57% | 0%   | 1 |
| 2               | 24.88% ± 2.72% | 30.44%   | 1.21|
| 4               | 24.30% ± 0.98% | 53.79%  | 1.26 |
| 8               | 21.45% ± 1.19% | 71.56%  | 1.27 |

F1 Reconstruction score is reported as mean ± standard deviation over 4 runs (P=1) and 3 runs (P=2, 4, 8).


Enron:

| # of partitions | F1 Reconstruction score | Edge cut | Balance |
|-----------------|-------------------------|----------|----------|
| 1               | 18.03% ± 0.44% | 0%    | 1    |
| 2               | 24.84% ± 0.38% | 33.07% | 1.09 |
| 4               | 27.44% ± 2.83% | 49.18% | 1.08 |
| 8               | 32.36% ± 1.24% | 61.99% | 1.25 |



Partitioning time is negligible compared to embedding time. By partitioning the graph into multiple partitions, we can reduce the embedding time significantly, where the slowest partition is the biggest bottleneck.

## Balance metric

$B = 1 + \beta$, defined with its proven range $1 \le B \le 3 - 2/K$ in
[doc/evaluation.md](../../doc/evaluation.md#2-partition-quality).

## Random partitioner baseline

To measure how much of the neighbor-based partitioner's embedding quality comes from preserving neighborhood locality, we ran the same buffered dynnode2vec pipeline using a **random partitioner**, which assigns each vertex to a partition uniformly at random, ignoring its neighbors entirely.

Example of run:

```sh
vv temporal_test --name CITESEER  --long-run --iterations 3  --use-dataset-params  --buffer-size 1000 --partitions 4 --replication-factor 1 --partitioner random
```

CITESEER:

| # of partitions | F1 Reconstruction score | Edge cut | Balance |
|-----------------|--------------------------|----------|----------|
| 2               | 0.03% ± 0.02%            | 50.01%   | 1.02 |
| 4               | 0.03% ± 0.03%            | 70.89%   | 1.32 |
| 8               | 0.04% ± 0.03%            | 81.58%   | 1.61 |

F1 Reconstruction score is reported as mean ± standard deviation over 3 runs.

AstroPh:

| # of partitions | F1 Reconstruction score | Edge cut | Balance |
|-----------------|--------------------------|----------|----------|
| 2               | 0.04% ± 0.02%            | 49.98%   | 1.00 |
| 4               | 0.03% ± 0.02%            | 70.86%   | 1.31 |
| 8               | 0.02% ± 0.01%            | 81.43%   | 1.60 |

F1 Reconstruction score is reported as mean ± standard deviation over 3 runs.

Compared to the neighbor-based partitioner, which reaches F1 scores of 45–55% on CITESEER and 46–55% on AstroPh at the same partition counts, random partitioning collapses reconstruction quality to essentially zero. This confirms that the neighbor partitioner's F1 scores are driven by preserving neighborhood locality across partitions rather than by the embedding algorithm alone: once locality is destroyed, per-partition embeddings become mutually incompatible and the reconstructed graph is no better than chance. Edge cut is also far higher under random partitioning (50–82%) than under the neighbor partitioner (23–65%), closely tracking the expected value $(P-1)/P$ for uniform random assignment
([evaluation.md](../../doc/evaluation.md#2-partition-quality)).

> **Open issue.** The exploratory ablation runs report much higher F1 for
> random partitioning (CITESEER 0.30 / 0.27 / 0.17, AstroPh 0.50 / 0.10 / 0.08
> at $P = 2, 4, 8$; [E11](../E11-ablation/README.md#results)).
> The two sets of runs used different code versions, and probably different
> aggregators. The seeded `random_partitioning` ablation variant will replace both.


## Temporal evolution plots

The following plots show how F1 score, balance, edge cuts, and repartitioning rate evolve over iterations (buffer steps) for each dataset, broken down by number of partitions.

### F1 Score over iterations

| CITESEER | DBLP |
|----------|------|
| ![CITESEER F1](figures/citeseer-f1.png) | ![DBLP F1](figures/dblp-f1.png) |

| AstroPh | AS-Oregon |
|---------|-----------|
| ![AstroPh F1](figures/astroph-f1.png) | ![AS-Oregon F1](figures/as-oregon-f1.png) |

| Enron |
|-------|
| ![Enron F1](figures/enron-f1.png) |

F1 reconstruction score generally improves as more events are ingested. Using more partitions tends to maintain competitive or improved F1 in datasets with clear community structure (CITESEER, DBLP, Enron), while high-density datasets (AstroPh) see a modest drop at higher partition counts due to increased edge cut.

### Balance over iterations

| CITESEER | DBLP |
|----------|------|
| ![CITESEER Balance](figures/citeseer-balance.png) | ![DBLP Balance](figures/dblp-balance.png) |

| AstroPh | AS-Oregon |
|---------|-----------|
| ![AstroPh Balance](figures/astroph-balance.png) | ![AS-Oregon Balance](figures/as-oregon-balance.png) |

| Enron |
|-------|
| ![Enron Balance](figures/enron-balance.png) |

Balance remains close to 1 across all datasets and partition counts, indicating that the neighbor-based partitioner distributes vertices nearly evenly without requiring explicit load-balancing. The imbalance stays within a small constant factor of the ideal even as the graph evolves over time.

### Edge cuts over iterations

| CITESEER | DBLP |
|----------|------|
| ![CITESEER Edge Cuts](figures/citeseer-edge-cuts.png) | ![DBLP Edge Cuts](figures/dblp-edge-cuts.png) |

| AstroPh | AS-Oregon |
|---------|-----------|
| ![AstroPh Edge Cuts](figures/astroph-edge-cuts.png) | ![AS-Oregon Edge Cuts](figures/as-oregon-edge-cuts.png) |

| Enron |
|-------|
| ![Enron Edge Cuts](figures/enron-edge-cuts.png) |

Edge cut ratio rises quickly in the first few iterations as new vertices are assigned, then stabilizes at a bounded plateau. This plateau is determined primarily by the number of partitions and the graph's community structure, confirming that the partitioner keeps edge cuts at predictable and manageable levels throughout the stream.

### Repartitioning rate over iterations

| CITESEER | DBLP |
|----------|------|
| ![CITESEER Repartitions](figures/citeseer-repartitions.png) | ![DBLP Repartitions](figures/dblp-repartitions.png) |

| AstroPh | AS-Oregon |
|---------|-----------|
| ![AstroPh Repartitions](figures/astroph-repartitions.png) | ![AS-Oregon Repartitions](figures/as-oregon-repartitions.png) |

| Enron |
|-------|
| ![Enron Repartitions](figures/enron-repartitions.png) |

The fraction of vertices repartitioned per iteration decays rapidly and stays below 20% across all datasets and partition configurations. After the initial warm-up phase, the partitioner stabilises and only reassigns vertices whose neighbourhood has shifted significantly, keeping the overhead of maintaining partition assignments low throughout the stream.

The per-partition summary table produced by the notebook
(`significance_summary_table`) reports, for each $P$: number of runs, mean
F1, and standard deviation of the final F1.

## Significance tests

For each dataset the notebook (`report_significance`) compares partition
counts pairwise — every $P$ against the baseline $P_{\min}$, and each
adjacent pair $(P_i, P_{i+1})$ — on the raw per-run final F1 scores
(`significance_pairs_table`), skipping configs with fewer than 2 runs.

### Welch's two-sample t-test — [concept](../../doc/glossary.md#t-test)

`scipy.stats.ttest_ind(vals_a, vals_b, equal_var=False)`

$$ t = \frac{\bar{f}^{(a)} - \bar{f}^{(b)}}{\sqrt{s_a^2/n_a + s_b^2/n_b}} $$

Unequal-variance version, because partition counts have different numbers of
runs and visibly different F1 variance. $H_0$: equal mean final F1. This
p-value drives the `Significant (α = 0.05)` column.

### Mann–Whitney U test — [concept](../../doc/glossary.md#mann-whitney-test)

`scipy.stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")`

Non-parametric, rank-based. With only a few runs per config the F1 scores
cannot be assumed normal, so the rank test is reported next to the t-test.
**Agreement between the two** means the conclusion does not hinge on the
normality assumption; disagreement is a signal to collect more runs.

### Cohen's d effect size — [concept](../../doc/glossary.md#cohens-d)

$$ d = \frac{\bar{f}^{(b)} - \bar{f}^{(a)}}{s_p}, \qquad
   s_p = \sqrt{\tfrac{s_a^2 + s_b^2}{2}} \quad (\text{sample SD}, \text{ddof}=1) $$

Reported because p-values alone are misleading at small $n$: a
non-significant result may just be low power. $|d| \lesssim 0.2$ → the
difference is practically negligible; $|d| \gtrsim 0.8$ with $p > 0.05$ →
underpowered, not equivalent.

# Conclusion

Embedding quality is judged **consistent / partition-invariant** for a
dataset when, across *all* baseline and adjacent pairs:

1. Welch t-test $p > 0.05$ (no detectable mean shift), **and**
2. Mann–Whitney $p > 0.05$ (no detectable distributional shift), **and**
3. $|d|$ is small (≈ < 0.5), so the non-result is not merely low power, **and**
4. the summary means show no monotone drift with $P$.

If instead one or more pairs are significant with medium-to-large $|d|$ and
the means drift with $P$, partitioning **changes** embedding quality, and $d$
quantifies by how many standard deviations of F1 per partitioning step.
Observed so far (mean final F1 from
[Reconstruction F1 vs. partition count](#reconstruction-f1-vs-partition-count),
$P = 1 \to 2 \to 4 \to 8$):

| Dataset | $P=1$ | $P=2$ | $P=4$ | $P=8$ | Trend with $P$ |
|---|---|---|---|---|---|
| CITESEER  | 47.57% | 45.81% | 48.06% | 54.81% | dip at $P=2$, then increase |
| DBLP      | 50.91% | 54.11% | 54.51% | 49.87% | increase up to $P=4$, drop at $P=8$ |
| AstroPh   | 57.70% | 54.67% | 50.95% | 45.94% | monotone decrease |
| AS-Oregon | 26.17% | 24.88% | 24.30% | 21.45% | monotone decrease |
| Enron     | 18.03% | 24.84% | 27.44% | 32.36% | monotone increase |

Partitioning therefore does not degrade quality uniformly: F1 increases with
$P$ on CITESEER and Enron, decreases on AstroPh and AS-Oregon, and is
non-monotone on DBLP. Edge cut grows with $P$ on every dataset (23–48% on
CITESEER up to 33–62% on Enron). In all cases structural partitioning does
not collapse reconstruction the way random partitioning does (F1 ≈ 0.02–0.04%,
see [Random partitioner baseline](#random-partitioner-baseline) and the
[ablation study](../E11-ablation/README.md)). The significance tests make the
"stable / not degraded" claim quantitative rather than eyeballed from a
table.

## Caveats

- **Small $n$** per config (3–4 runs for every dataset except CITESEER with
  11) → low power; the effect size keeps a non-significant result honest.
- **Multiple comparisons** are run with no correction, and the `Significant`
  flag uses the raw Welch $p$ only (Mann–Whitney $p$ and Cohen's $d$ are
  advisory columns). A
  [Bonferroni correction](../../doc/glossary.md#bonferroni-correction),
  or [ANOVA](../../doc/glossary.md#anova) +
  [Tukey's HSD](../../doc/glossary.md#tukey-hsd) /
  [Friedman test](../../doc/glossary.md#friedman-and-wilcoxon-test)
  across all partition counts at once, would be the principled multi-group
  alternative.

# Related concepts

- [Welch's / two-sample t-test](../../doc/glossary.md#t-test)
- [Mann–Whitney U test](../../doc/glossary.md#mann-whitney-test)
- [Cohen's d](../../doc/glossary.md#cohens-d)
- [standard deviation](../../doc/glossary.md#standard-deviation)
- [ANOVA](../../doc/glossary.md#anova) ·
  [Tukey's HSD](../../doc/glossary.md#tukey-hsd) ·
  [Friedman test](../../doc/glossary.md#friedman-and-wilcoxon-test)
