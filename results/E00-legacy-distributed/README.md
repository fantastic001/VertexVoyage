# E00 — Static distributed node2vec with community-aware partitioning (Seminars 1–2)

| | |
|---|---|
| **Research question** | Predecessor of [RQ1](../README.md#research-questions) |
| **Status** | ⚪ superseded by the streaming pipeline (E02–E07, E11) and the LPA-based static work (E08–E10) |
| **Period** | 2024-07 → 2025-11 |
| **Pipeline** | S, static, one epoch, 1 negative sample |
| **Datasets** | Zachary, Les Misérables, Davis Southern Women, Florentine families, Twitch, SBM, CITESEER, AstroPh, Cit-HepPh, Cit-HepTh |
| **Headline** | Label propagation (LPA) partitions give higher reconstruction F1 and much lower edge cut than modified LFM: at $P = 4$, CITESEER 44.3% vs 19.0% F1 and 22% vs 68.8% edge cut. DistGER (information-oriented walks) beat node2vec with LPA. |
| **Sources** | Seminar 1 paper [`doc/Seminar1/paper.md`](../../doc/Seminar1/paper.md); Seminar 2 paper [`doc/Seminar2/paper.md`](../../doc/Seminar2/paper.md) |

This page summarises the earlier, static line of work. The full methodology
and every table are in the two seminar papers. Balance in these papers is
$\beta$, where 0 means perfectly balanced, not $B = 1 + \beta$
([evaluation.md](../../doc/evaluation.md#2-partition-quality)).

## Question

Can a graph be partitioned along communities, embedded independently per
partition with node2vec, and still yield embeddings comparable to sequential
node2vec, while partitioning stays cheap?

## Key results

**Sequential vs distributed node2vec at $P = 2$** (10 walks per node, walk
length 80; Seminar 2, Table 2):

| Network | Dim | Sequential F1 | Distributed F1, $P = 2$ |
|---|---|---|---|
| CITESEER | 50 | 24% | 33% |
| AstroPh | 100 | 5.1% | 6.6% |
| Cit-HepPh | 100 | 2.5% | 2.8% |

**Partitioner comparison at $P = 4$** (Seminar 1, Tables 8–11 and 14):

| Network | F1, modified LFM | F1, random | F1, LPA | Edge cut LFM | Edge cut LPA | $\beta$ LFM | $\beta$ LPA |
|---|---|---|---|---|---|---|---|
| CITESEER | 18.99% | 26.67% | 44.34% | 68.8% | 22% | 0.0016 | 0.0006 |
| AstroPh | 8.97% | 8.23% | 19.48% | 74.5% | 8.4% | 0.0009 | 0.824 |
| Cit-HepPh | 3.43% | 3.08% | 4.8% | 73.8% | 20.6% | 0.0015 | 0.00005 |
| Cit-HepTh | 4.33% | 3.76% | 5.1% | 74.1% | 11.7% | 0.0023 | 0.483 |

**DistGER with LPA** (Seminar 2, Table 4):

| Network | $P = 1$ | $P = 2$ | $P = 4$ |
|---|---|---|---|
| CITESEER | 22.14% | 43.83% | 60.66% |
| AstroPh | 5.53% | 8.05% | 29.77% |
| Cit-HepPh | 2.68% | 4.04% | 4.39% |

**Clustering agreement** between sequential and distributed embeddings
(K-means, ARI; Seminar 1, Table 12): Zachary 1.0, Twitch 0.97, SBM 0.99,
Davis 0.65, Les Misérables 0.45, Florentine 0.38.

## What carried over and what did not

- **Carried over.** LPA as the static partitioner (E08–E10), reconstruction
  F1 as the main metric, and the observation that random partitioning hurts.
- **Revised later.** The absolute F1 values here use one epoch and one
  negative sample, so they are far below the later numbers (AstroPh 5–30%
  here vs 57–72% in E01/E03). The rise of F1 with $P$ seen here (CITESEER,
  DistGER) is the effect that [E08](../E08-f1-decomposition/README.md)
  later explains as within-partition recall, not a quality gain.
- **Mixed balance evidence.** LPA has the lower edge cut everywhere, but its
  balance is much worse than LFM's on AstroPh and Cit-HepTh, in contrast to
  the papers' summary sentence.
