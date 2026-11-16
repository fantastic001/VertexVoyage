# Datasets

Vertex and edge counts below were measured on 2026-11-16 from the files the
code loads (`experiments/datasets.py`). The script removes self-loops and
counts undirected edges $\{u, v\}$ once. *Events* is the length of the event
stream fed to `temporal_test`; it can exceed the edge count when the file
lists both directions or repeats edges across snapshots.

| Dataset | Vertices | Edges | Events | Type | Used in |
|---|---:|---:|---:|---|---|
| CITESEER | 3,264 | 4,536 | 4,536 | citation | E01–E11 |
| DBLP | 17,716 | 52,867 | 52,867 | citation | E01, E03–E05, E08, E11 |
| AstroPh | 18,772 | 198,050 (+60 self-loops) | 198,110 | co-authorship | E01–E11 |
| AS-Oregon | 11,806 | 38,781 | 284,040 | AS routing, 9 snapshots | E03–E05, E09, E10 |
| Enron | 36,692 | 183,831 | 367,662 | e-mail | E03–E05, E08, E10, E11 |
| Cit-HepPh | 34,546 | 420,877 (+44 self-loops) | 421,578 | citation | E00, E01, E02, E09 |
| Cit-HepTh | 27,770 | 352,285 (+39 self-loops) | 352,807 | citation | E00, E01 |
| Zachary karate club | 34 | 78 | 78 | social | E08 |
| StackOverflow | 2,601,977 | 63,497,050 (SNAP) | — | temporal Q&A | registered, not yet used |
| Erdős–Rényi $G(n,p)$ | $n \in \{200, 500, 1000, 5000\}$ | — | — | synthetic | E08 |

## Sources and corrections

- **CITESEER.** Planetoid CiteSeer (Z. Yang, W. W. Cohen, R. Salakhutdinov,
  "Revisiting Semi-Supervised Learning with Graph Embeddings", ICML 2016).
  Planetoid reports 3,327 vertices and 9,104 directed (4,552 undirected)
  edges. The file used here keeps the 3,264 non-isolated vertices and 4,536
  undirected edges. The README before 2026-11 quoted the Planetoid numbers.
- **DBLP.** 17,716 vertices and 52,867 edges match the DBLP *citation* graph
  of A. Bojchevski, S. Günnemann, "Deep Gaussian Embedding of Graphs", ICLR
  2018, as distributed by the
  [GRASPE repository](https://github.com/graphsinspace/graspe/tree/main/data).
  Earlier text cited SNAP and described it as a citation network of
  computer-science papers. The SNAP com-DBLP graph is a different,
  co-authorship network with 317,080 vertices. *To confirm against the
  download source.*
- **AstroPh.** SNAP ca-AstroPh, a co-authorship network (vertices are
  authors, not papers). J. Leskovec, J. Kleinberg, C. Faloutsos, "Graph
  Evolution: Densification and Shrinking Diameters", ACM TKDD 1(1), 2007.
- **AS-Oregon.** SNAP oregon2: the union of 9 weekly snapshots of
  autonomous-system peering. The last snapshot alone has 11,461 vertices and
  32,730 edges, the numbers quoted before 2026-11. J. Leskovec,
  J. Kleinberg, C. Faloutsos, "Graphs over Time", KDD 2005.
- **Enron.** SNAP email-Enron. Vertices are addresses, and an edge means at
  least one e-mail was exchanged. J. Leskovec, K. Lang, A. Dasgupta,
  M. Mahoney, "Community Structure in Large Networks", Internet Mathematics
  6(1), 2009; B. Klimt, Y. Yang, "Introducing the Enron corpus", CEAS 2004.
- **Cit-HepPh, Cit-HepTh.** SNAP arXiv high-energy-physics citation graphs
  (Leskovec et al., KDD 2005).
- **Zachary.** W. W. Zachary, "An information flow model for conflict and
  fission in small groups", J. Anthropol. Res. 33, 1977.
- **StackOverflow.** SNAP sx-stackoverflow. A. Paranjape, A. R. Benson,
  J. Leskovec, "Motifs in Temporal Networks", WSDM 2017.

## Embedding hyperparameters per dataset

`--use-dataset-params` reads $p$, $q$ and the dimension from
`dataset_params` in [`experiments/datasets.py`](../experiments/datasets.py):

| Dataset | $p$ | $q$ | dim |
|---|---|---|---|
| CITESEER | 0.5 | 0.25 | 50 |
| DBLP | 0.5 | 1 | 25 |
| AstroPh | 2 | 0.25 | 50 |
| AS-Oregon | 0.5 | 2 | 128 |
| Enron | 0.5 | 1 | 128 |
| Cit-HepPh | 4 | 0.25 | 100 |
| Cit-HepTh | 2 | 0.25 | 100 |
| Zachary | 0.25 | 4 | 100 |

## Obtaining the data

The data files are not stored in the repository. Each source above links to
the public original. [guide/adding-a-dataset.md](guide/adding-a-dataset.md)
describes how a dataset is registered.
