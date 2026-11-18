# References

Works the project builds on, compares against or follows for its protocols.
The BibTeX entries are in [`refs.bib`](refs.bib); each item gives its key.
"Used in" points to [ROADMAP](../ROADMAP.md) tasks and experiment pages.

## Embedding methods

- A. Grover, J. Leskovec. *node2vec: Scalable Feature Learning for
  Networks.* KDD 2016, 855–864.
  [arXiv:1607.00653](https://arxiv.org/abs/1607.00653) ·
  `grover2016node2vec` · Used in: every experiment.
- S. Mahdavi, S. Khoshraftar, A. An. *dynnode2vec: Scalable Dynamic Network
  Embedding.* IEEE Big Data 2018, 3762–3765.
  [arXiv:1812.02356](https://arxiv.org/abs/1812.02356) ·
  `mahdavi2018dynnode2vec` · Used in: pipeline T,
  [reference protocol](dynnode2vec-reference-protocol.md).
- B. Perozzi, R. Al-Rfou, S. Skiena. *DeepWalk: Online Learning of Social
  Representations.* KDD 2014, 701–710.
  [arXiv:1403.6652](https://arxiv.org/abs/1403.6652) ·
  `perozzi2014deepwalk` · Used in: M8.1, node-classification protocol
  (M7.2b).
- W. L. Hamilton, R. Ying, J. Leskovec. *Inductive Representation Learning
  on Large Graphs* (GraphSAGE). NeurIPS 2017.
  [arXiv:1706.02216](https://arxiv.org/abs/1706.02216) ·
  `hamilton2017inductive` · Used in: M8.3.
- J. Zhang, Y. Dong, Y. Wang, J. Tang, M. Ding. *ProNE: Fast and Scalable
  Network Representation Learning.* IJCAI 2019 · `zhang2019prone` ·
  Used in: M8.2.
- J. Qiu, Y. Dong, H. Ma, J. Li, K. Wang, J. Tang. *Network Embedding as
  Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec*
  (NetMF). WSDM 2018.
  [arXiv:1710.02971](https://arxiv.org/abs/1710.02971) ·
  `qiu2018network` · Used in: M8.2.

## Independent partition or patch embedding with alignment

These are the closest prior work to the partitioned pipeline and to the
aligner of [E10](../results/E10-partition-alignment/README.md).

- L. G. S. Jeub, G. Colavizza, X. Dong, M. Bazzi, M. Cucuringu.
  *Local2Global: a distributed approach for scaling representation learning
  on graphs.* Machine Learning 112(5):1663–1692, 2023.
  [doi:10.1007/s10994-022-06285-7](https://doi.org/10.1007/s10994-022-06285-7) ·
  `jeub2023local2global` · Overlapping patches embedded independently,
  then aligned by group synchronisation. Used in: M2.1, M6.3, M10.5.
- M. Imran, H. Yin, T. Chen, Z. Huang, X. Zhang, K. Zheng. *DDHH: A
  Decentralized Deep Learning Framework for Large-scale Heterogeneous
  Networks.* ICDE 2021, 2033–2038 · `imran2021ddhh` · Partitions embedded
  by independent workers, merged with Procrustes alignment. Used in: M2.1,
  M10.5.
- M. Imran, H. Yin, T. Chen, Z. Huang, K. Zheng. *DeHIN: A Decentralized
  Framework for Embedding Large-scale Heterogeneous Information Networks.*
  IEEE TKDE, 2023.
  [doi:10.1109/TKDE.2022.3141951](https://doi.org/10.1109/TKDE.2022.3141951),
  [arXiv:2201.02757](https://arxiv.org/abs/2201.02757) ·
  `imran2023dehin` · Successor of DDHH with distributed partitioning and
  context-aware alignment. Used in: M2.1, M10.5.

## Distributed embedding and training systems

- A. Lerer, L. Wu, J. Shen, T. Lacroix, L. Wehrstedt, A. Bose,
  A. Peysakhovich. *PyTorch-BigGraph: A Large-scale Graph Embedding System.*
  SysML 2019. [arXiv:1903.12287](https://arxiv.org/abs/1903.12287) ·
  `lerer2019pytorch` · Used in: M2.1.
- P. Fang, A. Khan, S. Luo, F. Wang, D. Feng, Z. Li, W. Yin, Y. Cao.
  *Distributed Graph Embedding with Information-Oriented Random Walks*
  (DistGER). PVLDB 16(7):1643–1656, 2023.
  [PDF](https://www.vldb.org/pvldb/vol16/p1643-fang.pdf) ·
  `fang2023distributed` · Used in:
  [E00](../results/E00-legacy-distributed/README.md), M2.1, M8.1.
- K. Yang, M. Zhang, K. Chen, X. Ma, Y. Bai, Y. Jiang. *KnightKing: A Fast
  Distributed Graph Random Walk Engine.* SOSP 2019 ·
  `yang2019knightking` · Used in: M2.1.
- D. Zheng, C. Ma, M. Wang, J. Zhou, Q. Su, X. Song, Q. Gan, Z. Zhang,
  G. Karypis. *DistDGL: Distributed Graph Neural Network Training for
  Billion-Scale Graphs.* IA3 workshop 2020.
  [arXiv:2010.05337](https://arxiv.org/abs/2010.05337) ·
  `zheng2020distdgl` · Used in: M2.1.

## Effect of partitioning on distributed training

- A. Kochsiek, R. Gemulla. *Parallel Training of Knowledge Graph Embedding
  Models: A Comparison of Techniques.* PVLDB 15(3):633–645, 2021.
  [doi:10.14778/3494124.3494144](https://doi.org/10.14778/3494124.3494144) ·
  `kochsiek2021parallel` · Random partitioning is competitive *with*
  parameter synchronisation, unlike [E03](../results/E03-partition-scaling/README.md)
  without it. Used in: M2.2, M5.3.
- N. Merkel, D. Stoll, R. Mayer, H.-A. Jacobsen. *An Experimental Comparison
  of Partitioning Strategies for Distributed Graph Neural Network Training.*
  EDBT 2025, 171–184.
  [doi:10.48786/edbt.2025.14](https://doi.org/10.48786/edbt.2025.14) ·
  `merkel2025experimental` · Partitioning effect on GNN training speed and
  memory, not on quality. Used in: M2.1.

## Graph partitioning

- I. Stanton, G. Kliot. *Streaming Graph Partitioning for Large Distributed
  Graphs* (LDG). KDD 2012, 1222–1230 · `stanton2012streaming` ·
  Used in: M2.3, [methods](methods.md#streaming-neighbour-partitioner).
- C. Tsourakakis, C. Gkantsidis, B. Radunovic, M. Vojnovic. *FENNEL:
  Streaming Graph Partitioning for Massive Scale Graphs.* WSDM 2014,
  333–342 · `tsourakakis2014fennel` · Used in: M2.3.
- F. Petroni, L. Querzoni, K. Daudjee, S. Kamali, G. Iacoboni. *HDRF:
  Stream-Based Partitioning for Power-Law Graphs.* CIKM 2015, 243–252.
  [PDF](https://cs.uwaterloo.ca/~kdaudjee/DaudjeeHDRF.pdf) ·
  `petroni2015hdrf` · Used in: M2.4.
- J. Nishimura, J. Ugander. *Restreaming Graph Partitioning: Simple
  Versatile Algorithms for Advanced Balancing.* KDD 2013 ·
  `nishimura2013restreaming` · Used in: M2.1.
- A. Pacaci, M. T. Özsu. *Experimental Analysis of Streaming Algorithms for
  Graph Partitioning.* SIGMOD 2019.
  [doi:10.1145/3299869.3300076](https://doi.org/10.1145/3299869.3300076) ·
  `pacaci2019experimental` · Used in: M2.1.
- G. Karypis, V. Kumar. *A Fast and High Quality Multilevel Scheme for
  Partitioning Irregular Graphs* (METIS). SIAM J. Sci. Comput.
  20(1):359–392, 1998 · `karypis1998fast` · Used in: M2.5.
- K. Andreev, H. Räcke. *Balanced Graph Partitioning.* Theory of Computing
  Systems 39(6):929–939, 2006 · `andreev2006balanced` · NP-hardness of
  balanced $k$-partitioning. Used in: M9.8.

## Link prediction and node classification protocols

- J. Li, H. Shomer, H. Mao, S. Zeng, Y. Ma, N. Shah, J. Tang, D. Yin.
  *Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls
  and New Benchmarking* (HeaRT). NeurIPS 2023 Datasets and Benchmarks.
  [arXiv:2306.10453](https://arxiv.org/abs/2306.10453) ·
  `li2023evaluating` · Heuristic hard negatives. Used in: M7.0a, M7.0f.
- F. Poursafaei, S. Huang, K. Pelrine, R. Rabbany. *Towards Better
  Evaluation for Dynamic Link Prediction.* NeurIPS 2022 Datasets and
  Benchmarks. [arXiv:2207.10128](https://arxiv.org/abs/2207.10128) ·
  `poursafaei2022towards` · Historical negatives and the EdgeBank
  baseline. Used in: M7.1c, M7.1e.
- S. Huang, F. Poursafaei, J. Danovitch, M. Fey, W. Hu, E. Rossi,
  J. Leskovec, M. Bronstein, G. Rabusseau, R. Rabbany. *Temporal Graph
  Benchmark for Machine Learning on Temporal Graphs* (TGB). NeurIPS 2023
  Datasets and Benchmarks.
  [arXiv:2307.01026](https://arxiv.org/abs/2307.01026) ·
  `huang2023temporal` · One-vs-many MRR with historical and random
  negatives. Used in: M3.1, M7.1c, M7.1f.
- J. Gastinger, S. Huang, M. Galkin, E. Loghmani, A. Parviz,
  F. Poursafaei, J. Danovitch, E. Rossi, I. Koutis, H. Stuckenschmidt,
  R. Rabbany, G. Rabusseau. *TGB 2.0: A Benchmark for Learning on Temporal
  Knowledge Graphs and Heterogeneous Graphs.* NeurIPS 2024 Datasets and
  Benchmarks. [arXiv:2406.09639](https://arxiv.org/abs/2406.09639) ·
  `gastinger2024tgb` · Used in: M3.1.
- W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta,
  J. Leskovec. *Open Graph Benchmark: Datasets for Machine Learning on
  Graphs* (OGB). NeurIPS 2020.
  [arXiv:2005.00687](https://arxiv.org/abs/2005.00687) · `hu2020open` ·
  Tie-averaged ranks; ogbn-arxiv. Used in: M3.2, M7.0b.
- J. Gama, R. Sebastião, P. P. Rodrigues. *On evaluating stream learning
  algorithms.* Machine Learning 90(3):317–346, 2013.
  [doi:10.1007/s10994-012-5320-9](https://doi.org/10.1007/s10994-012-5320-9) ·
  `gama2013evaluating` · Prequential (test-then-train) evaluation.
  Used in: M7.1b.
- O. Shchur, M. Mumme, A. Bojchevski, S. Günnemann. *Pitfalls of Graph
  Neural Network Evaluation.* R2L workshop, NeurIPS 2018.
  [arXiv:1811.05868](https://arxiv.org/abs/1811.05868) ·
  `shchur2018pitfalls` · Variance across splits; Coauthor and Amazon
  datasets. Used in: M3.6, M7.2b.
- Z. Yang, W. W. Cohen, R. Salakhutdinov. *Revisiting Semi-Supervised
  Learning with Graph Embeddings* (Planetoid). ICML 2016, 40–48.
  [arXiv:1603.08861](https://arxiv.org/abs/1603.08861) ·
  `yang2016revisiting` · CITESEER labels and split. Used in: M3.6,
  [datasets](datasets.md).
- T. N. Kipf, M. Welling. *Semi-Supervised Classification with Graph
  Convolutional Networks.* ICLR 2017.
  [arXiv:1609.02907](https://arxiv.org/abs/1609.02907) ·
  `kipf2017semi` · DeepWalk on CiteSeer: 43.2% (Table 2). Used in: M7.2f.

## Synthetic graph models

- P. W. Holland, K. B. Laskey, S. Leinhardt. *Stochastic blockmodels: First
  steps.* Social Networks 5(2):109–137, 1983 · `holland1983stochastic` ·
  Used in: M3.3, M9.1.
- E. Abbe. *Community detection and stochastic block models: recent
  developments.* JMLR 18(177):1–86, 2018 · `abbe2018community` ·
  Used in: M3.3.
- A. Lancichinetti, S. Fortunato, F. Radicchi. *Benchmark graphs for
  testing community detection algorithms* (LFR). Physical Review E 78(4),
  046110, 2008. [arXiv:0805.4770](https://arxiv.org/abs/0805.4770) ·
  `lancichinetti2008benchmark` · Used in: M3.3.

## Dataset sources

Details and corrections: [datasets.md](datasets.md).

- A. Bojchevski, S. Günnemann. *Deep Gaussian Embedding of Graphs:
  Unsupervised Inductive Learning via Ranking.* ICLR 2018 ·
  `bojchevski2018deep` · DBLP.
- J. Leskovec, J. Kleinberg, C. Faloutsos. *Graph Evolution: Densification
  and Shrinking Diameters.* ACM TKDD 1(1), 2007 · `leskovec2007graph` ·
  AstroPh.
- J. Leskovec, J. Kleinberg, C. Faloutsos. *Graphs over Time:
  Densification Laws, Shrinking Diameters and Possible Explanations.*
  KDD 2005, 177–187 · `leskovec2005graphs` · AS-Oregon, Cit-HepPh,
  Cit-HepTh.
- J. Leskovec, K. Lang, A. Dasgupta, M. Mahoney. *Community Structure in
  Large Networks.* Internet Mathematics 6(1):29–123, 2009 ·
  `leskovec2009community` · Enron.
- B. Klimt, Y. Yang. *Introducing the Enron Corpus.* CEAS 2004 ·
  `klimt2004introducing` · Enron.
- A. Paranjape, A. R. Benson, J. Leskovec. *Motifs in Temporal Networks.*
  WSDM 2017 · `paranjape2017motifs` · StackOverflow and the other SNAP
  temporal graphs (M3.1).
- S. Kumar, W. L. Hamilton, J. Leskovec, D. Jurafsky. *Community
  Interaction and Conflict on the Web.* WWW 2018 · `kumar2018community` ·
  Reddit hyperlinks (M3.1).
- W. W. Zachary. *An information flow model for conflict and fission in
  small groups.* J. Anthropol. Res. 33(4):452–473, 1977 ·
  `zachary1977information` · Zachary karate club.

## Statistics

- S. Holm. *A Simple Sequentially Rejective Multiple Test Procedure.*
  Scandinavian Journal of Statistics 6(2):65–70, 1979 ·
  `holm1979simple` · Used in: M0.1, M6.7, M11.1.
- D. J. Schuirmann. *A comparison of the two one-sided tests procedure and
  the power approach for assessing the equivalence of average
  bioavailability* (TOST). J. Pharmacokinetics and Biopharmaceutics
  15(6):657–680, 1987 · `schuirmann1987comparison` · Used in: M0.3,
  M11.1.
- J. Demšar. *Statistical Comparisons of Classifiers over Multiple Data
  Sets.* JMLR 7:1–30, 2006 · `demsar2006statistical` · Friedman and
  Wilcoxon tests over datasets. Used in: M11.1.
- E. D. Dolan, J. J. Moré. *Benchmarking optimization software with
  performance profiles.* Mathematical Programming 91(2):201–213, 2002 ·
  `dolan2002benchmarking` · Used in: M5.2.
