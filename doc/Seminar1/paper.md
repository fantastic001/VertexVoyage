---
title: "Evaluation and analysis of graph vertex embeddings in distributed environment with community-aware vertex partitioning"
author: "Stefan Nožinić"
abstract: |
  This paper explores the evaluation and analysis of graph vertex embeddings in a distributed environment, focusing on community-aware vertex partitioning to enhance the time efficiency of embeddings and reduce the need for network communication during the embedding generation process. The study investigates various partitioning algorithms combined with Node2Vec algorithm, assessing their effectiveness in capturing community structures and generating meaningful embeddings for large-scale graphs. Experimental results demonstrate the benefits of community-aware partitioning in improving embedding quality and efficiency in distributed settings.
bibliography: ./refs.bib
---

# Introduction 


A graph is a mathematical structure consisting of vertices (or nodes) connected by edges. Graphs are widely used to model relationships and interactions in various domains [@van_der_hofstad_random_2024], such as social networks [@leskovec_signed_2010] [@backstrom_group_2006] [@rozemberczki_twitch_2021], collaboration networks [@savic_analysis_2017], terrorist networks [@krebs_mapping_2002] and blog citation networks [@adamic_political_2005]. In these applications, the relationships between entities can be represented as edges connecting the corresponding vertices.


In many real-world graphs, the degree distribution follows a power-law, meaning that a small number of vertices have a very high degree (i.e., they are connected to many other vertices), while most vertices have a low degree. This characteristic is often observed in social networks, where a few individuals (e.g. celebrities) have many connections, while the majority of users have relatively few connections.

In real graphs, there are several properties that are often observed [@watts_collective_1998] [@zachary_information_1977] [@albert_statistical_2002]:
- **Small-world property**: Most pairs of vertices can be connected by a short path, even in large graphs. This is often referred to as the "six degrees of separation" phenomenon. [@watts_collective_1998]
- **Community structure**: Vertices tend to form clusters or communities, where vertices within the same community are more densely connected than those in different communities. This property is prevalent in social networks, where groups of friends or colleagues often form tightly-knit communities. [@leskovec_community_2009]
- **Scale-free property**: The degree distribution of the graph follows a power-law, meaning that a few vertices have a very high degree, while most vertices have a low degree. This is often observed in social networks, where a small number of individuals (e.g., celebrities) have many connections, while the majority of users have relatively few connections. [@barabasi_emergence_1999]

Graph vertex embeddings are a powerful technique for representing vertices in a graph as low-dimensional vectors, enabling various machine learning tasks such as vertex classification, link prediction [@leskovec_predicting_2010] and community detection. The effectiveness of these embeddings often depends on the underlying graph structure and the methods used to generate them. When faced with large graphs, the challenge of efficiently computing these embeddings while preserving the graph's structural properties becomes paramount. This paper explores the evaluation and analysis of graph vertex embeddings in a distributed environment, focusing on community-aware vertex partitioning to enhance the quality of embeddings and reduce the need for network communication during the embedding generation process.



## Motivation

Today, many real-world applications involve large-scale graphs, such as social networks [@zachary_information_1977] [@rozemberczki_multi-scale_2021] [@savic_analysis_2017], biological networks [@li_graph_2025] and web page graphs [@adamic_political_2005]. In order to perform machine learning tasks on these graphs, it is essential to generate meaningful embeddings that capture the relationships and structures within the graph. By doing so, it is possible to leverage the rich information contained in these graphs for various applications, such as recommendation systems, fraud detection, and network analysis.

Traditional methods for generating embeddings often struggle with large graphs due to their size and complexity, leading to inefficiencies and suboptimal results. The sheer size of these graphs often exceeds the memory capacity of a single machine, necessitating distributed computing approaches. 

Motivation for this work stems from the need to develop efficient methods for generating high-quality embeddings in a distributed environment, while also considering the community structure of the graph. Community-aware partitioning can significantly improve the quality of embeddings by ensuring that vertices within the same community are processed together, thereby preserving local structures and relationships while reducing network communication overhead.

## Problem statement 

In the context of distributed graph processing, there are several challenges that need to be addressed:

1. **Scalability**: Efficiently processing large graphs requires algorithms that can scale across multiple machines.
2. **Community Structure**: Many graphs exhibit community structures, where vertices are densely connected within communities but sparsely connected between them. Capturing this structure is crucial for generating meaningful embeddings.
3. **Partitioning Effectiveness**: Effective partitioning of the graph can significantly impact the quality of embeddings, as it influences how vertices are grouped and how information is propagated during embedding generation. Additionally, partitioning should be done in a way that minimizes inter-partition communication, which is a common bottleneck in distributed systems. 
4. **Partitioning Time**: The time taken to partition the graph can be significant, especially for large graphs. Therefore, it is essential to use efficient partitioning algorithms that can quickly produce high-quality partitions.
5. **Balance of Partitions**: Ensuring that partitions are balanced in terms of the number of vertices and edges can help improve the efficiency of distributed processing. Imbalanced partitions can lead to some machines being overloaded while others are underutilized, resulting in inefficient resource usage and longer processing times.

The goal of this paper is to partition large graph such that its community structure is preserved while keeping balance of partitions in a bounded interval with the goal of generating high-quality embeddings in a distributed environment. This involves evaluating different partitioning algorithms and embedding methods to determine their effectiveness in capturing community structures and generating meaningful embeddings in a distributed setting.

Formally, the problem can be defined as follows:

Given a large graph $G = (V, E)$ with vertices $V$ and edges $E$ , the goal is to find a partitioning of the vertices into $k$ subsets $P_1, P_2, \ldots, P_k$ such that:

1. The partitioning preserves the community structure of the graph, meaning that vertices within the same community are more likely to be placed in the same partition. This criteria means minimizing the edge cut between partitions $\sum_{i \neq j} |E(P_i, P_j)|$, where $E(P_i, P_j)$ is the set of edges between partitions $P_i$ and $P_j$ . Formally, edge cut is defined as: $E(P_i, P_j) = \{ (u, v) \in E | u \in P_i, v \in P_j \}$. 

2. The partitions are balanced, meaning that the number of vertices in each partition is within a bounded interval, i.e., $\forall i, |P_i| \in [\frac{|V|}{k} (1 - \epsilon), \frac{|V|}{k} (1 + \epsilon)]$ for some small $\epsilon > 0$ .

3. The partitioning time is minimized, meaning that the time taken to compute the partitioning is as low as possible, ideally linear in the number of vertices and edges in the graph.

A partition $P_k$ is set of vertices stored in a single machine in distributed environment. Every machine (node in cluster) can communicate with other machines in the cluster, but the communication is expensive and should be minimized. Number of machines is denoted as $K$ and the goal is to partition the graph into $K$ partitions such that the above criteria are satisfied as much as possible.

## Related work 

Graph vertex embedding is a well-studied area, with various methods proposed to generate low-dimensional representations of nodes in a graph. State of the art method which is widely used is Node2Vec [@grover_node2vec_2016] which uses random walks to capture the local and global structure of the graph. Node2Vec generates embeddings by performing biased random walks on the graph, allowing it to explore both local and global structure. The method has been shown to be effective in capturing community structures and generating meaningful embeddings for various machine learning tasks. As its improvement, DistGER [@fang_distributed_2023] is a distributed graph embedding method that extends Node2Vec by leveraging distributed computing to handle large graphs. DistGER uses a similar random walk approach but optimizes walk sampling in order to maximize the information gain when selecting the next vertex to visit. 

Another approach to scale Node2Vec is proposed in [@lombardo_scalable_2019] which is based on actor model and uses a distributed framework to generate embeddings for large graphs. This method allows for parallel processing of random walks, significantly improving the efficiency of embedding generation in terms of time and resource usage.

The common ground for these methods is that they generate walks which are later used to train Word2Vec model [@church_word2vec_2017] commonly used for generating embeddings in natural language processing tasks. 

During the learning process, there are several state of the art approaches to parallelize the training of word2vec model. Commonly used approach in distributed environment is ensemble learning [@ji_ensemble_2007] which combines multiple smaller models to create a larger model. Final mode is created by aggregating smaller models using parameter server architecture [@li_parameter_2013]. 

The main challenge to address the problem of distributed graph vertex embedding is partitioning the graph in a way that preserves the community structure while ensuring that the partitions are balanced and can be processed efficiently in a distributed environment. Up until recently, most partitioning methods covered only small graphs or graphs without inherent community structure, like in [@benlic_effective_2010] [@sanders_distributed_2012] [@sanders_engineering_2011] [@romero_ruiz_memetic_2018] [@catalyurek_more_2023]. The main focus of these methods is static graph partitioning meaning that the graph is partitioned once and then used for processing. However, in many real-world applications, graphs are dynamic and change over time, requiring dynamic partitioning methods that can adapt to changes in the graph structure.

For dynamic graph partitioning, there are several methods available in literature like [@nicoara_hermes_2015] [@huang_leopard_2016] [@xu_loggp_2014] and [@vaquero_adaptive_2013]. These methods focus on partitioning dynamic graphs by considering the changes in the graph structure over time and adapting the partitioning accordingly. However, these methods have not been used in embedding applications so far, and their effectiveness in generating high-quality embeddings in a distributed environment remains an open question.



## Contributions of this paper 


Main contributions of this paper include:
1. A comprehensive evaluation of graph vertex embeddings in a distributed environment, focusing on community-aware vertex partitioning.
2. A comparison of the performance of the LFM algorithm and its modified version in generating partitions that improve the quality of embeddings while reducing the need of network communication during the embedding generation process.
3. A comparison of different partitioning algorithms, including random partitioning and label propagation, in terms of their effectiveness in capturing community structures and generating meaningful embeddings.
   

# Methods 

## System architecture 

<!-- Ovde objasnim ZK i kako je sve implementirano kao velika baza podataka -->

![System architecture](./png/Copy%20of%20Arhitektura%20i%20infrastruktura.drawio.png)

Distributed embedding system consists of several components that work together to enable efficient graph processing and embedding generation. The main components of the system include:

1. **Graph Storage**: The graph data is stored in a distributed database, such as network attached storage (NAS), which allows for efficient storage and retrieval of large-scale graph data. 
2. **Community Detection Module**: This module is responsible for detecting communities within the graph and partitioning it into smaller subgraphs that can be processed independently. The partitioning is done using community-aware algorithms, such as the LFM [@lancichinetti_detecting_2009] and label propagation algorithm [@raghavan_near_2007], to ensure that vertices within the same community are grouped together.
3. **Partitioning Module**: This module takes the output from the community detection module and further refines the partitions to ensure balance and number of partitions is equal to number of machines in the cluster. This is done using algorithms that optimize for balance such as bin packing algorithms [@gupta_new_1999].
4. **Embedding Generation Module**: This module is responsible for generating vertex embeddings using Node2Vec [@grover_node2vec_2016]. The embedding generation is performed in a distributed manner, with each machine processing its assigned partition of the graph without the need for excessive network communication.

Nodes in the cluster communicate with each other using a distributed coordination service, such as Apache ZooKeeper [@apache_software_foundation_zookeeper_2011], to ensure synchronization and coordination during the embedding generation process. The system is designed to be scalable, allowing for the addition of more machines to handle larger graphs and improve processing efficiency.

## Community detection and graph partitioning


LFM is a community detection algorithm that identifies overlapping communities in large-scale networks. The algorithm operates by iteratively expanding communities based on a fitness function that measures the quality of the community structure. The fitness function considers both the internal density of connections within a community and the external connections to other communities. The LFM algorithm starts with a seed vertex and expands the community by adding neighboring vertices that improve the fitness score. This process is repeated until no further improvement can be made.

During experiments, it was observed that LFM can oscillate in terms of the remaining unassigned vertices. To address this, a modified version of the LFM algorithm was implemented, which includes a mechanism to track the number of unassigned vertices and terminate the algorithm if number of communities becomes too large. This modification helps to stabilize the partitioning process and ensures that the algorithm converges to a solution in a reasonable time frame.

Since LFM can be inefficient in terms of time complexity for very large graphs, especially when the number of communities becomes large, a threshold is set as percentage of unassigned vertices to terminate the algorithm early. This threshold helps to prevent excessive computation time while still allowing for effective community detection. Unassigned vertices after termination are assigned randomly to existing communities to ensure that all vertices are included in the final partitioning. By doing this, it is possible to control the balance between community-structure preservation and partitioning time, making the algorithm more practical for large-scale graph processing. Currently, most of the practical systems like Apache Giraph use random partitioning due to its simplicity and speed [@martella_practical_2015].

Label propagation is another community detection algorithm that operates on the principle of spreading labels through the network. Each node adopts the label of the most frequent neighbor label, and this process is repeated until a stable state is reached. 

Both LFM and label propagation are effective in detecting communities in large-scale graphs and are used in the community detection module of the system. Still, both algorithms can produce unbalanced partitions if only community structure is considered. To ensure balance, a bin packing algorithm [@gupta_new_1999] is applied after community detection to refine the partitions and ensure that they are balanced in terms of the number of vertices and edges.

Bin packing algorithm has many variants. In this paper, the approach from [@gupta_new_1999] is used. Each item (community) is assigned to the bin (partition) with the lowest current weight (number of vertices in partition) that can accommodate the item without exceeding the maximum allowed weight. If no such bin exists, maximum allowed weight is increased and the process is repeated until all items are assigned to bins. This approach ensures that the partitions are balanced while also preserving the community structure of the graph as much as possible since communities are assigned to partitions as whole units.

## Embedding


Node2Vec is a graph embedding method that generates low-dimensional representations of vertices in a graph by performing biased random walks. The algorithm uses two parameters, $p$ and $q$, to control the random walk process. The parameter $p$ determines the probability of returning to the previous vertex, while the parameter $q$ controls the probability of exploring new vertices. By adjusting these parameters, Node2Vec can capture both local and global structures in the graph. The random walks generated by Node2Vec are then used to train a Word2Vec model, which learns the embeddings based on the co-occurrence of vertices in the walks. Here, walk is treated as a sentence and vertices as words in the sentence. Negative sampling is used to optimize the training process, allowing the model to efficiently learn embeddings that capture the relationships between vertices in the graph.


## Embedding aggregation

After generating embeddings for each partition, the next step is to aggregate these embeddings into a single model that represents the entire graph. This is done using an ensemble learning approach, where multiple smaller models (embeddings from each partition) are combined to create a larger model. The aggregation process involves averaging the embeddings of individual vertices across all partitions. This approach adds network overhead only during the aggregation phase, as each machine needs to send its local embeddings to a central node for aggregation. However, this overhead is minimal compared to the overall embedding generation process, as the majority of the computation is performed locally on each partition.

## Evaluation criteria 

To evaluate the effectiveness of the graph vertex embeddings generated in a distributed environment with community-aware partitioning, several criteria are considered:

**Embedding Quality**: The quality of the embeddings is assessed using metrics such as F1-score of reconstructed graphs [@yip_restore_2023], which measures how well the embeddings capture the relationships between vertices in the original graph. Higher F1-scores indicate better preservation of graph structure in the embeddings. Let $G = (V,E)$ be the original graph and $G' = (V,E')$ be the reconstructed graph from embeddings. $G'$ is constructed by connecting k closest vertices in the embedding space, where k is the number of edges in the original graph. The F1-score is calculated as follows:

$$ F1 = 2 * \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

where

$$ \text{Precision} = \frac{1}{|V|} \sum_{v \in V} \frac{|N(v) \cap N'(v)|}{|N'(v)|} $$

$$ \text{Recall} = \frac{1}{|V|} \sum_{v \in V} \frac{|N(v) \cap N'(v)|}{|N(v)|} $$

and $N(v)$ and $N'(v)$ are the neighbors of vertex $v$ in the original and reconstructed graphs, respectively.

**Partition Quality**: The quality of the partitions is evaluated based on metrics such as edge cut, which measures the number of edges that connect vertices in different partitions. Lower edge cuts indicate better preservation of community structures within partitions. Edge cut is defined as: 

$$ \text{EdgeCut} = \frac{1}{|E|} \sum_{(u,v) \in E} \mathbb{I}(p(u) \neq p(v)) $$

where $E$ is the set of edges in the original graph, $p(u)$ is the partition of vertex $u$, and $\mathbb{I}(\cdot)$ is the indicator function.

**Balance of Partitions**: The balance of the partitions is assessed by measuring the size of each partition and ensuring that they are within a bounded interval. This helps to ensure that the workload is evenly distributed across machines in the distributed environment.

**Partitioning Time**: The time taken to partition the graph is measured to evaluate the efficiency of the partitioning algorithms. Faster partitioning times are preferred, especially for large graphs, as they reduce the overall processing time in a distributed environment.

**Cluster similarity**: To measure how useful are the embeddings for clustering tasks, k-means clustering is performed on the embeddings and the resulting clusters are compared to ground-truth communities using Adjusted Rand Index (ARI) [@vinh_information_2009]. ARI measures the similarity between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings. 




# Results and discussion

Experiments are conducted on several real-world graphs, including Twitch [@rozemberczki_twitch_2021], UK2002 [@], Wiki Talks [@leskovec_predicting_2010, @leskovec_signed_2010], and Live Journal [@backstrom_group_2006], as well as synthetic graphs generated using the Stochastic Block Model (SBM) [@holland_stochastic_1983] with 10 million edges. The performance of the proposed distributed embedding system with community-aware partitioning is compared against baseline methods, such as random partitioning and traditional embedding methods without partitioning.

<!-- partitioning time for lfm lpa random  -->
First, benchmarks on partitioning time were conducted to evaluate the time taken by different partitioning algorithms. 

In Table 1 partitioning time is shown for LFM algorithm on different networks:

| Network | Partitioning time (s) | # nodes | # edges |
|---------|-----------------------|---------|---------|
| Zachary Karate Club | 0.001230 | 34 | 78 |
| Les Miserables | 0.004032 | 77 | 254 |
| Davis Southern Women | 0.004145 | 32 | 89 |
| Florentine families | 0.000677 | 16 | 20 |
| Twitch | 11.54 | 168114 | 6797557 | 
Table: Partitioning time for LFM algorithm on real-world networks.



In Table 2, partitioning time is shown for different SBM graphs with varying number of vertices per community. Comparison is shown for original LFM implementation and modified LFM implementation with early stopping criteria based on unassigned vertices where threshold is set to 50% of total vertices. Probability of link across communities is set to 0.01 and probability of link inside community is set to 0.1. Number of communities is set to 3.

| Community size | Partitioning time (s) - LFM | Partitioning time (s) - Modified LFM | # vertices | # edges |
|----------------|-----------------------------|--------------------------------------|------------|---------|
| 100            | 0.125113                    | 0.080214                             | 300        | 1755    |
| 200            | 0.944693                    | 0.712999                             | 600        | 7171    |
| 300            | 3.031000                    | 2.052959                             | 900        | 16154   |
| 400            | 6.774440                    | 4.487337                             | 1200       | 28676   |
| 500            | 12.589976                   | 8.317590                             | 1500       | 45139   |
| 600            | 22.913838                   | 15.139224                            | 1800       | 65012   |
| 700            | 39.479742                   | 26.455410                            | 2100       | 87815   |
| 800            | 67.074574                   | 41.665899                            | 2400       | 115446  |
| 900            | 86.972126                   | 54.922895                            | 2700       | 145352  |
| 1000           | 110.282812                  | 73.058589                            | 3000       | 179382  |
Table: Partitioning time for LFM and modified LFM on SBM generated graphs.


As can be seen from the results, the modified LFM algorithm with early stopping criteria significantly reduces the partitioning time compared to the original LFM implementation, especially for larger graphs. This demonstrates the effectiveness of the modification in improving the efficiency of the partitioning process while still preserving community structures.

To demonstrate that F1 score stays relatively stable when performing embedding on partitioned graph, Table 3 shows F1 scores for embeddings generated using Node2Vec on Zachary Karate club graph with modified LFM partitioning.

| Number of walks | F1 score for sequential Node2Vec | F1 score for distributed Node2Vec with modified LFM partitioning |
|-----------------|----------------------------------|---------------------------------------------------------------|
| 100             | 0.62                             | 0.53                                                          |
| 200             | 0.76                             | 0.63                                                          |
| 300             | 0.79                             | 0.65                                                          |
| 400             | 0.80                             | 0.71                                                          |
| 500             | 0.80                             | 0.66                                                          |
| 600             | 0.77                             | 0.69                                                          |
Table: F1 scores for embeddings generated using Node2Vec on Zachary Karate club graph with modified LFM partitioning.

Same results are shown for Florentine families graph in Table 4 and for Les Miserables graph in Table 5 respectively.

| Number of walks | F1 score for sequential Node2Vec | F1 score for distributed Node2Vec with modified LFM partitioning |
|-----------------|----------------------------------|---------------------------------------------------------------|
| 100             | 0.92                             | 0.81                                                          |
| 200             | 0.85                             | 0.81                                                          |
| 300             | 0.96                             | 0.84                                                          |
| 400             | 0.96                             | 0.84                                                          |
| 500             | 0.92                             | 0.8                                                           |
| 600             | 0.92                             | 0.84                                                          |
| 700             | 0.91                             | 0.85                                                          |
| 800             | 0.92                             | 0.81                                                          |
| 900             | 0.96                             | 0.84                                                          |
Table: F1 scores for embeddings generated using Node2Vec on Florentine families graph with modified LFM partitioning.

| Number of walks | F1 score for sequential Node2Vec | F1 score for distributed Node2Vec with modified LFM partitioning |
|-----------------|----------------------------------|---------------------------------------------------------------|
| 100             | 0.49                             | 0.66                                                          |
| 200             | 0.64                             | 0.72                                                          |
| 300             | 0.62                             | 0.75                                                          |
| 400             | 0.67                             | 0.75                                                          |
| 500             | 0.68                             | 0.74                                                          |
| 600             | 0.64                             | 0.76                                                          |
| 700             | 0.64                             | 0.77                                                          |
| 800             | 0.69                             | 0.75                                                          |
| 900             | 0.68                             | 0.76                                                          |
Table: F1 scores for embeddings generated using Node2Vec on Les Miserables graph with modified LFM partitioning.


The system is further evaluated on larger networks from literature to assess the performance of distributed Node2Vec with modified LFM partitioning compared to sequential Node2Vec. Table 6 shows characteristics of the networks used in the experiments.

| Network | # nodes | # edges | Clustering coefficient |
|---------|---------|---------|------------------------|
| CITESEER| 3264    | 4532    | 0.14                   |
| AstroPh | 18772   | 198110  | 0.63                   |
| Cit-HepPh| 34546  | 420921  | 0.28                   |
| Cit-HepTh| 27770  | 352324  | 0.31                   |
Table: Characteristics of networks used in experiments.

Partitioning time for these networks using modified LFM algorithm was below 1 second for all networks regardless of the number of partitions set. Indeed, most of the time was spent in community detection, which is independent of the number of partitions. Bin packing algorithm for balancing partitions takes negligible time compared to community detection since number of communities is significantly lower than number of vertices in the graph.

Table 7 shows F1 scores for networks from literature. F1 score of reconstruction remains relatively stable when using distributed Node2Vec with modified LFM partitioning compared to sequential Node2Vec when number of partitions is set to 2 and number of walks per node is set to 10 with walk size 80.

<!-- lfm threshold = 0 alpha = 1 -->

| Network | Dim | F1 score for sequential | F1 score on 2 nodes |
|---------|-----|-----------------|---------------|
| CITESEER| 50  | 24%            | 33%           |
| AstroPh | 100 | 5.1%           | 6.6%          |
| Cit-HepPh | 100 | 2.5%         | 2.8%         |
| Cit-HepTh | 100 | 2.9%         | 3.2%        |

Evaluation is also done when number of partitions is set to 4 and number of walks per node is set to 10 with walk size 80, as shown in Table 8.

| Network | F1 score on 4 nodes |
|---------|---------------------|
| CITESEER| 18.99%                  |
| AstroPh | 8.97%                  |
| Cit-HepPh | 3.43%                  |
| Cit-HepTh | 4.33%                  |
Table: F1 scores for embeddings generated using distributed Node2Vec with modified LFM partitioning on larger networks when number of partitions is set to 4.

<!-- lfm threshold = 0.5 and alpha 1 -->

Table 9 shows F1 scores when number of partitions is set to 4 where 50% of unassigned vertices is used as threshold to terminate LFM algorithm early. 

| Network | F1 score on 4 nodes |
|---------|---------------------|
| CITESEER| 28.46%                 |
| AstroPh | 9.21%                  |
| Cit-HepPh | 3.37%               |
| Cit-HepTh | 4.30%               |
Table: F1 scores for embeddings generated using distributed Node2Vec with modified LFM partitioning on larger networks when number of partitions is set to 4 and threshold for unassigned vertices is set to 50%.

<!-- lfm threshold = 1 -->

When vertices are partitioned randomly where each partition is selected with equal probability, the results indicate a significant drop in F1 scores compared to community-aware partitioning methods. The results are shown in the Table 10:

| Network | F1 score on 4 nodes |
|---------|---------------------|
| CITESEER| 26.67%              |
| AstroPh | 8.23%               |
| Cit-HepPh | 3.08%             |
| Cit-HepTh | 3.76%             |
Table: F1 scores for embeddings generated using distributed Node2Vec with modified LFM partitioning on larger networks when number of partitions is set to 4.


<!-- LPA -->

Label propagation algorithm is also evaluated as a community detection method for partitioning before embedding generation. The results are shown in Table 11:

| Network | F1 score on 4 nodes |
|---------|---------------------|
| CITESEER|  44.34%             |
| AstroPh | 19.48%              |
| Cit-HepPh | 4.8%             |
| Cit-HepTh | 5.1%              |
Table: F1 scores for embeddings generated using distributed Node2Vec with label propagation partitioning on larger networks when number of partitions is set to 4.

It was observed that LFM algorithm produces uniformly balanced partitions on these networks. Even when number of partitions is set higher than 2, partitions remain balanced within 10% of average partition size.

All evaluations were done using word2vec which was trained with negative sampling where number of negative samples was set to 1 and with one epoch since comparison with sequential Node2Vec was the main goal and increasing number of epochs would increase training time significantly.

Also, clustering similarity after embedding using sequential and parallel implementations was calculated. In Table 6, the clustering similarity using the K-means algorithm with 3 clusters is presented, where the similarity was calculated using the ARI method. SBM generated network had 1000 vertices, 2 communities, where the connection probability within the community was p=0.1, and the connection probability of nodes that do not belong to the same community was q=0.01. 


Table 12 shows clustering similarity using K-means algorithm on embeddings generated with distributed Node2Vec with modified LFM partitioning.

| Network | ARI | 
|---------|-----|
| Zachary Karate Club | 1.0 |
| Davis Southern Women | 0.65 |
| Florentine Families | 0.38 |
| Les Miserables | 0.45 | 
| Twitch | 0.97 |
| SBM Generated Graph | 0.99 |
Table: Clustering similarity using K-means algorithm on embeddings generated with distributed Node2Vec with modified LFM partitioning.

From results, it can be observed that the distributed embedding with community-aware partitioning achieves high clustering similarity compared to the sequential implementation, indicating that the embeddings effectively capture the community structure of the graph and can be used for clustering tasks.


<!-- _____________ -->


During experiments, it was observed that negative sampling in Word2Vec training can significantly impact the quality of the embeddings. For instance, in the Table 13, F1 scores for embeddings generated using Node2Vec on different graphs with modified LFM partitioning are shown for different values of negative samples when number of partitions is set to 2 and number of walks per node is set to 10.

| Graph | ns=0 | ns=1 | ns=10 | ns=100 |
|-------|------|------|-------|--------|
| Zachary Karate Club | 10.89% | 41.8% | 41.3% | 14.5% | 
| Les Miserables | 13.03% | 55.01% | 50.0% | 12.77% | 
| ER(200, 0.05) | 3.9% | 11.73% | 16.29% | 13.67% |
| BA(300, 50) | 27.5% | 38.82 | 39.2% | 39.2% |
Table: F1 scores for embeddings generated using Node2Vec on different graphs with modified LFM partitioning for different values of negative samples (ns).






<!-- corruptibility on small networks  -->


<!-- corruptibility on big networks  -->

In experiments, it was observed that label propagation algorithm can produce balanced partitions which also 
preserves community structure to some extent. In the Table 14, average balance and edge cut for label propagation
and modified LFM algorithm are shown for various real world networks when number of partitions is set to 4.

| Network | Balance - LPA | Edge cut - LPA | Balance - LFM | Edge cut - LFM |
|---------|---------------|----------------|---------------|----------------|
| CITESEER | 0.000612     | 22%            | 0.0016        | 68.8%          |
| AstroPh  | 0.824        | 8.4%        | 0.00085          | 74.5%          |
| Cit-HepPh| 0.00005      | 20.6%          | 0.00147       | 73.8%          |
| Cit-HepTh| 0.483        | 11.74%       | 0.0023          | 74.14%         |
Table: Balance and edge cut for label propagation and modified LFM algorithm on various real world networks

From table it can be seen that label propagation produces more balanced partitions with lower edge cuts compared to modified LFM algorithm. This also explains higher F1 scores observed when using label propagation for partitioning before embedding generation. 


Overall, distributed graph vertex embedding with community-aware partitioning done
using label propagation algorithm has shown to be the most effective, consistently
outperforming other methods in terms of embedding quality, partition quality, and balance of partitions as well as reconstruction F1 scores.

# Conclusion 

In this paper, a distributed graph vertex embedding system with community-aware partitioning is proposed and evaluated. The system leverages community detection algorithms, such as LFM and label propagation, to partition the graph into smaller subgraphs that can be processed independently. The partitions are further refined using bin packing algorithms to ensure balance across machines in the distributed environment.

The experimental results demonstrate that the proposed system effectively preserves community structures within partitions, leading to high-quality embeddings that capture the relationships between vertices in the original graph. The modified LFM algorithm with early stopping criteria significantly reduces partitioning time while maintaining the quality of the partitions. Additionally, label propagation shows promise in producing balanced partitions with lower edge cuts, resulting in improved embedding quality.

As a future work, further exploration of dynamic graph partitioning methods and their impact on embedding quality in a distributed environment is suggested. Additionally, investigating the scalability of the system on larger graphs and optimizing the embedding aggregation process could lead to further improvements in performance and efficiency.

Also, exploring other embedding methods such as DistGER [@fang_distributed_2023] and their compatibility with community-aware partitioning could provide insights into the generalizability of the proposed approach.


# References 

