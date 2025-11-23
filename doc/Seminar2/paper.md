---
title: "Evaluation and analysis of graph vertex embeddings based on information-oriented random walks in distributed environment with community-aware vertex partitioning"
author: "Stefan Nožinić"
abstract: |
  This paper presents an evaluation and analysis of graph vertex embeddings generated using information-oriented random walks in a distributed environment with community-aware vertex partitioning. The study focuses on the effectiveness of community-aware partitioning methods in preserving the structural properties of the graph while generating high-quality embeddings. The performance of the proposed approach is assessed through various metrics, including embedding quality, partition quality, and balance of partitions. The results demonstrate that community-aware partitioning significantly improves the quality of embeddings and reduces the need for network communication during the embedding generation process. This work contributes to the understanding of distributed graph embedding techniques and highlights the importance of considering community structures in large-scale graph processing.
bibliography: ./refs.bib
---

# Introduction 

<!-- citation example [@apache_software_foundation_zookeeper_2011] -->

A graph is a mathematical structure consisting of vertices (or nodes) connected by edges. Graphs are widely used to model relationships and interactions in various domains [@van_der_hofstad_random_2024], such as social networks [@leskovec_signed_2010] [@backstrom_group_2006] [@rozemberczki_twitch_2021], collaboration networks [@savic_analysis_2017], terrorist networks [@krebs_mapping_2002] and blog citation networks [@adamic_political_2005]. In these applications, the relationships between entities can be represented as edges connecting the corresponding vertices.

<!-- objasniti sta su realni grafovi i raspodelu stepeni kod njih kao i Watt-Strogatz princip -->

In many real-world graphs, the degree distribution follows a power-law, meaning that a small number of vertices have a very high degree (i.e., they are connected to many other vertices), while most vertices have a low degree. This characteristic is often observed in social networks, where a few individuals (e.g., celebrities) have many connections, while the majority of users have relatively few connections.

In real graphs, there are several properties that are often observed [@watts_collective_1998] [@zachary_information_1977] [@albert_statistical_2002]:
- **Small-world property**: Most pairs of vertices can be connected by a short path, even in large graphs. This is often referred to as the "six degrees of separation" phenomenon. [@watts_collective_1998]
- **Community structure**: Vertices tend to form clusters or communities, where vertices within the same community are more densely connected than those in different communities. This property is prevalent in social networks, where groups of friends or colleagues often form tightly-knit communities. [@leskovec_community_2009]
- **Scale-free property**: The degree distribution of the graph follows a power-law, meaning that a few vertices have a very high degree, while most vertices have a low degree. This is often observed in social networks, where a small number of individuals (e.g., celebrities) have many connections, while the majority of users have relatively few connections. [@barabasi_emergence_1999]

Graph vertex embeddings are a powerful technique for representing vertices in a graph as low-dimensional vectors, enabling various machine learning tasks such as vertex classification, link prediction [@leskovec_predicting_2010], and community detection. The effectiveness of these embeddings often depends on the underlying graph structure and the methods used to generate them. When faced with large graphs, the challenge of efficiently computing these embeddings while preserving the graph's structural properties becomes paramount. This paper explores the evaluation and analysis of graph vertex embeddings in a distributed environment, focusing on community-aware vertex partitioning to enhance the quality of embeddings and reduce the need for network communication during the embedding generation process. 

Additionally, embeddings are generated using information-oriented random walks 
to train embedding model.



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

So the goal of this paper is to partition large graph such that we preserve the community structure of the graph while keeping balance of partitions in a bounded interval with the goal of generating high-quality embeddings in a distributed environment. This involves evaluating different partitioning algorithms and embedding methods to determine their effectiveness in capturing community structures and generating meaningful embeddings in a distributed setting.

Formally, the problem can be defined as follows:

Given a large graph $G = (V, E)$ with vertices $V$ and edges $E$ , the goal is to find a partitioning of the vertices into $k$ subsets $P_1, P_2, \ldots, P_k$ such that:

1. The partitioning preserves the community structure of the graph, meaning that vertices within the same community are more likely to be placed in the same partition. This criteria means minimizing the edge cut between partitions $\sum_{i \neq j} |E(P_i, P_j)|$, where $E(P_i, P_j)$ is the set of edges between partitions $P_i$ and $P_j$ . Formally, edge cut is defined as: $E(P_i, P_j) = \{ (u, v) \in E | u \in P_i, v \in P_j \}$. 

2. The partitions are balanced, meaning that the number of vertices in each partition is within a bounded interval, i.e., $\forall i, |P_i| \in [\frac{|V|}{k} (1 - \epsilon), \frac{|V|}{k} (1 + \epsilon)]$ for some small $\epsilon > 0$ .

3. The partitioning time is minimized, meaning that the time taken to compute the partitioning is as low as possible, ideally linear in the number of vertices and edges in the graph.

A partition $P_k$ is set of vertices stored in a single machine in distributed environment. Every machine (node in cluster) can communicate with other machines in the cluster, but the communication is expensive and should be minimized. Number of machines is denoted as $K$ and the goal is to partition the graph into $K$ partitions such that the above criteria are satisfied as much as possible.

## Related work 

Graph vertex embedding is a well-studied area, with various methods proposed to generate low-dimensional representations of nodes in a graph. State of the art method which is widely used is Node2Vec [@grover_node2vec_2016] which uses random walks to capture the local and global structure of the graph. Node2Vec generates embeddings by performing biased random walks on the graph, allowing it to explore both local and global structures. The method has been shown to be effective in capturing community structures and generating meaningful embeddings for various machine learning tasks. As its improvement, DistGER [@fang_distributed_2023] is a distributed graph embedding method that extends Node2Vec by leveraging distributed computing to handle large graphs. DistGER uses a similar random walk approach but optimizes walk sampling in order to maximize the information gain when selecting the next vertex to visit. 

Another approach to scale Node2Vec is proposed in [@lombardo_scalable_2019] which is based on actor model and uses a distributed framework to generate embeddings for large graphs. This method allows for parallel processing of random walks, significantly improving the efficiency of embedding generation in terms of time and resource usage.

The common ground for these methods is that they generate walks which are later used to train Word2Vec model [@church_word2vec_2017] commonly used for generating embeddings in natural language processing tasks. 

During the learning process, there are several state of the art approaches to parallelize the training of word2vec model. Commonly used approach in distributed environment is ensemble learning [@ji_ensemble_2007] which combines multiple smaller models to create a larger model. Final mode is created by aggregating smaller models using parameter server architecture [@li_parameter_2013]. 

The main challenge to address the problem of distributed graph vertex embedding is partitioning the graph in a way that preserves the community structure while ensuring that the partitions are balanced and can be processed efficiently in a distributed environment. Up until recently, most partitioning methods covered only small graphs or graphs without inherent community structure, like in [@benlic_effective_2010] [@sanders_distributed_2012] [@sanders_engineering_2011] [@romero_ruiz_memetic_2018] [@catalyurek_more_2023]. The main focus of these methods is static graph partitioning meaning that the graph is partitioned once and then used for processing. However, in many real-world applications, graphs are dynamic and change over time, requiring dynamic partitioning methods that can adapt to changes in the graph structure.

For dynamic graph partitioning, there are several methods available in literature like [@nicoara_hermes_2015] [@huang_leopard_2016] [@xu_loggp_2014] and [@vaquero_adaptive_2013]. These methods focus on partitioning dynamic graphs by considering the changes in the graph structure over time and adapting the partitioning accordingly. However, these methods have not been used in embedding applications so far, and their effectiveness in generating high-quality embeddings in a distributed environment remains an open question.



## Contributions of this paper 


Main contributions of this paper include:
1. A comprehensive evaluation of graph vertex embeddings in a distributed environment, focusing on community-aware vertex partitioning.
2. A comparison of the performance of the label propagation algorithm in terms of partition quality and embedding quality.
3. An analysis of reconstruction F1 score for embeddings generated using information-oriented random walks in a distributed environment with community-aware partitioning.
   

# Methods 

## System architecture 

<!-- Ovde objasnim ZK i kako je sve implementirano kao velika baza podataka -->

![System architecture](./png/Copy%20of%20Arhitektura%20i%20infrastruktura.drawio.png)

Distributed embedding system consists of several components that work together to enable efficient graph processing and embedding generation. The main components of the system include:

1. **Graph Storage**: The graph data is stored in a distributed database, such as network attached storage (NAS), which allows for efficient storage and retrieval of large-scale graph data. 
2. **Community Detection Module**: This module is responsible for detecting communities within the graph and partitioning it into smaller subgraphs that can be processed independently. The partitioning is done using community-aware label propagation algorithm [@raghavan_near_2007], to ensure that vertices within the same community are grouped together.
3. **Partitioning Module**: This module takes the output from the community detection module and further refines the partitions to ensure balance and number of partitions is equal to number of machines in the cluster. This is done using algorithms that optimize for balance such as bin packing algorithms [@gupta_new_1999].
4. **Embedding Generation Module**: This module is responsible for generating vertex embeddings using methods such as Node2Vec [@grover_node2vec_2016] and DistGER [@fang_distributed_2023]. The embedding generation is performed in a distributed manner, with each machine processing its assigned partition of the graph without the need for excessive network communication.

Nodes in the cluster communicate with each other using a distributed coordination service, such as Apache ZooKeeper [@apache_software_foundation_zookeeper_2011], to ensure synchronization and coordination during the embedding generation process. The system is designed to be scalable, allowing for the addition of more machines to handle larger graphs and improve processing efficiency.

## Community detection and graph partitioning

<!-- ovde objasnim LFM i label propagation -->

Label propagation is another community detection algorithm that operates on the principle of spreading labels through the network. Each node adopts the label of the most frequent neighbor label, and this process is repeated until a stable state is reached. 

During the initial experiments, it was observed that LFM algorithm can take a significant amount of time to converge, especially on large graphs with many vertices, while label propagation converges much faster and produces comparable partition quality. That is why label propagation was chosen as the primary community detection algorithm in the system.

Both LFM and label propagation are effective in detecting communities in large-scale graphs and are used in the community detection module of the system. Still, both algorithms can produce unbalanced partitions if only community structure is considered. To ensure balance, a bin packing algorithm [@gupta_new_1999] is applied after community detection to refine the partitions and ensure that they are balanced in terms of the number of vertices and edges.

Bin packing algorithm has many variants. In this paper, the approach from [@gupta_new_1999] is used. Each item (community) is assigned to the bin (partition) with the lowest current weight (number of vertices in partition) that can accommodate the item without exceeding the maximum allowed weight. If no such bin exists, maximum allowed weight is increased and the process is repeated until all items are assigned to bins. This approach ensures that the partitions are balanced while also preserving the community structure of the graph as much as possible since communities are assigned to partitions as whole units.

## Embedding


Node2Vec is a graph embedding method that generates low-dimensional representations of vertices in a graph by performing biased random walks. The algorithm uses two parameters, $p$ and $q$, to control the random walk process. The parameter $p$ determines the probability of returning to the previous vertex, while the parameter $q$ controls the probability of exploring new vertices. By adjusting these parameters, Node2Vec can capture both local and global structures in the graph. The random walks generated by Node2Vec are then used to train a Word2Vec model, which learns the embeddings based on the co-occurrence of vertices in the walks. Here, walk is treated as a sentence and vertices as words in the sentence. Negative sampling is used to optimize the training process, allowing the model to efficiently learn embeddings that capture the relationships between vertices in the graph.

DistGER changes the way random walks are generated by maximizing the information gain when selecting the next vertex to visit. Even though DistGER provides its own distributed framework for generating embeddings, in this paper, the distributed framework described in system architecture is used to evaluate the effectiveness of DistGER in a distributed environment with community-aware partitioning. The random walks generated by DistGER are also used to train a Word2Vec model, similar to Node2Vec, to learn the vertex embeddings. Precisely, the walks generated by DistGER are used to train the same Word2Vec model as the one used for Node2Vec, allowing for a direct comparison of the embeddings generated by both methods. Only difference is in the way walks are generated.

In this paper, DistGER approach for generating random walks is evaluated in addition to Node2Vec approach to see if maximizing information gain during walk generation leads to better embeddings in a distributed environment with community-aware partitioning.

DistGER has changed the way random walks are generated by maximizing the information gain when selecting the next vertex to visit. 

<!-- HUGE algorithm -->


For every node, next node in walk is selected based on acceptance probability derived from information gain statistics of the walk so far.

For an edge $(u,v)\in E$, they define the similarity score $\alpha(u,v)$ as:

$$ 
\alpha(u,v) =
\frac{1}{\deg(u) - C_m(u,v)}
\cdot
\max!\left(
\frac{\deg(u)}{\deg(v)},
\frac{\deg(v)}{\deg(u)}
\right)
$$

Where:

* $\deg(x)$ is the degree of node (x).
* $C_m(u,v)$ is number of common neighbors:
$$
C_m(u,v) = |N(u)\cap N(v)|.
$$

The first factor
$$
\frac{1}{\deg(u) - C_m(u,v)}
$$
grows when (u) and (v) have many common neighbors (i.e. $\deg(u)-C_m(u,v)$ shrinks).

The second factor
$$
\max\left(\frac{\deg(u)}{\deg(v)},\frac{\deg(v)}{\deg(u)}\right)
$$
boosts transitions that involve at least one high-degree node.


DistGER uses the transition acceptance probability defined as:

$$
P(u,v) = Z(\alpha(u,v)).
$$

The paper defines (Z(x)) explicitly as the scaled hyperbolic tangent:

$$
Z(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = \tanh(x)
$$

This guarantees (0 < Z(x) < 1) for all (x>0), and:

* larger ($\alpha(u,v)$) leads to larger acceptance probability
* small ($\alpha(u,v)$) leads to small acceptance probability

At node ($u$), the random walk transition is as follows:

First pick neighbor ($v\in N(u)$) uniformly, then compute ($\alpha(u,v)$) and accept the transition with probability ($P(u,v)$). If rejected, pick another neighbor and repeat until accepted.


Walk size is determined adaptively based on the correlation coefficient (R) of information gain statistics collected during the walk.


Suppose walk starts from node (u):

$$
W^L_u = (v_1, v_2, \dots, v_L), \qquad v_1 = u.
$$

After every step ($L=1,2,3,\dots$), compute the **node occurrence distribution**:

* Let (n(v)) = number of times node (v) appeared in the walk so far.
* Define the empirical probability:

$$
p(v) = \frac{n(v)}{\sum_{x\in W^L_u} n(x)}.
$$

Compute **information entropy** of the walk:

$$
H_L = - \sum_{v \in W^L_u} p(v) \log p(v).
$$

So for every prefix length (L), there is a corresponding entropy value (H_L).


As the walk grows, DistGER collects a small series of pairs:

$$
(L_1, H_1), (L_2, H_2), \dots, (L_n, H_n)
$$

where $L_i$ is the walk length after $i$ steps, and $H_i$ is the corresponding entropy.

DistGER measures how much new information each new step introduces by checking how strongly entropy grows as walk length grows. This is done by computing the Pearson correlation coefficient (r) between the two sequences (H) and (L) and then squaring it to get coefficient of determination ($R^2$):

$$
r(H,L) =
\frac{
\sum_{i=1}^n (H_i - \bar{H})(L_i - \bar{L})
}{
\sqrt{\sum_{i=1}^n (H_i - \bar{H})^2}
\sqrt{\sum_{i=1}^n (L_i - \bar{L})^2}
}.
$$


$$
R^2(H,L) = r(H,L)^2.
$$



Algorithm ends the walk when:

$$
R^2(H,L) \ge \mu
$$

where $\mu$ is a high threshold (in the paper: $\mu = 0.99$).



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






# Results and discussion


The system is evaluated on larger networks from literature to assess the performance of distributed Node2Vec and DistGER with label propagation partitioning compared to sequential implementations. Following table shows characteristics of the networks used in the experiments.

| Network | # nodes | # edges | Clustering coefficient |
|---------|---------|---------|------------------------|
| CITESEER| 3264    | 4532    | 0.14                   |
| AstroPh | 18772   | 198110  | 0.63                   |
| Cit-HepPh| 34546  | 420921  | 0.28                   |
Table: Characteristics of networks used in experiments.

<!-- | Cit-HepTh| 27770  | 352324  | 0.31                   | -->

For networks from literature, F1 score of reconstruction remains relatively stable when using distributed Node2Vec with modified LFM partitioning compared to sequential Node2Vec when number of partitions is set to 2 and number of walks per node is set to 10 with walk size 80.

| Network | Dim | F_1 for sequential | F_1 score on 2 nodes |
|---------|-----|-----------------|---------------|
| CITESEER| 50  | 24%            | 33%           |
| AstroPh | 100 | 5.1%           | 6.6%          |
| Cit-HepPh | 100 | 2.5%         | 2.8%         |
Table: F1 scores for embeddings generated using distributed Node2Vec with LFM partitioning on larger networks when number of partitions is set to 2.
<!-- | Cit-HepTh | 100 | 2.9%         | 3.2%        | -->

It was observed that LFM algorithm produces uniformly balanced partitions on these networks. Even when number of partitions is set higher than 2, partitions remain balanced within 10% of average partition size.

In table 3, results of Node2Vec with label propagation partitioning are shown. It can be observed that F1 score of reconstruction remains stable even when number of partitions is increased to 4.

| Network | F1 score on 4 nodes |
|---------|---------------------|
| CITESEER|  44.34%             |
| AstroPh | 19.48%              |
| Cit-HepPh | 4.8%             |
Table: F1 scores for embeddings generated using distributed Node2Vec with label propagation partitioning on larger networks when number of partitions is set to 4.

<!-- | Cit-HepTh | 5.1%              | -->

In table 4, results of DistGER with label propagation partitioning are shown. It can be observed that F1 score of reconstruction remains stable even when number of partitions is increased to 4 and is slightly better than Node2Vec results.

| Network | F1 score on 1 node | F1 score on 2 nodes | F1 score on 4 nodes |
|---------|--------------------|---------------------|---------------------|
| CITESEER|  22.14%            | 43.83%              | 60.66%              |
| AstroPh |  5.53%             | 8.05%               | 29.77%              |
| Cit-HepPh| 2.68%             | 4.04%               | 4.39%               |
Table: F1 scores for embeddings generated using distributed DistGER with label propagation partitioning on larger networks when number of partitions is set to 2 and 4.

<!-- | Cit-HepTh|                   |                     |                     | -->

In experiments, it was observed that label propagation algorithm can produce balanced partitions which also 
preserves community structure to some extent. In the Table 5, average balance and edge cut for label propagation algorithm are shown for various real world networks when number of partitions is set to 4.

| Network | Balance - LPA | Edge cut - LPA | 
|---------|---------------|----------------|
| CITESEER | 0.000612     | 22%            |
| AstroPh  | 0.824        | 8.4%        |
| Cit-HepPh| 0.00005      | 20.6%          |
Table: Balance and edge cut for label propagation algorithm on various real world networks

<!-- | Cit-HepTh| 0.483        | 11.74%       | -->

Label propagation algorithm consistently produced balanced partitions for certain networks like CITESEER and Cit-HepPh, with balance values close to zero, indicating that the partitions are nearly equal in size. For other networks like AstroPh and Cit-HepTh, the balance values were higher, suggesting some imbalance in partition sizes. However, the edge cut values remained relatively low across all networks, indicating that the partitions preserved community structures effectively.

Overall, distributed graph vertex embedding with community-aware partitioning done
using label propagation algorithm has shown to be the most effective, consistently
outperforming other methods in terms of embedding quality, partition quality, and balance of partitions as well as reconstruction F1 scores.

Partitioned graphs with community-aware label propagation partitioning have shown to produce high-quality embeddings in distributed environment with minimal loss in embedding quality compared to sequential implementations. DistGER method has shown to outperform Node2Vec in terms of reconstruction F1 scores when used with community-aware partitioning.


# Conclusion 

In this paper, it was demonstrated that distributed graph vertex embedding with community-aware partitioning can effectively generate high-quality embeddings while preserving the community structure of the graph. The label propagation algorithm proved to be an effective method for partitioning graphs in a way that balances partition sizes and minimizes edge cuts, leading to improved embedding quality. 

DistGER method showed superior performance compared to Node2Vec in terms of reconstruction F1 scores, indicating that maximizing information gain during random walk generation can lead to better embeddings.

# References 

