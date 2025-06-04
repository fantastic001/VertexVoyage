---
bibliography: ./refs.bib
---

# Evaluation and analysis of graph vertex embeddings in distributed environment with community-aware vertex partitioning

## Introduction 

<!-- citation example [@apache_software_foundation_zookeeper_2011] -->

A graph is a mathematical structure consisting of vertices (or nodes) connected by edges. Graphs are widely used to model relationships and interactions in various domains [@van_der_hofstad_random_2024], such as social networks [@leskovec_signed_2010] [@backstrom_group_2006] [@rozemberczki_twitch_2021], collaboration networks [@savic_analysis_2017], terrorist networks [@krebs_mapping_2002] and blog citation networks [@adamic_political_2005]. In these applications, the relationships between entities can be represented as edges connecting the corresponding vertices.

<!-- objasniti sta su realni grafovi i raspodelu stepeni kod njih kao i Watt-Strogatz princip -->

In many real-world graphs, the degree distribution follows a power-law, meaning that a small number of vertices have a very high degree (i.e., they are connected to many other vertices), while most vertices have a low degree. This characteristic is often observed in social networks, where a few individuals (e.g., celebrities) have many connections, while the majority of users have relatively few connections.

In real graphs, there are several properties that are often observed [@watts_collective_1998] [@zachary_information_1977] [@albert_statistical_2002]:
- **Small-world property**: Most pairs of vertices can be connected by a short path, even in large graphs. This is often referred to as the "six degrees of separation" phenomenon. [@watts_collective_1998]
- **Community structure**: Vertices tend to form clusters or communities, where vertices within the same community are more densely connected than those in different communities. This property is prevalent in social networks, where groups of friends or colleagues often form tightly-knit communities. [@leskovec_community_2009]
- **Scale-free property**: The degree distribution of the graph follows a power-law, meaning that a few vertices have a very high degree, while most vertices have a low degree. This is often observed in social networks, where a small number of individuals (e.g., celebrities) have many connections, while the majority of users have relatively few connections. [@barabasi_emergence_1999]

Graph vertex embeddings are a powerful technique for representing nodes in a graph as low-dimensional vectors , enabling various machine learning tasks such as node classification, link prediction [@leskovec_predicting_2010], and community detection. The effectiveness of these embeddings often depends on the underlying graph structure and the methods used to generate them. When faced with large graphs, the challenge of efficiently computing these embeddings while preserving the graph's structural properties becomes paramount. This paper explores the evaluation and analysis of graph vertex embeddings in a distributed environment, focusing on community-aware vertex partitioning to enhance the quality of embeddings.



### Motivation

Today, many real-world applications involve large-scale graphs, such as social networks, biological networks, and knowledge graphs. In order to perform machine learning tasks on these graphs, it is essential to generate meaningful embeddings that capture the relationships and structures within the graph. By doing so, it is possible to leverage the rich information contained in these graphs for various applications, such as recommendation systems, fraud detection, and network analysis.

Traditional methods for generating embeddings often struggle with large graphs due to their size and complexity, leading to inefficiencies and suboptimal results. The sheer size of these graphs often exceeds the memory capacity of a single machine, necessitating distributed computing approaches. 

Motivation for this work stems from the need to develop efficient methods for generating high-quality embeddings in a distributed environment, while also considering the community structure of the graph. Community-aware partitioning can significantly improve the quality of embeddings by ensuring that nodes within the same community are processed together, thereby preserving local structures and relationships while reducing network communication overhead.

### Problem statement 

In the context of distributed graph processing, there are several challenges that need to be addressed:

1. **Scalability**: Efficiently processing large graphs requires algorithms that can scale across multiple machines.
2. **Community Structure**: Many graphs exhibit community structures, where nodes are densely connected within communities but sparsely connected between them. Capturing this structure is crucial for generating meaningful embeddings.
3. **Partitioning Effectiveness**: Effective partitioning of the graph can significantly impact the quality of embeddings, as it influences how nodes are grouped and how information is propagated during embedding generation. Additionally, partitioning should be done in a way that minimizes inter-partition communication, which is a common bottleneck in distributed systems. 
4. **Partitioning Time**: The time taken to partition the graph can be significant, especially for large graphs. Therefore, it is essential to use efficient partitioning algorithms that can quickly produce high-quality partitions.
5. **Balance of Partitions**: Ensuring that partitions are balanced in terms of the number of nodes and edges can help improve the efficiency of distributed processing. Imbalanced partitions can lead to some machines being overloaded while others are underutilized, resulting in inefficient resource usage and longer processing times.

### Related work 



### Contributions of this paper 

<!-- 
evaluacija na velikim mrezama
node2vec i distger evaluacija
lfm i modifikovani lfm i njihova evaluacija
 -->

Main contributions of this paper include:
1. A comprehensive evaluation of graph vertex embeddings in a distributed environment, focusing on community-aware vertex partitioning.
2. An analysis of the effectiveness of different embedding methods, including node2vec and DistGER, in capturing community structures.
3. A comparison of the performance of the LFM algorithm and its modified version in generating partitions that improve the quality of embeddings while reducing the need of network communication during the embedding generation process.
   

## Methods 

### System architecture 

<!-- Ovde objasnim ZK i kako je sve implementirano kao velika baza podataka -->

### Partitioning 

<!-- ovde objasnim LFM -->

### Embedding

<!-- ukratko objasnim node2vec i distger  -->

### Evaluation criteria 

<!-- Ovde objasnim kako su benchmarkovi radjeni -->

## Results and discussion

<!-- ovde one moje benchmarkove stavim -->

## Conclusion 

## References 

<!-- 
sve iz prethodnog rada
KaHP 
node2vec 
distger 
lfm algoritam 
 -->


