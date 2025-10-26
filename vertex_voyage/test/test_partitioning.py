

import unittest
from vertex_voyage.partitioning import calculate_corruptability, calculate_partitioning_corruption, partition_graph, modified__lfm
import networkx as nx
from vertex_voyage.node2vec import Node2Vec
import numpy as np 
from vertex_voyage.reconstruction import get_f1_score, reconstruct
import vertex_voyage_native as vvn 

class TestPartitioning(unittest.TestCase):
    
    def test_partitioning(self):
        zachary = nx.karate_club_graph()
        partition_num = 2
        communities = partition_graph(zachary, partition_num)
        self.assertEqual(len(communities), partition_num)
        # approximately small differences between partitions )less than 10%_
        self.assertLessEqual(len(communities[0]), len(zachary.nodes) * 0.75)
        self.assertLessEqual(len(communities[1]), len(zachary.nodes) * 0.75)
    
    def test_partitioning_on_sbm(self):
        n = 100
        p = 0.1
        q = 0.01
        G = nx.planted_partition_graph(2, n, p, q, seed=42)
        partition_num = 2
        communities = partition_graph(G, partition_num)
        self.assertEqual(len(communities), partition_num)
        # approximately small differences between partitions )less than 10%_
        self.assertLessEqual(len(communities[0]), len(G.nodes) * 0.60)
    
    def test_partitioned_n2v(self):
        dim = 128
        zachary = nx.karate_club_graph()
        nodes = list(zachary.nodes())
        # rename nodes to be 0,...,N-1
        mapping = {}
        for i, node in enumerate(nodes):
            mapping[node] = i
        zachary = nx.relabel_nodes(zachary, mapping)
        partition_num = 2
        communities = partition_graph(zachary, partition_num, use_modified_lfm=True, threshold=0)
        n2v_full = Node2Vec(dim=dim, negative_sample_num=1)
        n2v_full.fit(zachary)

        n2v_partitioned = []
        for community in communities:
            n2v_partitioned.append(Node2Vec(
                dim=dim,
                walk_size=n2v_full.walk_size,
                n_walks=n2v_full.n_walks,
                epochs=n2v_full.epochs,
                p=n2v_full.p,
                q=n2v_full.q,
                learning_rate=n2v_full.learning_rate,
                negative_sample_num=n2v_full.negative_sample_num,
                seed=n2v_full.seed,
                window_size=n2v_full.window_size
            ))
            pg = zachary.subgraph(community)
            n2v_partitioned[-1].fit(pg, nodes = zachary.nodes())
        
        embeddings = {}
        for comm in communities:
            for node in comm:
                myemb = n2v_partitioned[communities.index(comm)]
                if node not in embeddings:
                    embeddings[node] = [] 
                embeddings[node].append(myemb.embed_node(node))
        for node in embeddings:
            embeddings[node] = np.mean(embeddings[node], axis=0)
        k = len(zachary.edges())
        G = zachary
        result = [] 
        for node in G.nodes():
            result.append(embeddings[node])
        reconstructed_graph = reconstruct(k, result, list(G.nodes()))
        nodes = G.nodes()
        f1 = get_f1_score(G, reconstructed_graph)
        self.assertGreaterEqual(f1, 0.40)

    def test_partitioning_corruption(self):
        zachary = nx.karate_club_graph()
        partition_num = 2
        communities = partition_graph(zachary, partition_num)
        corruption = calculate_partitioning_corruption(zachary, communities)
        self.assertGreaterEqual(corruption, 0)
        self.assertLessEqual(corruption, 1.0)
    
    def test_calculate_corruptability(self):
        zachary = nx.karate_club_graph()
        partition_num = 2
        corruptability = calculate_corruptability(zachary, partition_num)
        self.assertGreaterEqual(corruptability, 0)
        self.assertLessEqual(corruptability, 1.0)

        self.assertEqual(calculate_corruptability(zachary, 1), 0)

        x = list(range(1, 100))
        y = [calculate_corruptability(zachary, i) for i in x]
        self.assertEqual(len(x), len(y))
        # do linear regression to check if the corruptability is increasing 
        # with the number of partitions
        from sklearn.linear_model import LinearRegression
        x = np.array(x).reshape(-1, 1)-1
        y = np.array(y).reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        self.assertGreaterEqual(reg.coef_[0][0], 0)
        print(reg.coef_)
    
    def test_modified_lfm(self):
        n = 100
        p = 0.1
        q = 0.01
        G = nx.planted_partition_graph(2, n, p, q, seed=42)
        partition_num = 2
        communities = modified__lfm(G, partition_num, seed=42, pm_k=10)
        self.assertEqual(len(communities), partition_num)
    
    def test_modified_lfm_two_node_graph(self):
        G = nx.Graph()
        G.add_node(0)
        G.add_node(1)
        G.add_edge(0, 1)
        partition_num = 2
        communities = modified__lfm(G, partition_num, seed=42, pm_k=10)
        self.assertEqual(len(communities), partition_num)
        self.assertEqual(len(communities[0]), 2)

    def test_modified_lfm_complete_graph(self):
        G = nx.complete_graph(10)
        partition_num = 2
        communities = modified__lfm(G, partition_num, seed=42, pm_k=10)
        self.assertEqual(len(communities), partition_num)
        self.assertEqual(len(communities[0]), 10)
        self.assertEqual(len(communities[1]), 0)

if __name__ == '__main__':
    unittest.main()