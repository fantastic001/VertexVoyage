

import unittest
from vertex_voyage.partitioning import calculate_corruptability, calculate_partitioning_corruption, partition_graph
import networkx as nx
from vertex_voyage.node2vec import Node2Vec
import numpy as np 
from vertex_voyage.reconstruction import reconstruct


class TestPartitioning(unittest.TestCase):
    
    def test_partitioning(self):
        zachary = nx.karate_club_graph()
        partition_num = 2
        communities = partition_graph(zachary, partition_num)
        self.assertEqual(len(communities), partition_num)
        # approximately small differences between partitions )less than 10%_
        self.assertLessEqual(len(communities[0]), len(zachary.nodes) * 0.60)
        self.assertLessEqual(len(communities[1]), len(zachary.nodes) * 0.60)
    
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
        partition_num = 2
        communities = partition_graph(zachary, partition_num)
        n2v_full = Node2Vec(
            dim=dim, 
            walk_size=80, 
            n_walks=100, 
            window_size=10,
            epochs=10, 
            p = .25,
            q = 4,
            negative_sample_num=10, # in practice, should be 500
            seed=42,
            learning_rate=0.01,
        )
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
            pg = nx.Graph()
            pg.add_nodes_from(community)
            pg.add_edges_from(zachary.subgraph(community).edges)
            n2v_partitioned[-1].fit(pg, nodes = zachary.nodes())
        
        W = np.zeros([len(zachary.nodes()), dim], dtype=np.float32)
        for part in n2v_partitioned:
            W = W + part.W
        W /= partition_num
        node = np.zeros([len(zachary.nodes())], dtype=np.float32)
        node[0] = 1
        emb = np.dot(node, W)
        emb_full = np.dot(node, n2v_full.W)
        embeddings = []
        for i in range(len(zachary.nodes())):
            code = np.zeros([len(zachary.nodes())], dtype=np.float32)
            code[i] = 1
            embeddings.append(np.dot(code, W))
        k = len(zachary.edges())
        G = zachary
        reconstructed_graph = reconstruct(k, embeddings)
        nodes = G.nodes()
        recall = sum([len(set(G.neighbors(n)).intersection(reconstructed_graph.neighbors(n))) / len(list(G.neighbors(n))) for n in nodes]) / len(G.nodes())
        precision = sum([len(set(G.neighbors(n)).intersection(reconstructed_graph.neighbors(n))) / len(list(reconstructed_graph.neighbors(n))) for n in nodes if len(list(reconstructed_graph.neighbors(n))) > 0]) / len([n for n in G.nodes() if len(list(reconstructed_graph.neighbors(n))) > 0])
        f1 = 2 * (precision * recall) / (precision + recall)
        self.assertGreaterEqual(f1, 0.41)

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