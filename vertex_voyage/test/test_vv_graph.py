
from unittest import TestCase

from vertex_voyage.reconstruction import reconstruct
from vertex_voyage.vv_graph import VVGraph
from vertex_voyage.partitioning import random_partitioning, label_propagation_partitioner

import networkx as nx
from vertex_voyage.node2vec import Node2Vec

class TestVVGraph(TestCase):
    def test_add_and_remove_edge(self):
        g = VVGraph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        self.assertEqual(g.degree(1), 2)
        self.assertEqual(g.degree(2), 1)
        self.assertEqual(g.degree(3), 1)
        g.remove_edge(1, 2)
        self.assertEqual(g.degree(1), 1)
        self.assertEqual(g.degree(2), 0)
        self.assertEqual(g.degree(3), 1)
        self.assertEqual(g.number_of_nodes(), 3)
    
    def test_subgraph(self):
        g = VVGraph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        subg = g.subgraph({1, 2, 3})
        self.assertEqual(subg.number_of_nodes(), 3)
        self.assertEqual(subg.number_of_edges(), 3)
        self.assertEqual(subg.degree(1), 2)
        self.assertEqual(subg.degree(2), 2)
        self.assertEqual(subg.degree(3), 2)
        self.assertEqual(subg.degree(4), 0)  # Node 4 is not in the subgraph
    
    def test_random_partitioning(self):
        g = VVGraph()
        edges = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6)]
        for u, v in edges:
            g.add_edge(u, v)
        partitions = random_partitioning(g, 2)
        self.assertEqual(len(partitions), 2)
        all_nodes = set()
        for part in partitions:
            all_nodes.update(part)
        self.assertEqual(all_nodes, g.nodes)
    
    def test_label_propagation_partitioner(self):
        g = VVGraph()
        edges = [(1, 2), (1, 3), (2, 3), (4, 5), (5, 6), (6, 4)]
        for u, v in edges:
            g.add_edge(u, v)
        partitions = label_propagation_partitioner(g, partition_num=2)
        self.assertEqual(len(partitions), 2)
        all_nodes = set()
        for part in partitions:
            all_nodes.update(part)
        self.assertEqual(all_nodes, g.nodes)

    def test_zachary_node2vec(self):
        G = nx.karate_club_graph()
        vv_graph = VVGraph()
        for u, v in G.edges():
            vv_graph.add_edge(u, v)
        node2vec = Node2Vec(
            dim=128, 
            walk_size=80, 
            n_walks=100, 
            window_size=10,
            epochs=10, 
            p = .25,
            q = 4,
            negative_sample_num=10, # in practice, should be 500
            seed=42,
            learning_rate=0.01,
            use_threads=False
        )
        node2vec.fit(vv_graph)
        # calculate embeddings
        embeddings = node2vec.embed_nodes(list(vv_graph.nodes))
        # reconstruct graph
        k = vv_graph.number_of_edges()
        reconstructed_graph = reconstruct(k, embeddings, list(vv_graph.nodes))
        recall = sum([len(set(vv_graph.neighbors(n)).intersection(reconstructed_graph.neighbors(n))) / len(list(vv_graph.neighbors(n))) for n in list(vv_graph.nodes)]) / len(vv_graph.nodes)
        precision = sum([len(set(vv_graph.neighbors(n)).intersection(reconstructed_graph.neighbors(n))) / len(list(reconstructed_graph.neighbors(n))) for n in list(vv_graph.nodes) if len(list(reconstructed_graph.neighbors(n))) > 0]) / len([n for n in list(vv_graph.nodes) if len(list(reconstructed_graph.neighbors(n))) > 0])
        f1 = 2 * (precision * recall) / (precision + recall)
        self.assertGreaterEqual(f1, 0.57)
    
        node2vec.fit(vv_graph)
        for node in vv_graph.nodes:
            embedding = node2vec.embed_node(node)
            self.assertEqual(len(embedding), node2vec.dim)
        node2vec.fit(vv_graph.subgraph(list(vv_graph.nodes)[:10]), nodes=list(vv_graph.nodes))
        for node in list(vv_graph.nodes)[:10]:
            embedding = node2vec.embed_node(node)
            self.assertEqual(len(embedding), node2vec.dim)
        