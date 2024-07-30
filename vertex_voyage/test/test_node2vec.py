import unittest
import networkx as nx
import numpy as np
from vertex_voyage.node2vec import Node2Vec
from sklearn.cluster import KMeans
from vertex_voyage.reconstruction import reconstruct

class TestNode2Vec(unittest.TestCase):

    def setUp(self):
        # Create a simple graph for testing
        self.G = nx.Graph()
        self.G.add_edge(1, 2, weight=1.0)
        self.G.add_edge(2, 3, weight=1.0)
        self.G.add_edge(3, 4, weight=1.0)
        self.G.add_edge(4, 1, weight=1.0)
        self.G.add_edge(1, 3, weight=1.0)
        self.node2vec = Node2Vec(dim=2, walk_size=10, n_walks=10, window_size=2, epochs=1, seed=42)

    def test_random_walks(self):
        self.node2vec.G = self.G
        self.node2vec.nodes = {node: self.node2vec._encode(node) for node in self.G.nodes()}
        walks = self.node2vec._random_walks()
        self.assertEqual(len(walks), self.node2vec.n_walks)
        for walk in walks:
            self.assertEqual(len(walk), self.node2vec.walk_size)
        
        G = nx.Graph()
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(4, 3, weight=1.0)
        node2vec = Node2Vec(dim=2, walk_size=10, n_walks=10, window_size=2, epochs=1, seed=42)
        node2vec.G = G
        node2vec.nodes = {node: node2vec._encode(node) for node in G.nodes()}
        walks = node2vec._random_walks()
        self.assertEqual(len(walks), node2vec.n_walks)
        for walk in walks:
            self.assertEqual(len(walk), node2vec.walk_size)
            decoded_walk = [] 
            for w in walk:
                decoded = [k for k, v in node2vec.nodes.items() if (v == w).all()][0]
                decoded_walk.append(decoded)
            if 1 in decoded_walk:
                self.assertTrue(2 in decoded_walk)
                self.assertFalse(3 in decoded_walk)
                self.assertFalse(4 in decoded_walk)
            if 4 in decoded_walk:
                self.assertTrue(3 in decoded_walk)
                self.assertFalse(1 in decoded_walk)
                self.assertFalse(2 in decoded_walk)


    def test_embed_node(self):
        self.node2vec.fit(self.G)
        for node in self.G.nodes():
            embedding = self.node2vec.embed_node(node)
            self.assertEqual(len(embedding), self.node2vec.dim)

    def test_embed_nodes(self):
        self.node2vec.fit(self.G)
        nodes = list(self.G.nodes())
        embeddings = self.node2vec.embed_nodes(nodes)
        self.assertEqual(len(embeddings), len(nodes))
        for embedding in embeddings:
            self.assertEqual(len(embedding), self.node2vec.dim)

    def test_reconstruct(self):
        # create graph with 3 nodes, where 2 nodes are connected and one is isolated 
        G = nx.Graph()
        G.add_edge(1, 2, weight=1.0)
        G.add_node(3)
        # fit node2vec model
        node2vec = Node2Vec(dim=5, walk_size=10, n_walks=10, window_size=2, epochs=1, seed=42)
        node2vec.fit(G)
        # calculate embeddings
        embeddings = node2vec.embed_nodes(G.nodes())
        # reconstruct graph
        reconstructed_graph = reconstruct(1, embeddings)
        self.assertEqual(reconstructed_graph.number_of_edges(), 1)

        # create graph with 4 nodes where 2 nodes are connected and 2 other nodes are connected 
        G = nx.Graph()
        nodes = [1, 2, 3, 4]
        G.add_edge(nodes[0], nodes[1], weight=1.0)
        G.add_edge(nodes[2], nodes[3], weight=1.0)
        # fit node2vec model
        node2vec = Node2Vec(dim=4, walk_size=10, n_walks=100, window_size=5, seed=42, epochs=10, p=.25, q=4)
        node2vec.fit(G)
        # calculate embeddings
        embeddings = node2vec.embed_nodes(nodes)
        # reconstruct graph
        reconstructed_graph = reconstruct(2, embeddings, nodes)
        print(reconstructed_graph.edges())
        self.assertEqual(reconstructed_graph.number_of_edges(), 2)

        # test model prediction of context neighbor 
        # for n in nodes:
        #     x = node2vec.nodes[n]
        #     pred = node2vec.model.predict(np.array([x]))
        #     predicted_node = pred.argmax()
        #     self.assertTrue(nodes[predicted_node] in G.neighbors(n))

        precision = len(set(G.edges()).intersection(reconstructed_graph.edges())) / len(reconstructed_graph.edges())
        self.assertEqual(precision, 1.0)
        

    def test_zachary_recall_reconstructed_graph(self):
        # load karate club graph
        G = nx.karate_club_graph()
        nodes = list(G.nodes())
        # fit node2vec model
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
            learning_rate=0.01
        )
        node2vec.fit(G)
        # calculate embeddings
        embeddings = node2vec.embed_nodes(nodes)
        # reconstruct graph
        k = len(G.edges())
        reconstructed_graph = reconstruct(k, embeddings, nodes)
        recall = sum([len(set(G.neighbors(n)).intersection(reconstructed_graph.neighbors(n))) / len(list(G.neighbors(n))) for n in nodes]) / len(G.nodes())
        precision = sum([len(set(G.neighbors(n)).intersection(reconstructed_graph.neighbors(n))) / len(list(reconstructed_graph.neighbors(n))) for n in nodes if len(list(reconstructed_graph.neighbors(n))) > 0]) / len([n for n in G.nodes() if len(list(reconstructed_graph.neighbors(n))) > 0])
        f1 = 2 * (precision * recall) / (precision + recall)
        self.assertGreaterEqual(f1, 0.6)
