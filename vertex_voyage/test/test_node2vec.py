import unittest
import networkx as nx
import numpy as np
from vertex_voyage.node2vec import Node2Vec
from sklearn.cluster import KMeans

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

    def test_fit(self):
        model = self.node2vec.fit(self.G)
        self.assertEqual(model.shape[0], len(self.G.nodes()))
        self.assertEqual(model.shape[1], self.node2vec.dim)

    def test_random_walks(self):
        self.node2vec.G = self.G
        self.node2vec.nodes = {node: self.node2vec._encode(node) for node in self.G.nodes()}
        walks = self.node2vec._random_walks()
        self.assertEqual(len(walks), self.node2vec.n_walks)
        for walk in walks:
            self.assertEqual(len(walk), self.node2vec.walk_size)

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

    def test_embedding_node_distance(self):
        # create graph with 3 nodes, where 2 nodes are connected and one is isolated 
        G = nx.Graph()
        G.add_edge(1, 2, weight=1.0)
        G.add_node(3)
        # fit node2vec model
        node2vec = Node2Vec(dim=2, walk_size=10, n_walks=10, window_size=2, epochs=1, seed=42)
        node2vec.fit(G)
        # calculate embeddings
        embeddings = node2vec.embed_nodes(G.nodes())
        # calculate distance between embeddings
        distance = np.linalg.norm(embeddings[0] - embeddings[1])
        self.assertGreater(distance, 0)
        # calculate distance between isolated node and connected node
        distance2 = np.linalg.norm(embeddings[0] - embeddings[2])
        self.assertGreater(distance2, 0)
        self.assertGreater(distance2, distance)

        # now create graph with 4 nodes where 2 nodes are connected and 2 other nodes are connected 
        G = nx.Graph()
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(3, 4, weight=1.0)
        # fit node2vec model
        node2vec = Node2Vec(dim=2, walk_size=10, n_walks=10, window_size=50, epochs=1, seed=42)
        node2vec.fit(G)
        # calculate embeddings
        embeddings = node2vec.embed_nodes(G.nodes())
        # now lets do k-means on embeddings and see if clsters are assigned properly
        kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
        self.assertEqual(kmeans.labels_[0], kmeans.labels_[1])
        self.assertEqual(kmeans.labels_[2], kmeans.labels_[3])
        self.assertNotEqual(kmeans.labels_[0], kmeans.labels_[2])
