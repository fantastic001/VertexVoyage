import unittest 

from vertex_voyage.tasks.link_prediction import Ranks, heart_benchmark
import networkx as nx
from torch import tensor

class DummyEmbeddingModel:
    def embed_nodes(self, nodes):
        # Return a fixed embedding for testing
        return {node: [0.1, 0.2, 0.3] for node in nodes}
    def embed_node(self, node):
        # Return a fixed embedding for testing
        return [0.1, 0.2, 0.3]

class DummyModel:
    def predict(self, X, Y):
        # Return a fixed probability for testing
        return tensor([0.9 for _ in range(len(X))])
    
    def __call__(self, X, Y):
        return self.predict(X, Y)

class TestLinkPredictionBenchmark(unittest.TestCase):
    def setUp(self):
        # Create a dummy graph and positive edges for testing
        self.graph = nx.Graph()
        self.graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
        self.graph.remove_edge(1, 2)  # Remove a positive edge to simulate testing
        self.graph.remove_edge(2, 3)  # Remove another positive edge to simulate testing
        self.positive_edges = {(1, 2), (2, 3)}
        self.model = DummyModel()

    def test_heart_benchmark(self):
        self.embedding_model = DummyEmbeddingModel()
        ranks = heart_benchmark(self.embedding_model, self.model, self.graph, self.positive_edges, ns=5, ps=2)
        self.assertIsInstance(ranks, Ranks)
        self.assertTrue(len(ranks.ranks) > 0)