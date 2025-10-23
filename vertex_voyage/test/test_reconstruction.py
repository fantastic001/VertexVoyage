
from random import shuffle
from experiments import embedding
from vertex_voyage.reconstruction import reconstruct
import unittest
import numpy as np 
import networkx as nx
from vertex_voyage_native import get_reconstructed_edges


class TestReconstruction(unittest.TestCase):

    def test_native_reconstruction(self):
        embeddings = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
        k = 2
        nodes = [1, 2, 3]
        embeddings = np.array([np.array(emb) for emb in embeddings])
        reconstructed_edges = get_reconstructed_edges(np.array(embeddings, dtype=np.float64), k)
        reconstructed_edges = [(nodes[e[0]], nodes[e[1]]) for e in reconstructed_edges]
        reconstructed_graph = nx.Graph()
        reconstructed_graph.add_edges_from(reconstructed_edges)
        reconstructed_non_native_graph = reconstruct(k, embeddings, nodes)
        print(reconstructed_graph.edges())
        print(reconstructed_non_native_graph.edges())
        self.assertEqual(len(reconstructed_graph.edges()), len(reconstructed_non_native_graph.edges()))
        self.assertEqual(set(reconstructed_graph.edges()), set(reconstructed_non_native_graph.edges()))
    
    def test_reconstruction_shuffle(self):
        embeddings = [
            np.array([np.sin(i * np.pi / 100), np.cos(i * np.pi / 100)]) for i in range(100)
        ]
        k = 2
        shuffle(embeddings)
        nodes = [i for i in range(100)]
        embeddings = np.array(embeddings, dtype=np.float64)
        expected_graph = reconstruct(k, embeddings, nodes)
        reconstructed_edges_native = get_reconstructed_edges(embeddings, k)
        reconstructed_edges_native = [(nodes[e[0]], nodes[e[1]]) for e in reconstructed_edges_native]
        print("Expected graph edges:", expected_graph.edges())
        print("Reconstructed graph edges (native):", reconstructed_edges_native)
        self.assertEqual(len(reconstructed_edges_native), len(expected_graph.edges()))
        self.assertEqual(set(reconstructed_edges_native), set(expected_graph.edges()))

if __name__ == '__main__':
    unittest.main()