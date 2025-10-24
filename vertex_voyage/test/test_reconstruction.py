
from random import shuffle
from experiments import embedding
from vertex_voyage.reconstruction import reconstruct
import unittest
import numpy as np 
import networkx as nx


def myreconstruct(k: int, embedding: list[np.array], nodes = None) -> nx.Graph:
    """
    Reconstructs graph from embedding by taking k most closest nodes and creating links between them.

    Parameters:
    k (int): Number of closest nodes to connect.
    embedding (list[np.array]): List of embeddings of nodes.
    nodes (list): List of nodes. If None, nodes are assumed to be [0, 1, ..., n].

    Returns:
    nx.Graph: Reconstructed undirected graph.
    """
    n = len(embedding)
    graph = nx.Graph()
    if nodes is None:
        for i in range(n):
            graph.add_node(i)
        nodes = list(range(n))
    else:
        for i, node in enumerate(nodes):
            graph.add_node(node)
    distances = [] 
    for i in range(n):
            for j in range(i+1, n):
                distances.append((np.linalg.norm(embedding[i] - embedding[j]), i, j))
    distances = sorted(distances, key=lambda x: x[0])
    for _, x, y in distances[:k]:
        graph.add_edge(nodes[x], nodes[y])
    return graph
class TestReconstruction(unittest.TestCase):

    def test_native_reconstruction(self):
        embeddings = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
        embeddings = np.array(embeddings)
        k = 2
        nodes = [1, 2, 3]
        reconstructed_graph = reconstruct(k, embeddings, nodes)
        reconstructed_non_native_graph = myreconstruct(k, embeddings, nodes)
        print(reconstructed_graph.edges())
        print(reconstructed_non_native_graph.edges())
        self.assertEqual(len(reconstructed_graph.edges()), len(reconstructed_non_native_graph.edges()))
        edges_1 = sorted([tuple(sorted(edge)) for edge in reconstructed_graph.edges()])
        edges_2 = sorted([tuple(sorted(edge)) for edge in reconstructed_non_native_graph.edges()])
        self.assertEqual(edges_1, edges_2)

    def test_reconstruction_shuffle(self):
        embeddings = [
            np.array([np.sin(i * np.pi / 100), np.cos(i * np.pi / 100)]) for i in range(40000)
        ]
        k = 2
        shuffle(embeddings)
        nodes = [i for i in range(40000)]
        # expected_graph = reconstruct(k, embeddings, nodes)
        G = reconstruct(k, embeddings)
        # print("Expected graph edges:", expected_graph.edges())
        self.assertEqual(len(G.edges()), k)
        
if __name__ == '__main__':
    unittest.main()