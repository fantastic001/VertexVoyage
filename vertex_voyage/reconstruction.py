
import networkx as nx 
import numpy as np 
from vertex_voyage_native import get_reconstructed_edges


def reconstruct(k: int, embedding: list[np.array], nodes = None) -> nx.Graph:
    """
    Reconstructs graph from embedding by taking k most closest nodes and creating links between them.

    Parameters:
    k (int): Number of closest nodes to connect.
    embedding (list[np.array]): List of embeddings of nodes.
    nodes (list): List of nodes. If None, nodes are assumed to be [0, 1, ..., n].

    Returns:
    nx.Graph: Reconstructed undirected graph.
    """
    if nodes is None:
        nodes = [i for i in range(len(embedding))]
    reconstructed_edges = get_reconstructed_edges(np.array(embedding, dtype=np.float64), k)
    reconstructed_edges = [(nodes[e[0]], nodes[e[1]]) for e in reconstructed_edges]
    reconstructed_graph = nx.Graph()
    reconstructed_graph.add_edges_from(reconstructed_edges)
    return reconstructed_graph

def get_f1_score(G, reconstructed_graph):
    nodes = G.nodes()
    recall = sum([len(set(G.neighbors(n)).intersection(reconstructed_graph.neighbors(n))) / len(list(G.neighbors(n))) for n in nodes]) / len(G.nodes())
    precision = sum([len(set(G.neighbors(n)).intersection(reconstructed_graph.neighbors(n))) / len(list(reconstructed_graph.neighbors(n))) for n in nodes if len(list(reconstructed_graph.neighbors(n))) > 0]) / len([n for n in G.nodes() if len(list(reconstructed_graph.neighbors(n))) > 0])
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1