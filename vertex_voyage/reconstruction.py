
import networkx as nx 
import numpy as np 


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