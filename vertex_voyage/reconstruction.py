
import networkx as nx 
import numpy as np 


def reconstruct(k: int, embedding: list[np.array]) -> nx.Graph:
    """
    Reconstructs graph from embedding by taking k most closest nodes and creating links between them.

    Parameters:
    k (int): Number of closest nodes to connect.
    embedding (list[np.array]): List of embeddings of nodes.

    Returns:
    nx.Graph: Reconstructed undirected graph.
    """
    n = len(embedding)
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(i)
    distances = [] 
    for i in range(n):
            for j in range(i+1, n):
                distances.append((np.linalg.norm(embedding[i] - embedding[j]), i, j))
    distances = sorted(distances, key=lambda x: x[0])
    for _, x, y in distances[:k]:
        graph.add_edge(x, y)
    return graph