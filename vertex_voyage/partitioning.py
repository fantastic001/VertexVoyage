

import networkx as nx 
from cdlib.algorithms import lfm 
from binpacking import to_constant_bin_number

def partition_graph(G: nx.Graph, partition_num: int) -> list:
    """
    Partition the graph into a given number of partitions using LFM algorithm.
    """
    # create a LFM object
    communities = lfm(G, alpha=1).communities
    # partition the graph into a given number of partitions
    partitions = to_constant_bin_number(communities, partition_num, key=len)
    partitions = [list(sum(part, [])) for part in partitions]
    return partitions

def calculate_partitioning_corruption(G: nx.Graph, partitions: list):
    """
    Partitioning corruption is 1 - ratio of size of edges of union of subgraphs of G induced by partitions and number of edges in original graph G  
    """
    # calculate the number of edges in original graph G
    original_edges = G.number_of_edges()
    # calculate the number of edges in union of subgraphs of G induced by partitions
    partitions_edges = set()
    for partition in partitions:
        subgraph = G.subgraph(partition)
        partitions_edges ^= set(subgraph.edges)
    # calculate the partitioning corruption
    partitioning_corruption = 1-len(partitions_edges) / original_edges
    return partitioning_corruption

def calculate_corruptability(G: nx.Graph, partition_num: int):
    """
    Calculate the corruptability of the graph.
    """
    # partition the graph into a given number of partitions
    partitions = partition_graph(G, partition_num)
    # calculate the partitioning corruption
    corruption = calculate_partitioning_corruption(G, partitions)
    return corruption