

import networkx as nx 
from cdlib.algorithms import lfm
from cdlib.algorithms.internal.lfm import Community
from binpacking import to_constant_bin_number
import random 
import vertex_voyage.config as cfg 

def modified__lfm(G: nx.Graph, alpha: float = 1, threshold: float = 0.5) -> list:
    communities = []
    node_not_include = list(G.nodes.keys())[:]
    node_num = len(node_not_include)
    while len(node_not_include) > node_num * threshold:
        c = Community(G, alpha)
        # randomly select a seed node
        seed = random.choice(node_not_include)
        c.add_node(seed)

        to_be_examined = c.get_neighbors()
        while to_be_examined:
            # largest fitness to be added
            m = {}
            for node in to_be_examined:
                fitness = c.cal_add_fitness(node)
                m[node] = fitness
            to_be_add = sorted(m.items(), key=lambda x: x[1], reverse=True)[0]

            # stop condition
            if to_be_add[1] < 0.0:
                break
            c.add_node(to_be_add[0])

            to_be_remove = c.recalculate()
            while to_be_remove is not None:
                c.remove_vertex(to_be_remove)
                to_be_remove = c.recalculate()

            to_be_examined = c.get_neighbors()

        for node in c.nodes:
            if node in node_not_include:
                node_not_include.remove(node)
        communities.append(list(c.nodes))
    for node in node_not_include:
        random_comm = random.choice(communities)
        random_comm.append(node)
    return list(communities)

@cfg.pluggable
def partition_graph(G: nx.Graph, partition_num: int, use_modified_lfm: bool = False, threshold: float = 0.5) -> list:
    """
    Partition the graph into a given number of partitions using LFM algorithm.
    """
    # create a LFM object
    communities = None 
    if use_modified_lfm:
        communities = modified__lfm(G, alpha=1, threshold=threshold)
    else:
        communities = lfm(G, alpha=1).communities
    # partition the graph into a given number of partitions
    partitions = to_constant_bin_number(communities, partition_num, key=len)
    partitions = [list(sum(part, [])) for part in partitions]
    return partitions

@cfg.pluggable
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

@cfg.pluggable
def calculate_corruptability(G: nx.Graph, partition_num: int, use_modified_lfm = False, threshold = 0.5, partitions = None):
    """
    Calculate the corruptability of the graph.
    """
    # partition the graph into a given number of partitions
    if partitions is None:
        partitions = partition_graph(G, partition_num, use_modified_lfm=use_modified_lfm, threshold=threshold)
    # calculate the partitioning corruption
    corruption = calculate_partitioning_corruption(G, partitions)
    return corruption

@cfg.pluggable
def calculate_graph_corruptability(G: nx.Graph, max_partition_num: int, use_modified_lfm = False, threshold=0.5):
    """
    Calculate the corruptability of the graph for all partition numbers from 1 to max_partition_num and returns linear coefficient of function corruptability(pnum) = k * (pnum-1).
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np
    # calculate the corruptability of the graph for all partition numbers from 1 to max_partition_num
    x = np.array(range(1, max_partition_num+1)).reshape(-1, 1)-1
    y = np.array([calculate_corruptability(G, pnum, use_modified_lfm=use_modified_lfm, threshold=threshold) for pnum in range(1, max_partition_num+1)]).reshape(-1, 1)
    # fit the linear regression model
    model = LinearRegression().fit(x, y)
    return model.coef_[0][0]