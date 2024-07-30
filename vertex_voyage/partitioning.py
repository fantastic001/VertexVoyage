

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
