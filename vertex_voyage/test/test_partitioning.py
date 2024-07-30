

import unittest
from vertex_voyage.partitioning import partition_graph
import networkx as nx

class TestPartitioning(unittest.TestCase):
    
    def test_partitioning(self):
        zachary = nx.karate_club_graph()
        partition_num = 2
        communities = partition_graph(zachary, partition_num)
        self.assertEqual(len(communities), partition_num)
        # approximately small differences between partitions )less than 10%_
        self.assertLessEqual(len(communities[0]), len(zachary.nodes) * 0.60)
    
    def test_partitioning_on_sbm(self):
        n = 100
        p = 0.1
        q = 0.01
        G = nx.planted_partition_graph(2, n, p, q, seed=42)
        partition_num = 2
        communities = partition_graph(G, partition_num)
        self.assertEqual(len(communities), partition_num)
        # approximately small differences between partitions )less than 10%_
        self.assertLessEqual(len(communities[0]), len(G.nodes) * 0.60)