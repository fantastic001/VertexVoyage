

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