
from unittest import TestCase

from vertex_voyage.vv_graph import VVGraph
from vertex_voyage.partitioning import random_partitioning, label_propagation_partitioner


class TestVVGraph(TestCase):
    def test_add_and_remove_edge(self):
        g = VVGraph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        self.assertEqual(g.degree(1), 2)
        self.assertEqual(g.degree(2), 1)
        self.assertEqual(g.degree(3), 1)
        g.remove_edge(1, 2)
        self.assertEqual(g.degree(1), 1)
        self.assertEqual(g.degree(2), 0)
        self.assertEqual(g.degree(3), 1)
        self.assertEqual(g.number_of_nodes(), 3)
    
    def test_subgraph(self):
        g = VVGraph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        subg = g.subgraph({1, 2, 3})
        self.assertEqual(subg.number_of_nodes(), 3)
        self.assertEqual(subg.number_of_edges(), 3)
        self.assertEqual(subg.degree(1), 2)
        self.assertEqual(subg.degree(2), 2)
        self.assertEqual(subg.degree(3), 2)
        self.assertEqual(subg.degree(4), 0)  # Node 4 is not in the subgraph
    
    def test_random_partitioning(self):
        g = VVGraph()
        edges = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6)]
        for u, v in edges:
            g.add_edge(u, v)
        partitions = random_partitioning(g, 2)
        self.assertEqual(len(partitions), 2)
        all_nodes = set()
        for part in partitions:
            all_nodes.update(part)
        self.assertEqual(all_nodes, g.nodes)
    
    def test_label_propagation_partitioner(self):
        g = VVGraph()
        edges = [(1, 2), (1, 3), (2, 3), (4, 5), (5, 6), (6, 4)]
        for u, v in edges:
            g.add_edge(u, v)
        partitions = label_propagation_partitioner(g, partition_num=2)
        self.assertEqual(len(partitions), 2)
        all_nodes = set()
        for part in partitions:
            all_nodes.update(part)
        self.assertEqual(all_nodes, g.nodes)