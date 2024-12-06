from vertex_voyage.cluster import parse_slurm_nodelist_spec
import unittest

class TestSlurm(unittest.TestCase):
    def test_parse_slurm_nodelist_spec(self):
        self.assertEqual(parse_slurm_nodelist_spec("node[1-3,5]"), ["node1", "node2", "node3", "node5"])
        self.assertEqual(parse_slurm_nodelist_spec("node[1-3,5,7-8]"), ["node1", "node2", "node3", "node5", "node7", "node8"])
        self.assertEqual(parse_slurm_nodelist_spec("node[1,3,5]"), ["node1", "node3", "node5"])
        self.assertEqual(parse_slurm_nodelist_spec("node[1-3]"), ["node1", "node2", "node3"])
        self.assertEqual(parse_slurm_nodelist_spec("node[1,3]"), ["node1", "node3"])
        self.assertEqual(parse_slurm_nodelist_spec("node1"), ["node1"])