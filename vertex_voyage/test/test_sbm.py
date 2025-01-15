import unittest

from vertex_voyage.plugins.sbm import LinearCongruentialGenerator, SBMNetwork

class TestLinearCongruentialGenerator(unittest.TestCase):
    def setUp(self):
        # Initialize the generator with known parameters
        self.lcg = LinearCongruentialGenerator(seed=1, a=2, c=3, m=5)

    def test_initial_seed(self):
        # Test the initial seed value
        self.assertEqual(self.lcg.get_number(0), 1 % 5)

    def test_first_number(self):
        # Test the first number in the sequence
        self.assertEqual(self.lcg.get_number(1), (2 * 1 + 3) % 5)

    def test_second_number(self):
        # Test the second number in the sequence
        self.assertEqual(self.lcg.get_number(2), (2 * ((2 * 1 + 3) % 5) + 3) % 5)

    def test_modular_exponentiation(self):
        # Test the modular exponentiation function
        self.assertEqual(self.lcg._modular_exponentiation(2, 3, 5), pow(2, 3, 5))

    def test_large_index(self):
        # Test a larger index to ensure correctness
        self.assertEqual(self.lcg.get_number(10), (2**10 * 1 + 3 * ((1 - 2**10) * pow(1 - 2, -1, 5)) % 5) % 5)

    def test_special_case_a_equals_1(self):
        # Test the special case when a == 1
        lcg_special = LinearCongruentialGenerator(seed=1, a=1, c=3, m=5)
        self.assertEqual(lcg_special.get_number(10), (1 + 10 * 3) % 5)

    def test_performance(self):
        # Test the performance of the generator
        for i in range(1000000):
            self.lcg.get_number(i)
        self.lcg.get_number(1000000**2)


class TestSBMNetwork(unittest.TestCase):
    def setUp(self):
        sizes = [2, 3]
        probabilities = [
            [0.9, 0.1],
            [0.1, 0.8]
        ]
        lcg_params = {'seed': 1, 'a': 2, 'c': 3, 'm': 5}
        self.sbm = SBMNetwork(sizes, probabilities, lcg_params)

    def test_initialization(self):
        self.assertEqual(self.sbm.sizes, [2, 3])
        self.assertEqual(self.sbm.probabilities, [
            [0.9, 0.1],
            [0.1, 0.8]
        ])
        self.assertEqual(self.sbm.node_to_block, [0, 0, 1, 1, 1])
        self.assertEqual(self.sbm.total_nodes, 5)

    def test_get_edge_index(self):
        self.assertEqual(self.sbm._get_edge_index(1, 3), 1 * 5 + 3)
        self.assertEqual(self.sbm._get_edge_index(3, 1), 1 * 5 + 3)

    def test_edge_exists(self):
        # Mock the LCG to control the random values
        self.sbm.lcg.get_number = lambda x: 1  # Always return 1
        self.assertFalse(self.sbm.edge_exists(0, 1))  # Probability is 0.9, but random value is 1
        self.assertFalse(self.sbm.edge_exists(2, 3))  # Probability is 0.8, but random value is 1

        self.sbm.lcg.get_number = lambda x: 0  # Always return 0
        self.assertTrue(self.sbm.edge_exists(0, 1))  # Probability is 0.9, and random value is 0
        self.assertTrue(self.sbm.edge_exists(2, 3))  # Probability is 0.8, and random value is 0

    def test_get_neighbors(self):
        # Mock the LCG to control the random values
        self.sbm.lcg.get_number = lambda x: 0  # Always return 0
        self.assertEqual(self.sbm.get_neighbors(0), [1, 2, 3, 4])
        self.assertEqual(self.sbm.get_neighbors(2), [0, 1, 3, 4])

        self.sbm.lcg.get_number = lambda x: 1  # Always return 1
        self.assertEqual(self.sbm.get_neighbors(0), [])
        self.assertEqual(self.sbm.get_neighbors(2), [])
