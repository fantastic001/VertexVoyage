import random
import unittest

import numpy as np

from vertex_voyage.seeding import MAX_SEED_EXCLUSIVE, derive_iteration_seed, seed_shared_random_state

class TestDeriveIterationSeed(unittest.TestCase):
    def test_same_inputs_give_same_seed(self):
        self.assertEqual(derive_iteration_seed(42, 3), derive_iteration_seed(42, 3))

    def test_iterations_get_distinct_seeds(self):
        seeds = {derive_iteration_seed(42, iteration) for iteration in range(10)}
        self.assertEqual(len(seeds), 10)

    def test_derived_seed_is_valid_seed(self):
        self.assertTrue(0 <= derive_iteration_seed(42, 0) < MAX_SEED_EXCLUSIVE)

class TestSeedSharedRandomState(unittest.TestCase):
    def test_same_seed_reproduces_builtin_and_numpy_draws(self):
        seed_shared_random_state(7)
        first = (random.random(), np.random.rand())
        seed_shared_random_state(7)
        second = (random.random(), np.random.rand())
        self.assertEqual(first, second)

    def test_different_seeds_produce_different_draws(self):
        seed_shared_random_state(7)
        first = random.random()
        seed_shared_random_state(8)
        second = random.random()
        self.assertNotEqual(first, second)

    def test_rejects_seed_outside_numpy_range(self):
        for invalid_seed in (-1, MAX_SEED_EXCLUSIVE):
            with self.subTest(seed=invalid_seed):
                with self.assertRaises(ValueError):
                    seed_shared_random_state(invalid_seed)

    def test_rejects_non_integer_seed(self):
        with self.assertRaises(ValueError):
            seed_shared_random_state("42")

    def test_torch_draws_reproduce_when_torch_installed(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        seed_shared_random_state(7)
        first = torch.rand(3)
        seed_shared_random_state(7)
        second = torch.rand(3)
        self.assertTrue(torch.equal(first, second))
