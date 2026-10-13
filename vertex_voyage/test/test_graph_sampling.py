import unittest
from collections import Counter

import networkx as nx
import numpy as np

from vertex_voyage.graph_sampling import (
    BiasedSampler,
    GraphSampler,
    Node2VecBias,
    RoundRobinSampler,
    RoundRobinUniform,
    SamplingError,
    UniformSampler,
)
from vertex_voyage.graph_sampling.choice import (
    CumulativeDistribution,
    choose_weighted,
)
from vertex_voyage.vv_graph import VVGraph

SEED = 7
DRAW_COUNT = 6000
FREQUENCY_TOLERANCE = 0.03


def path_graph() -> nx.Graph:
    return nx.path_graph(4)


def frequencies(draws) -> dict:
    counts = Counter(draws)
    return {item: count / len(draws) for item, count in counts.items()}


def prefer_larger_candidate(previous: int, candidate: int) -> float:
    return float(candidate)


class FrequencyAssertions(unittest.TestCase):
    def assert_frequencies(self, draws, expected: dict):
        observed = frequencies(draws)
        self.assertEqual(set(observed), set(expected))
        for item, probability in expected.items():
            self.assertAlmostEqual(
                observed[item],
                probability,
                delta=FREQUENCY_TOLERANCE,
            )


class TestCumulativeDistribution(FrequencyAssertions):
    def test_never_draws_zero_weight_index(self):
        rng = np.random.default_rng(SEED)
        distribution = CumulativeDistribution([0, 0, 3, 0, 1, 0])
        draws = [distribution.draw_index(rng) for _ in range(DRAW_COUNT)]
        self.assert_frequencies(draws, {2: 0.75, 4: 0.25})

    def test_rejects_invalid_weights(self):
        for weights in ([1, -1], [1, float("nan")], [float("inf")]):
            with self.subTest(weights=weights):
                with self.assertRaises(SamplingError):
                    CumulativeDistribution(weights)

    def test_zero_total_weight_cannot_be_drawn(self):
        rng = np.random.default_rng(SEED)
        for weights in ([], [0, 0]):
            with self.subTest(weights=weights):
                with self.assertRaises(SamplingError):
                    CumulativeDistribution(weights).draw_index(rng)

    def test_choose_weighted_rejects_length_mismatch(self):
        rng = np.random.default_rng(SEED)
        with self.assertRaises(SamplingError):
            choose_weighted(rng, ["a", "b"], [1.0])


class TestUniformSampler(FrequencyAssertions):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        self.sampler = UniformSampler(path_graph())

    def test_is_graph_sampler(self):
        self.assertIsInstance(self.sampler, GraphSampler)

    def test_node_is_uniform(self):
        draws = [self.sampler.node(self.rng) for _ in range(DRAW_COUNT)]
        self.assert_frequencies(draws, {n: 0.25 for n in range(4)})

    def test_neighbor_is_uniform(self):
        draws = [
            self.sampler.neighbor(self.rng, 1) for _ in range(DRAW_COUNT)
        ]
        self.assert_frequencies(draws, {0: 0.5, 2: 0.5})

    def test_non_neighbor_is_uniform(self):
        draws = [
            self.sampler.non_neighbor(self.rng, 0) for _ in range(DRAW_COUNT)
        ]
        self.assert_frequencies(draws, {2: 0.5, 3: 0.5})

    def test_edge_is_uniform_over_arcs(self):
        arcs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
        draws = [self.sampler.edge(self.rng) for _ in range(DRAW_COUNT)]
        self.assert_frequencies(draws, {arc: 1 / 6 for arc in arcs})

    def test_non_edge_is_uniform_over_non_adjacent_pairs(self):
        pairs = [(0, 2), (2, 0), (0, 3), (3, 0), (1, 3), (3, 1)]
        draws = [self.sampler.non_edge(self.rng) for _ in range(DRAW_COUNT)]
        self.assert_frequencies(draws, {pair: 1 / 6 for pair in pairs})

    def test_non_neighbor_of_dense_node_uses_enumeration(self):
        star_with_outsider = nx.star_graph(200)
        star_with_outsider.add_node("outsider")
        sampler = UniformSampler(star_with_outsider)
        for _ in range(50):
            self.assertEqual(sampler.non_neighbor(self.rng, 0), "outsider")

    def test_self_loop_is_never_a_non_neighbor(self):
        graph = path_graph()
        graph.add_edge(0, 0)
        sampler = UniformSampler(graph)
        draws = {sampler.non_neighbor(self.rng, 0) for _ in range(200)}
        self.assertEqual(draws, {2, 3})

    def test_directed_graph_follows_successors(self):
        sampler = UniformSampler(nx.DiGraph([(0, 1), (1, 2)]))
        draws = {sampler.edge(self.rng) for _ in range(200)}
        self.assertEqual(draws, {(0, 1), (1, 2)})

    def test_same_seed_reproduces_draws(self):
        first = [
            self.sampler.edge(np.random.default_rng(SEED)) for _ in range(5)
        ]
        second = [
            self.sampler.edge(np.random.default_rng(SEED)) for _ in range(5)
        ]
        self.assertEqual(first, second)

    def test_supports_vv_graph(self):
        graph = VVGraph()
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        sampler = UniformSampler(graph)
        self.assertEqual(sampler.neighbor(self.rng, 0), 1)
        self.assertEqual(sampler.non_neighbor(self.rng, 0), 2)

    def test_infeasible_draws_raise(self):
        complete = UniformSampler(nx.complete_graph(3))
        edgeless = UniformSampler(nx.empty_graph(3))
        empty = UniformSampler(nx.Graph())
        infeasible_draws = {
            "node of empty graph": lambda: empty.node(self.rng),
            "edge of edgeless graph": lambda: edgeless.edge(self.rng),
            "neighbor of isolated": lambda: edgeless.neighbor(self.rng, 0),
            "non-edge of complete": lambda: complete.non_edge(self.rng),
            "non-neighbor in complete": (
                lambda: complete.non_neighbor(self.rng, 0)
            ),
            "unknown node": lambda: complete.neighbor(self.rng, "missing"),
        }
        for name, draw in infeasible_draws.items():
            with self.subTest(draw=name):
                with self.assertRaises(SamplingError):
                    draw()

    def test_rejects_non_generator_rng(self):
        with self.assertRaises(TypeError):
            self.sampler.node(SEED)


class TestBiasedSampler(FrequencyAssertions):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        self.sampler = BiasedSampler(path_graph(), prefer_larger_candidate)

    def test_is_graph_sampler(self):
        self.assertIsInstance(self.sampler, GraphSampler)

    def test_neighbor_weighs_candidates_by_bias_of_previous(self):
        draws = [
            self.sampler.neighbor(self.rng, 2, 1) for _ in range(DRAW_COUNT)
        ]
        self.assert_frequencies(draws, {1: 0.25, 3: 0.75})

    def test_bias_receives_previous_and_candidate(self):
        received = []

        def recording_bias(previous: int, candidate: int) -> float:
            received.append((previous, candidate))
            return 1.0

        BiasedSampler(path_graph(), recording_bias).neighbor(self.rng, 2, 1)
        self.assertEqual(sorted(received), [(1, 1), (1, 3)])

    def test_neighbor_without_previous_is_uniform(self):
        draws = [
            self.sampler.neighbor(self.rng, 2) for _ in range(DRAW_COUNT)
        ]
        self.assert_frequencies(draws, {1: 0.5, 3: 0.5})

    def test_non_neighbor_weighs_candidates_by_bias_of_previous(self):
        draws = [
            self.sampler.non_neighbor(self.rng, 0, 1)
            for _ in range(DRAW_COUNT)
        ]
        self.assert_frequencies(draws, {2: 0.4, 3: 0.6})

    def test_zero_bias_candidate_is_never_drawn(self):
        draws = {self.sampler.neighbor(self.rng, 1, 2) for _ in range(200)}
        self.assertEqual(draws, {2})

    def test_edge_is_uniform_over_arcs(self):
        arcs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
        draws = [self.sampler.edge(self.rng) for _ in range(DRAW_COUNT)]
        self.assert_frequencies(draws, {arc: 1 / 6 for arc in arcs})

    def test_non_edge_only_returns_non_adjacent_pairs(self):
        graph = path_graph()
        for _ in range(200):
            source, target = self.sampler.non_edge(self.rng)
            self.assertNotEqual(source, target)
            self.assertFalse(graph.has_edge(source, target))

    def test_invalid_bias_raises(self):
        negative = BiasedSampler(path_graph(), lambda previous, node: -1.0)
        vanishing = BiasedSampler(path_graph(), lambda previous, node: 0.0)
        for name, sampler in (("negative", negative), ("zero", vanishing)):
            with self.subTest(bias=name):
                with self.assertRaises(SamplingError):
                    sampler.neighbor(self.rng, 1, 0)

    def test_infeasible_draws_raise(self):
        isolated = BiasedSampler(nx.empty_graph(2), prefer_larger_candidate)
        complete = BiasedSampler(
            nx.complete_graph(3),
            prefer_larger_candidate,
        )
        with self.assertRaises(SamplingError):
            isolated.neighbor(self.rng, 0, 1)
        with self.assertRaises(SamplingError):
            complete.non_neighbor(self.rng, 0, 1)

    def test_rejects_non_callable_bias(self):
        with self.assertRaises(SamplingError):
            BiasedSampler(path_graph(), 1.0)


class TestNode2VecBias(FrequencyAssertions):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        self.graph = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3)])
        self.bias = Node2VecBias(self.graph, p=0.5, q=2.0)

    def test_weights_follow_distance_from_previous(self):
        self.assertEqual(self.bias(0, 0), 2.0)
        self.assertEqual(self.bias(0, 1), 1.0)
        self.assertEqual(self.bias(0, 3), 0.5)

    def test_biased_walk_step_matches_node2vec_transition(self):
        sampler = BiasedSampler(self.graph, self.bias)
        draws = [sampler.neighbor(self.rng, 2, 0) for _ in range(DRAW_COUNT)]
        self.assert_frequencies(draws, {0: 4 / 7, 1: 2 / 7, 3: 1 / 7})

    def test_rejects_non_positive_parameters(self):
        for p, q in ((0.0, 1.0), (1.0, -1.0), (float("inf"), 1.0)):
            with self.subTest(p=p, q=q):
                with self.assertRaises(SamplingError):
                    Node2VecBias(self.graph, p, q)


class TestRoundRobinUniform(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        self.first = nx.Graph([("a", "b")])
        self.second = nx.Graph([("c", "d")])
        self.third = nx.Graph([("e", "f")])
        self.sampler = RoundRobinUniform(self.first, self.second, self.third)

    def test_nodes_rotate_through_graphs(self):
        graphs = [self.first, self.second, self.third] * 2
        for graph in graphs:
            self.assertIn(self.sampler.node(self.rng), graph)

    def test_rotation_is_shared_between_methods(self):
        self.assertIn(self.sampler.node(self.rng), self.first)
        self.assertTrue(self.second.has_edge(*self.sampler.edge(self.rng)))
        self.assertIn(self.sampler.node(self.rng), self.third)

    def test_skips_graphs_that_cannot_serve_the_draw(self):
        for _ in range(3):
            self.assertEqual(self.sampler.neighbor(self.rng, "c"), "d")
        self.assertIn(self.sampler.node(self.rng), self.third)

    def test_raises_when_no_graph_can_serve(self):
        with self.assertRaises(SamplingError):
            self.sampler.neighbor(self.rng, "missing")
        with self.assertRaises(SamplingError):
            self.sampler.non_edge(self.rng)

    def test_requires_at_least_one_graph(self):
        with self.assertRaises(SamplingError):
            RoundRobinUniform()

    def test_round_robin_accepts_any_sampler(self):
        sampler = RoundRobinSampler(
            [
                UniformSampler(self.first),
                BiasedSampler(self.second, lambda source, target: 1.0),
            ]
        )
        self.assertIn(sampler.node(self.rng), self.first)
        self.assertIn(sampler.node(self.rng), self.second)


if __name__ == "__main__":
    unittest.main()
