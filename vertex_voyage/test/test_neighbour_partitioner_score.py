import math
import random
import unittest

from vertex_voyage.partition_neighbours import (
    LARGEST_FINITE_EXPONENT,
    BufferNeighbours,
    DecayedNeighbours,
    decayed_weight,
    neighbour_weights_of,
)
from vertex_voyage.temporal import Event
from vertex_voyage.temporal_partitioning import (
    InMemoryPartition,
    MostCommonNeighborPartitioner,
)

DECAY = 0.5
EPSILON = 0.1
TIE_BREAK_SEEDS = range(20)


def partitions_holding(*memberships):
    partitions = []
    for identifier, vertices in enumerate(memberships):
        partition = InMemoryPartition(identifier)
        for vertex in vertices:
            partition.add(vertex)
        partitions.append(partition)
    return partitions


def partitioner_of(partitions, mu=0.0, alpha=1.0, decay=None):
    return MostCommonNeighborPartitioner.all_neighbors(
        set(partitions),
        replication_factor=1,
        mu=mu,
        epsilon=EPSILON,
        alpha=alpha,
        decay=decay,
    )


def events_at(timestamp, *pairs):
    return [Event(src, dest, timestamp) for src, dest in pairs]


class TestNeighbourScore(unittest.TestCase):
    def test_score_counts_the_neighbours_of_the_vertex(self):
        left, right = partitions_holding("ab", "cd")
        partitioner = partitioner_of([left, right])
        partitioner.neighbours.observe(
            events_at(0, ("a", "b"), ("c", "d"), ("x", "a"), ("y", "c")), 0
        )
        expected = {
            ("x", left): 1.0,
            ("x", right): 0.0,
            ("y", left): 0.0,
            ("y", right): 1.0,
        }
        for (vertex, partition), score in expected.items():
            with self.subTest(vertex=vertex, partition=partition.id):
                self.assertEqual(
                    partitioner.score_partition(vertex, partition), score
                )

    def test_repeated_events_and_self_loops_add_no_neighbour(self):
        left, = partitions_holding("ax")
        partitioner = partitioner_of([left])
        partitioner.neighbours.observe(
            events_at(0, ("x", "a"), ("a", "x"), ("x", "x")), 0
        )
        self.assertEqual(partitioner.score_partition("x", left), 1.0)

    def test_without_decay_only_the_current_buffer_counts(self):
        left, = partitions_holding("a")
        partitioner = partitioner_of([left])
        partitioner.neighbours.observe(events_at(0, ("x", "a")), 0)
        partitioner.neighbours.observe(events_at(1, ("y", "a")), 1)
        self.assertEqual(partitioner.score_partition("x", left), 0.0)
        self.assertEqual(partitioner.score_partition("y", left), 1.0)

    def test_capacity_penalty_follows_the_documented_formula(self):
        full, empty = partitions_holding("abcd", "")
        cases = {
            1.0: 4 - 1.0 * (1 + EPSILON) * 2,
            0.5: 4 - 0.5 * (1 + EPSILON) * 2,
        }
        for alpha, penalty in cases.items():
            with self.subTest(alpha=alpha):
                partitioner = partitioner_of([full, empty], mu=1.0, alpha=alpha)
                for _ in range(2):
                    self.assertAlmostEqual(
                        partitioner.capacity_penalty_of(full), penalty
                    )
                self.assertEqual(partitioner.capacity_penalty_of(empty), 0.0)

    def test_decay_weights_every_earlier_neighbour_by_its_age(self):
        left, = partitions_holding("ab")
        partitioner = partitioner_of([left], decay=DECAY)
        partitioner.neighbours.observe(events_at(1, ("x", "a")), 1)
        partitioner.neighbours.observe(
            events_at(3, ("x", "b")) + events_at(4, ("y", "z")), 4
        )
        self.assertAlmostEqual(
            partitioner.score_partition("x", left),
            math.exp(-DECAY * 3) + math.exp(-DECAY * 1),
        )

    def test_decay_uses_the_last_time_an_edge_was_seen(self):
        neighbours = DecayedNeighbours(DECAY)
        neighbours.observe(events_at(1, ("x", "a")), 1)
        neighbours.observe(events_at(5, ("a", "x")), 5)
        self.assertEqual(neighbours.weights_of("x"), {"a": 1.0})

    def test_ties_are_broken_at_random_and_reproducibly(self):
        partitions = partitions_holding("", "", "")
        partitioner = partitioner_of(partitions)
        chosen = set()
        for seed in TIE_BREAK_SEEDS:
            random.seed(seed)
            first = partitioner.best_partitions_of("x")
            random.seed(seed)
            self.assertEqual(partitioner.best_partitions_of("x"), first)
            chosen.add(first[0].id)
        self.assertEqual(chosen, {0, 1, 2})

    def test_push_places_vertices_with_their_neighbours(self):
        left, right = partitions_holding("ab", "cd")
        partitioner = partitioner_of([left, right])
        partitioner.push(events_at(0, ("x", "a"), ("y", "c")))
        self.assertEqual(partitioner.get("x"), {left})
        self.assertEqual(partitioner.get("y"), {right})
        self.assertTrue(left.connects("x", "a"))
        self.assertTrue(right.connects("y", "c"))

    def test_empty_buffer_changes_nothing(self):
        left, = partitions_holding("a")
        partitioner = partitioner_of([left])
        partitioner.push([])
        self.assertEqual(left.size(), 1)


class TestNeighbourWeights(unittest.TestCase):
    def test_strategy_follows_the_decay_setting(self):
        self.assertIsInstance(neighbour_weights_of(None), BufferNeighbours)
        self.assertIsInstance(neighbour_weights_of(DECAY), DecayedNeighbours)

    def test_unknown_vertex_has_no_neighbours(self):
        for neighbours in (BufferNeighbours(), DecayedNeighbours(DECAY)):
            with self.subTest(strategy=type(neighbours).__name__):
                self.assertEqual(dict(neighbours.weights_of("x")), {})

    def test_invalid_decay_is_rejected(self):
        for decay in (-DECAY, math.inf, math.nan):
            with self.subTest(decay=decay):
                with self.assertRaises(ValueError):
                    DecayedNeighbours(decay)

    def test_weight_of_a_negative_age_saturates_instead_of_overflowing(self):
        self.assertEqual(
            decayed_weight(DECAY, -1e9), math.exp(LARGEST_FINITE_EXPONENT)
        )


if __name__ == "__main__":
    unittest.main()
