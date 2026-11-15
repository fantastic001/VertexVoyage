import json
import os
import random
import tempfile
import unittest
from unittest import mock

from vertex_voyage.walk_strategy import (
    NODE2VEC_WALK,
    RANDOM_VERTEX_SEQUENCE_WALK,
    REPEAT_START_VERTEX_WALK,
    Node2VecWalk,
    RandomVertexSequenceWalk,
    RepeatStartVertexWalk,
    configured_walk_strategy,
    walk_strategy_names,
    walk_strategy_of,
)

WALK_SIZE = 6
CONFIG_ENVVAR = "VERTEX_VOYAGE_CONFIG"
STRATEGY_ENVVAR = "VERTEX_VOYAGE_WALK_STRATEGY"


class PathContext:
    def __init__(self, vertices, successor):
        self._vertices = vertices
        self._successor = successor

    def vertices(self):
        return self._vertices

    def next_vertex(self, previous, current):
        return self._successor[current]


class EdgelessContext:
    def __init__(self, vertices):
        self._vertices = vertices

    def vertices(self):
        return self._vertices

    def next_vertex(self, previous, current):
        raise AssertionError("A walk without edges must not follow one")


def empty_config_environment() -> dict:
    handle, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(handle, "w") as config_file:
        json.dump({}, config_file)
    return {CONFIG_ENVVAR: path}


class TestNode2VecWalk(unittest.TestCase):
    def setUp(self):
        self.context = PathContext(
            ["a", "b", "c"],
            {"a": "b", "b": "c", "c": "a"},
        )

    def test_walk_follows_the_successor_of_every_vertex(self):
        self.assertEqual(
            Node2VecWalk().walk_from("a", WALK_SIZE, self.context),
            ["a", "b", "c", "a", "b", "c"],
        )

    def test_walk_of_one_vertex_is_the_start(self):
        self.assertEqual(
            Node2VecWalk().walk_from("b", 1, self.context),
            ["b"],
        )

    def test_walk_shorter_than_one_vertex_raises(self):
        with self.assertRaises(ValueError):
            Node2VecWalk().walk_from("a", 0, self.context)


class TestRandomVertexSequenceWalk(unittest.TestCase):
    def setUp(self):
        self.vertices = ["a", "b", "c"]
        self.context = EdgelessContext(self.vertices)
        self.strategy = RandomVertexSequenceWalk()

    def test_walk_starts_at_the_start_vertex(self):
        random.seed(0)
        walk = self.strategy.walk_from("a", WALK_SIZE, self.context)
        self.assertEqual(walk[0], "a")

    def test_walk_has_the_requested_size(self):
        random.seed(0)
        walk = self.strategy.walk_from("a", WALK_SIZE, self.context)
        self.assertEqual(len(walk), WALK_SIZE)

    def test_walk_only_visits_vertices_of_the_partition(self):
        random.seed(0)
        walk = self.strategy.walk_from("a", WALK_SIZE, self.context)
        self.assertTrue(set(walk).issubset(set(self.vertices)))

    def test_walk_without_vertices_repeats_the_start(self):
        walk = self.strategy.walk_from("a", WALK_SIZE, EdgelessContext([]))
        self.assertEqual(walk, ["a"] * WALK_SIZE)

    def test_walk_visits_more_than_the_start_vertex(self):
        random.seed(0)
        walks = [
            self.strategy.walk_from("a", WALK_SIZE, self.context)
            for _ in range(20)
        ]
        self.assertTrue(any(set(walk) != {"a"} for walk in walks))

    def test_walk_shorter_than_one_vertex_raises(self):
        with self.assertRaises(ValueError):
            self.strategy.walk_from("a", 0, self.context)


class TestRepeatStartVertexWalk(unittest.TestCase):
    def test_walk_repeats_the_start_vertex(self):
        walk = RepeatStartVertexWalk().walk_from(
            "a",
            WALK_SIZE,
            EdgelessContext(["a", "b"]),
        )
        self.assertEqual(walk, ["a"] * WALK_SIZE)

    def test_walk_shorter_than_one_vertex_raises(self):
        with self.assertRaises(ValueError):
            RepeatStartVertexWalk().walk_from("a", 0, EdgelessContext(["a"]))


class TestWalkStrategyRegistry(unittest.TestCase):
    def test_every_name_creates_its_strategy(self):
        self.assertEqual(
            walk_strategy_names(),
            [
                NODE2VEC_WALK,
                RANDOM_VERTEX_SEQUENCE_WALK,
                REPEAT_START_VERTEX_WALK,
            ],
        )
        self.assertIsInstance(
            walk_strategy_of(NODE2VEC_WALK),
            Node2VecWalk,
        )
        self.assertIsInstance(
            walk_strategy_of(RANDOM_VERTEX_SEQUENCE_WALK),
            RandomVertexSequenceWalk,
        )
        self.assertIsInstance(
            walk_strategy_of(REPEAT_START_VERTEX_WALK),
            RepeatStartVertexWalk,
        )

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            walk_strategy_of("missing")

    def test_configured_strategy_defaults_to_node2vec(self):
        with mock.patch.dict(
            os.environ,
            empty_config_environment(),
            clear=True,
        ):
            self.assertIsInstance(configured_walk_strategy(), Node2VecWalk)

    def test_configured_strategy_is_read_from_the_environment(self):
        environment = empty_config_environment()
        environment[STRATEGY_ENVVAR] = REPEAT_START_VERTEX_WALK
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertIsInstance(
                configured_walk_strategy(),
                RepeatStartVertexWalk,
            )


if __name__ == "__main__":
    unittest.main()
