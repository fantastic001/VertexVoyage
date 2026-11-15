import json
import os
import tempfile
import unittest
from unittest import mock

import networkx as nx
import numpy as np

from vertex_voyage.dynnode2vec import DynNode2Vec
from vertex_voyage.node2vec import Node2Vec
from vertex_voyage.plugins import test_commands
from vertex_voyage.plugins.test_commands import (
    TestCustomCLICommandExecutor,
    WalkParameters,
    persisted_walk_overrides,
    retrain_threshold_of,
)
from vertex_voyage.temporal import Event
from vertex_voyage.walk_strategy import (
    RandomVertexSequenceWalk,
    RepeatStartVertexWalk,
)
from vertex_voyage.word2vec_training import RandomInitialisedWord2Vec

CONFIG_ENVVAR = "VERTEX_VOYAGE_CONFIG"
WALK_ENVVAR = "VERTEX_VOYAGE_WALK_STRATEGY"
WORD2VEC_ENVVAR = "VERTEX_VOYAGE_WORD2VEC_TRAINING"
RETRAIN_ENVVAR = "VERTEX_VOYAGE_RETRAIN_THRESHOLD"
DIM = 4
WALK_SIZE = 10
N_WALKS = 10
WINDOW_SIZE = 3
EPOCHS = 1
SEED = 7


def environment_with(**settings) -> dict:
    handle, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(handle, "w") as config_file:
        json.dump({}, config_file)
    environment = {CONFIG_ENVVAR: path}
    environment.update(settings)
    return environment


def cycle_graph() -> nx.Graph:
    graph = nx.Graph()
    for src, dest in [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]:
        graph.add_edge(src, dest)
    return graph


def cycle_events() -> list:
    return [
        Event(src=src, dest=dest, timestamp=timestamp)
        for timestamp, (src, dest) in enumerate(cycle_graph().edges())
    ]


def node2vec_of(**settings) -> Node2Vec:
    with mock.patch.dict(os.environ, environment_with(**settings), clear=True):
        return Node2Vec(
            dim=DIM,
            walk_size=WALK_SIZE,
            n_walks=N_WALKS,
            window_size=WINDOW_SIZE,
            epochs=EPOCHS,
            seed=SEED,
            use_threads=False,
        )


class TestWalkStrategySetting(unittest.TestCase):
    def test_repeated_start_vertex_walks_never_leave_the_start(self):
        node2vec = node2vec_of(**{WALK_ENVVAR: "repeat_start_vertex"})
        self.assertIsInstance(node2vec.walk_strategy, RepeatStartVertexWalk)
        node2vec.fit(cycle_graph())
        for walk in node2vec.walks:
            self.assertEqual(len(set(walk)), 1)

    def test_random_vertex_sequence_walks_ignore_the_edges(self):
        node2vec = node2vec_of(**{WALK_ENVVAR: "random_vertex_sequence"})
        self.assertIsInstance(
            node2vec.walk_strategy,
            RandomVertexSequenceWalk,
        )
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(3, 4)
        node2vec.fit(graph)
        visited_after_start = {
            vertex for walk in node2vec.walks for vertex in walk[1:]
        }
        self.assertEqual(len(visited_after_start), graph.number_of_nodes())

    def test_embeddings_keep_their_dimension(self):
        node2vec = node2vec_of(**{WALK_ENVVAR: "repeat_start_vertex"})
        node2vec.fit(cycle_graph())
        self.assertEqual(
            np.asarray(node2vec.embed_nodes(cycle_graph().nodes)).shape,
            (cycle_graph().number_of_nodes(), DIM),
        )


class TestWord2VecTrainingSetting(unittest.TestCase):
    def test_node2vec_keeps_its_random_vectors(self):
        node2vec = node2vec_of(**{WORD2VEC_ENVVAR: "random_init_only"})
        self.assertIsInstance(
            node2vec.word2vec_training,
            RandomInitialisedWord2Vec,
        )
        node2vec.fit(cycle_graph())
        embeddings = np.asarray(node2vec.embed_nodes(cycle_graph().nodes))
        self.assertEqual(
            embeddings.shape,
            (cycle_graph().number_of_nodes(), DIM),
        )
        self.assertTrue(np.any(embeddings != 0))

    def test_dynnode2vec_updates_without_training(self):
        environment = environment_with(
            **{WORD2VEC_ENVVAR: "random_init_only"}
        )
        with mock.patch.dict(os.environ, environment, clear=True):
            dynnode2vec = DynNode2Vec(
                dim=DIM,
                walk_size=WALK_SIZE,
                n_walks=N_WALKS,
                window_size=WINDOW_SIZE,
                epochs=EPOCHS,
                seed=SEED,
                use_threads=False,
                retrain_threshold=0,
            )
        dynnode2vec.update(cycle_events())
        embeddings = np.asarray(
            dynnode2vec.embed_nodes(cycle_graph().nodes)
        )
        self.assertEqual(
            embeddings.shape,
            (cycle_graph().number_of_nodes(), DIM),
        )


class TestSparseWalkCoverage(unittest.TestCase):
    def test_every_node_is_embedded_after_a_single_short_walk(self):
        node2vec = node2vec_of()
        node2vec.n_walks = 1
        node2vec.walk_size = 3
        graph = cycle_graph()
        graph.add_edge(5, 6)
        node2vec.fit(graph, graph.nodes)
        self.assertEqual(
            np.asarray(node2vec.embed_nodes(graph.nodes)).shape,
            (graph.number_of_nodes(), DIM),
        )


class TestWalkParameterOverrides(unittest.TestCase):
    def setUp(self):
        self.executor = TestCustomCLICommandExecutor()
        self.long_run = self.executor._walk_params(True)
        self.short_run = self.executor._walk_params(False)

    def test_long_run_and_short_run_defaults_differ(self):
        self.assertEqual(self.long_run, WalkParameters(10, 80, 10))
        self.assertEqual(self.short_run, WalkParameters(1, 10, 3))

    def test_unset_overrides_keep_the_defaults(self):
        self.assertEqual(self.long_run.overridden_by(0, 0, 0), self.long_run)

    def test_one_walk_overrides_only_the_walk_count(self):
        self.assertEqual(
            self.long_run.overridden_by(1, 0, 0),
            WalkParameters(1, 80, 10),
        )

    def test_every_parameter_can_be_overridden(self):
        self.assertEqual(
            self.short_run.overridden_by(2, 20, 5),
            WalkParameters(2, 20, 5),
        )

    def test_negative_override_raises(self):
        for override in [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    self.long_run.overridden_by(*override)

    def test_dataset_without_walk_params_keeps_the_long_run_values(self):
        self.assertEqual(
            self.executor._dataset_walk_params("CITESEER", True, True),
            self.long_run,
        )

    def test_dataset_walk_params_replace_the_long_run_values(self):
        with mock.patch.dict(
            test_commands.dataset_params,
            {"CITESEER": {"n_walks": 4, "walk_size": 40}},
        ):
            self.assertEqual(
                self.executor._dataset_walk_params("CITESEER", True, True),
                WalkParameters(4, 40, self.long_run.window_size),
            )

    def test_dataset_walk_params_are_ignored_when_not_requested(self):
        with mock.patch.dict(
            test_commands.dataset_params,
            {"CITESEER": {"n_walks": 4}},
        ):
            self.assertEqual(
                self.executor._dataset_walk_params("CITESEER", True, False),
                self.long_run,
            )

    def test_explicit_override_wins_over_the_dataset(self):
        with mock.patch.dict(
            test_commands.dataset_params,
            {"CITESEER": {"n_walks": 4}},
        ):
            resolved = self.executor._dataset_walk_params(
                "CITESEER",
                True,
                True,
            ).overridden_by(1, 0, 0)
        self.assertEqual(
            resolved,
            WalkParameters(1, self.long_run.walk_size,
                           self.long_run.window_size),
        )

    def test_only_overridden_parameters_are_persisted(self):
        self.assertEqual(persisted_walk_overrides(0, 0, 0), {})
        self.assertEqual(persisted_walk_overrides(1, 0, 0), {"n_walks": 1})
        self.assertEqual(
            persisted_walk_overrides(1, 20, 5),
            {"n_walks": 1, "walk_size": 20, "window_size": 5},
        )


class TestRetrainThresholdSetting(unittest.TestCase):
    def test_default_threshold_is_a_tenth_of_the_vertices(self):
        with mock.patch.dict(os.environ, environment_with(), clear=True):
            self.assertEqual(retrain_threshold_of(cycle_graph()), 0)

    def test_threshold_above_one_retrains_every_update(self):
        environment = environment_with(**{RETRAIN_ENVVAR: "100"})
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertEqual(
                retrain_threshold_of(cycle_graph()),
                100 * cycle_graph().number_of_nodes(),
            )


if __name__ == "__main__":
    unittest.main()
