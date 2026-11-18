import unittest

import networkx as nx
import numpy as np

from vertex_voyage.aggregation import create_aggregator
from vertex_voyage.aggregation.registry import TRIPLET_AGGREGATOR_NAME
from vertex_voyage.partition_training import PerPartitionTraining
import vertex_voyage.plugins.test_commands as test_commands
from vertex_voyage.plugins.test_commands import (
    ITERATION_RESULT_NAMES,
    WalkParameters,
)
from vertex_voyage.seeding import seed_shared_random_state
from vertex_voyage.tasks.prequential_link_prediction import (
    POOLED_RESULT_NAME,
    REPORTS_RESULT_NAME,
    DisabledLinkPrediction,
    PrequentialLinkPrediction,
)
from vertex_voyage.temporal import Event, to_nx_graph
from vertex_voyage.temporal_partitioning import (
    PartitionerProfile,
    RandomPartitioner,
)

SEED = 17
DIM = 8
PARTITIONS = 2
BUFFER_SIZE = 10


def karate_events():
    edges = list(nx.karate_club_graph().edges)
    return [Event(src=u, dest=v, timestamp=t) for t, (u, v) in enumerate(edges)]


class StreamRun:
    def __init__(self, link_prediction_model: str):
        self.executor = test_commands.TestCustomCLICommandExecutor()
        events = karate_events()
        original_graph = to_nx_graph(events)
        seed_shared_random_state(SEED)
        training = PerPartitionTraining()
        self.models = self.executor._create_temporal_models(
            algorithm="dynnode2vec",
            partitions=PARTITIONS,
            dim=DIM,
            epochs=1,
            default_p=1,
            default_q=1,
            walk_params=WalkParameters(2, 5, 2),
            original_graph=original_graph,
            seed=SEED,
            training=training,
        )
        self.partitioner = PartitionerProfile(
            RandomPartitioner.uniform(set(self.models.keys()))
        )
        self.aggregator = create_aggregator(TRIPLET_AGGREGATOR_NAME)
        self.link_prediction = self.executor._create_link_prediction(
            link_prediction_model, DIM, SEED
        )
        self.scores = self.executor._process_temporal_buffers(
            nx=nx,
            sorted_events=events,
            buffer_size=BUFFER_SIZE,
            partitioner=self.partitioner,
            models=self.models,
            original_graph=original_graph,
            iteration=0,
            aggregator=self.aggregator,
            training=training,
            link_prediction=self.link_prediction,
        )
        self.vertices = sorted(original_graph.nodes)

    def final_embedding(self) -> np.ndarray:
        return np.array(
            self.partitioner.get_distributed_embedding(
                self.models, self.vertices, self.aggregator
            )
        )


class TestLinkPredictionDoesNotChangeTheEmbedding(unittest.TestCase):
    def test_same_scores_and_embedding_with_and_without_link_prediction(self):
        without = StreamRun("")
        with_link_prediction = StreamRun("hadamard")
        self.assertIsInstance(without.link_prediction, DisabledLinkPrediction)
        self.assertIsInstance(
            with_link_prediction.link_prediction, PrequentialLinkPrediction
        )
        self.assertEqual(without.scores, with_link_prediction.scores)
        np.testing.assert_array_equal(
            without.final_embedding(), with_link_prediction.final_embedding()
        )
        results = with_link_prediction.link_prediction.results()
        self.assertEqual(
            len(results[REPORTS_RESULT_NAME]),
            (len(karate_events()) + BUFFER_SIZE - 1) // BUFFER_SIZE,
        )
        self.assertIn("all", results[POOLED_RESULT_NAME])


class TestTemporalLinkPredictionConfiguration(unittest.TestCase):
    def setUp(self):
        self.executor = test_commands.TestCustomCLICommandExecutor()

    def test_unsupported_models_fail_before_any_work(self):
        for model in ("random_forest", "unknown"):
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    self.executor.temporal_test(
                        name="NO_SUCH_DATASET", link_prediction_model=model
                    )

    def test_supported_models_are_accepted(self):
        for model in ("", "hadamard", "bilinear"):
            with self.subTest(model=model):
                self.executor._validate_temporal_link_prediction_model(model)

    def test_checkpoint_parameters_only_name_an_enabled_model(self):
        self.assertEqual(
            self.executor._persisted_link_prediction_params(""), {}
        )
        self.assertEqual(
            self.executor._persisted_link_prediction_params("bilinear"),
            {"link_prediction_model": "bilinear"},
        )

    def test_cached_iterations_need_link_prediction_results(self):
        disabled = self.executor._create_link_prediction("", DIM, SEED)
        enabled = self.executor._create_link_prediction("bilinear", DIM, SEED)
        self.assertEqual(
            self.executor._iteration_result_keys(3, disabled),
            [f"{name}_3" for name in ITERATION_RESULT_NAMES],
        )
        self.assertEqual(
            self.executor._iteration_result_keys(3, enabled)[-2:],
            [f"{REPORTS_RESULT_NAME}_3", f"{POOLED_RESULT_NAME}_3"],
        )

    def test_results_are_stored_and_read_back(self):
        run = {}
        enabled = self.executor._create_link_prediction("hadamard", DIM, SEED)
        pooled = self.executor._persist_link_prediction_results(run, 0, enabled)
        self.assertEqual(pooled, {})
        self.assertEqual(
            sorted(run), [f"{POOLED_RESULT_NAME}_0", f"{REPORTS_RESULT_NAME}_0"]
        )
        self.assertEqual(
            self.executor._stored_pooled_link_prediction(run, 0, enabled), {}
        )
        disabled = self.executor._create_link_prediction("", DIM, SEED)
        self.assertEqual(
            self.executor._persist_link_prediction_results({}, 0, disabled), {}
        )
        self.assertEqual(
            self.executor._stored_pooled_link_prediction({}, 0, disabled), {}
        )

    def test_summary_logs_every_pooled_scope(self):
        pooled = {"all": {"auc": 0.5, "average_precision": 0.6, "f1": 0.7}}
        self.executor._log_link_prediction_summary([pooled, pooled])
        self.executor._log_link_prediction_summary([{}, {}])


if __name__ == "__main__":
    unittest.main()
