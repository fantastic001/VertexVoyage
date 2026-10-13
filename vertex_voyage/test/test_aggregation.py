import unittest

import networkx as nx
import numpy as np

from vertex_voyage.aggregation import (
    AggregationError,
    Aggregator,
    MeanAggregator,
)
from vertex_voyage.graph_sampling import UniformSampler
from vertex_voyage.temporal_partitioning import (
    InMemoryPartition,
    TemporalGraphPartitioner,
)

DIM = 3


class ConstantModel:
    def __init__(self, embeddings: dict):
        self.dim = DIM
        self._embeddings = embeddings

    def embed_node(self, node) -> np.ndarray:
        return np.asarray(self._embeddings.get(node, np.zeros(DIM)))


class FixedPartitioner(TemporalGraphPartitioner):
    def __init__(self, vertex_partitions: dict):
        self._vertex_partitions = vertex_partitions

    def push(self, buffer):
        pass

    def get(self, vertex):
        return self._vertex_partitions.get(vertex, set())


class PartitionCountAggregator(Aggregator):
    def update(self, sampler, embedding_model, partitioner, buffer) -> None:
        pass

    def get(self, embedding_model, partitioner, vertex) -> np.ndarray:
        return np.array([float(len(partitioner.get(vertex)))])


class TestMeanAggregator(unittest.TestCase):
    def setUp(self):
        self.first = InMemoryPartition.empty(id=0)
        self.second = InMemoryPartition.empty(id=1)
        self.models = {
            self.first: ConstantModel({"a": [1.0, 2.0, 3.0]}),
            self.second: ConstantModel(
                {"a": [3.0, 4.0, 5.0], "b": [6.0, 6.0, 6.0]}
            ),
        }
        self.partitioner = FixedPartitioner(
            {"a": {self.first, self.second}, "b": {self.second}}
        )
        self.aggregator = MeanAggregator()

    def test_is_aggregator(self):
        self.assertIsInstance(self.aggregator, Aggregator)

    def test_averages_embeddings_across_partitions_of_vertex(self):
        embedding = self.aggregator.get(self.models, self.partitioner, "a")
        np.testing.assert_allclose(embedding, [2.0, 3.0, 4.0])

    def test_single_partition_embedding_is_unchanged(self):
        embedding = self.aggregator.get(self.models, self.partitioner, "b")
        np.testing.assert_allclose(embedding, [6.0, 6.0, 6.0])

    def test_vertex_in_no_partition_gets_zero_embedding(self):
        with self.assertLogs("Aggregation", level="WARNING"):
            embedding = self.aggregator.get(
                self.models,
                self.partitioner,
                "unseen",
            )
        np.testing.assert_array_equal(embedding, np.zeros(DIM))

    def test_partitioner_distributed_embedding_uses_mean_aggregation(
        self,
    ):
        embeddings = self.partitioner.get_distributed_embedding(
            self.models,
            ["a", "b", "unseen"],
        )
        expected = [[2.0, 3.0, 4.0], [6.0, 6.0, 6.0], [0.0, 0.0, 0.0]]
        self.assertEqual(len(embeddings), len(expected))
        for embedding, expected_embedding in zip(embeddings, expected):
            np.testing.assert_allclose(embedding, expected_embedding)

    def test_partitioner_distributed_embedding_uses_given_aggregator(self):
        embeddings = self.partitioner.get_distributed_embedding(
            self.models,
            ["a", "b"],
            aggregator=PartitionCountAggregator(),
        )
        np.testing.assert_array_equal(embeddings, [[2.0], [1.0]])

    def test_update_leaves_aggregation_unchanged(self):
        before = self.aggregator.get(self.models, self.partitioner, "a")
        sampler = UniformSampler(nx.Graph([("a", "b")]))
        returned = self.aggregator.update(
            sampler,
            self.models,
            self.partitioner,
            [("a", "b")],
        )
        after = self.aggregator.get(self.models, self.partitioner, "a")
        self.assertIsNone(returned)
        np.testing.assert_array_equal(before, after)

    def test_missing_partition_model_raises(self):
        models = {self.first: self.models[self.first]}
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                self.aggregator.get(models, self.partitioner, "a")

    def test_zero_embedding_without_models_raises(self):
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                self.aggregator.get({}, self.partitioner, "unseen")


if __name__ == "__main__":
    unittest.main()
