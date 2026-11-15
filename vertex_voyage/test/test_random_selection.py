import random
import unittest

import numpy as np

from vertex_voyage.aggregation import (
    AggregationError,
    RandomPartitionAggregator,
    RandomReplicaAggregator,
    create_aggregator,
)
from vertex_voyage.temporal_partitioning import (
    InMemoryPartition,
    TemporalGraphPartitioner,
)

DIM = 2
DRAWS = 50
SEED = 0


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


class PartitionFixture(unittest.TestCase):
    def setUp(self):
        self.first = InMemoryPartition.empty(id=0)
        self.second = InMemoryPartition.empty(id=1)
        self.first_embedding = [1.0, 1.0]
        self.second_embedding = [2.0, 2.0]
        self.models = {
            self.first: ConstantModel(
                {"shared": self.first_embedding, "only_first": [3.0, 3.0]}
            ),
            self.second: ConstantModel({"shared": self.second_embedding}),
        }
        self.partitioner = FixedPartitioner(
            {
                "shared": {self.first, self.second},
                "only_first": {self.first},
            }
        )

    def embeddings_of(self, aggregator, vertex, draws=DRAWS):
        random.seed(SEED)
        return [
            tuple(aggregator.get(self.models, self.partitioner, vertex))
            for _ in range(draws)
        ]


class TestRandomPartitionAggregator(PartitionFixture):
    def test_the_same_partition_embeds_every_vertex(self):
        aggregator = RandomPartitionAggregator()
        shared = set(self.embeddings_of(aggregator, "shared"))
        self.assertEqual(len(shared), 1)
        self.assertIn(
            shared.pop(),
            {tuple(self.first_embedding), tuple(self.second_embedding)},
        )

    def test_vertex_outside_the_chosen_partition_is_embedded_by_it(self):
        aggregator = RandomPartitionAggregator()
        random.seed(SEED)
        aggregator.update(None, self.models, self.partitioner, [])
        chosen = aggregator.get(self.models, self.partitioner, "only_first")
        self.assertIn(
            tuple(chosen),
            {(3.0, 3.0), (0.0, 0.0)},
        )

    def test_choice_does_not_change_between_buffers(self):
        aggregator = RandomPartitionAggregator()
        random.seed(SEED)
        aggregator.update(None, self.models, self.partitioner, [])
        first = tuple(aggregator.get(self.models, self.partitioner, "shared"))
        aggregator.update(None, self.models, self.partitioner, [])
        second = tuple(aggregator.get(self.models, self.partitioner, "shared"))
        self.assertEqual(first, second)

    def test_choosing_without_models_raises(self):
        aggregator = RandomPartitionAggregator()
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                aggregator.get({}, self.partitioner, "shared")

    def test_registered_under_its_name(self):
        self.assertIsInstance(
            create_aggregator("random_partition"),
            RandomPartitionAggregator,
        )


class TestRandomReplicaAggregator(PartitionFixture):
    def test_every_replica_of_a_vertex_is_drawn(self):
        aggregator = RandomReplicaAggregator()
        drawn = set(self.embeddings_of(aggregator, "shared"))
        self.assertEqual(
            drawn,
            {tuple(self.first_embedding), tuple(self.second_embedding)},
        )

    def test_vertex_of_one_partition_uses_that_partition(self):
        aggregator = RandomReplicaAggregator()
        drawn = set(self.embeddings_of(aggregator, "only_first"))
        self.assertEqual(drawn, {(3.0, 3.0)})

    def test_unplaced_vertex_gets_a_zero_embedding(self):
        aggregator = RandomReplicaAggregator()
        with self.assertLogs("Aggregation", level="WARNING"):
            embedding = aggregator.get(
                self.models,
                self.partitioner,
                "unplaced",
            )
        self.assertEqual(tuple(embedding), (0.0, 0.0))

    def test_replica_without_a_model_raises(self):
        aggregator = RandomReplicaAggregator()
        partitioner = FixedPartitioner(
            {"shared": {InMemoryPartition.empty(id=7)}}
        )
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                aggregator.get(self.models, partitioner, "shared")

    def test_update_keeps_no_state(self):
        aggregator = RandomReplicaAggregator()
        aggregator.update(None, self.models, self.partitioner, [])
        self.assertEqual(
            set(self.embeddings_of(aggregator, "only_first")),
            {(3.0, 3.0)},
        )

    def test_registered_under_its_name(self):
        self.assertIsInstance(
            create_aggregator("random_replica"),
            RandomReplicaAggregator,
        )


if __name__ == "__main__":
    unittest.main()
