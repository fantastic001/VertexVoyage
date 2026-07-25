import unittest

import numpy as np

from vertex_voyage.semantic_temporal_partitioning import (
    CosineSimilarityMetric,
    DBSCANAssignment,
    DistanceMetric,
    DynNode2VecSemanticEmbeddingModel,
    KNearestNeighborsAssignment,
    Metric,
    Node2VecSemanticEmbeddingModel,
    PartitionAssignment,
    SemanticEmbeddingModel,
    SemanticTemporalGraphPartitioner,
)
from vertex_voyage.temporal import Event, EventType
from vertex_voyage.temporal_partitioning import InMemoryPartition, Partition


class FakeEmbeddingModel(SemanticEmbeddingModel):
    def __init__(self, vectors: dict[object, np.ndarray], dim: int = 2):
        self._vectors = vectors
        self._dim = dim
        self.update_calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def update(self, buffer: list[Event]) -> None:
        self.update_calls += 1

    def embed_node(self, node) -> np.ndarray:
        if node in self._vectors:
            return self._vectors[node]
        return np.zeros(self._dim)


class DeterministicAssignment(PartitionAssignment):
    def __init__(self, mapping: dict[object, set[Partition]]):
        self.mapping = mapping

    def assign(self, node_embeddings, partitions, metric: Metric, partition_embedding_fn):
        return {node: self.mapping.get(node, set()) for node in node_embeddings}


class TestSemanticFactories(unittest.TestCase):
    def test_node2vec_factory_defaults(self):
        partitions = {InMemoryPartition.empty(1), InMemoryPartition.empty(2)}
        partitioner = SemanticTemporalGraphPartitioner.node2vec(partitions)

        self.assertIsInstance(partitioner.embedding_model, Node2VecSemanticEmbeddingModel)
        self.assertIsInstance(partitioner.metric, CosineSimilarityMetric)
        self.assertIsInstance(partitioner.assignment, KNearestNeighborsAssignment)

    def test_dynnode2vec_factory_defaults(self):
        partitions = {InMemoryPartition.empty(1), InMemoryPartition.empty(2)}
        partitioner = SemanticTemporalGraphPartitioner.dynnode2vec(partitions)

        self.assertIsInstance(partitioner.embedding_model, DynNode2VecSemanticEmbeddingModel)
        self.assertIsInstance(partitioner.metric, DistanceMetric)
        self.assertIsInstance(partitioner.assignment, DBSCANAssignment)

    def test_factory_invalid_metric_raises(self):
        partitions = {InMemoryPartition.empty(1)}
        with self.assertRaises(ValueError):
            SemanticTemporalGraphPartitioner.node2vec(partitions, metric="unknown")

    def test_factory_invalid_assignment_raises(self):
        partitions = {InMemoryPartition.empty(1)}
        with self.assertRaises(ValueError):
            SemanticTemporalGraphPartitioner.node2vec(partitions, assignment="unknown")


class TestSemanticAssignments(unittest.TestCase):
    def test_knn_balances_when_partitions_have_no_centroids_and_mu_positive(self):
        p0 = InMemoryPartition.empty(0)
        p1 = InMemoryPartition.empty(1)
        p2 = InMemoryPartition.empty(2)
        p3 = InMemoryPartition.empty(3)

        vectors = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.8, 0.2]),
            "c": np.array([0.2, 0.8]),
            "d": np.array([0.0, 1.0]),
        }
        embedding_model = FakeEmbeddingModel(vectors)
        partitioner = SemanticTemporalGraphPartitioner(
            partitions={p0, p1, p2, p3},
            embedding_model=embedding_model,
            metric=CosineSimilarityMetric(),
            assignment=KNearestNeighborsAssignment(k=1, mu=1.0),
        )

        events = [
            Event(src="a", dest="b", timestamp=0),
            Event(src="c", dest="d", timestamp=1),
        ]
        partitioner.push(events)

        sizes = sorted([p.size() for p in [p0, p1, p2, p3]])
        self.assertEqual(sizes, [1, 1, 1, 1])

    def test_knn_respects_similarity_metric_direction(self):
        p_left = InMemoryPartition.empty(1)
        p_right = InMemoryPartition.empty(2)
        p_left.add("anchor_left")
        p_right.add("anchor_right")

        vectors = {
            "anchor_left": np.array([-1.0, 0.0]),
            "anchor_right": np.array([1.0, 0.0]),
            "u": np.array([-0.9, 0.0]),
            "v": np.array([0.9, 0.0]),
        }

        embedding_model = FakeEmbeddingModel(vectors)
        partitioner = SemanticTemporalGraphPartitioner(
            partitions={p_left, p_right},
            embedding_model=embedding_model,
            metric=CosineSimilarityMetric(),
            assignment=KNearestNeighborsAssignment(k=1),
        )

        events = [Event(src="u", dest="v", timestamp=0)]
        partitioner.push(events)

        self.assertEqual(partitioner.get("u"), {p_left})
        self.assertEqual(partitioner.get("v"), {p_right})

    def test_dbscan_assigns_every_node_with_similarity_metric(self):
        p1 = InMemoryPartition.empty(1)
        p2 = InMemoryPartition.empty(2)
        p3 = InMemoryPartition.empty(3)

        # Make partition size ordering deterministic for cluster-to-partition mapping.
        p2.add("p2_a")
        p3.add("p3_a")
        p3.add("p3_b")

        vectors = {
            "p2_a": np.array([0.0, 1.0]),
            "p3_a": np.array([1.0, 0.0]),
            "p3_b": np.array([1.0, 0.0]),
            "n1": np.array([0.0, 1.0]),
            "n2": np.array([0.0, 0.95]),
            "n3": np.array([1.0, 0.0]),
            "n4": np.array([0.95, 0.0]),
        }
        embedding_model = FakeEmbeddingModel(vectors)
        partitioner = SemanticTemporalGraphPartitioner(
            partitions={p1, p2, p3},
            embedding_model=embedding_model,
            metric=CosineSimilarityMetric(),
            assignment=DBSCANAssignment(eps=0.2, min_samples=2, reassign_noise=True),
        )

        events = [
            Event(src="n1", dest="n2", timestamp=0),
            Event(src="n3", dest="n4", timestamp=1),
        ]
        partitioner.push(events)

        for node in ["n1", "n2", "n3", "n4"]:
            self.assertTrue(partitioner.get(node))

    def test_dbscan_respects_replication_factor_via_k(self):
        p0 = InMemoryPartition.empty(0)
        p1 = InMemoryPartition.empty(1)
        p2 = InMemoryPartition.empty(2)
        p3 = InMemoryPartition.empty(3)

        vectors = {
            "u": np.array([1.0, 0.0]),
            "v": np.array([1.0, 0.0]),
            "w": np.array([1.0, 0.0]),
        }
        embedding_model = FakeEmbeddingModel(vectors)
        assignment = DBSCANAssignment(eps=0.2, min_samples=2, reassign_noise=True)
        assignment.k = 3
        partitioner = SemanticTemporalGraphPartitioner(
            partitions={p0, p1, p2, p3},
            embedding_model=embedding_model,
            metric=CosineSimilarityMetric(),
            assignment=assignment,
        )

        partitioner.push([Event(src="u", dest="v", timestamp=0), Event(src="v", dest="w", timestamp=1)])

        self.assertEqual(len(partitioner.get("u")), 3)
        self.assertEqual(len(partitioner.get("v")), 3)
        self.assertEqual(len(partitioner.get("w")), 3)


class TestSemanticPushGet(unittest.TestCase):
    def test_push_and_remove_edge(self):
        p1 = InMemoryPartition.empty(1)
        p2 = InMemoryPartition.empty(2)
        vectors = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([1.0, 0.0]),
        }
        embedding_model = FakeEmbeddingModel(vectors)
        assignment = DeterministicAssignment(mapping={"a": {p1}, "b": {p1}})
        partitioner = SemanticTemporalGraphPartitioner(
            partitions={p1, p2},
            embedding_model=embedding_model,
            metric=DistanceMetric(),
            assignment=assignment,
        )

        partitioner.push([Event(src="a", dest="b", timestamp=0, type=EventType.ADD)])

        self.assertEqual(partitioner.get("a"), {p1})
        self.assertEqual(partitioner.get("b"), {p1})
        self.assertTrue(p1.connects("a", "b"))
        self.assertEqual(embedding_model.update_calls, 1)

        partitioner.push([Event(src="a", dest="b", timestamp=1, type=EventType.REMOVE)])

        self.assertFalse(p1.connects("a", "b"))
        self.assertEqual(embedding_model.update_calls, 2)


if __name__ == "__main__":
    unittest.main()