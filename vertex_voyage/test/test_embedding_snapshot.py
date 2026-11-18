import unittest

import numpy as np

from vertex_voyage.tasks.link_prediction import (
    DatasetGenerator,
    EmbeddingSnapshot,
)
import networkx as nx

DIM = 3


def vectors_by_vertex():
    return {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0]),
    }


class TestEmbeddingSnapshot(unittest.TestCase):
    def test_snapshot_does_not_follow_later_changes_of_the_source(self):
        source = vectors_by_vertex()
        snapshot = EmbeddingSnapshot.of(source)
        source["a"][0] = 42.0
        np.testing.assert_array_equal(snapshot.embed_node("a"), [1, 0, 0])

    def test_returned_vectors_cannot_change_the_snapshot(self):
        snapshot = EmbeddingSnapshot.of(vectors_by_vertex())
        with self.assertRaises(ValueError):
            snapshot.embed_node("a")[0] = 42.0

    def test_embed_nodes_follows_the_requested_order(self):
        snapshot = EmbeddingSnapshot.of(vectors_by_vertex())
        np.testing.assert_array_equal(
            snapshot.embed_nodes(["c", "a"]), [[0, 0, 1], [1, 0, 0]]
        )

    def test_aligned_vertices_and_vectors(self):
        snapshot = EmbeddingSnapshot(
            [7, 3], [np.ones(DIM), np.zeros(DIM)], DIM
        )
        self.assertEqual(snapshot.vertices(), [7, 3])
        self.assertIn(7, snapshot)
        self.assertNotIn(5, snapshot)
        self.assertEqual(len(snapshot), 2)
        np.testing.assert_array_equal(snapshot.embed_node(3), np.zeros(DIM))

    def test_unknown_vertex_is_rejected(self):
        snapshot = EmbeddingSnapshot.of(vectors_by_vertex())
        with self.assertRaises(KeyError):
            snapshot.embed_node("z")
        with self.assertRaises(KeyError):
            snapshot.embed_nodes(["a", "z"])

    def test_empty_snapshot(self):
        snapshot = EmbeddingSnapshot.empty(DIM)
        self.assertEqual(len(snapshot), 0)
        self.assertEqual(snapshot.dim, DIM)
        self.assertEqual(snapshot.embed_nodes([]).shape, (0, DIM))

    def test_invalid_snapshots_are_rejected(self):
        invalid = {
            "non-positive dimension": lambda: EmbeddingSnapshot.empty(0),
            "fewer vectors than vertices": lambda: EmbeddingSnapshot(
                [1, 2], [np.ones(DIM)], DIM
            ),
            "wrong dimension": lambda: EmbeddingSnapshot(
                [1], [np.ones(DIM + 1)], DIM
            ),
            "duplicate vertex": lambda: EmbeddingSnapshot(
                [1, 1], [np.ones(DIM), np.ones(DIM)], DIM
            ),
            "empty mapping": lambda: EmbeddingSnapshot.of({}),
        }
        for case, create in invalid.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    create()

    def test_dataset_generator_reads_the_snapshot(self):
        graph = nx.Graph([("a", "b"), ("b", "c")])
        generator = DatasetGenerator(
            graph, EmbeddingSnapshot.of(vectors_by_vertex())
        )
        u, v, y = generator.generate_data([("a", "b")], [("a", "c")])
        np.testing.assert_array_equal(u, [[1, 0, 0], [1, 0, 0]])
        np.testing.assert_array_equal(v, [[0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(y, [1, 0])


if __name__ == "__main__":
    unittest.main()
