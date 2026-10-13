import pickle
import unittest

import networkx as nx
import numpy as np
import torch

from vertex_voyage.aggregation import (
    AggregationError,
    Aggregator,
    create_aggregator,
)
from vertex_voyage.aggregation.alignment import (
    AlignmentAggregator,
    AlignmentSettings,
    ContrastiveAlignmentAggregator,
    TripletAlignmentAggregator,
)
from vertex_voyage.aggregation.procrustes import (
    ProcrustesAggregator,
    ProcrustesSettings,
    anchor_links,
    procrustes_rotation,
)
from vertex_voyage.aggregation.replica_links import (
    PartitionIndex,
    ReplicaTable,
    endpoints_of,
    links_between_replicas_of_edge,
    links_between_replicas_of_vertex,
    replica_links_of_buffer,
)
from vertex_voyage.aggregation.residual_aligner import (
    ResidualAligner,
    contrastive_loss,
    triplet_loss,
    weights_of,
)
from vertex_voyage.graph_sampling import UniformSampler
from vertex_voyage.temporal import Event

SEED = 5
DIM = 8
COMMUNITIES = 4
COMMUNITY_SIZE = 12
NOISE = 0.15
SETTINGS = AlignmentSettings(train_steps=300, negative_pool=32)


class DictModel:
    def __init__(self, vectors: dict, dim: int = DIM):
        self.dim = dim
        self._vectors = vectors

    def embed_node(self, node) -> np.ndarray:
        return np.asarray(self._vectors.get(node, np.zeros(self.dim)))


class FixedPartitioner:
    def __init__(self, vertex_partitions: dict):
        self._vertex_partitions = vertex_partitions

    def get(self, vertex):
        return self._vertex_partitions.get(vertex, set())


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    rotation, _ = np.linalg.qr(rng.normal(size=(DIM, DIM)))
    return rotation


class MisalignedReplicas:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.nodes = list(range(COMMUNITIES * COMMUNITY_SIZE))
        self.community = [node // COMMUNITY_SIZE for node in self.nodes]
        centres = unit_rows(rng.normal(size=(COMMUNITIES, DIM)))
        truth = unit_rows(
            centres[self.community]
            + NOISE * rng.normal(size=(len(self.nodes), DIM))
        )
        rotated = truth @ random_rotation(rng).T
        self.models = {
            "left": DictModel(dict(zip(self.nodes, truth))),
            "right": DictModel(dict(zip(self.nodes, rotated))),
        }
        self.partitioner = FixedPartitioner(
            {node: {"left", "right"} for node in self.nodes}
        )
        self.graph = nx.Graph(
            (u, v)
            for u in self.nodes
            for v in self.nodes
            if u < v and self.community[u] == self.community[v]
        )
        self.buffer = list(self.graph.edges)
        self.sampler = UniformSampler(self.graph)

    def replica_cosine(self, aggregator) -> float:
        cosines = [
            float(
                aggregator._aligned_replica(0, self.unit("left", node))
                @ aggregator._aligned_replica(1, self.unit("right", node))
            )
            for node in self.nodes
        ]
        return float(np.mean(cosines))

    def unit(self, partition: str, node: int) -> np.ndarray:
        return self.models[partition].embed_node(node)

    def train(self, aggregator) -> None:
        aggregator.update(
            self.sampler,
            self.models,
            self.partitioner,
            self.buffer,
        )


class TestReplicaLinks(unittest.TestCase):
    def test_endpoints_of_events_and_pairs(self):
        self.assertEqual(endpoints_of(Event(1, 2, 0)), (1, 2))
        self.assertEqual(endpoints_of((3, 4)), (3, 4))
        self.assertEqual(endpoints_of([5, 6]), (5, 6))

    def test_unreadable_edge_raises(self):
        for edge in ((1, 2, 3), 7):
            with self.subTest(edge=edge):
                with self.assertLogs("Aggregation", level="ERROR"):
                    with self.assertRaises(AggregationError):
                        endpoints_of(edge)

    def test_covered_edge_links_replicas_inside_shared_partitions(self):
        links = links_between_replicas_of_edge("u", "v", {0, 1}, {1, 2})
        self.assertEqual(links, [((1, "u"), (1, "v"))])

    def test_cut_edge_links_every_replica_pair(self):
        links = links_between_replicas_of_edge("u", "v", {0, 1}, {2})
        self.assertEqual(
            links,
            [((0, "u"), (2, "v")), ((1, "u"), (2, "v"))],
        )

    def test_vertex_links_each_pair_of_its_replicas_once(self):
        links = links_between_replicas_of_vertex("u", {2, 0, 1})
        self.assertEqual(
            links,
            [
                ((0, "u"), (1, "u")),
                ((0, "u"), (2, "u")),
                ((1, "u"), (2, "u")),
            ],
        )
        self.assertEqual(links_between_replicas_of_vertex("u", {0}), [])

    def test_buffer_links_cover_intra_cut_and_replica_pairs(self):
        models = {"a": DictModel({}), "b": DictModel({})}
        partitioner = FixedPartitioner(
            {1: {"a"}, 2: {"a", "b"}, 3: {"b"}, 4: set()}
        )
        index = PartitionIndex(models, partitioner)
        links = replica_links_of_buffer(index, [(1, 2), (1, 3), (3, 4)])
        self.assertEqual(
            sorted(links),
            [
                ((0, 1), (0, 2)),
                ((0, 1), (1, 3)),
                ((0, 2), (1, 2)),
            ],
        )

    def test_table_skips_replicas_without_embedding(self):
        table = ReplicaTable(
            [DictModel({"u": [3.0, 4.0]}, dim=2), DictModel({}, dim=2)]
        )
        self.assertEqual(table.row_of((0, "u")), 0)
        self.assertIsNone(table.row_of((1, "u")))
        self.assertEqual(
            table.rows_of_links([((0, "u"), (1, "u")), ((0, "u"), (0, "u"))]),
            [(0, 0)],
        )
        np.testing.assert_allclose(table.vectors(), [[0.6, 0.8]])
        np.testing.assert_array_equal(table.partitions(), [0])
        self.assertEqual(len(table), 1)
        self.assertEqual(table.partition_count(), 2)


class TestResidualAligner(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.model = ResidualAligner(2, DIM, 2 * DIM)
        self.vectors = torch.nn.functional.normalize(
            torch.randn(6, DIM),
            dim=-1,
        )
        self.partitions = torch.tensor([0, 0, 0, 1, 1, 1])

    def test_outputs_are_unit_vectors(self):
        out = self.model(self.vectors, self.partitions)
        np.testing.assert_allclose(
            out.norm(dim=1).detach().numpy(),
            np.ones(6),
            rtol=1e-5,
        )

    def test_partition_selects_first_layer(self):
        same_vector = self.vectors[:1].repeat(2, 1)
        out = self.model(same_vector, torch.tensor([0, 1]))
        self.assertFalse(torch.allclose(out[0], out[1]))

    def test_numpy_weights_match_torch_forward(self):
        out = self.model(self.vectors, self.partitions).detach().numpy()
        weights = weights_of(self.model)
        for row, partition in enumerate(self.partitions.tolist()):
            np.testing.assert_allclose(
                weights.align(partition, self.vectors[row].numpy()),
                out[row],
                atol=1e-5,
            )

    def test_losses_vanish_for_separated_clusters(self):
        out = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        links = torch.tensor([[0, 1]])
        self.assertAlmostEqual(
            float(contrastive_loss(out, links, 1e-3)),
            0.0,
            places=5,
        )

    def test_triplet_loss_orders_distances(self):
        out = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        torch.manual_seed(SEED)
        close = float(triplet_loss(out, torch.tensor([[0, 1]]), 0.5))
        torch.manual_seed(SEED)
        far = float(triplet_loss(out, torch.tensor([[0, 2]]), 0.5))
        self.assertLess(close, far)

    def test_losses_are_finite_and_differentiable(self):
        links = torch.tensor([[0, 3], [1, 4], [2, 2]])
        for loss in (contrastive_loss, triplet_loss):
            with self.subTest(loss=loss.__name__):
                value = loss(
                    self.model(self.vectors, self.partitions),
                    links,
                    1.0,
                )
                value.backward()
                self.assertTrue(torch.isfinite(value))
                for parameter in self.model.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())


class TestAlignmentAggregators(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.problem = MisalignedReplicas()
        self.aggregators = {
            "contrastive": ContrastiveAlignmentAggregator(SETTINGS, SEED),
            "triplet": TripletAlignmentAggregator(SETTINGS, SEED),
        }

    def test_are_aggregators(self):
        for name, aggregator in self.aggregators.items():
            with self.subTest(aggregator=name):
                self.assertIsInstance(aggregator, Aggregator)

    def test_registered_by_name(self):
        self.assertIsInstance(
            create_aggregator("contrastive"),
            ContrastiveAlignmentAggregator,
        )
        self.assertIsInstance(
            create_aggregator("triplet"),
            TripletAlignmentAggregator,
        )

    def test_update_brings_replicas_of_a_vertex_together(self):
        for name, aggregator in self.aggregators.items():
            with self.subTest(aggregator=name):
                before = self.problem.replica_cosine(aggregator)
                self.problem.train(aggregator)
                after = self.problem.replica_cosine(aggregator)
                self.assertLess(before, 0.3)
                self.assertGreater(after, before + 0.3)

    def test_contrastive_alignment_makes_replicas_nearly_coincide(self):
        aggregator = self.aggregators["contrastive"]
        self.problem.train(aggregator)
        self.assertGreater(self.problem.replica_cosine(aggregator), 0.8)

    def test_merged_embedding_is_unit_and_keeps_communities(self):
        aggregator = self.aggregators["contrastive"]
        self.problem.train(aggregator)
        merged = np.array(
            [
                aggregator.get(
                    self.problem.models,
                    self.problem.partitioner,
                    node,
                )
                for node in self.problem.nodes
            ]
        )
        np.testing.assert_allclose(
            np.linalg.norm(merged, axis=1),
            np.ones(len(merged)),
        )
        cosines = merged @ merged.T
        community = np.array(self.problem.community)
        same = community[:, None] == community[None, :]
        self.assertGreater(cosines[same].mean(), cosines[~same].mean() + 0.5)

    def test_get_before_update_is_normalised_sum_of_replicas(self):
        models = {
            "left": DictModel({"u": [2.0, 0.0]}, dim=2),
            "right": DictModel({"u": [0.0, 5.0]}, dim=2),
        }
        partitioner = FixedPartitioner({"u": {"left", "right"}})
        embedding = self.aggregators["contrastive"].get(
            models,
            partitioner,
            "u",
        )
        np.testing.assert_allclose(embedding, np.ones(2) / np.sqrt(2))

    def test_vertex_without_replicas_gets_zero_embedding(self):
        for vertex in ("unseen", "unembedded"):
            with self.subTest(vertex=vertex):
                with self.assertLogs("Aggregation", level="WARNING"):
                    embedding = self.aggregators["triplet"].get(
                        {"left": DictModel({})},
                        FixedPartitioner({"unembedded": {"left"}}),
                        vertex,
                    )
                np.testing.assert_array_equal(embedding, np.zeros(DIM))

    def test_update_without_embedded_links_leaves_identity(self):
        aggregator = self.aggregators["contrastive"]
        aggregator.update(
            UniformSampler(nx.Graph()),
            {"left": DictModel({})},
            FixedPartitioner({}),
            [("a", "b")],
        )
        self.assertIsNone(aggregator._weights)

    def test_aligner_survives_pickling(self):
        aggregator = self.aggregators["contrastive"]
        self.problem.train(aggregator)
        restored = pickle.loads(pickle.dumps(aggregator))
        np.testing.assert_allclose(
            restored.get(self.problem.models, self.problem.partitioner, 0),
            aggregator.get(self.problem.models, self.problem.partitioner, 0),
        )

    def test_link_memory_is_bounded(self):
        aggregator = ContrastiveAlignmentAggregator(
            AlignmentSettings(train_steps=1, link_memory=10),
            SEED,
        )
        self.problem.train(aggregator)
        self.assertEqual(len(aggregator._links), 10)

    def test_rejects_margin_outside_unit_sphere_range(self):
        for margin in (0.0, -1.0, 2.5):
            with self.subTest(margin=margin):
                with self.assertLogs("Aggregation", level="ERROR"):
                    with self.assertRaises(AggregationError):
                        AlignmentAggregator(contrastive_loss, margin)


class TestProcrustesAggregator(unittest.TestCase):
    def setUp(self):
        self.problem = MisalignedReplicas()
        self.aggregator = ProcrustesAggregator()

    def test_is_registered_aggregator(self):
        self.assertIsInstance(self.aggregator, Aggregator)
        self.assertIsInstance(
            create_aggregator("procrustes"),
            ProcrustesAggregator,
        )

    def test_rotation_maps_source_onto_target(self):
        rng = np.random.default_rng(SEED)
        source = unit_rows(rng.normal(size=(20, DIM)))
        rotation = random_rotation(rng)
        fitted = procrustes_rotation(source, source @ rotation)
        np.testing.assert_allclose(fitted, rotation, atol=1e-8)
        np.testing.assert_allclose(fitted @ fitted.T, np.eye(DIM), atol=1e-8)

    def test_anchors_prefer_replica_links_over_cut_links(self):
        intra = ((0, "u"), (0, "v"))
        cut = ((0, "u"), (1, "w"))
        replica = ((0, "u"), (1, "u"))
        self.assertEqual(anchor_links([intra, cut, replica]), [replica])
        self.assertEqual(anchor_links([intra, cut]), [cut])
        self.assertEqual(anchor_links([intra]), [])

    def test_update_recovers_exact_rotation_between_partitions(self):
        self.assertLess(self.problem.replica_cosine(self.aggregator), 0.3)
        self.problem.train(self.aggregator)
        self.assertAlmostEqual(
            self.problem.replica_cosine(self.aggregator),
            1.0,
            places=6,
        )

    def test_reference_partition_is_left_unrotated(self):
        self.problem.train(self.aggregator)
        unit = self.problem.unit("left", 0)
        np.testing.assert_allclose(
            self.aggregator._aligned_replica(0, unit),
            unit,
        )

    def test_merged_embedding_equals_reference_embedding(self):
        self.problem.train(self.aggregator)
        for node in self.problem.nodes[:5]:
            np.testing.assert_allclose(
                self.aggregator.get(
                    self.problem.models,
                    self.problem.partitioner,
                    node,
                ),
                self.problem.unit("left", node),
                atol=1e-6,
            )

    def test_cut_edges_align_partitions_without_replicas(self):
        left = [n for n in self.problem.nodes if n % 2 == 0]
        partitioner = FixedPartitioner(
            {
                n: {"left"} if n in left else {"right"}
                for n in self.problem.nodes
            }
        )
        before = self.cross_partition_edge_cosine(partitioner)
        self.aggregator.update(
            self.problem.sampler,
            self.problem.models,
            partitioner,
            self.problem.buffer,
        )
        after = self.cross_partition_edge_cosine(partitioner)
        self.assertGreater(after, before + 0.4)

    def cross_partition_edge_cosine(self, partitioner) -> float:
        embedding = {
            node: self.aggregator.get(
                self.problem.models,
                partitioner,
                node,
            )
            for node in self.problem.nodes
        }
        return float(
            np.mean(
                [
                    embedding[u] @ embedding[v]
                    for u, v in self.problem.buffer
                    if u % 2 != v % 2
                ]
            )
        )

    def test_three_partitions_align_through_a_chain(self):
        rng = np.random.default_rng(SEED)
        truth = unit_rows(rng.normal(size=(30, DIM)))
        nodes = list(range(30))
        models = {
            name: DictModel(dict(zip(nodes, truth @ random_rotation(rng).T)))
            for name in ("a", "b", "c")
        }
        partitioner = FixedPartitioner(
            {n: {"a", "b"} if n < 15 else {"b", "c"} for n in nodes}
        )
        self.aggregator.update(
            UniformSampler(nx.path_graph(30)),
            models,
            partitioner,
            list(nx.path_graph(30).edges),
        )
        for node, (first, second) in ((0, (0, 1)), (20, (1, 2))):
            names = list(models)
            cosine = float(
                self.aggregator._aligned_replica(
                    first,
                    models[names[first]].embed_node(node),
                )
                @ self.aggregator._aligned_replica(
                    second,
                    models[names[second]].embed_node(node),
                )
            )
            self.assertAlmostEqual(cosine, 1.0, places=6)

    def test_update_without_anchors_keeps_identity(self):
        self.aggregator.update(
            UniformSampler(nx.Graph()),
            {"left": DictModel({"a": np.ones(DIM), "b": np.ones(DIM)})},
            FixedPartitioner({"a": {"left"}, "b": {"left"}}),
            [("a", "b")],
        )
        self.assertEqual(self.aggregator._rotations, [])

    def test_survives_pickling(self):
        self.problem.train(self.aggregator)
        restored = pickle.loads(pickle.dumps(self.aggregator))
        np.testing.assert_allclose(
            restored.get(self.problem.models, self.problem.partitioner, 0),
            self.aggregator.get(
                self.problem.models,
                self.problem.partitioner,
                0,
            ),
        )

    def test_sweeps_are_configurable(self):
        aggregator = ProcrustesAggregator(ProcrustesSettings(sweeps=1))
        self.problem.train(aggregator)
        self.assertAlmostEqual(
            self.problem.replica_cosine(aggregator),
            1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
