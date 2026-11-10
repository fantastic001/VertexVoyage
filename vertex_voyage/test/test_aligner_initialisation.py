import dataclasses
import unittest

import numpy as np
import torch

from vertex_voyage.aggregation import AggregationError
from vertex_voyage.aggregation.aligner_initialisation import (
    IDENTITY_INITIALISATION,
    PROCRUSTES_INITIALISATION,
    RANDOM_INITIALISATION,
    IdentityInitialisation,
    ProcrustesInitialisation,
    RandomInitialisation,
    aligner_initialisation_of,
    load_rotations_into,
)
from vertex_voyage.aggregation.alignment import (
    ContrastiveAlignmentAggregator,
    TripletAlignmentAggregator,
)
from vertex_voyage.aggregation.replica_links import ReplicaTable
from vertex_voyage.aggregation.residual_aligner import ResidualAligner
from vertex_voyage.test.test_alignment_aggregation import (
    DIM,
    SEED,
    SETTINGS,
    MisalignedReplicas,
    random_rotation,
    unit_rows,
)

PARTITIONS = 2
ROWS = 20
NO_TRAINING = 0
TOLERANCE = 1e-5


def unit_inputs() -> torch.Tensor:
    rng = np.random.default_rng(SEED)
    vectors = unit_rows(rng.normal(size=(ROWS, DIM)))
    return torch.as_tensor(vectors, dtype=torch.float32)


def alternating_partitions() -> torch.Tensor:
    return torch.arange(ROWS) % PARTITIONS


def fresh_aligner(hidden_width: int = 2 * DIM) -> ResidualAligner:
    torch.manual_seed(SEED)
    return ResidualAligner(PARTITIONS, DIM, hidden_width)


def untrained_aggregator(factory, initialisation: str):
    settings = dataclasses.replace(
        SETTINGS,
        train_steps=NO_TRAINING,
        initialisation=initialisation,
    )
    aggregator = factory(settings, SEED)
    MisalignedReplicas().train(aggregator)
    return aggregator


class TestInitialisationSelection(unittest.TestCase):
    def test_names_select_initialisations(self):
        cases = (
            (RANDOM_INITIALISATION, RandomInitialisation),
            (IDENTITY_INITIALISATION, IdentityInitialisation),
            (PROCRUSTES_INITIALISATION, ProcrustesInitialisation),
        )
        for name, kind in cases:
            with self.subTest(name=name):
                self.assertIsInstance(aligner_initialisation_of(name), kind)

    def test_unknown_name_raises(self):
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                aligner_initialisation_of("orthogonal")

    def test_aggregator_rejects_unknown_initialisation(self):
        settings = dataclasses.replace(SETTINGS, initialisation="orthogonal")
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                TripletAlignmentAggregator(settings, SEED)

    def test_non_positive_sweeps_raise(self):
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                ProcrustesInitialisation(0)


class TestRandomInitialisation(unittest.TestCase):
    def test_parameters_are_left_unchanged(self):
        model = fresh_aligner()
        before = [p.detach().clone() for p in model.parameters()]
        RandomInitialisation().initialise(model, None, [])
        for old, new in zip(before, model.parameters()):
            torch.testing.assert_close(old, new)


class TestIdentityInitialisation(unittest.TestCase):
    def test_aligner_returns_its_unit_input(self):
        model = fresh_aligner()
        IdentityInitialisation().initialise(model, None, [])
        vectors = unit_inputs()
        with torch.no_grad():
            out = model(vectors, alternating_partitions())
        torch.testing.assert_close(out, vectors)


class TestLoadRotations(unittest.TestCase):
    def rotations(self):
        rng = np.random.default_rng(SEED)
        return [random_rotation(rng) for _ in range(PARTITIONS)]

    def expected_rotated(self, vectors, partitions, rotations):
        expected = vectors.clone()
        for partition, rotation in enumerate(rotations):
            rows = partitions == partition
            matrix = torch.as_tensor(rotation, dtype=torch.float32)
            expected[rows] = vectors[rows] @ matrix
        return expected

    def test_aligner_applies_each_partition_rotation(self):
        for hidden_width in (2 * DIM, 3 * DIM):
            with self.subTest(hidden_width=hidden_width):
                model = fresh_aligner(hidden_width)
                rotations = self.rotations()
                load_rotations_into(model, rotations)
                vectors = unit_inputs()
                partitions = alternating_partitions()
                with torch.no_grad():
                    out = model(vectors, partitions)
                torch.testing.assert_close(
                    out,
                    self.expected_rotated(vectors, partitions, rotations),
                    atol=TOLERANCE,
                    rtol=TOLERANCE,
                )

    def test_identity_rotations_keep_every_layer_trainable(self):
        cases = {
            "all identity": [np.eye(DIM)] * PARTITIONS,
            "reference identity": [np.eye(DIM), self.rotations()[1]],
        }
        for label, rotations in cases.items():
            with self.subTest(label):
                model = fresh_aligner()
                with self.assertLogs("Aggregation", level="INFO"):
                    load_rotations_into(model, rotations)
                vectors = unit_inputs()
                partitions = alternating_partitions()
                out = model(vectors, partitions)
                torch.testing.assert_close(
                    out.detach(),
                    self.expected_rotated(vectors, partitions, rotations),
                    atol=TOLERANCE,
                    rtol=TOLERANCE,
                )
                (out * torch.roll(out, 1, dims=0)).sum().backward()
                for layer in model.first:
                    self.assertGreater(float(layer.weight.grad.abs().max()), 0)
                self.assertGreater(
                    float(model.second.weight.grad.abs().max()),
                    0,
                )

    def test_identity_rotations_are_counted_in_the_log(self):
        model = fresh_aligner()
        rotations = [np.eye(DIM), self.rotations()[1]]
        with self.assertLogs("Aggregation", level="INFO") as logs:
            load_rotations_into(model, rotations)
        self.assertIn("1 of 2 Procrustes rotations", logs.output[0])

    def test_narrow_hidden_layer_is_rejected(self):
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                load_rotations_into(fresh_aligner(DIM), self.rotations())

    def test_wrong_rotation_count_is_rejected(self):
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                load_rotations_into(fresh_aligner(), self.rotations()[:1])


class TestProcrustesInitialisation(unittest.TestCase):
    def test_self_loops_alone_fall_back_to_identity(self):
        replicas = MisalignedReplicas()
        model = fresh_aligner()
        node = replicas.nodes[0]
        with self.assertLogs("Aggregation", level="WARNING"):
            ProcrustesInitialisation(1).initialise(
                model,
                ReplicaTable(list(replicas.models.values())),
                [((0, node), (0, node))],
            )
        vectors = unit_inputs()
        with torch.no_grad():
            out = model(vectors, alternating_partitions())
        torch.testing.assert_close(out, vectors)

    def test_no_anchors_falls_back_to_identity(self):
        model = fresh_aligner()
        with self.assertLogs("Aggregation", level="WARNING"):
            ProcrustesInitialisation(1).initialise(
                model,
                ReplicaTable(list(MisalignedReplicas().models.values())),
                [],
            )
        vectors = unit_inputs()
        with torch.no_grad():
            out = model(vectors, alternating_partitions())
        torch.testing.assert_close(out, vectors)


class TestAggregatorInitialisation(unittest.TestCase):
    def test_procrustes_start_aligns_replicas_before_training(self):
        replicas = MisalignedReplicas()
        cosines = {
            name: replicas.replica_cosine(
                untrained_aggregator(TripletAlignmentAggregator, name)
            )
            for name in (IDENTITY_INITIALISATION, PROCRUSTES_INITIALISATION)
        }
        self.assertGreater(cosines[PROCRUSTES_INITIALISATION], 0.999)
        self.assertLess(cosines[IDENTITY_INITIALISATION], 0.5)

    def test_identity_start_keeps_replicas_unchanged(self):
        replicas = MisalignedReplicas()
        aggregator = untrained_aggregator(
            ContrastiveAlignmentAggregator,
            IDENTITY_INITIALISATION,
        )
        for node in replicas.nodes[:5]:
            unit = replicas.unit("right", node)
            np.testing.assert_allclose(
                aggregator._aligned_replica(1, unit),
                unit,
                atol=TOLERANCE,
            )

    def test_initialisation_is_logged_in_configuration(self):
        settings = dataclasses.replace(
            SETTINGS,
            initialisation=PROCRUSTES_INITIALISATION,
        )
        aggregator = TripletAlignmentAggregator(settings, SEED)
        self.assertIn(
            "initialisation='procrustes'",
            aggregator.configuration().report(),
        )


if __name__ == "__main__":
    unittest.main()
