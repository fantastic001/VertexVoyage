import json
import os
import tempfile
import unittest
from unittest import mock

from vertex_voyage.partition_training import (
    PER_PARTITION_TRAINING,
    SINGLE_MODEL_WITHOUT_CUT_EDGES_TRAINING,
    PerPartitionTraining,
    SingleModelWithoutCutEdgesTraining,
    configured_partition_training,
    partition_training_names,
    partition_training_of,
    shared_model_of,
    within_partition_events,
)
from vertex_voyage.temporal import Event

CONFIG_ENVVAR = "VERTEX_VOYAGE_CONFIG"
TRAINING_ENVVAR = "VERTEX_VOYAGE_PARTITION_TRAINING"
FIRST_PARTITION = "P0"
SECOND_PARTITION = "P1"


class RecordingModel:
    def __init__(self):
        self.updates = []

    def update(self, events):
        self.updates.append(list(events))


class ReplicatedPartitioner:
    def __init__(self, partitions_of_event):
        self._partitions_of_event = partitions_of_event

    def get_partition_buffers(self, buffer):
        buffers = {}
        for event in buffer:
            for partition in self._partitions_of_event(event):
                buffers.setdefault(partition, []).append(event)
        return list(buffers.items())


def event_of(src, dest) -> Event:
    return Event(src=src, dest=dest, timestamp=0)


def empty_config_environment() -> dict:
    handle, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(handle, "w") as config_file:
        json.dump({}, config_file)
    return {CONFIG_ENVVAR: path}


class PartitionBufferFixture(unittest.TestCase):
    def setUp(self):
        self.within_first = event_of("a", "b")
        self.replicated = event_of("b", "c")
        self.cut = event_of("c", "d")
        self.buffer = [self.within_first, self.replicated, self.cut]
        self.partitioner = ReplicatedPartitioner(self._partitions_of_event)

    def _partitions_of_event(self, event):
        return {
            id(self.within_first): [FIRST_PARTITION],
            id(self.replicated): [FIRST_PARTITION, SECOND_PARTITION],
            id(self.cut): [],
        }[id(event)]


class TestWithinPartitionEvents(PartitionBufferFixture):
    def test_cut_edges_are_dropped_and_replicas_counted_once(self):
        self.assertEqual(
            within_partition_events(self.partitioner, self.buffer),
            [self.within_first, self.replicated],
        )

    def test_buffer_of_cut_edges_only_is_empty(self):
        partitioner = ReplicatedPartitioner(lambda event: [])
        self.assertEqual(
            within_partition_events(partitioner, self.buffer),
            [],
        )


class TestPerPartitionTraining(PartitionBufferFixture):
    def setUp(self):
        super().setUp()
        self.training = PerPartitionTraining()
        self.models = self.training.models_of_partitions(
            [FIRST_PARTITION, SECOND_PARTITION],
            RecordingModel,
        )

    def test_every_partition_gets_its_own_model(self):
        self.assertIsNot(
            self.models[FIRST_PARTITION],
            self.models[SECOND_PARTITION],
        )

    def test_every_model_is_updated_with_its_own_edges(self):
        self.training.update_models(self.models, self.partitioner, self.buffer)
        self.assertEqual(
            self.models[FIRST_PARTITION].updates,
            [[self.within_first, self.replicated]],
        )
        self.assertEqual(
            self.models[SECOND_PARTITION].updates,
            [[self.replicated]],
        )


class TestSingleModelWithoutCutEdgesTraining(PartitionBufferFixture):
    def setUp(self):
        super().setUp()
        self.training = SingleModelWithoutCutEdgesTraining()
        self.models = self.training.models_of_partitions(
            [FIRST_PARTITION, SECOND_PARTITION],
            RecordingModel,
        )

    def test_every_partition_shares_one_model(self):
        self.assertIs(
            self.models[FIRST_PARTITION],
            self.models[SECOND_PARTITION],
        )

    def test_shared_model_is_updated_once_without_cut_edges(self):
        self.training.update_models(self.models, self.partitioner, self.buffer)
        self.assertEqual(
            shared_model_of(self.models).updates,
            [[self.within_first, self.replicated]],
        )

    def test_buffer_of_cut_edges_only_leaves_the_model_untouched(self):
        partitioner = ReplicatedPartitioner(lambda event: [])
        self.training.update_models(self.models, partitioner, self.buffer)
        self.assertEqual(shared_model_of(self.models).updates, [])

    def test_training_without_a_model_raises(self):
        with self.assertRaises(ValueError):
            shared_model_of({})


class TestPartitionTrainingRegistry(unittest.TestCase):
    def test_every_name_creates_its_training(self):
        self.assertEqual(
            partition_training_names(),
            [PER_PARTITION_TRAINING, SINGLE_MODEL_WITHOUT_CUT_EDGES_TRAINING],
        )
        self.assertIsInstance(
            partition_training_of(PER_PARTITION_TRAINING),
            PerPartitionTraining,
        )
        self.assertIsInstance(
            partition_training_of(SINGLE_MODEL_WITHOUT_CUT_EDGES_TRAINING),
            SingleModelWithoutCutEdgesTraining,
        )

    def test_unknown_training_raises(self):
        with self.assertRaises(ValueError):
            partition_training_of("missing")

    def test_configured_training_defaults_to_per_partition(self):
        with mock.patch.dict(
            os.environ,
            empty_config_environment(),
            clear=True,
        ):
            self.assertIsInstance(
                configured_partition_training(),
                PerPartitionTraining,
            )

    def test_configured_training_is_read_from_the_environment(self):
        environment = empty_config_environment()
        environment[TRAINING_ENVVAR] = SINGLE_MODEL_WITHOUT_CUT_EDGES_TRAINING
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertIsInstance(
                configured_partition_training(),
                SingleModelWithoutCutEdgesTraining,
            )


if __name__ == "__main__":
    unittest.main()
