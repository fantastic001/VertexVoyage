import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Sequence

from vertex_voyage.config import get_config_str
from vertex_voyage.timing import TimeMetric

logger = logging.getLogger(__name__)

PER_PARTITION_TRAINING = "per_partition"
SINGLE_MODEL_WITHOUT_CUT_EDGES_TRAINING = "single_model_without_cut_edges"

PartitionModels = Dict[Any, Any]


def within_partition_events(partitioner: Any, buffer: Sequence) -> List[Any]:
    kept = {
        id(event)
        for _, partition_buffer in partitioner.get_partition_buffers(buffer)
        for event in partition_buffer
    }
    return [event for event in buffer if id(event) in kept]


def shared_model_of(models: Mapping[Any, Any]) -> Any:
    if len(models) > 0:
        return next(iter(models.values()))
    else:
        logger.error("No partition model to train on the whole buffer")
        raise ValueError("Cannot train without any partition model")


class PartitionTraining(ABC):
    @abstractmethod
    def models_of_partitions(
        self,
        partitions: Sequence[Any],
        create_model: Callable[[], Any],
    ) -> PartitionModels:
        ...

    @abstractmethod
    def update_models(
        self,
        models: PartitionModels,
        partitioner: Any,
        buffer: Sequence,
    ) -> None:
        ...


class PerPartitionTraining(PartitionTraining):
    def models_of_partitions(
        self,
        partitions: Sequence[Any],
        create_model: Callable[[], Any],
    ) -> PartitionModels:
        return {partition: create_model() for partition in partitions}

    def update_models(
        self,
        models: PartitionModels,
        partitioner: Any,
        buffer: Sequence,
    ) -> None:
        for partition, partition_buffer in partitioner.get_partition_buffers(
            buffer
        ):
            with TimeMetric("model_update_part"):
                models[partition].update(partition_buffer)


class SingleModelWithoutCutEdgesTraining(PartitionTraining):
    def models_of_partitions(
        self,
        partitions: Sequence[Any],
        create_model: Callable[[], Any],
    ) -> PartitionModels:
        shared_model = create_model()
        logger.info(
            "Training a single model on the within-partition edges of %d "
            "partitions",
            len(partitions),
        )
        return {partition: shared_model for partition in partitions}

    def update_models(
        self,
        models: PartitionModels,
        partitioner: Any,
        buffer: Sequence,
    ) -> None:
        events = within_partition_events(partitioner, buffer)
        if len(events) > 0:
            with TimeMetric("model_update_part"):
                shared_model_of(models).update(events)
        else:
            logger.debug(
                "Every edge of the buffer is cut, nothing to train on"
            )


PARTITION_TRAINING_OF_NAME: Dict[str, Callable[[], PartitionTraining]] = {
    PER_PARTITION_TRAINING: PerPartitionTraining,
    SINGLE_MODEL_WITHOUT_CUT_EDGES_TRAINING: (
        SingleModelWithoutCutEdgesTraining
    ),
}


def partition_training_names() -> List[str]:
    return sorted(PARTITION_TRAINING_OF_NAME)


def partition_training_of(name: str) -> PartitionTraining:
    if name in PARTITION_TRAINING_OF_NAME:
        return PARTITION_TRAINING_OF_NAME[name]()
    else:
        logger.error("Unknown partition training %r", name)
        raise ValueError(
            f"Unknown partition training {name!r}, expected one of: "
            f"{', '.join(partition_training_names())}"
        )


def configured_partition_training() -> PartitionTraining:
    return partition_training_of(
        get_config_str(
            "partition_training",
            PER_PARTITION_TRAINING,
            "Whether every partition trains its own model or one model is "
            "trained on the union of the within-partition edges: "
            f"{', '.join(partition_training_names())}",
        )
    )
