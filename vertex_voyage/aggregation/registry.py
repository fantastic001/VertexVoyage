from typing import Callable, Dict, List

from vertex_voyage.aggregation.interface import (
    Aggregator,
    aggregation_error,
    logger,
)
from vertex_voyage.aggregation.mean import MeanAggregator
from vertex_voyage.aggregation.procrustes import ProcrustesAggregator
from vertex_voyage.config import get_config_int, get_config_str

AggregatorFactory = Callable[[], Aggregator]

MEAN_AGGREGATOR_NAME = "mean"
CONTRASTIVE_AGGREGATOR_NAME = "contrastive"
TRIPLET_AGGREGATOR_NAME = "triplet"
PROCRUSTES_AGGREGATOR_NAME = "procrustes"
DEFAULT_AGGREGATOR_NAME = get_config_str(
    "default_aggregator",
    TRIPLET_AGGREGATOR_NAME,
    "Aggregator combining partition embeddings when none is given",
)
UNALIGNED_AGGREGATOR_NAME = MEAN_AGGREGATOR_NAME
SINGLE_PARTITION = 1
ALIGNMENT_MINIMUM_PARTITIONS = get_config_int(
    "alignment_minimum_partitions",
    2,
    "Fewest partitions at which the requested aggregator is used; with "
    "fewer there is nothing to align and the mean aggregator is used",
)



def create_contrastive_alignment_aggregator() -> Aggregator:
    from vertex_voyage.aggregation.alignment import (
        ContrastiveAlignmentAggregator,
    )
    return ContrastiveAlignmentAggregator()


def create_triplet_alignment_aggregator() -> Aggregator:
    from vertex_voyage.aggregation.alignment import TripletAlignmentAggregator
    return TripletAlignmentAggregator()


_AGGREGATOR_FACTORIES: Dict[str, AggregatorFactory] = {
    MEAN_AGGREGATOR_NAME: MeanAggregator,
    CONTRASTIVE_AGGREGATOR_NAME: create_contrastive_alignment_aggregator,
    TRIPLET_AGGREGATOR_NAME: create_triplet_alignment_aggregator,
    PROCRUSTES_AGGREGATOR_NAME: ProcrustesAggregator,
}


def aggregator_names() -> List[str]:
    return sorted(_AGGREGATOR_FACTORIES)


def is_aggregator_registered(name: str) -> bool:
    return name in _AGGREGATOR_FACTORIES


def register_aggregator(name: str, factory: AggregatorFactory) -> None:
    if is_aggregator_registered(name):
        raise aggregation_error(f"Aggregator {name!r} is already registered")
    elif not callable(factory):
        raise aggregation_error(
            f"Aggregator factory {factory!r} is not callable"
        )
    else:
        _AGGREGATOR_FACTORIES[name] = factory
        logger.info("Registered aggregator %r", name)


def create_aggregator(name: str) -> Aggregator:
    if is_aggregator_registered(name):
        return _AGGREGATOR_FACTORIES[name]()
    else:
        raise aggregation_error(
            f"Unknown aggregator {name!r}, expected one of: "
            f"{', '.join(aggregator_names())}"
        )


def aggregator_name_for_partitions(
    name: str,
    partition_count: int,
    minimum_partitions: int = ALIGNMENT_MINIMUM_PARTITIONS,
) -> str:
    if not is_aggregator_registered(name):
        raise aggregation_error(
            f"Unknown aggregator {name!r}, expected one of: "
            f"{', '.join(aggregator_names())}"
        )
    elif partition_count < SINGLE_PARTITION:
        raise aggregation_error(
            f"Partition count must be at least {SINGLE_PARTITION}, "
            f"got {partition_count!r}"
        )
    elif partition_count >= minimum_partitions:
        return name
    else:
        logger.info(
            "Aggregating %d partition(s) with %r instead of %r: alignment "
            "needs at least %d partitions",
            partition_count,
            UNALIGNED_AGGREGATOR_NAME,
            name,
            minimum_partitions,
        )
        return UNALIGNED_AGGREGATOR_NAME
