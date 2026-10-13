from typing import Callable, Dict, List

from vertex_voyage.aggregation.interface import (
    Aggregator,
    aggregation_error,
    logger,
)
from vertex_voyage.aggregation.mean import MeanAggregator
from vertex_voyage.aggregation.procrustes import ProcrustesAggregator
from vertex_voyage.config import get_config_str

AggregatorFactory = Callable[[], Aggregator]

MEAN_AGGREGATOR_NAME = "mean"
CONTRASTIVE_AGGREGATOR_NAME = "contrastive"
TRIPLET_AGGREGATOR_NAME = "triplet"
PROCRUSTES_AGGREGATOR_NAME = "procrustes"
DEFAULT_AGGREGATOR_NAME = get_config_str(
    "default_aggregator",
    MEAN_AGGREGATOR_NAME,
    "Aggregator combining partition embeddings when none is given",
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
