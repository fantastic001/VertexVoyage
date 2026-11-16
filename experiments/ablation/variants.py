import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from experiments.ablation.grid import RF1_DATASETS, Cell, cells_of
from experiments.ablation.settings import BASE_BUFFER_SIZE

logger = logging.getLogger(__name__)

Flags = Dict[str, Optional[str]]
SWITCH = None
PARTITION_TRAINING_ENVVAR = "VERTEX_VOYAGE_PARTITION_TRAINING"
WALK_STRATEGY_ENVVAR = "VERTEX_VOYAGE_WALK_STRATEGY"
WORD2VEC_TRAINING_ENVVAR = "VERTEX_VOYAGE_WORD2VEC_TRAINING"
RETRAIN_THRESHOLD_ENVVAR = "VERTEX_VOYAGE_RETRAIN_THRESHOLD"
ALIGNMENT_TRAIN_STEPS_ENVVAR = "VERTEX_VOYAGE_ALIGNMENT_TRAIN_STEPS"
RETRAIN_EVERY_UPDATE_FRACTION = "100"
UNTRAINED_ALIGNMENT_STEPS = "0"

BASE_FLAGS: Flags = {
    "long-run": SWITCH,
    "use-dataset-params": SWITCH,
    "ordering": "bfs",
    "track-seen": SWITCH,
    "buffer-size": str(BASE_BUFFER_SIZE),
    "partitioner-name": "neighbors.all",
    "mu": "1",
    "alpha": "1",
    "aggregator": "triplet",
    "iterations": "1",
    "skip-full-graph-baseline": SWITCH,
}


@lru_cache(maxsize=None)
def edge_count_of(dataset: str) -> int:
    from experiments.datasets import datasets

    count = sum(1 for _ in datasets[dataset]())
    logger.info("Dataset %s has %d events", dataset, count)
    return count


def dataset_dim_of(dataset: str) -> int:
    from experiments.datasets import dataset_params

    return int(dataset_params[dataset]["dim"])


def no_dataset_flags(dataset: str) -> Flags:
    return {}


def single_buffer_flags(dataset: str) -> Flags:
    return {"buffer-size": str(edge_count_of(dataset))}


def unbiased_walk_flags(dataset: str) -> Flags:
    return {"dim": str(dataset_dim_of(dataset))}


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    flags: Mapping[str, Optional[str]] = field(default_factory=dict)
    removed_flags: FrozenSet[str] = frozenset()
    env: Mapping[str, str] = field(default_factory=dict)
    dataset_flags: Callable[[str], Flags] = no_dataset_flags
    cells: Tuple[Cell, ...] = tuple(cells_of())


VARIANT_LIST: List[Variant] = [
    Variant("baseline", "Full pipeline"),
    Variant(
        "random_edge_order",
        "Randomized edge order",
        flags={"ordering": "random"},
    ),
    Variant(
        "no_edge_order",
        "No edge ordering, events in dataset file order",
        removed_flags=frozenset({"ordering", "track-seen"}),
    ),
    Variant(
        "single_buffer",
        "No stream buffering, one buffer of all events",
        dataset_flags=single_buffer_flags,
    ),
    Variant(
        "unit_buffer",
        "Buffer size is 1",
        flags={"buffer-size": "1"},
    ),
    Variant(
        "random_partitioning",
        "Random partitioning",
        flags={"partitioner-name": "random"},
        cells=tuple(cells_of(RF1_DATASETS)),
    ),
    Variant(
        "no_partitioning",
        "No partitioning, all nodes in a single partition",
        flags={"partitions": "1"},
    ),
    Variant(
        "single_model_without_cut_edges",
        "Single model trained on within-partition edges",
        env={PARTITION_TRAINING_ENVVAR: "single_model_without_cut_edges"},
    ),
    Variant(
        "unbiased_walk",
        "Unbiased random walk, p = q = 1",
        flags={"default-p": "1", "default-q": "1"},
        removed_flags=frozenset({"use-dataset-params"}),
        dataset_flags=unbiased_walk_flags,
    ),
    Variant(
        "random_vertex_sequence",
        "No walk, random sequence of vertices",
        env={WALK_STRATEGY_ENVVAR: "random_vertex_sequence"},
    ),
    Variant(
        "repeat_start_vertex",
        "Walk repeating its start vertex",
        env={WALK_STRATEGY_ENVVAR: "repeat_start_vertex"},
    ),
    Variant("one_walk", "Just one walk", flags={"n-walks": "1"}),
    Variant(
        "random_init_only",
        "Random word2vec weights only",
        env={WORD2VEC_TRAINING_ENVVAR: "random_init_only"},
    ),
    Variant("one_epoch", "Only one epoch", flags={"epochs": "1"}),
    Variant(
        "retrain_every_update",
        "Retrain word2vec on every update",
        env={RETRAIN_THRESHOLD_ENVVAR: RETRAIN_EVERY_UPDATE_FRACTION},
    ),
    Variant(
        "random_partition_aggregator",
        "Select random partition for final embedding",
        flags={"aggregator": "random_partition"},
    ),
    Variant(
        "random_replica_aggregator",
        "Select random partition embedding per vertex",
        flags={"aggregator": "random_replica"},
    ),
    Variant(
        "mean_aggregator",
        "Aggregate with average",
        flags={"aggregator": "mean"},
    ),
    Variant(
        "untrained_triplet_aggregator",
        "Aggregate with triplet without training",
        env={ALIGNMENT_TRAIN_STEPS_ENVVAR: UNTRAINED_ALIGNMENT_STEPS},
    ),
]

VARIANTS: Dict[str, Variant] = {
    variant.name: variant for variant in VARIANT_LIST
}


def variant_names() -> List[str]:
    return list(VARIANTS)


def variant_of(name: str) -> Variant:
    if name in VARIANTS:
        return VARIANTS[name]
    else:
        logger.error("Unknown ablation variant %r", name)
        raise ValueError(
            f"Unknown ablation variant {name!r}, expected one of: "
            f"{', '.join(variant_names())}"
        )
