import logging
from typing import List, NamedTuple, Sequence

logger = logging.getLogger(__name__)

SINGLE_REPLICA = 1
TRIPLE_REPLICA = 3
RF1_DATASETS = ("CITESEER", "AstroPh", "DBLP", "Enron")
RF1_PARTITIONS = (1, 2, 4)
RF3_DATASETS = ("CITESEER", "AstroPh")
RF3_PARTITIONS = (4,)


class Cell(NamedTuple):
    dataset: str
    partitions: int
    replication_factor: int


def cells_with_replication(
    datasets: Sequence[str],
    partition_counts: Sequence[int],
    replication_factor: int,
) -> List[Cell]:
    return [
        Cell(dataset, partitions, replication_factor)
        for dataset in datasets
        for partitions in partition_counts
    ]


def cells_of(rf3_datasets: Sequence[str] = RF3_DATASETS) -> List[Cell]:
    return cells_with_replication(
        RF1_DATASETS,
        RF1_PARTITIONS,
        SINGLE_REPLICA,
    ) + cells_with_replication(
        rf3_datasets,
        RF3_PARTITIONS,
        TRIPLE_REPLICA,
    )


def datasets_of(cells: Sequence[Cell]) -> List[str]:
    return list(dict.fromkeys(cell.dataset for cell in cells))


def restricted_cells(
    cells: Sequence[Cell],
    datasets: Sequence[str],
) -> List[Cell]:
    unknown = sorted(set(datasets) - set(datasets_of(cells)))
    if not unknown:
        selected = set(datasets) or set(datasets_of(cells))
        return [cell for cell in cells if cell.dataset in selected]
    else:
        logger.error("Datasets %s are not part of this variant", unknown)
        raise ValueError(
            f"Datasets {unknown} are not part of this variant, expected "
            f"some of: {', '.join(datasets_of(cells))}"
        )
