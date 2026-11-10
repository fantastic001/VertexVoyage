from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.linalg import orthogonal_procrustes

from vertex_voyage.aggregation.interface import (
    Embedding,
    PartitionKey,
    V,
    logger,
)
from vertex_voyage.aggregation.replica_alignment import (
    REPLICA_NORM_OUTPUT_SCALE,
    ReplicaAlignmentAggregator,
    ReplicaAlignmentSettings,
)
from vertex_voyage.aggregation.replica_links import (
    PartitionIndex,
    ReplicaLink,
    ReplicaTable,
    RowPair,
)
from vertex_voyage.config import get_config_int
from vertex_voyage.graph_sampling import GraphSampler

REFERENCE_PARTITION = 0
ROTATION_CHANGE_TOLERANCE = 1e-9
NO_CHANGE = 0.0
LEFT = 0
RIGHT = 1


@dataclass(frozen=True)
class ProcrustesSettings(ReplicaAlignmentSettings):
    sweeps: int = get_config_int(
        "alignment_procrustes_sweeps",
        10,
        "Passes over the partitions when fitting their rotations",
    )


def is_cut_link(link: ReplicaLink) -> bool:
    (left_partition, _), (right_partition, _) = link
    return left_partition != right_partition


def is_replica_link(link: ReplicaLink) -> bool:
    (_, left_vertex), (_, right_vertex) = link
    return is_cut_link(link) and left_vertex == right_vertex


def anchor_links(links: Sequence[ReplicaLink]) -> List[ReplicaLink]:
    replica_links = [link for link in links if is_replica_link(link)]
    if len(replica_links) > 0:
        return replica_links
    else:
        return [link for link in links if is_cut_link(link)]


class RotationFit:
    def __init__(self, table: ReplicaTable, row_pairs: Sequence[RowPair]):
        self._raw = table.vectors().astype(float)
        self._partitions = table.partitions()
        pairs = np.asarray(row_pairs, dtype=np.int64).reshape(-1, 2)
        self._pairs = np.vstack([pairs, pairs[:, ::-1]])
        self._aligned = self._raw.copy()
        self._settled_partitions = {REFERENCE_PARTITION}
        self.rotations = [
            np.eye(self._raw.shape[1])
            for _ in range(table.partition_count())
        ]

    def sweep_over_partitions(self) -> float:
        return max(
            [NO_CHANGE]
            + [
                self._refit_rotation_of(partition)
                for partition in range(len(self.rotations))
                if partition != REFERENCE_PARTITION
            ]
        )

    def _anchors_from_settled_partitions(self, partition: int) -> np.ndarray:
        starts_here = self._partitions[self._pairs[:, LEFT]] == partition
        ends_settled = np.isin(
            self._partitions[self._pairs[:, RIGHT]],
            sorted(self._settled_partitions),
        )
        return self._pairs[starts_here & ends_settled]

    def _refit_rotation_of(self, partition: int) -> float:
        mine = self._anchors_from_settled_partitions(partition)
        if len(mine) > 0:
            rotation, _ = orthogonal_procrustes(
                self._raw[mine[:, LEFT]],
                self._aligned[mine[:, RIGHT]],
            )
            change = float(np.abs(rotation - self.rotations[partition]).max())
            rows = self._partitions == partition
            self._aligned[rows] = self._raw[rows] @ rotation
            self.rotations[partition] = rotation
            self._settled_partitions.add(partition)
            return change
        else:
            logger.debug(
                "Partition %d has no anchors to settled partitions yet",
                partition,
            )
            return NO_CHANGE

    def mean_anchor_cosine(self) -> float:
        left = self._aligned[self._pairs[:, LEFT]]
        right = self._aligned[self._pairs[:, RIGHT]]
        return float((left * right).sum(axis=1).mean())


def fitted_rotations_of(
    table: ReplicaTable,
    links: Sequence[ReplicaLink],
    sweeps: int,
) -> Optional[List[np.ndarray]]:
    row_pairs = table.rows_of_links(anchor_links(links))
    if len(row_pairs) > 0:
        fit = RotationFit(table, row_pairs)
        for _ in range(sweeps):
            if fit.sweep_over_partitions() <= ROTATION_CHANGE_TOLERANCE:
                break
            else:
                continue
        logger.info(
            "Rotations fitted on %d anchors, mean anchor cosine %f",
            len(row_pairs),
            fit.mean_anchor_cosine(),
        )
        return fit.rotations
    else:
        logger.debug("No cross-partition anchors, no rotations fitted")
        return None


class ProcrustesAggregator(ReplicaAlignmentAggregator[V, PartitionKey]):
    def __init__(self, settings: Optional[ProcrustesSettings] = None):
        self._settings = ProcrustesSettings() if settings is None else settings
        super().__init__(self._settings, REPLICA_NORM_OUTPUT_SCALE)
        self._rotations: List[np.ndarray] = []

    def _fit_alignment(
        self,
        table: ReplicaTable,
        sampler: GraphSampler,
        partition_index: PartitionIndex,
    ) -> None:
        rotations = fitted_rotations_of(
            table,
            list(self._links),
            self._settings.sweeps,
        )
        if rotations is not None:
            self._rotations = rotations
        else:
            logger.debug("Rotations kept from the previous fit")

    def _aligned_replica(self, partition: int, unit: Embedding) -> Embedding:
        if partition < len(self._rotations):
            return unit @ self._rotations[partition]
        else:
            return unit
