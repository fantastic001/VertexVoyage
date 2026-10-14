from typing import (
    Any,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from vertex_voyage.aggregation.embeddings import unit_vector_or_none
from vertex_voyage.aggregation.interface import (
    EdgeBuffer,
    Embedding,
    NodeEmbedder,
    PartitionModels,
    VertexPartitioner,
    aggregation_error,
)

Replica = Tuple[int, Hashable]
ReplicaLink = Tuple[Replica, Replica]
RowPair = Tuple[int, int]

SOURCE_ATTRIBUTE = "src"
DESTINATION_ATTRIBUTE = "dest"
ENDPOINT_COUNT = 2


def endpoints_of(edge: Any) -> Tuple[Hashable, Hashable]:
    if hasattr(edge, SOURCE_ATTRIBUTE) and hasattr(edge, DESTINATION_ATTRIBUTE):
        return (
            getattr(edge, SOURCE_ATTRIBUTE),
            getattr(edge, DESTINATION_ATTRIBUTE),
        )
    elif isinstance(edge, Sequence) and len(edge) == ENDPOINT_COUNT:
        return (edge[0], edge[1])
    else:
        raise aggregation_error(f"Cannot read endpoints of edge {edge!r}")


class PartitionIndex:
    def __init__(
        self,
        embedding_model: PartitionModels,
        partitioner: VertexPartitioner,
    ):
        self._index_of = {
            partition: index
            for index, partition in enumerate(embedding_model)
        }
        self._partitioner = partitioner

    def indexes_holding(self, vertex: Hashable) -> FrozenSet[int]:
        return frozenset(
            self._index_of[partition]
            for partition in self._partitioner.get(vertex)
            if partition in self._index_of
        )


def links_between_replicas_of_edge(
    source: Hashable,
    target: Hashable,
    source_partitions: Iterable[int],
    target_partitions: Iterable[int],
) -> List[ReplicaLink]:
    shared = sorted(set(source_partitions) & set(target_partitions))
    if len(shared) > 0:
        return [((p, source), (p, target)) for p in shared]
    else:
        return [
            ((p, source), (q, target))
            for p in sorted(source_partitions)
            for q in sorted(target_partitions)
        ]


def links_between_replicas_of_vertex(
    vertex: Hashable,
    partitions: Iterable[int],
) -> List[ReplicaLink]:
    ordered = sorted(partitions)
    return [
        ((p, vertex), (q, vertex))
        for position, p in enumerate(ordered)
        for q in ordered[position + 1:]
    ]


def replica_links_of_buffer(
    partition_index: PartitionIndex,
    buffer: EdgeBuffer,
) -> List[ReplicaLink]:
    edges = [endpoints_of(edge) for edge in buffer]
    partitions_of: Dict[Hashable, FrozenSet[int]] = {
        vertex: partition_index.indexes_holding(vertex)
        for edge in edges
        for vertex in edge
    }
    links: List[ReplicaLink] = []
    for source, target in edges:
        links.extend(
            links_between_replicas_of_edge(
                source,
                target,
                partitions_of[source],
                partitions_of[target],
            )
        )
    for vertex, partitions in partitions_of.items():
        links.extend(links_between_replicas_of_vertex(vertex, partitions))
    return links


class ReplicaTable:
    def __init__(self, models: Sequence[NodeEmbedder]):
        self._models = models
        self._row_of: Dict[Replica, Optional[int]] = {}
        self._vectors: List[Embedding] = []
        self._partitions: List[int] = []

    def __len__(self) -> int:
        return len(self._vectors)

    def partition_count(self) -> int:
        return len(self._models)

    def row_of(self, replica: Replica) -> Optional[int]:
        if replica not in self._row_of:
            self._row_of[replica] = self._append_row_of(replica)
        else:
            pass
        return self._row_of[replica]

    def _append_row_of(self, replica: Replica) -> Optional[int]:
        partition, vertex = replica
        unit = unit_vector_or_none(self._models[partition].embed_node(vertex))
        if unit is None:
            return None
        else:
            self._vectors.append(unit)
            self._partitions.append(partition)
            return len(self._vectors) - 1

    def rows_of_links(self, links: Iterable[ReplicaLink]) -> List[RowPair]:
        row_pairs = [
            (self.row_of(left), self.row_of(right)) for left, right in links
        ]
        return [
            (left, right)
            for left, right in row_pairs
            if left is not None and right is not None
        ]

    def vectors(self) -> np.ndarray:
        return np.asarray(self._vectors, dtype=np.float32)

    def partitions(self) -> np.ndarray:
        return np.asarray(self._partitions, dtype=np.int64)
