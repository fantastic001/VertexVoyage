from abc import abstractmethod
from collections import deque
from typing import Deque, List

import numpy as np

from vertex_voyage.aggregation.embeddings import (
    unit_vector_or_none,
    zero_embedding_of_unplaced_vertex,
)
from vertex_voyage.aggregation.interface import (
    Aggregator,
    EdgeBuffer,
    Embedding,
    PartitionKey,
    PartitionModels,
    V,
    VertexPartitioner,
)
from vertex_voyage.aggregation.replica_links import (
    PartitionIndex,
    ReplicaLink,
    ReplicaTable,
    replica_links_of_buffer,
)
from vertex_voyage.config import get_config_int
from vertex_voyage.graph_sampling import GraphSampler

DEFAULT_LINK_MEMORY = get_config_int(
    "alignment_link_memory",
    20000,
    "Most recent replica links an aligning aggregator keeps fitting on",
)
REPLICA_AXIS = 0


class ReplicaAlignmentAggregator(Aggregator[V, PartitionKey]):
    def __init__(self, link_memory: int):
        self._links: Deque[ReplicaLink] = deque(maxlen=link_memory)

    def update(
        self,
        sampler: GraphSampler,
        embedding_model: PartitionModels,
        partitioner: VertexPartitioner,
        buffer: EdgeBuffer,
    ) -> None:
        partition_index = PartitionIndex(embedding_model, partitioner)
        self._links.extend(replica_links_of_buffer(partition_index, buffer))
        table = ReplicaTable(list(embedding_model.values()))
        self._fit_alignment(table, sampler, partition_index)

    def get(
        self,
        embedding_model: PartitionModels,
        partitioner: VertexPartitioner,
        vertex: V,
    ) -> Embedding:
        partition_index = PartitionIndex(embedding_model, partitioner)
        models = list(embedding_model.values())
        aligned = [
            self._aligned_replica(partition, unit)
            for partition, unit in (
                (p, unit_vector_or_none(models[p].embed_node(vertex)))
                for p in sorted(partition_index.indexes_holding(vertex))
            )
            if unit is not None
        ]
        if len(aligned) > 0:
            return self._merge_aligned_replicas(aligned)
        else:
            return zero_embedding_of_unplaced_vertex(embedding_model, vertex)

    @staticmethod
    def _merge_aligned_replicas(aligned: List[Embedding]) -> Embedding:
        total = np.sum(aligned, axis=REPLICA_AXIS)
        merged = unit_vector_or_none(total)
        if merged is None:
            return aligned[0]
        else:
            return merged

    @abstractmethod
    def _fit_alignment(
        self,
        table: ReplicaTable,
        sampler: GraphSampler,
        partition_index: PartitionIndex,
    ) -> None:
        ...

    @abstractmethod
    def _aligned_replica(self, partition: int, unit: Embedding) -> Embedding:
        ...
