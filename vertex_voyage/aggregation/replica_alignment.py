from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Sequence

import numpy as np

from vertex_voyage.aggregation.embeddings import (
    NORM_FLOOR,
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
    aggregation_error,
)
from vertex_voyage.aggregation.replica_links import (
    PartitionIndex,
    ReplicaLink,
    ReplicaTable,
    replica_links_of_buffer,
)
from vertex_voyage.config import get_config_int, get_config_str
from vertex_voyage.graph_sampling import GraphSampler

REPLICA_AXIS = 0
UNIT_LENGTH = 1.0

OutputScale = Callable[[Sequence[float]], float]

UNIT_OUTPUT_SCALE = "unit"
REPLICA_NORM_OUTPUT_SCALE = "replica_norm"
OUTPUT_SCALES: Dict[str, OutputScale] = {
    UNIT_OUTPUT_SCALE: lambda replica_norms: UNIT_LENGTH,
    REPLICA_NORM_OUTPUT_SCALE: np.mean,
}
AGGREGATOR_OUTPUT_SCALE = "auto"


@dataclass(frozen=True)
class ReplicaAlignmentSettings:
    link_memory: int = get_config_int(
        "alignment_link_memory",
        20000,
        "Most recent replica links an aligning aggregator keeps fitting on",
    )
    output_scale: str = get_config_str(
        "alignment_output_scale",
        AGGREGATOR_OUTPUT_SCALE,
        "Length of a merged embedding: unit, replica_norm (mean norm of its "
        "replicas) or auto (the choice of each aggregator)",
    )


class ReplicaAlignmentAggregator(Aggregator[V, PartitionKey]):
    def __init__(
        self,
        settings: ReplicaAlignmentSettings,
        aggregator_output_scale: str,
    ):
        if settings.output_scale == AGGREGATOR_OUTPUT_SCALE:
            self._output_scale_name = aggregator_output_scale
        else:
            self._output_scale_name = settings.output_scale
        if self._output_scale_name in OUTPUT_SCALES:
            self._links: Deque[ReplicaLink] = deque(
                maxlen=settings.link_memory
            )
        else:
            raise aggregation_error(
                f"Unknown output scale {self._output_scale_name!r}, "
                f"expected one of: {', '.join(sorted(OUTPUT_SCALES))}"
            )

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
        replicas = [
            (partition, raw, float(np.linalg.norm(raw)))
            for partition, raw in (
                (p, np.asarray(models[p].embed_node(vertex), dtype=float))
                for p in sorted(partition_index.indexes_holding(vertex))
            )
        ]
        embedded = [replica for replica in replicas if replica[2] > NORM_FLOOR]
        if len(embedded) > 0:
            merged = self._merge_aligned_replicas(
                [
                    self._aligned_replica(partition, raw / norm)
                    for partition, raw, norm in embedded
                ]
            )
            output_scale = OUTPUT_SCALES[self._output_scale_name]
            return merged * output_scale([norm for _, _, norm in embedded])
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
