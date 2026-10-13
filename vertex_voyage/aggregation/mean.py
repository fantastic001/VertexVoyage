from typing import Collection, List

import numpy as np

from vertex_voyage.aggregation.interface import (
    Aggregator,
    EdgeBuffer,
    Embedding,
    PartitionKey,
    PartitionModels,
    V,
    VertexPartitioner,
    aggregation_error,
    logger,
)
from vertex_voyage.graph_sampling import GraphSampler

PARTITION_AXIS = 0


class MeanAggregator(Aggregator[V, PartitionKey]):
    def update(
        self,
        sampler: GraphSampler,
        embedding_model: PartitionModels,
        partitioner: VertexPartitioner,
        buffer: EdgeBuffer,
    ) -> None:
        logger.debug(
            "Mean aggregation keeps no state, ignoring %d buffered edges",
            len(buffer),
        )

    def get(
        self,
        embedding_model: PartitionModels,
        partitioner: VertexPartitioner,
        vertex: V,
    ) -> Embedding:
        partitions = partitioner.get(vertex)
        if partitions:
            return np.mean(
                self._embed_in_each_partition(
                    embedding_model,
                    partitions,
                    vertex,
                ),
                axis=PARTITION_AXIS,
            )
        else:
            logger.warning(
                "Vertex %r is in no partition, returning zero embedding",
                vertex,
            )
            return np.zeros(self._embedding_dimension_of(embedding_model))

    @staticmethod
    def _embed_in_each_partition(
        embedding_model: PartitionModels,
        partitions: Collection[PartitionKey],
        vertex: V,
    ) -> List[Embedding]:
        missing = [p for p in partitions if p not in embedding_model]
        if len(missing) == 0:
            return [embedding_model[p].embed_node(vertex) for p in partitions]
        else:
            raise aggregation_error(
                f"No embedding model for partitions {missing!r}"
            )

    @staticmethod
    def _embedding_dimension_of(embedding_model: PartitionModels) -> int:
        if len(embedding_model) > 0:
            return next(iter(embedding_model.values())).dim
        else:
            raise aggregation_error(
                "Cannot aggregate without any embedding model"
            )
