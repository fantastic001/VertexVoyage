import random
from typing import Collection, List, Optional

from vertex_voyage.aggregation.embeddings import (
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
    logger,
)
from vertex_voyage.graph_sampling import GraphSampler


def partitions_in_sampling_order(
    partitions: Collection[PartitionKey],
) -> List[PartitionKey]:
    return sorted(partitions, key=str)


def model_of_partition(
    embedding_model: PartitionModels,
    partition: PartitionKey,
):
    if partition in embedding_model:
        return embedding_model[partition]
    else:
        raise aggregation_error(
            f"No embedding model for partition {partition!r}"
        )


class RandomPartitionAggregator(Aggregator[V, PartitionKey]):
    """Embeds every vertex with the model of one partition drawn once."""

    def __init__(self) -> None:
        self._chosen: Optional[PartitionKey] = None

    def update(
        self,
        sampler: GraphSampler,
        embedding_model: PartitionModels,
        partitioner: VertexPartitioner,
        buffer: EdgeBuffer,
    ) -> None:
        self._choose_once(embedding_model)

    def get(
        self,
        embedding_model: PartitionModels,
        partitioner: VertexPartitioner,
        vertex: V,
    ) -> Embedding:
        self._choose_once(embedding_model)
        return model_of_partition(embedding_model, self._chosen).embed_node(
            vertex
        )

    def _choose_once(self, embedding_model: PartitionModels) -> None:
        if self._chosen is not None:
            return
        elif len(embedding_model) > 0:
            self._chosen = random.choice(
                partitions_in_sampling_order(embedding_model)
            )
            logger.info(
                "Embedding every vertex with the model of partition %r out "
                "of %d partitions",
                self._chosen,
                len(embedding_model),
            )
        else:
            raise aggregation_error(
                "Cannot choose a random partition without any embedding model"
            )


class RandomReplicaAggregator(Aggregator[V, PartitionKey]):
    """Embeds a vertex with one of its replicas drawn on every read."""

    def update(
        self,
        sampler: GraphSampler,
        embedding_model: PartitionModels,
        partitioner: VertexPartitioner,
        buffer: EdgeBuffer,
    ) -> None:
        logger.debug(
            "Random replica aggregation keeps no state, ignoring %d "
            "buffered edges",
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
            replica = random.choice(partitions_in_sampling_order(partitions))
            return model_of_partition(embedding_model, replica).embed_node(
                vertex
            )
        else:
            return zero_embedding_of_unplaced_vertex(embedding_model, vertex)
