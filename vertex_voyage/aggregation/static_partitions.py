from typing import (
    Any,
    Collection,
    Dict,
    FrozenSet,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    Tuple,
)

from vertex_voyage.aggregation.interface import (
    Aggregator,
    Embedding,
    NodeEmbedder,
    PartitionKey,
    PartitionModels,
    V,
    aggregation_error,
    logger,
)
from vertex_voyage.graph_sampling import GraphSampler

NO_PARTITIONS: FrozenSet[Any] = frozenset()


class StaticPartitioner(Generic[V, PartitionKey]):
    def __init__(self, parts: Mapping[PartitionKey, Collection[V]]):
        partitions_of: Dict[V, set] = {}
        for partition, vertices in parts.items():
            for vertex in vertices:
                partitions_of.setdefault(vertex, set()).add(partition)
        self._partitions_of: Dict[V, FrozenSet[PartitionKey]] = {
            vertex: frozenset(partitions)
            for vertex, partitions in partitions_of.items()
        }
        logger.debug(
            "Static partitioner holds %d vertices in %d partitions",
            len(self._partitions_of),
            len(parts),
        )

    def get(self, vertex: V) -> FrozenSet[PartitionKey]:
        return self._partitions_of.get(vertex, NO_PARTITIONS)


def models_by_partition_position(
    models: Sequence[NodeEmbedder],
) -> Dict[int, NodeEmbedder]:
    return {
        position: model
        for position, model in enumerate(models)
        if model is not None
    }


def partitioner_of_modelled_parts(
    parts: Sequence[Collection[V]],
    embedding_model: PartitionModels,
) -> StaticPartitioner:
    if all(0 <= position < len(parts) for position in embedding_model):
        return StaticPartitioner(
            {position: parts[position] for position in embedding_model}
        )
    else:
        raise aggregation_error(
            f"Models {sorted(embedding_model)!r} do not match "
            f"{len(parts)} partitions"
        )


def aggregate_static_embeddings(
    aggregator: Aggregator,
    sampler: GraphSampler,
    embedding_model: PartitionModels,
    parts: Sequence[Collection[V]],
    edges: Iterable[Tuple[Hashable, Hashable]],
) -> "StaticAggregation":
    partitioner = partitioner_of_modelled_parts(parts, embedding_model)
    edge_buffer = list(edges)
    logger.info(
        "Updating aggregator on %d edges of %d static partitions",
        len(edge_buffer),
        len(embedding_model),
    )
    aggregator.update(sampler, embedding_model, partitioner, edge_buffer)
    return StaticAggregation(aggregator, embedding_model, partitioner)


class StaticAggregation(Generic[V]):
    def __init__(
        self,
        aggregator: Aggregator,
        embedding_model: PartitionModels,
        partitioner: StaticPartitioner,
    ):
        self._aggregator = aggregator
        self._embedding_model = embedding_model
        self._partitioner = partitioner

    def embed_node(self, vertex: V) -> Embedding:
        return self._aggregator.get(
            self._embedding_model,
            self._partitioner,
            vertex,
        )

    def embed_nodes(self, vertices: Iterable[V]) -> list:
        return [self.embed_node(vertex) for vertex in vertices]
