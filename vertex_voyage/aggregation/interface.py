import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Collection,
    Generic,
    Hashable,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
)

import numpy as np

from vertex_voyage.graph_sampling import GraphSampler

logger = logging.getLogger("Aggregation")

V = TypeVar("V", bound=Hashable)
PartitionKey = TypeVar("PartitionKey", bound=Hashable)
Embedding = np.ndarray
EdgeBuffer = Sequence[Any]


class AggregationError(ValueError):
    pass


def aggregation_error(message: str) -> AggregationError:
    logger.error(message)
    return AggregationError(message)


class NodeEmbedder(Protocol[V]):
    dim: int

    def embed_node(self, node: V) -> Embedding:
        ...


class VertexPartitioner(Protocol[V, PartitionKey]):
    def get(self, vertex: V) -> Collection[PartitionKey]:
        ...


PartitionModels = Mapping[PartitionKey, NodeEmbedder]


class Aggregator(ABC, Generic[V, PartitionKey]):
    @abstractmethod
    def update(
        self,
        sampler: GraphSampler,
        embedding_model: PartitionModels,
        partitioner: VertexPartitioner,
        buffer: EdgeBuffer,
    ) -> None:
        ...

    @abstractmethod
    def get(
        self,
        embedding_model: PartitionModels,
        partitioner: VertexPartitioner,
        vertex: V,
    ) -> Embedding:
        ...
