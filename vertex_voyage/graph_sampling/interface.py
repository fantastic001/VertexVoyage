import logging
from abc import ABC, abstractmethod
from typing import Generic, Hashable, Iterable, Protocol, Tuple, TypeVar

import numpy as np

logger = logging.getLogger("GraphSampling")

Rng = np.random.Generator
V = TypeVar("V", bound=Hashable)
Edge = Tuple[V, V]


class SamplingError(ValueError):
    pass


def sampling_error(message: str) -> SamplingError:
    logger.debug("Infeasible draw: %s", message)
    return SamplingError(message)


def require_rng(rng: Rng) -> None:
    if isinstance(rng, np.random.Generator):
        return
    else:
        message = f"Expected numpy.random.Generator, got {type(rng)!r}"
        logger.error(message)
        raise TypeError(message)


class AdjacencyGraph(Protocol[V]):
    @property
    def nodes(self) -> Iterable[V]:
        ...

    def neighbors(self, node: V) -> Iterable[V]:
        ...


class GraphSampler(ABC, Generic[V]):
    @abstractmethod
    def node(self, rng: Rng) -> V:
        ...

    @abstractmethod
    def neighbor(self, rng: Rng, node: V) -> V:
        ...

    @abstractmethod
    def non_neighbor(self, rng: Rng, node: V) -> V:
        ...

    @abstractmethod
    def edge(self, rng: Rng) -> Edge:
        ...

    @abstractmethod
    def non_edge(self, rng: Rng) -> Edge:
        ...
