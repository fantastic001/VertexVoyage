import logging
from abc import ABC, abstractmethod
from typing import (
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

import numpy as np

logger = logging.getLogger("GraphSampling")

Rng = np.random.Generator
V = TypeVar("V", bound=Hashable)
Edge = Tuple[V, V]

SHORTEST_WALK_LENGTH = 1


class SamplingError(ValueError):
    pass


class NoCandidatesError(SamplingError):
    pass


def sampling_error(message: str) -> SamplingError:
    logger.debug("Infeasible draw: %s", message)
    return SamplingError(message)


def no_candidates_error(message: str) -> NoCandidatesError:
    logger.debug("No candidates: %s", message)
    return NoCandidatesError(message)


def require_walk_length(length: int) -> None:
    is_integer = isinstance(length, int) and not isinstance(length, bool)
    if is_integer and length >= SHORTEST_WALK_LENGTH:
        return
    else:
        message = (
            f"Walk length must be an integer of at least "
            f"{SHORTEST_WALK_LENGTH}, got {length!r}"
        )
        logger.error(message)
        raise SamplingError(message)


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

    def walk(self, rng: Rng, start: V, length: int) -> List[V]:
        require_rng(rng)
        require_walk_length(length)
        visited: List[V] = [start]
        previous: Optional[V] = None
        while len(visited) < length:
            current = visited[-1]
            try:
                following = self._next_walk_node(rng, current, previous)
            except NoCandidatesError:
                logger.debug("Walk stopped at dead end %r", current)
                break
            else:
                visited.append(following)
                previous = current
        assert SHORTEST_WALK_LENGTH <= len(visited) <= length
        return visited

    def _next_walk_node(
        self,
        rng: Rng,
        node: V,
        previous: Optional[V],
    ) -> V:
        return self.neighbor(rng, node)
