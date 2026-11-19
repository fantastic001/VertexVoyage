import logging
import math
import sys
from abc import ABC, abstractmethod
from typing import Dict, Hashable, Iterator, Mapping, Optional, Sequence, Tuple

from vertex_voyage.temporal import Event

logger = logging.getLogger(__name__)

UNIT_NEIGHBOUR_WEIGHT = 1.0
SMALLEST_DECAY = 0.0
LARGEST_FINITE_EXPONENT = math.log(sys.float_info.max)

Vertex = Hashable
NeighbourWeightsByVertex = Dict[Vertex, Dict[Vertex, float]]


def neighbour_pairs_of(
    events: Sequence[Event],
) -> Iterator[Tuple[Vertex, Vertex, Event]]:
    for event in events:
        if event.src != event.dest:
            yield event.src, event.dest, event
            yield event.dest, event.src, event
        else:
            logger.debug("Self-loop at %r has no neighbour", event.src)


def decayed_weight(decay: float, age: float) -> float:
    exponent = min(-decay * age, LARGEST_FINITE_EXPONENT)
    return math.exp(exponent)


def require_valid_decay(decay: float) -> None:
    if math.isfinite(decay) and decay >= SMALLEST_DECAY:
        return
    else:
        logger.error("Decay %r is not a finite number >= 0", decay)
        raise ValueError(f"Decay must be a finite number >= 0, got {decay!r}")


class NeighbourWeights(ABC):
    @abstractmethod
    def observe(
        self,
        events: Sequence[Event],
        current_timestamp: float,
    ) -> None:
        ...

    @abstractmethod
    def weights_of(self, vertex: Vertex) -> Mapping[Vertex, float]:
        ...


class BufferNeighbours(NeighbourWeights):
    def __init__(self) -> None:
        self.weights_by_vertex: NeighbourWeightsByVertex = {}

    def observe(
        self,
        events: Sequence[Event],
        current_timestamp: float,
    ) -> None:
        self.weights_by_vertex = {}
        for vertex, neighbour, _ in neighbour_pairs_of(events):
            self.weights_by_vertex.setdefault(vertex, {})[neighbour] = (
                UNIT_NEIGHBOUR_WEIGHT
            )

    def weights_of(self, vertex: Vertex) -> Mapping[Vertex, float]:
        return self.weights_by_vertex.get(vertex, {})


class DecayedNeighbours(NeighbourWeights):
    def __init__(self, decay: float) -> None:
        require_valid_decay(decay)
        self.decay = decay
        self.timestamps_by_vertex: NeighbourWeightsByVertex = {}
        self.current_timestamp = 0.0

    def observe(
        self,
        events: Sequence[Event],
        current_timestamp: float,
    ) -> None:
        for vertex, neighbour, event in neighbour_pairs_of(events):
            self.timestamps_by_vertex.setdefault(vertex, {})[neighbour] = (
                event.timestamp
            )
        self.current_timestamp = current_timestamp

    def weights_of(self, vertex: Vertex) -> Mapping[Vertex, float]:
        return {
            neighbour: decayed_weight(
                self.decay,
                self.current_timestamp - timestamp,
            )
            for neighbour, timestamp in self.timestamps_by_vertex.get(
                vertex, {}
            ).items()
        }


def neighbour_weights_of(decay: Optional[float]) -> NeighbourWeights:
    if decay is None:
        return BufferNeighbours()
    else:
        return DecayedNeighbours(decay)
