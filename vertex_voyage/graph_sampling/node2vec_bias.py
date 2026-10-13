import math
from typing import Generic

from vertex_voyage.graph_sampling.interface import (
    AdjacencyGraph,
    V,
    sampling_error,
)
from vertex_voyage.graph_sampling.snapshot import GraphSnapshot

COMMON_NEIGHBOR_WEIGHT = 1.0


def inverse_of_walk_parameter(name: str, value: float) -> float:
    if math.isfinite(value) and value > 0:
        return 1.0 / value
    else:
        raise sampling_error(
            f"Node2Vec parameter {name} must be positive, got {value!r}"
        )


class Node2VecBias(Generic[V]):
    def __init__(self, graph: AdjacencyGraph, p: float, q: float):
        self._snapshot: GraphSnapshot = GraphSnapshot(graph)
        self._return_weight = inverse_of_walk_parameter("p", p)
        self._in_out_weight = inverse_of_walk_parameter("q", q)

    def __call__(self, previous: V, candidate: V) -> float:
        if candidate == previous:
            return self._return_weight
        elif candidate in self._snapshot.neighbor_set_of(previous):
            return COMMON_NEIGHBOR_WEIGHT
        else:
            return self._in_out_weight
