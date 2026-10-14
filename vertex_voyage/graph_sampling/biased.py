from typing import Callable, Optional, Sequence

from vertex_voyage.graph_sampling.choice import (
    choose_uniformly,
    choose_weighted,
)
from vertex_voyage.graph_sampling.interface import (
    AdjacencyGraph,
    Rng,
    V,
    no_candidates_error,
    require_rng,
    sampling_error,
)
from vertex_voyage.graph_sampling.snapshot import SnapshotSampler

Bias = Callable[[V, V], float]


class BiasedSampler(SnapshotSampler[V]):
    def __init__(self, graph: AdjacencyGraph, beta: Bias):
        if callable(beta):
            self._beta = beta
        else:
            raise sampling_error(f"Bias {beta!r} is not callable")
        super().__init__(graph)

    def neighbor(
        self,
        rng: Rng,
        node: V,
        previous: Optional[V] = None,
    ) -> V:
        require_rng(rng)
        candidates = self._snapshot.neighbors_of(node)
        if len(candidates) > 0:
            return self._choose_next_node(rng, previous, candidates)
        else:
            raise no_candidates_error(f"Node {node!r} has no neighbors")

    def non_neighbor(
        self,
        rng: Rng,
        node: V,
        previous: Optional[V] = None,
    ) -> V:
        require_rng(rng)
        candidates = self._snapshot.non_neighbors_of(node)
        if len(candidates) > 0:
            return self._choose_next_node(rng, previous, candidates)
        else:
            raise no_candidates_error(
                f"Node {node!r} has no non-neighbors"
            )

    def _choose_next_node(
        self,
        rng: Rng,
        previous: Optional[V],
        candidates: Sequence[V],
    ) -> V:
        if previous is None:
            return choose_uniformly(rng, candidates)
        else:
            weights = [
                self._beta(previous, candidate) for candidate in candidates
            ]
            return choose_weighted(rng, candidates, weights)

    def _next_walk_node(
        self,
        rng: Rng,
        node: V,
        previous: Optional[V],
    ) -> V:
        return self.neighbor(rng, node, previous)
