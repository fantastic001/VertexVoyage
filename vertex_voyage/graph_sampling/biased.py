from typing import Callable, Optional, Sequence

from vertex_voyage.graph_sampling.choice import (
    choose_uniformly,
    choose_weighted,
)
from vertex_voyage.graph_sampling.interface import (
    AdjacencyGraph,
    Edge,
    GraphSampler,
    Rng,
    V,
    require_rng,
    sampling_error,
)
from vertex_voyage.graph_sampling.snapshot import (
    ArcSourceSampler,
    GraphSnapshot,
)

Bias = Callable[[V, V], float]


class BiasedSampler(GraphSampler[V]):
    def __init__(self, graph: AdjacencyGraph, beta: Bias):
        if callable(beta):
            self._beta = beta
        else:
            raise sampling_error(f"Bias {beta!r} is not callable")
        self._snapshot: GraphSnapshot = GraphSnapshot(graph)
        self._sources: ArcSourceSampler = ArcSourceSampler(self._snapshot)

    def node(self, rng: Rng) -> V:
        require_rng(rng)
        if len(self._snapshot.nodes) > 0:
            return choose_uniformly(rng, self._snapshot.nodes)
        else:
            raise sampling_error("Graph has no nodes")

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
            raise sampling_error(f"Node {node!r} has no neighbors")

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
            raise sampling_error(f"Node {node!r} has no non-neighbors")

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

    def edge(self, rng: Rng) -> Edge:
        require_rng(rng)
        source = self._sources.edge_source(rng)
        return (source, self.neighbor(rng, source))

    def non_edge(self, rng: Rng) -> Edge:
        require_rng(rng)
        source = self._sources.non_edge_source(rng)
        return (source, self.non_neighbor(rng, source))
