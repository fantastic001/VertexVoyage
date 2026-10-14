from typing import Dict, FrozenSet, Generic, Optional, Tuple

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
    logger,
    require_rng,
    sampling_error,
)


class GraphSnapshot(Generic[V]):
    def __init__(self, graph: AdjacencyGraph):
        self.nodes: Tuple[V, ...] = tuple(graph.nodes)
        self._neighbors: Dict[V, Tuple[V, ...]] = {
            node: tuple(graph.neighbors(node)) for node in self.nodes
        }
        self._neighbor_sets: Dict[V, FrozenSet[V]] = {}
        logger.debug("Snapshotted graph with %d nodes", len(self.nodes))

    def neighbors_of(self, node: V) -> Tuple[V, ...]:
        if node in self._neighbors:
            return self._neighbors[node]
        else:
            raise sampling_error(f"Node {node!r} is not in the graph")

    def neighbor_set_of(self, node: V) -> FrozenSet[V]:
        if node not in self._neighbor_sets:
            self._neighbor_sets[node] = frozenset(self.neighbors_of(node))
        else:
            pass
        return self._neighbor_sets[node]

    def is_non_neighbor(self, node: V, candidate: V) -> bool:
        is_other_node = candidate != node
        return is_other_node and candidate not in self.neighbor_set_of(node)

    def non_neighbors_of(self, node: V) -> Tuple[V, ...]:
        return tuple(
            candidate
            for candidate in self.nodes
            if self.is_non_neighbor(node, candidate)
        )

    def non_neighbor_count_of(self, node: V) -> int:
        excluded = len(self.neighbor_set_of(node) | {node})
        return len(self.nodes) - excluded


class SnapshotSampler(GraphSampler[V]):
    def __init__(self, graph: AdjacencyGraph):
        self._snapshot: GraphSnapshot = GraphSnapshot(graph)
        self._arc_counts_by_source = tuple(
            len(self._snapshot.neighbors_of(node))
            for node in self._snapshot.nodes
        )
        self._non_arc_counts_by_source = tuple(
            self._snapshot.non_neighbor_count_of(node)
            for node in self._snapshot.nodes
        )

    def node(self, rng: Rng) -> V:
        require_rng(rng)
        return choose_uniformly(rng, self._snapshot.nodes)

    def edge(self, rng: Rng) -> Edge:
        require_rng(rng)
        source = choose_weighted(
            rng,
            self._snapshot.nodes,
            self._arc_counts_by_source,
        )
        return (source, self.neighbor(rng, source))

    def non_edge(self, rng: Rng) -> Edge:
        require_rng(rng)
        source = choose_weighted(
            rng,
            self._snapshot.nodes,
            self._non_arc_counts_by_source,
        )
        return (source, self.non_neighbor(rng, source))
