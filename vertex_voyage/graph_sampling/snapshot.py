from typing import Dict, FrozenSet, Generic, List, Tuple

from vertex_voyage.graph_sampling.choice import CumulativeDistribution
from vertex_voyage.graph_sampling.interface import (
    AdjacencyGraph,
    Rng,
    V,
    logger,
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

    def degree_of(self, node: V) -> int:
        return len(self.neighbors_of(node))

    def non_neighbor_count_of(self, node: V) -> int:
        excluded = len(self.neighbor_set_of(node) | {node})
        return len(self.nodes) - excluded


class ArcSourceSampler(Generic[V]):
    def __init__(self, snapshot: GraphSnapshot):
        self._snapshot = snapshot
        self._edge_sources = CumulativeDistribution(
            self._count_per_node(snapshot.degree_of)
        )
        self._non_edge_sources = CumulativeDistribution(
            self._count_per_node(snapshot.non_neighbor_count_of)
        )

    def _count_per_node(self, count_of) -> List[int]:
        return [count_of(node) for node in self._snapshot.nodes]

    def edge_source(self, rng: Rng) -> V:
        if self._edge_sources.total_weight > 0:
            return self._snapshot.nodes[self._edge_sources.draw_index(rng)]
        else:
            raise sampling_error("Graph has no edges")

    def non_edge_source(self, rng: Rng) -> V:
        if self._non_edge_sources.total_weight > 0:
            index = self._non_edge_sources.draw_index(rng)
            return self._snapshot.nodes[index]
        else:
            raise sampling_error("Graph has no non-edges")
