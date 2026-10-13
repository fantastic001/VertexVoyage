from vertex_voyage.config import get_config_int
from vertex_voyage.graph_sampling.choice import choose_uniformly
from vertex_voyage.graph_sampling.interface import (
    AdjacencyGraph,
    Edge,
    GraphSampler,
    Rng,
    V,
    no_candidates_error,
    require_rng,
    sampling_error,
)
from vertex_voyage.graph_sampling.snapshot import (
    ArcSourceSampler,
    GraphSnapshot,
)

MAX_REJECTION_ATTEMPTS = get_config_int(
    "graph_sampling_max_rejection_attempts",
    16,
    "Rejection draws of a non-neighbor before enumerating all of them",
)


class UniformSampler(GraphSampler[V]):
    def __init__(self, graph: AdjacencyGraph):
        self._snapshot: GraphSnapshot = GraphSnapshot(graph)
        self._sources: ArcSourceSampler = ArcSourceSampler(self._snapshot)

    def node(self, rng: Rng) -> V:
        require_rng(rng)
        if len(self._snapshot.nodes) > 0:
            return choose_uniformly(rng, self._snapshot.nodes)
        else:
            raise sampling_error("Graph has no nodes")

    def neighbor(self, rng: Rng, node: V) -> V:
        require_rng(rng)
        neighbors = self._snapshot.neighbors_of(node)
        if len(neighbors) > 0:
            return choose_uniformly(rng, neighbors)
        else:
            raise no_candidates_error(f"Node {node!r} has no neighbors")

    def non_neighbor(self, rng: Rng, node: V) -> V:
        require_rng(rng)
        if self._snapshot.non_neighbor_count_of(node) > 0:
            return self._draw_existing_non_neighbor(rng, node)
        else:
            raise no_candidates_error(
                f"Node {node!r} has no non-neighbors"
            )

    def _draw_existing_non_neighbor(self, rng: Rng, node: V) -> V:
        for _ in range(MAX_REJECTION_ATTEMPTS):
            candidate = choose_uniformly(rng, self._snapshot.nodes)
            if self._snapshot.is_non_neighbor(node, candidate):
                return candidate
            else:
                continue
        return choose_uniformly(rng, self._snapshot.non_neighbors_of(node))

    def edge(self, rng: Rng) -> Edge:
        require_rng(rng)
        source = self._sources.edge_source(rng)
        return (source, self.neighbor(rng, source))

    def non_edge(self, rng: Rng) -> Edge:
        require_rng(rng)
        source = self._sources.non_edge_source(rng)
        return (source, self.non_neighbor(rng, source))
