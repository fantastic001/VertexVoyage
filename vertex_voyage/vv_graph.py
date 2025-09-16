
from typing import Optional

class VVGraph:
    def __init__(self, edges: Optional[dict[int, list[int]]] = None, induced_nodes: Optional[set[int]] = None):
        if edges is None:
            self._edges = dict()
        else:
            self._edges = edges
        self.induced_nodes = set(induced_nodes) if induced_nodes is not None else None


    def add_edge(self, u: int, v: int):
        if u not in self._edges:
            self._edges[u] = []
        if v not in self._edges:
            self._edges[v] = []
        if v not in self._edges[u]:
            self._edges[u].append(v)
        if u not in self._edges[v]:
            self._edges[v].append(u)

    def remove_edge(self, u: int, v: int):
        if self.induced_nodes is not None:
            if u not in self.induced_nodes or v not in self.induced_nodes:
                return
        if u in self._edges and v in self._edges[u]:
            self._edges[u].remove(v)
        if v in self._edges and u in self._edges[v]:
            self._edges[v].remove(u)

    @property
    def nodes(self):
        if self.induced_nodes is not None:
            return self.induced_nodes & set(self._edges.keys())
        return set(self._edges.keys())
    
    def degree(self, node: int) -> int:
        if self.induced_nodes is not None and node not in self.induced_nodes:
            return 0
        if node not in self._edges:
            return 0
        return len(set(self._edges.get(node, [])) & self.nodes)
    
    @property
    def edges(self):
        seen = set()
        for u, neighbors in self._edges.items():
            for v in neighbors:
                if (u, v) not in seen and (v, u) not in seen:
                    if self.induced_nodes is not None:
                        if u in self.induced_nodes and v in self.induced_nodes:
                            yield (u, v)
                            seen.add((u, v))
                    else:
                        yield (u, v)
                    seen.add((u, v))
    def number_of_edges(self) -> int:
        return sum(len(list(filter((lambda x: x in self.nodes), neighbors))) for neighbors in self._edges.values()) // 2
    def number_of_nodes(self) -> int:
        return len(self.nodes)
    
    def subgraph(self, induced_nodes: set[int]) -> 'VVGraph':
        return VVGraph(
            edges=self._edges, 
            induced_nodes=induced_nodes if self.induced_nodes is None else self.induced_nodes & induced_nodes
        )
    def neighbors(self, node: int) -> set[int]:
        if self.induced_nodes is None:
            return set(self._edges.get(node, []))
        if self.induced_nodes is not None and node not in self.induced_nodes:
            return set()
        return set(self._edges.get(node, [])) & self.nodes
    def __iter__(self):
        return iter(self.nodes)
    def __contains__(self, node: int) -> bool:
        if self.induced_nodes is not None and node not in self.induced_nodes:
            return False
        return node in self._edges
    def __getitem__(self, node: int) -> list[int]:
        if self.induced_nodes is not None and node not in self.induced_nodes:
            return []
        if not self.induced_nodes:
            return self._edges.get(node, [])
        return list(set(self._edges.get(node, [])) & self.nodes)