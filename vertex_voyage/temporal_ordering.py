"""
Edge ordering strategies for temporal graph loading.

Given an EventStream (or EventSequence), these classes yield the same
Event objects back out in a different order: breadth-first, depth-first
(both delegated to networkx's edge traversal algorithms over an
undirected multigraph), or uniformly at random. Because BFS/DFS ordering
depends on the full graph structure, the input is eagerly consumed once
at construction time.
"""

from random import shuffle
from typing import Union

import networkx as nx

from vertex_voyage.temporal import Event, EventSequence, EventStream, FromIterable


def _as_event_stream(sequence: Union[EventSequence, EventStream]) -> EventStream:
    if isinstance(sequence, EventStream):
        return sequence
    return EventStream(sequence)


def _build_graph(events: list[Event]) -> nx.MultiGraph:
    """
    Builds an undirected networkx MultiGraph from a list of events, one
    edge per event, keeping the originating Event as edge data so it can
    be recovered after traversal.
    """
    G = nx.MultiGraph()
    for event in events:
        G.add_edge(event.src, event.dest, event=event)
    return G


def _traverse(events: list[Event], edge_traversal: callable):
    G = _build_graph(events)
    visited_nodes = set()
    for root in G.nodes():
        if root in visited_nodes:
            continue
        for u, v, key in edge_traversal(G, root):
            visited_nodes.add(u)
            visited_nodes.add(v)
            yield G.edges[u, v, key]["event"]


class BFSOrdering(FromIterable):
    """
    Reorders events using a breadth-first traversal of the underlying
    graph (networkx's edge_bfs). Disconnected components are visited in
    the order their root vertex first appeared in the input stream.
    """

    def __init__(self, sequence: Union[EventSequence, EventStream]):
        events = list(_as_event_stream(sequence))
        super().__init__(_traverse(events, nx.edge_bfs))


class DFSOrdering(FromIterable):
    """
    Reorders events using a depth-first traversal of the underlying
    graph (networkx's edge_dfs). Disconnected components are visited in
    the order their root vertex first appeared in the input stream.
    """

    def __init__(self, sequence: Union[EventSequence, EventStream]):
        events = list(_as_event_stream(sequence))
        super().__init__(_traverse(events, nx.edge_dfs))


class RandomOrdering(FromIterable):
    """
    Reorders events uniformly at random.
    """

    def _shuffled(self, events: list[Event]):
        shuffled = events.copy()
        shuffle(shuffled)
        yield from shuffled

    def __init__(self, sequence: Union[EventSequence, EventStream]):
        events = list(_as_event_stream(sequence))
        super().__init__(self._shuffled(events))
