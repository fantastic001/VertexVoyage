
import math
from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Iterable, Optional

import networkx as nx
import numpy as np
from vertex_voyage_native import get_reconstructed_edges

import logging
logger = logging.getLogger(__name__)

UNDEFINED_SCORE = math.nan

def reconstruct(k: int, embedding: list[np.array], nodes = None) -> nx.Graph:
    """
    Reconstructs graph from embedding by taking k most closest nodes and creating links between them.

    Parameters:
    k (int): Number of closest nodes to connect.
    embedding (list[np.array]): List of embeddings of nodes.
    nodes (list): List of nodes. If None, nodes are assumed to be [0, 1, ..., n].

    Returns:
    nx.Graph: Reconstructed undirected graph.
    """
    if nodes is None:
        nodes = [i for i in range(len(embedding))]
    reconstructed_graph = nx.Graph()
    for n in nodes:
        reconstructed_graph.add_node(n)
    reconstructed_edges = get_reconstructed_edges(np.array(embedding, dtype=np.float64), k)
    reconstructed_edges = [(nodes[e[0]], nodes[e[1]]) for e in reconstructed_edges]
    reconstructed_graph.add_edges_from(reconstructed_edges)
    return reconstructed_graph

def get_f1_score(G, reconstructed_graph, weighted: bool = False) -> tuple[float, float, float]:
    def _weight(n):
        return len(list(G.neighbors(n))) if weighted else 1
    nodes = G.nodes()
    weights = {n: _weight(n) for n in nodes}
    N = lambda n: set(G.neighbors(n))
    RN = lambda n: set(reconstructed_graph.neighbors(n))
    recall = sum(
        [
            w*len(N(n).intersection(RN(n))) / len(N(n)) if len(N(n)) > 0 else 0 
            for n, w in weights.items()
        ]
    ) / sum(weights.values())
    nodes_with_reconstructed_neighbors = [n for n in G.nodes() if len(list(reconstructed_graph.neighbors(n))) > 0]
    if len(nodes_with_reconstructed_neighbors) == 0:
        return 0, 0, 0
    precision = sum(
        [
            w * len(N(n).intersection(RN(n))) / len(RN(n)) if len(RN(n)) > 0 else 0
            for n, w in weights.items() 
        ]
    ) / sum(weights.values())
    if precision + recall == 0:
        return 0, 0, 0
    f1 = 2 * (precision * recall) / (precision + recall)
    logger.info(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    return precision, recall, f1

class EdgeScope(Enum):
    WITHIN_PARTITION = "within_partition"
    CROSS_PARTITION = "cross_partition"


def _node_partition_ids(
    partitions: Iterable[Iterable[Hashable]],
) -> dict[Hashable, frozenset[int]]:
    ids: dict[Hashable, set[int]] = {}
    for partition_id, partition in enumerate(partitions):
        for node in partition:
            ids.setdefault(node, set()).add(partition_id)
    return {node: frozenset(node_ids) for node, node_ids in ids.items()}


def _edge_scope(
    u: Hashable,
    v: Hashable,
    node_partition_ids: dict[Hashable, frozenset[int]],
) -> EdgeScope:
    shared_ids = node_partition_ids.get(u, frozenset()) & node_partition_ids.get(
        v, frozenset()
    )
    if len(shared_ids) == 0:
        return EdgeScope.CROSS_PARTITION
    else:
        return EdgeScope.WITHIN_PARTITION


def _scoped_edge_sets(
    graph: nx.Graph,
    node_partition_ids: dict[Hashable, frozenset[int]],
) -> dict[EdgeScope, set[frozenset]]:
    scoped_edges: dict[EdgeScope, set[frozenset]] = {
        scope: set() for scope in EdgeScope
    }
    for u, v in graph.edges():
        scope = _edge_scope(u, v, node_partition_ids)
        scoped_edges[scope].add(frozenset((u, v)))
    return scoped_edges


@dataclass(frozen=True)
class EdgeCounts:
    true_positives: int
    true_edges: int
    reconstructed_edges: int


def _edge_counts(
    original_edges: set[frozenset],
    reconstructed_edges: set[frozenset],
) -> EdgeCounts:
    return EdgeCounts(
        true_positives=len(original_edges & reconstructed_edges),
        true_edges=len(original_edges),
        reconstructed_edges=len(reconstructed_edges),
    )


def _ratio_or_undefined(numerator: float, denominator: float) -> float:
    if math.isnan(numerator) or math.isnan(denominator) or denominator <= 0:
        return UNDEFINED_SCORE
    return numerator / denominator


def _scoped_edge_counts(
    G: nx.Graph,
    reconstructed_graph: nx.Graph,
    node_partition_ids: dict[Hashable, frozenset[int]],
) -> dict[EdgeScope, EdgeCounts]:
    original = _scoped_edge_sets(G, node_partition_ids)
    reconstructed = _scoped_edge_sets(reconstructed_graph, node_partition_ids)
    return {
        scope: _edge_counts(original[scope], reconstructed[scope])
        for scope in EdgeScope
    }


def _pair_count(node_count: int) -> int:
    return node_count * (node_count - 1) // 2


def _edges_within_node_set(graph: nx.Graph, nodes: set) -> set[frozenset]:
    return {
        frozenset((u, v))
        for u, v in graph.edges()
        if u in nodes and v in nodes
    }


@dataclass(frozen=True)
class PairProbabilityTerms:
    p_true_edge: float
    p_reconstructed_edge: float
    p_true_positive: float
    pair_count: int


def _pair_probability_terms(
    counts: EdgeCounts, pair_count: int
) -> PairProbabilityTerms:
    return PairProbabilityTerms(
        p_true_edge=_ratio_or_undefined(counts.true_edges, pair_count),
        p_reconstructed_edge=_ratio_or_undefined(
            counts.reconstructed_edges, pair_count
        ),
        p_true_positive=_ratio_or_undefined(counts.true_positives, pair_count),
        pair_count=pair_count,
    )


def f1_from_pair_probability_terms(terms: PairProbabilityTerms) -> float:
    denominator = terms.p_true_edge + terms.p_reconstructed_edge
    if math.isnan(denominator) or denominator == 0:
        return UNDEFINED_SCORE
    return 2 * terms.p_true_positive / denominator


@dataclass(frozen=True)
class PartitionPairContribution:
    partition_id: int
    weight: float
    terms: PairProbabilityTerms


@dataclass(frozen=True)
class PairProbabilityF1Report:
    within: PairProbabilityTerms
    cross: PairProbabilityTerms
    within_by_partition: tuple[PartitionPairContribution, ...]
    s: float
    cross_f1: float
    alpha: float
    delta: float


def _partition_pair_contribution(
    partition_id: int,
    nodes: set,
    pair_count: int,
    within_pair_count: int,
    G: nx.Graph,
    reconstructed_graph: nx.Graph,
) -> PartitionPairContribution:
    counts = _edge_counts(
        _edges_within_node_set(G, nodes),
        _edges_within_node_set(reconstructed_graph, nodes),
    )
    return PartitionPairContribution(
        partition_id=partition_id,
        weight=_ratio_or_undefined(pair_count, within_pair_count),
        terms=_pair_probability_terms(counts, pair_count),
    )


def get_pair_probability_f1_report(
    G: nx.Graph,
    reconstructed_graph: nx.Graph,
    partitions: Iterable[Iterable[Hashable]],
) -> PairProbabilityF1Report:
    partition_node_sets = [set(partition) for partition in partitions]
    node_partition_ids = _node_partition_ids(partition_node_sets)
    counts_by_scope = _scoped_edge_counts(G, reconstructed_graph, node_partition_ids)

    partition_pair_counts = [
        _pair_count(len(nodes)) for nodes in partition_node_sets
    ]
    within_pair_count = sum(partition_pair_counts)
    total_pair_count = _pair_count(G.number_of_nodes())
    cross_pair_count = total_pair_count - within_pair_count

    within_terms = _pair_probability_terms(
        counts_by_scope[EdgeScope.WITHIN_PARTITION], within_pair_count
    )
    cross_terms = _pair_probability_terms(
        counts_by_scope[EdgeScope.CROSS_PARTITION], cross_pair_count
    )
    within_by_partition = tuple(
        _partition_pair_contribution(
            partition_id,
            nodes,
            pair_count,
            within_pair_count,
            G,
            reconstructed_graph,
        )
        for partition_id, (nodes, pair_count) in enumerate(
            zip(partition_node_sets, partition_pair_counts)
        )
    )

    s = f1_from_pair_probability_terms(within_terms)
    cross_f1 = f1_from_pair_probability_terms(cross_terms)
    total_edge_count = sum(
        counts.true_edges for counts in counts_by_scope.values()
    )
    report = PairProbabilityF1Report(
        within=within_terms,
        cross=cross_terms,
        within_by_partition=within_by_partition,
        s=s,
        cross_f1=cross_f1,
        alpha=_ratio_or_undefined(cross_f1, s),
        delta=_ratio_or_undefined(
            counts_by_scope[EdgeScope.CROSS_PARTITION].true_edges, total_edge_count
        ),
    )
    logger.info(f"Pair probability F1 report: {report}")
    return report


def _split_neighbors_by_scope(
    graph: nx.Graph,
    node: Hashable,
    node_partition_ids: dict[Hashable, frozenset[int]],
) -> tuple[set, set]:
    within: set = set()
    cross: set = set()
    if not graph.has_node(node):
        return within, cross
    for neighbor in graph.neighbors(node):
        if _edge_scope(node, neighbor, node_partition_ids) == EdgeScope.WITHIN_PARTITION:
            within.add(neighbor)
        else:
            cross.add(neighbor)
    return within, cross


def _macro_mean_with_default(ratios: list[tuple[int, int]]) -> float:
    if not ratios:
        return UNDEFINED_SCORE
    total = sum(
        numerator / denominator if denominator > 0 else 0.0
        for numerator, denominator in ratios
    )
    return total / len(ratios)

def _macro_stddev_with_default(ratios: list[tuple[int, int]]) -> float:
    if not ratios:
        return UNDEFINED_SCORE
    stddev = float(np.std([
            numerator / denominator if denominator > 0 else 0.0
            for numerator, denominator in ratios
    ]))
    return stddev 

def _macro_mean_excluding_undefined(ratios: list[tuple[int, int]]) -> float:
    defined = [
        numerator / denominator for numerator, denominator in ratios if denominator > 0
    ]
    if not defined:
        return UNDEFINED_SCORE
    return sum(defined) / len(defined)

def _macro_stddev_excluding_undefined(ratios: list[tuple[int, int]]) -> float:
    defined = [
        numerator / denominator for numerator, denominator in ratios if denominator > 0
    ]
    if not defined:
        return UNDEFINED_SCORE
    return float(np.std(defined))

@dataclass(frozen=True)
class MacroScopedRate:
    within: float
    cross: float
    within_stddev: float
    cross_stddev: float


@dataclass(frozen=True)
class MacroF1Score:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class MacroF1Report:
    global_f1: MacroF1Score
    recall_by_scope: MacroScopedRate
    precision_by_scope: MacroScopedRate
    true_edge_partition_share: MacroScopedRate
    reconstructed_edge_partition_share: MacroScopedRate
    vertices_with_true_neighbors: int
    vertices_with_reconstructed_neighbors: int
    # F1 score weighted by vertex degree
    weighted_f1: MacroF1Score

    jaccard: float


def get_macro_f1_report(
    G: nx.Graph,
    reconstructed_graph: nx.Graph,
    partitions: Iterable[Iterable[Hashable]],
    global_f1: Optional[tuple[float, float, float]] = None,
) -> MacroF1Report:
    node_partition_ids = _node_partition_ids([set(partition) for partition in partitions])

    recall_within: list[tuple[int, int]] = []
    recall_cross: list[tuple[int, int]] = []
    precision_within: list[tuple[int, int]] = []
    precision_cross: list[tuple[int, int]] = []
    true_edge_within_share: list[tuple[int, int]] = []
    reconstructed_edge_within_share: list[tuple[int, int]] = []
    vertices_with_true_neighbors = 0
    vertices_with_reconstructed_neighbors = 0

    weighted_f1s = []

    for node in G.nodes():
        true_within, true_cross = _split_neighbors_by_scope(G, node, node_partition_ids)
        reconstructed_within, reconstructed_cross = _split_neighbors_by_scope(
            reconstructed_graph, node, node_partition_ids
        )
        true_neighbors = true_within | true_cross
        reconstructed_neighbors = reconstructed_within | reconstructed_cross

        recall_within.append((len(true_within & reconstructed_neighbors), len(true_within)))
        recall_cross.append((len(true_cross & reconstructed_neighbors), len(true_cross)))

        precision_within.append(
            (len(reconstructed_within & true_neighbors), len(reconstructed_within))
        )
        precision_cross.append(
            (len(reconstructed_cross & true_neighbors), len(reconstructed_cross))
        )

        true_edge_within_share.append((len(true_within), len(true_neighbors)))
        reconstructed_edge_within_share.append(
            (len(reconstructed_within), len(reconstructed_neighbors))
        )

        if true_neighbors:
            vertices_with_true_neighbors += 1
        if reconstructed_neighbors:
            vertices_with_reconstructed_neighbors += 1

    if global_f1 is None:
        global_f1 = get_f1_score(G, reconstructed_graph)
    weighted_f1 = get_f1_score(G, reconstructed_graph, weighted=True)
    global_precision, global_recall, global_f1_score = global_f1
    weighted_precision, weighted_recall, weighted_f1_score = weighted_f1

    weighted_f1_score_obj = MacroF1Score(
        precision=weighted_precision,
        recall=weighted_recall,
        f1=weighted_f1_score,
    )

    report = MacroF1Report(
        global_f1=MacroF1Score(
            precision=global_precision,
            recall=global_recall,
            f1=global_f1_score,
        ),
        weighted_f1=weighted_f1_score_obj,
        recall_by_scope=MacroScopedRate(
            within=_macro_mean_with_default(recall_within),
            cross=_macro_mean_with_default(recall_cross),
            within_stddev=_macro_stddev_with_default(recall_within),
            cross_stddev=_macro_stddev_with_default(recall_cross)
        ),
        precision_by_scope=MacroScopedRate(
            within=_macro_mean_excluding_undefined(precision_within),
            cross=_macro_mean_excluding_undefined(precision_cross),
            within_stddev=_macro_stddev_with_default(precision_within),
            cross_stddev=_macro_stddev_with_default(precision_cross)
        ),
        true_edge_partition_share=MacroScopedRate(
            within=_macro_mean_excluding_undefined(true_edge_within_share),
            cross=_macro_mean_excluding_undefined(
                [(denom - num, denom) for num, denom in true_edge_within_share]
            ),
            within_stddev=_macro_stddev_excluding_undefined(true_edge_within_share),
            cross_stddev=_macro_stddev_excluding_undefined(
                [(denom - num, denom) for num, denom in true_edge_within_share]
            )

        ),
        reconstructed_edge_partition_share=MacroScopedRate(
            within=_macro_mean_excluding_undefined(reconstructed_edge_within_share),
            cross=_macro_mean_excluding_undefined(
                [(denom - num, denom) for num, denom in reconstructed_edge_within_share]
            ),
            within_stddev=_macro_stddev_excluding_undefined(reconstructed_edge_within_share),
            cross_stddev=_macro_stddev_excluding_undefined(
                [(denom - num, denom) for num, denom, in reconstructed_edge_within_share]
            )
        ),
        vertices_with_true_neighbors=vertices_with_true_neighbors,
        vertices_with_reconstructed_neighbors=vertices_with_reconstructed_neighbors,
        jaccard=(
            len(set(G.edges) & set(reconstructed_graph.edges)) / len(set(G.edges) | set(reconstructed_graph.edges))
        ),
    )
    logger.info(f"Macro F1 report: {report}")
    return report
