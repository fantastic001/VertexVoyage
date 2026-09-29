import math

import networkx as nx
import pytest

from vertex_voyage.reconstruction import get_f1_score, get_macro_f1_report

PATH_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)]
RECONSTRUCTED_EDGES = [(0, 1), (1, 2), (2, 4)]
TWO_PARTITIONS = [{0, 1, 2}, {3, 4}]


def build_graph(edges: list) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def build_reconstructed_graph(original: nx.Graph, edges: list) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(original.nodes())
    graph.add_edges_from(edges)
    return graph


def test_precomputed_global_f1_is_reused_not_recomputed():
    original = build_graph(PATH_EDGES)
    reconstructed = build_reconstructed_graph(original, RECONSTRUCTED_EDGES)
    deliberately_wrong_global_f1 = (0.1, 0.2, 0.3)

    report = get_macro_f1_report(
        original, reconstructed, TWO_PARTITIONS, global_f1=deliberately_wrong_global_f1
    )

    assert report.global_f1.precision == pytest.approx(0.1)
    assert report.global_f1.recall == pytest.approx(0.2)
    assert report.global_f1.f1 == pytest.approx(0.3)


def test_global_f1_matches_get_f1_score_regardless_of_partitions():
    original = build_graph(PATH_EDGES)
    reconstructed = build_reconstructed_graph(original, RECONSTRUCTED_EDGES)
    expected_precision, expected_recall, expected_f1 = get_f1_score(
        original, reconstructed
    )

    report = get_macro_f1_report(original, reconstructed, TWO_PARTITIONS)

    assert report.global_f1.precision == pytest.approx(expected_precision)
    assert report.global_f1.recall == pytest.approx(expected_recall)
    assert report.global_f1.f1 == pytest.approx(expected_f1)


def test_global_f1_does_not_depend_on_partitioning():
    original = build_graph(PATH_EDGES)
    reconstructed = build_reconstructed_graph(original, RECONSTRUCTED_EDGES)

    single_partition = get_macro_f1_report(original, reconstructed, [set(original.nodes())])
    two_partitions = get_macro_f1_report(original, reconstructed, TWO_PARTITIONS)

    assert single_partition.global_f1 == two_partitions.global_f1


def test_decomposition_matches_hand_computed_values():
    original = build_graph(PATH_EDGES)
    reconstructed = build_reconstructed_graph(original, RECONSTRUCTED_EDGES)

    report = get_macro_f1_report(original, reconstructed, TWO_PARTITIONS)

    assert report.global_f1.precision == pytest.approx(0.625)
    assert report.global_f1.recall == pytest.approx(0.5)
    assert report.global_f1.f1 == pytest.approx(5 / 9)

    assert report.recall_by_scope.within == pytest.approx(0.6)
    assert report.recall_by_scope.cross == pytest.approx(0.0)

    assert report.precision_by_scope.within == pytest.approx(1.0)
    assert report.precision_by_scope.cross == pytest.approx(0.0)

    assert report.true_edge_partition_share.within == pytest.approx(0.8)
    assert report.true_edge_partition_share.cross == pytest.approx(0.2)

    assert report.reconstructed_edge_partition_share.within == pytest.approx(0.625)
    assert report.reconstructed_edge_partition_share.cross == pytest.approx(0.375)

    assert report.vertices_with_true_neighbors == 5
    assert report.vertices_with_reconstructed_neighbors == 4


def test_true_and_reconstructed_shares_sum_to_one():
    original = build_graph(PATH_EDGES)
    reconstructed = build_reconstructed_graph(original, RECONSTRUCTED_EDGES)

    report = get_macro_f1_report(original, reconstructed, TWO_PARTITIONS)

    assert report.true_edge_partition_share.within + report.true_edge_partition_share.cross == pytest.approx(1.0)
    assert (
        report.reconstructed_edge_partition_share.within
        + report.reconstructed_edge_partition_share.cross
    ) == pytest.approx(1.0)


def test_single_partition_has_no_cross_scope():
    original = build_graph(PATH_EDGES)
    reconstructed = build_reconstructed_graph(original, RECONSTRUCTED_EDGES)

    report = get_macro_f1_report(original, reconstructed, [set(original.nodes())])

    assert report.recall_by_scope.cross == pytest.approx(0.0)
    assert math.isnan(report.precision_by_scope.cross)
    assert report.true_edge_partition_share.cross == pytest.approx(0.0)
    assert report.reconstructed_edge_partition_share.cross == pytest.approx(0.0)
    assert report.true_edge_partition_share.within == pytest.approx(1.0)
    assert report.reconstructed_edge_partition_share.within == pytest.approx(1.0)


def test_no_edges_leaves_global_scores_at_zero():
    original = nx.Graph()
    original.add_nodes_from([0, 1])
    reconstructed = nx.Graph()
    reconstructed.add_nodes_from([0, 1])

    report = get_macro_f1_report(original, reconstructed, [{0, 1}])

    assert report.global_f1.recall == pytest.approx(0.0)
    assert report.global_f1.precision == pytest.approx(0.0)
    assert report.global_f1.f1 == pytest.approx(0.0)
    assert report.vertices_with_true_neighbors == 0
    assert report.vertices_with_reconstructed_neighbors == 0


def test_get_f1_score_does_not_raise_without_reconstructed_edges():
    original = nx.Graph()
    original.add_nodes_from([0, 1])
    reconstructed = nx.Graph()
    reconstructed.add_nodes_from([0, 1])

    assert get_f1_score(original, reconstructed) == (0, 0, 0)
