import math

import networkx as nx
import pytest

from vertex_voyage.reconstruction import (
    PairProbabilityTerms,
    f1_from_pair_probability_terms,
    get_pair_probability_f1_report,
)

NAN = math.nan
PATH_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)]
RECONSTRUCTED_EDGES = [(0, 1), (1, 2), (2, 4)]
TWO_PARTITIONS = [{0, 1, 2}, {3, 4}]


def build_graph(edges: list) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def pooled_edge_f1(graph: nx.Graph, reconstructed: nx.Graph) -> float:
    original_edges = {frozenset(e) for e in graph.edges()}
    reconstructed_edges = {frozenset(e) for e in reconstructed.edges()}
    true_positives = len(original_edges & reconstructed_edges)
    denominator = len(original_edges) + len(reconstructed_edges)
    return 2 * true_positives / denominator


def approx_terms(terms: PairProbabilityTerms, expected: tuple) -> None:
    expected_terms = PairProbabilityTerms(*expected)
    assert terms.pair_count == expected_terms.pair_count
    assert (terms.p_true_edge, terms.p_reconstructed_edge, terms.p_true_positive) == pytest.approx(
        (
            expected_terms.p_true_edge,
            expected_terms.p_reconstructed_edge,
            expected_terms.p_true_positive,
        ),
        nan_ok=True,
    )


def test_within_and_cross_terms_match_hand_computed_probabilities():
    report = get_pair_probability_f1_report(
        build_graph(PATH_EDGES), build_graph(RECONSTRUCTED_EDGES), TWO_PARTITIONS
    )

    approx_terms(report.within, (0.75, 0.5, 0.5, 4))
    approx_terms(report.cross, (1 / 6, 1 / 6, 0.0, 6))


def test_s_and_cross_f1_fields_match_formula_applied_to_terms():
    report = get_pair_probability_f1_report(
        build_graph(PATH_EDGES), build_graph(RECONSTRUCTED_EDGES), TWO_PARTITIONS
    )

    assert report.s == pytest.approx(
        f1_from_pair_probability_terms(report.within)
    )
    assert report.cross_f1 == pytest.approx(
        f1_from_pair_probability_terms(report.cross)
    )
    assert report.s == pytest.approx(0.8)
    assert report.cross_f1 == pytest.approx(0.0)


def test_alpha_and_delta_match_hand_computed_edge_based_values():
    report = get_pair_probability_f1_report(
        build_graph(PATH_EDGES), build_graph(RECONSTRUCTED_EDGES), TWO_PARTITIONS
    )

    assert report.alpha == pytest.approx(report.cross_f1 / report.s)
    assert report.alpha == pytest.approx(0.0)
    assert report.delta == pytest.approx(0.25)


def test_per_partition_contributions_match_hand_computed_probabilities():
    report = get_pair_probability_f1_report(
        build_graph(PATH_EDGES), build_graph(RECONSTRUCTED_EDGES), TWO_PARTITIONS
    )

    first, second = report.within_by_partition
    assert first.partition_id == 0
    assert first.weight == pytest.approx(0.75)
    approx_terms(first.terms, (2 / 3, 2 / 3, 2 / 3, 3))

    assert second.partition_id == 1
    assert second.weight == pytest.approx(0.25)
    approx_terms(second.terms, (1.0, 0.0, 0.0, 1))


def test_per_partition_weights_sum_to_one_when_partitions_do_not_overlap():
    report = get_pair_probability_f1_report(
        build_graph(PATH_EDGES), build_graph(RECONSTRUCTED_EDGES), TWO_PARTITIONS
    )

    assert sum(c.weight for c in report.within_by_partition) == pytest.approx(1.0)


def test_micro_averaged_terms_equal_weighted_sum_of_per_partition_terms():
    report = get_pair_probability_f1_report(
        build_graph(PATH_EDGES), build_graph(RECONSTRUCTED_EDGES), TWO_PARTITIONS
    )

    weighted_true_edge = sum(
        c.weight * c.terms.p_true_edge for c in report.within_by_partition
    )
    weighted_reconstructed_edge = sum(
        c.weight * c.terms.p_reconstructed_edge for c in report.within_by_partition
    )
    weighted_true_positive = sum(
        c.weight * c.terms.p_true_positive for c in report.within_by_partition
    )

    assert weighted_true_edge == pytest.approx(report.within.p_true_edge)
    assert weighted_reconstructed_edge == pytest.approx(
        report.within.p_reconstructed_edge
    )
    assert weighted_true_positive == pytest.approx(report.within.p_true_positive)
    assert (
        2
        * weighted_true_positive
        / (weighted_true_edge + weighted_reconstructed_edge)
    ) == pytest.approx(report.s)


def test_macro_average_of_per_partition_f1_does_not_match_s():
    report = get_pair_probability_f1_report(
        build_graph(PATH_EDGES), build_graph(RECONSTRUCTED_EDGES), TWO_PARTITIONS
    )

    macro_average = sum(
        c.weight * f1_from_pair_probability_terms(c.terms)
        for c in report.within_by_partition
    )

    assert macro_average != pytest.approx(report.s)


def test_single_partition_within_terms_equal_pooled_edge_f1():
    graph = build_graph(PATH_EDGES)
    reconstructed = build_graph(RECONSTRUCTED_EDGES)

    report = get_pair_probability_f1_report(
        graph, reconstructed, [set(graph.nodes())]
    )

    assert report.cross.pair_count == 0
    assert math.isnan(report.cross.p_true_edge)
    assert report.s == pytest.approx(pooled_edge_f1(graph, reconstructed))
    assert report.s == pytest.approx(4 / 7)


def test_global_f1_matches_weighted_average_of_within_and_cross_f1():
    graph = build_graph(PATH_EDGES)
    reconstructed = build_graph(RECONSTRUCTED_EDGES)

    report = get_pair_probability_f1_report(graph, reconstructed, TWO_PARTITIONS)
    within_edge_mass = (
        report.within.p_true_edge + report.within.p_reconstructed_edge
    ) * report.within.pair_count
    cross_edge_mass = (
        report.cross.p_true_edge + report.cross.p_reconstructed_edge
    ) * report.cross.pair_count
    total_edge_mass = within_edge_mass + cross_edge_mass
    weight_within = within_edge_mass / total_edge_mass
    weight_cross = cross_edge_mass / total_edge_mass

    weighted_average = weight_within * report.s + weight_cross * report.cross_f1

    assert weighted_average == pytest.approx(pooled_edge_f1(graph, reconstructed))
    assert weighted_average == pytest.approx(4 / 7)


def test_f1_from_pair_probability_terms_is_undefined_without_pairs():
    assert math.isnan(
        f1_from_pair_probability_terms(PairProbabilityTerms(NAN, NAN, NAN, 0))
    )


def test_f1_from_pair_probability_terms_is_undefined_without_any_edge():
    assert math.isnan(
        f1_from_pair_probability_terms(PairProbabilityTerms(0.0, 0.0, 0.0, 5))
    )


def test_overlapping_partitions_do_not_raise_and_stay_within_report_shape():
    graph = build_graph(PATH_EDGES)
    reconstructed = build_graph(RECONSTRUCTED_EDGES)

    report = get_pair_probability_f1_report(
        graph, reconstructed, [{0, 1, 2}, {2, 3, 4}]
    )

    assert len(report.within_by_partition) == 2
    assert all(
        c.terms.pair_count >= 0 for c in report.within_by_partition
    )


def test_no_partitions_puts_all_pairs_in_cross_scope():
    graph = build_graph(PATH_EDGES)
    reconstructed = build_graph(RECONSTRUCTED_EDGES)

    report = get_pair_probability_f1_report(graph, reconstructed, [])

    assert report.within_by_partition == ()
    assert report.within.pair_count == 0
    assert math.isnan(report.within.p_true_edge)
    assert report.cross.pair_count == _pair_count(graph.number_of_nodes())


def _pair_count(node_count: int) -> int:
    return node_count * (node_count - 1) // 2
