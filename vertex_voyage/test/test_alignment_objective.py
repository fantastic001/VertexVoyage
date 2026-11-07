import dataclasses
import pickle
import unittest

import torch

from vertex_voyage.aggregation import AggregationError
from vertex_voyage.aggregation.aligner_training import (
    GradientReport,
    term_gradient_of,
)
from vertex_voyage.aggregation.alignment import ContrastiveAlignmentAggregator
from vertex_voyage.aggregation.alignment_objective import (
    CROSS_SCOPE,
    INVERSE_CROSS_FRACTION_LINK_WEIGHTING,
    UNIFORM_LINK_WEIGHTING,
    WITHIN_SCOPE,
    InverseCrossFractionObjective,
    UniformObjective,
    cross_partition_fraction_of,
    cross_partition_mask_of,
    objective_of,
)
from vertex_voyage.aggregation.residual_aligner import (
    ATTRACTION_TERM,
    REPULSION_TERM,
    LossTerm,
    ResidualAligner,
    contrastive_loss,
    mean_loss_of,
    pair_distance,
    triplet_loss,
)
from vertex_voyage.test.test_alignment_aggregation import (
    SEED,
    SETTINGS,
    MisalignedReplicas,
)

TINY_MARGIN = 1e-3
PARTITIONS = torch.tensor([0, 0, 0, 1, 1, 1])
LINKS = torch.tensor([[0, 1], [1, 2], [3, 4], [2, 3]])
WITHIN_LINKS = torch.tensor([[0, 1], [1, 2], [3, 4]])
CROSS_LINKS = torch.tensor([[2, 3]])


def spread_points() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.nn.functional.normalize(torch.randn(6, 4), dim=-1)


def attraction_of(terms) -> float:
    return sum(
        float(term.weighted_mean())
        for term in terms
        if term.name.startswith(ATTRACTION_TERM)
    )


class TestCrossPartitionMask(unittest.TestCase):
    def test_mask_marks_links_between_partitions(self):
        mask = cross_partition_mask_of(PARTITIONS, LINKS)
        self.assertEqual(mask.tolist(), [False, False, False, True])

    def test_fraction_of_cross_links(self):
        self.assertEqual(cross_partition_fraction_of(PARTITIONS, LINKS), 0.25)

    def test_fraction_of_no_links_is_zero(self):
        empty = torch.zeros((0, 2), dtype=torch.long)
        self.assertEqual(cross_partition_fraction_of(PARTITIONS, empty), 0.0)


class TestLossTermWeight(unittest.TestCase):
    def test_unweighted_term_reports_without_weight(self):
        term = LossTerm(ATTRACTION_TERM, torch.tensor(0.5), 2)
        self.assertNotIn("weight", term.report())
        self.assertEqual(float(term.weighted_mean()), 0.5)

    def test_weighted_term_scales_mean_and_reports_weight(self):
        term = LossTerm(ATTRACTION_TERM, torch.tensor(0.5), 2, 0.25)
        self.assertEqual(float(term.weighted_mean()), 0.125)
        self.assertIn("weight 0.250000", term.report())
        self.assertEqual(term.summed_value(), 1.0)
        self.assertEqual(float(mean_loss_of((term, term))), 0.25)


class TestUniformObjective(unittest.TestCase):
    def test_equals_the_plain_loss(self):
        out = spread_points()
        objective = UniformObjective(contrastive_loss, TINY_MARGIN)
        terms = objective.terms_of(out, PARTITIONS, LINKS)
        self.assertEqual(
            [term.name for term in terms],
            [ATTRACTION_TERM, REPULSION_TERM],
        )
        expected = float((pair_distance(out, LINKS) ** 2).mean())
        self.assertAlmostEqual(attraction_of(terms), expected, places=6)


class TestInverseCrossFractionObjective(unittest.TestCase):
    def setUp(self):
        self.objective = InverseCrossFractionObjective(
            contrastive_loss,
            TINY_MARGIN,
        )

    def test_terms_are_split_by_scope(self):
        terms = self.objective.terms_of(spread_points(), PARTITIONS, LINKS)
        self.assertEqual(
            [term.name for term in terms],
            [
                f"{ATTRACTION_TERM} {WITHIN_SCOPE}",
                f"{REPULSION_TERM} {WITHIN_SCOPE}",
                f"{ATTRACTION_TERM} {CROSS_SCOPE}",
                f"{REPULSION_TERM} {CROSS_SCOPE}",
            ],
        )
        self.assertEqual([term.pair_count for term in terms], [3, 3, 1, 1])

    def test_scope_weights_follow_inverse_cross_fraction(self):
        delta = 0.25
        terms = self.objective.terms_of(spread_points(), PARTITIONS, LINKS)
        within_share = (1 - delta) / (2 - delta)
        cross_share = 1 / (2 - delta)
        expected = [within_share, within_share, cross_share, cross_share]
        for term, weight in zip(terms, expected):
            self.assertAlmostEqual(term.weight, weight)
        self.assertAlmostEqual(within_share + cross_share, 1.0)

    def test_attraction_is_the_inverse_fraction_weighted_mean(self):
        out = spread_points()
        delta = 0.25
        within = (pair_distance(out, WITHIN_LINKS) ** 2).sum()
        cross = (pair_distance(out, CROSS_LINKS) ** 2).sum()
        weighted_mean = (within + cross / delta) / (3 + 1 / delta)
        terms = self.objective.terms_of(out, PARTITIONS, LINKS)
        self.assertAlmostEqual(
            attraction_of(terms),
            float(weighted_mean),
            places=6,
        )

    def test_one_scope_falls_back_to_the_plain_loss(self):
        for links in (WITHIN_LINKS, CROSS_LINKS):
            with self.subTest(links=links.tolist()):
                terms = self.objective.terms_of(
                    spread_points(),
                    PARTITIONS,
                    links,
                )
                self.assertEqual(
                    [term.name for term in terms],
                    [ATTRACTION_TERM, REPULSION_TERM],
                )

    def test_works_for_triplet_loss(self):
        objective = InverseCrossFractionObjective(triplet_loss, 0.5)
        terms = objective.terms_of(spread_points(), PARTITIONS, LINKS)
        self.assertEqual(len(terms), 2)
        self.assertTrue(all(0 < term.weight < 1 for term in terms))

    def test_term_gradients_sum_to_weighted_backward(self):
        torch.manual_seed(SEED)
        model = ResidualAligner(2, 4, 8)
        vectors = spread_points()[:, :4]
        torch.manual_seed(SEED)
        terms = self.objective.terms_of(
            model(vectors, PARTITIONS),
            PARTITIONS,
            LINKS,
        )
        parameters = list(model.parameters())
        report = GradientReport(
            tuple(term_gradient_of(term, parameters) for term in terms),
            1e-7,
        )
        mean_loss_of(terms).backward()
        for parameter, summed in zip(parameters, report.total):
            torch.testing.assert_close(parameter.grad, summed)

    def test_survives_pickling(self):
        restored = pickle.loads(pickle.dumps(self.objective))
        self.assertEqual(restored.margin, TINY_MARGIN)


class TestObjectiveSelection(unittest.TestCase):
    def test_names_select_objectives(self):
        self.assertIsInstance(
            objective_of(UNIFORM_LINK_WEIGHTING, contrastive_loss, 1.0),
            UniformObjective,
        )
        self.assertIsInstance(
            objective_of(
                INVERSE_CROSS_FRACTION_LINK_WEIGHTING,
                contrastive_loss,
                1.0,
            ),
            InverseCrossFractionObjective,
        )

    def test_unknown_name_raises(self):
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                objective_of("sometimes", contrastive_loss, 1.0)

    def test_aggregator_rejects_unknown_weighting(self):
        settings = dataclasses.replace(SETTINGS, link_weighting="sometimes")
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                ContrastiveAlignmentAggregator(settings, SEED)


class TestAggregatorWeighting(unittest.TestCase):
    def epoch_lines(self, link_weighting):
        torch.manual_seed(SEED)
        settings = dataclasses.replace(
            SETTINGS,
            train_steps=2,
            link_weighting=link_weighting,
        )
        aggregator = ContrastiveAlignmentAggregator(settings, SEED)
        with self.assertLogs("Aggregation", level="INFO") as captured:
            MisalignedReplicas().train(aggregator)
        return [record.getMessage() for record in captured.records]

    def test_weighted_training_logs_scoped_terms_and_fraction(self):
        lines = self.epoch_lines(INVERSE_CROSS_FRACTION_LINK_WEIGHTING)
        self.assertTrue(
            any(
                "link weighting inverse_cross_fraction" in line
                for line in lines
            )
        )
        epoch_line = [line for line in lines if "Aligner epoch" in line][0]
        for expected in (
            f"{ATTRACTION_TERM} {CROSS_SCOPE}",
            f"{ATTRACTION_TERM} {WITHIN_SCOPE}",
            f"gradient norm {ATTRACTION_TERM} {WITHIN_SCOPE}",
        ):
            self.assertIn(expected, epoch_line)

    def test_uniform_training_keeps_unscoped_terms(self):
        lines = self.epoch_lines(UNIFORM_LINK_WEIGHTING)
        epoch_line = [line for line in lines if "Aligner epoch" in line][0]
        self.assertNotIn(CROSS_SCOPE, epoch_line.split("validation")[0])
        self.assertIn(f"gradient norm {ATTRACTION_TERM} ", epoch_line)


if __name__ == "__main__":
    unittest.main()
