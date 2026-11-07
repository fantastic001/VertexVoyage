import dataclasses
import math
import unittest

import torch

from vertex_voyage.aggregation import AggregationError
from vertex_voyage.aggregation.aligner_training import (
    TOTAL_GRADIENT_NAME,
    VANISHING_MARKER,
    AlignerTrainer,
    EdgeHashSplitter,
    GradientReport,
    TermGradient,
    TrainingHistory,
    TrainingOnlySplitter,
    ValidationMode,
    link_splitter_of,
    row_pair_tensor_of,
    term_gradient_of,
)
from vertex_voyage.aggregation.alignment import ContrastiveAlignmentAggregator
from vertex_voyage.aggregation.residual_aligner import (
    ATTRACTION_TERM,
    REPULSION_TERM,
    LossTerm,
    ResidualAligner,
    contrastive_loss,
    mean_loss_of,
)
from vertex_voyage.test.test_alignment_aggregation import (
    SEED,
    SETTINGS,
    MisalignedReplicas,
)

DIM = 4
VANISHING_NORM = 1e-7
LINKS = [((0, u), (1, u + 1)) for u in range(200)]


def small_aligner() -> ResidualAligner:
    torch.manual_seed(SEED)
    return ResidualAligner(2, DIM, 2 * DIM)


def small_batch():
    torch.manual_seed(SEED)
    vectors = torch.nn.functional.normalize(torch.randn(10, DIM), dim=-1)
    partitions = torch.tensor([0, 1] * 5)
    return vectors, partitions


def small_trainer(model: ResidualAligner) -> AlignerTrainer:
    return AlignerTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        contrastive_loss,
        1.0,
        VANISHING_NORM,
    )


class TestLinkSplitters(unittest.TestCase):
    def test_training_only_keeps_every_link(self):
        training, validation = TrainingOnlySplitter().split(LINKS)
        self.assertEqual(training, LINKS)
        self.assertEqual(validation, [])

    def test_edge_hash_split_is_disjoint_complete_and_near_fraction(self):
        training, validation = EdgeHashSplitter(0.25).split(LINKS)
        self.assertEqual(sorted(training + validation), sorted(LINKS))
        self.assertFalse(set(training) & set(validation))
        self.assertGreater(len(validation), 0.1 * len(LINKS))
        self.assertLess(len(validation), 0.4 * len(LINKS))

    def test_edge_hash_split_is_stable_across_calls(self):
        splitter = EdgeHashSplitter(0.3)
        self.assertEqual(splitter.split(LINKS), splitter.split(LINKS))

    def test_links_of_one_edge_land_in_one_split(self):
        same_edge = [((p, "u"), (q, "v")) for p in range(4) for q in range(4)]
        training, validation = EdgeHashSplitter(0.5).split(same_edge)
        self.assertIn(0, (len(training), len(validation)))

    def test_fraction_outside_unit_interval_raises(self):
        for fraction in (0.0, 1.0, -0.5, 2.0):
            with self.subTest(fraction=fraction):
                with self.assertLogs("Aggregation", level="ERROR"):
                    with self.assertRaises(AggregationError):
                        EdgeHashSplitter(fraction)

    def test_mode_selects_splitter(self):
        self.assertIsInstance(
            link_splitter_of(ValidationMode.DISABLED, 0.2),
            TrainingOnlySplitter,
        )
        self.assertIsInstance(
            link_splitter_of(ValidationMode.ENABLED, 0.2),
            EdgeHashSplitter,
        )

    def test_unknown_mode_raises(self):
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                link_splitter_of("sometimes", 0.2)

    def test_empty_row_pairs_make_two_column_tensor(self):
        self.assertEqual(tuple(row_pair_tensor_of([]).shape), (0, 2))
        self.assertEqual(tuple(row_pair_tensor_of([(1, 2)]).shape), (1, 2))


class TestGradientReport(unittest.TestCase):
    def test_term_gradients_sum_to_backward_gradient(self):
        model = small_aligner()
        vectors, partitions = small_batch()
        links = torch.tensor([[0, 1], [2, 3]])
        torch.manual_seed(SEED)
        terms = contrastive_loss(model(vectors, partitions), links, 1.0)
        parameters = list(model.parameters())
        report = GradientReport(
            tuple(term_gradient_of(term, parameters) for term in terms),
            VANISHING_NORM,
        )
        mean_loss_of(terms).backward()
        for parameter, summed in zip(parameters, report.total):
            torch.testing.assert_close(parameter.grad, summed)

    def test_norms_name_positive_negative_and_total(self):
        report = GradientReport(
            (
                TermGradient(ATTRACTION_TERM, (torch.tensor([3.0, 0.0]),)),
                TermGradient(REPULSION_TERM, (torch.tensor([0.0, 4.0]),)),
            ),
            VANISHING_NORM,
        )
        self.assertEqual(
            report.norms(),
            {
                ATTRACTION_TERM: 3.0,
                REPULSION_TERM: 4.0,
                TOTAL_GRADIENT_NAME: 5.0,
            },
        )
        self.assertFalse(report.total_vanishes())
        self.assertNotIn(VANISHING_MARKER, report.report())

    def test_vanishing_terms_are_flagged(self):
        report = GradientReport(
            (
                TermGradient(ATTRACTION_TERM, (torch.tensor([1.0]),)),
                TermGradient(REPULSION_TERM, (torch.tensor([0.0]),)),
            ),
            VANISHING_NORM,
        )
        self.assertEqual(report.vanishing_names(), [REPULSION_TERM])
        self.assertIn(
            REPULSION_TERM + " 0.000000e+00" + VANISHING_MARKER,
            report.report(),
        )

    def test_cancelling_terms_vanish_in_total(self):
        report = GradientReport(
            (
                TermGradient(ATTRACTION_TERM, (torch.tensor([1.0]),)),
                TermGradient(REPULSION_TERM, (torch.tensor([-1.0]),)),
            ),
            VANISHING_NORM,
        )
        self.assertTrue(report.total_vanishes())
        self.assertEqual(report.vanishing_names(), [TOTAL_GRADIENT_NAME])

    def test_constant_term_has_zero_gradient(self):
        model = small_aligner()
        parameters = list(model.parameters())
        gradient = term_gradient_of(
            LossTerm(REPULSION_TERM, torch.tensor(0.0), 1),
            parameters,
        )
        self.assertEqual(gradient.norm(), 0.0)
        self.assertEqual(len(gradient.gradients), len(parameters))

    def test_no_terms_have_zero_norm(self):
        self.assertEqual(GradientReport((), VANISHING_NORM).total_norm(), 0.0)


class TestAlignerTrainer(unittest.TestCase):
    def test_epoch_reports_training_gradients_and_validation(self):
        model = small_aligner()
        vectors, partitions = small_batch()
        outcome = small_trainer(model).run_epoch(
            vectors,
            partitions,
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([[4, 5]]),
        )
        names = [term.name for term in outcome.terms]
        self.assertEqual(names, [ATTRACTION_TERM, REPULSION_TERM])
        self.assertEqual(len(outcome.validation_terms), 2)
        self.assertGreater(outcome.gradients.total_norm(), 0.0)
        for term in outcome.validation_terms:
            self.assertFalse(term.mean.requires_grad)

    def test_epoch_moves_parameters(self):
        model = small_aligner()
        before = [p.detach().clone() for p in model.parameters()]
        vectors, partitions = small_batch()
        small_trainer(model).run_epoch(
            vectors,
            partitions,
            torch.tensor([[0, 1], [2, 3]]),
            row_pair_tensor_of([]),
        )
        moved = any(
            not torch.equal(old, new)
            for old, new in zip(before, model.parameters())
        )
        self.assertTrue(moved)

    def test_validation_does_not_move_parameters(self):
        model = small_aligner()
        before = [p.detach().clone() for p in model.parameters()]
        vectors, partitions = small_batch()
        terms = small_trainer(model).validation_terms_of(
            vectors,
            partitions,
            torch.tensor([[0, 1]]),
        )
        self.assertTrue(math.isfinite(float(mean_loss_of(terms))))
        for old, new in zip(before, model.parameters()):
            self.assertTrue(torch.equal(old, new))

    def test_no_validation_links_give_no_terms(self):
        vectors, partitions = small_batch()
        terms = small_trainer(small_aligner()).validation_terms_of(
            vectors,
            partitions,
            row_pair_tensor_of([]),
        )
        self.assertEqual(terms, ())

    def test_epoch_without_training_links_raises(self):
        vectors, partitions = small_batch()
        with self.assertLogs("Aggregation", level="ERROR"):
            with self.assertRaises(AggregationError):
                small_trainer(small_aligner()).run_epoch(
                    vectors,
                    partitions,
                    row_pair_tensor_of([]),
                    row_pair_tensor_of([]),
                )


class TestTrainingHistory(unittest.TestCase):
    def outcome_with_norm(self, norm: float):
        model = small_aligner()
        vectors, partitions = small_batch()
        outcome = small_trainer(model).run_epoch(
            vectors,
            partitions,
            torch.tensor([[0, 1]]),
            row_pair_tensor_of([]),
        )
        return dataclasses.replace(
            outcome,
            gradients=GradientReport(
                (TermGradient(ATTRACTION_TERM, (torch.tensor([norm]),)),),
                VANISHING_NORM,
            ),
        )

    def test_empty_history_reports_no_loss(self):
        self.assertIn("not computed", TrainingHistory().report())

    def test_vanishing_epochs_are_counted_and_warned(self):
        history = TrainingHistory()
        history.record(self.outcome_with_norm(0.0))
        history.record(self.outcome_with_norm(1.0))
        self.assertEqual(history.vanishing_epoch_count, 1)
        with self.assertLogs("Aggregation", level="WARNING") as captured:
            history.log_vanishing_summary()
        self.assertIn("1 of 2", captured.output[0])
        self.assertIn("gradient norm", history.report())

    def test_healthy_history_does_not_warn(self):
        history = TrainingHistory()
        history.record(self.outcome_with_norm(1.0))
        with self.assertLogs("Aggregation", level="DEBUG") as captured:
            history.log_vanishing_summary()
        levels = {record.levelname for record in captured.records}
        self.assertEqual(levels, {"DEBUG"})


class TestAggregatorValidation(unittest.TestCase):
    def train_lines(self, settings):
        torch.manual_seed(SEED)
        problem = MisalignedReplicas()
        aggregator = ContrastiveAlignmentAggregator(settings, SEED)
        with self.assertLogs("Aggregation", level="INFO") as captured:
            problem.train(aggregator)
        return [record.getMessage() for record in captured.records]

    def test_enabled_validation_logs_validation_loss_each_epoch(self):
        settings = dataclasses.replace(
            SETTINGS,
            train_steps=3,
            validation=ValidationMode.ENABLED,
            validation_fraction=0.3,
        )
        lines = self.train_lines(settings)
        epoch_lines = [line for line in lines if "Aligner epoch" in line]
        self.assertEqual(len(epoch_lines), 3)
        for line in epoch_lines:
            self.assertIn("validation loss", line)
            self.assertIn(ATTRACTION_TERM, line.split("validation")[1])
            self.assertIn("gradient norm " + ATTRACTION_TERM, line)
        summary = [line for line in lines if "validated on" in line][0]
        self.assertNotIn("validated on 0,", summary)

    def test_disabled_validation_holds_out_nothing(self):
        settings = dataclasses.replace(SETTINGS, train_steps=2)
        lines = self.train_lines(settings)
        summary = [line for line in lines if "validated on" in line][0]
        self.assertIn("validated on 0,", summary)
        self.assertIn("validation loss not computed", summary)

    def test_vanishing_gradient_is_warned(self):
        settings = dataclasses.replace(
            SETTINGS,
            train_steps=2,
            vanishing_gradient_norm=float("inf"),
        )
        lines = self.train_lines(settings)
        self.assertTrue(
            any("Vanishing aligner gradient at epoch" in line for line in lines)
        )
        self.assertTrue(
            any("Vanishing total gradient in 2 of 2" in line for line in lines)
        )


if __name__ == "__main__":
    unittest.main()
