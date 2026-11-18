import math
import unittest
from collections import Counter
from unittest import mock

import networkx as nx
import numpy as np

import vertex_voyage.tasks.prequential_link_prediction as prequential
from vertex_voyage.tasks.link_prediction import (
    EmbeddingSnapshot,
    HadamardLogitsNet,
    IncrementalLogitsDecoder,
    TorchLogitsLinkPredictionModelTrainer,
    evaluate_predictions,
)
from vertex_voyage.tasks.prequential_link_prediction import (
    POOLED_RESULT_NAME,
    REPORTS_RESULT_NAME,
    DisabledLinkPrediction,
    EdgeNovelty,
    PartitionScope,
    PrequentialLinkPrediction,
    ScoredPairs,
    UniformComplementSampler,
    decoder_scores_of,
    description_of_report,
    edge_counts_by_partition_scope,
    edges_by_novelty,
    evaluate_decoder,
    frozen_vertex_partitions,
    partition_scope_of,
    train_decoder,
)
from vertex_voyage.temporal import Event, EventType, buffered

DIM = 4
SEED = 3
ATTEMPTS_PER_NEGATIVE = 100
BUFFER_SIZE = 10
DECODER_EPOCHS = 2


def events_of(edges, type=EventType.ADD):
    return [
        Event(src=u, dest=v, timestamp=t, type=type)
        for t, (u, v) in enumerate(edges)
    ]


def vector_of(vertex) -> np.ndarray:
    return np.random.default_rng(abs(hash(vertex)) % 1000).normal(size=DIM)


def snapshot_of(vertices) -> EmbeddingSnapshot:
    vertices = list(vertices)
    return EmbeddingSnapshot(vertices, [vector_of(v) for v in vertices], DIM)


class ParityPartitioner:
    def get(self, vertex):
        return {vertex % 2}


def new_decoder(seed: int = SEED) -> IncrementalLogitsDecoder:
    trainer = TorchLogitsLinkPredictionModelTrainer(
        HadamardLogitsNet, epochs=DECODER_EPOCHS
    )
    return IncrementalLogitsDecoder(trainer, DIM, seed)


def new_sampler(seed: int = SEED) -> UniformComplementSampler:
    return UniformComplementSampler(
        np.random.default_rng(seed), ATTEMPTS_PER_NEGATIVE
    )


def canonical(edge) -> frozenset:
    return frozenset(edge)


class TestEdgesByNovelty(unittest.TestCase):
    def test_worked_example(self):
        known_graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
        edges = edges_by_novelty(
            events_of([(0, 2), (3, 4), (1, 2), (2, 2)]),
            known_graph,
            {0, 1, 2, 3},
        )
        self.assertEqual(edges[EdgeNovelty.NEW_EDGE], [(0, 2)])
        self.assertEqual(edges[EdgeNovelty.NEW_VERTEX], [(3, 4)])
        self.assertEqual(edges[EdgeNovelty.REPEATED_EDGE], [(1, 2)])
        self.assertEqual(edges[EdgeNovelty.SELF_LOOP], [(2, 2)])

    def test_duplicates_in_one_buffer_count_once_as_new(self):
        edges = edges_by_novelty(
            events_of([(0, 2), (2, 0), (0, 2)]), nx.Graph(), {0, 1, 2}
        )
        self.assertEqual(edges[EdgeNovelty.NEW_EDGE], [(0, 2)])
        self.assertEqual(
            edges[EdgeNovelty.REPEATED_EDGE], [(2, 0), (0, 2)]
        )

    def test_rules_apply_in_order(self):
        edges = edges_by_novelty(
            events_of([(5, 5), (5, 6), (5, 6)]), nx.Graph(), {0}
        )
        self.assertEqual(edges[EdgeNovelty.SELF_LOOP], [(5, 5)])
        self.assertEqual(edges[EdgeNovelty.NEW_VERTEX], [(5, 6), (5, 6)])

    def test_classification_leaves_the_known_graph_unchanged(self):
        known_graph = nx.Graph([(0, 1)])
        edges_by_novelty(events_of([(0, 2)]), known_graph, {0, 1, 2})
        self.assertEqual(list(known_graph.edges), [(0, 1)])

    def test_edge_removals_are_rejected(self):
        with self.assertRaises(ValueError):
            edges_by_novelty(
                events_of([(0, 1)], type=EventType.REMOVE), nx.Graph(), {0, 1}
            )


class TestPartitionScope(unittest.TestCase):
    def test_scope_follows_shared_partitions(self):
        partitions = {
            "a": frozenset({1}),
            "b": frozenset({1, 2}),
            "c": frozenset({3}),
        }
        self.assertEqual(
            partition_scope_of("a", "b", partitions), PartitionScope.WITHIN
        )
        self.assertEqual(
            partition_scope_of("a", "c", partitions), PartitionScope.CROSS
        )
        self.assertEqual(
            partition_scope_of("a", "unplaced", partitions),
            PartitionScope.CROSS,
        )

    def test_frozen_partitions_do_not_follow_the_partitioner(self):
        placement = {0: {"p"}, 1: {"q"}}

        class MutablePartitioner:
            def get(self, vertex):
                return placement[vertex]

        frozen = frozen_vertex_partitions(MutablePartitioner(), [0, 1])
        placement[0].add("q")
        self.assertEqual(frozen[0], frozenset({"p"}))

    def test_edge_counts_by_partition_scope(self):
        partitions = frozen_vertex_partitions(ParityPartitioner(), range(4))
        self.assertEqual(
            edge_counts_by_partition_scope(
                [(0, 2), (1, 3), (0, 1)], partitions
            ),
            {"within": 2, "cross": 1},
        )


class StubGeneratorDrawingAnEdge:
    def __init__(self):
        self.real = np.random.default_rng(SEED)

    def integers(self, high, size):
        return np.array([0, 1])

    def choice(self, *args, **kwargs):
        return self.real.choice(*args, **kwargs)


class TestUniformComplementSampler(unittest.TestCase):
    def assert_valid_negatives(self, graph, vertices, negatives):
        self.assertTrue(all(u != v for u, v in negatives))
        self.assertTrue(all(not graph.has_edge(u, v) for u, v in negatives))
        self.assertTrue(
            all(u in vertices and v in vertices for u, v in negatives)
        )
        self.assertEqual(
            len({canonical(edge) for edge in negatives}), len(negatives)
        )

    def test_negatives_are_distinct_non_edges(self):
        graph = nx.path_graph(20)
        vertices = list(range(20))
        negatives = new_sampler().sample(graph, vertices, 30)
        self.assertEqual(len(negatives), 30)
        self.assert_valid_negatives(graph, vertices, negatives)

    def test_same_seed_gives_the_same_negatives(self):
        graph = nx.path_graph(20)
        first = new_sampler().sample(graph, list(range(20)), 10)
        second = new_sampler().sample(graph, list(range(20)), 10)
        self.assertEqual(first, second)

    def test_small_complement_is_returned_whole(self):
        graph = nx.complete_graph(4)
        graph.remove_edges_from([(0, 3), (1, 3)])
        negatives = new_sampler().sample(graph, [0, 1, 2, 3], 5)
        self.assertEqual(
            {canonical(edge) for edge in negatives},
            {canonical((0, 3)), canonical((1, 3))},
        )

    def test_complement_of_the_exact_size(self):
        graph = nx.complete_graph(3)
        graph.remove_edge(0, 2)
        negatives = new_sampler().sample(graph, [0, 1, 2], 1)
        self.assertEqual(
            [canonical(edge) for edge in negatives], [canonical((0, 2))]
        )

    def test_enumeration_after_failed_rejection(self):
        graph = nx.Graph([(0, 1)])
        vertices = list(range(6))
        sampler = UniformComplementSampler(StubGeneratorDrawingAnEdge(), 1)
        negatives = sampler.sample(graph, vertices, 4)
        self.assertEqual(len(negatives), 4)
        self.assert_valid_negatives(graph, vertices, negatives)

    def test_enumerated_complement_of_the_exact_size(self):
        graph = nx.Graph([(0, 1)])
        sampler = UniformComplementSampler(StubGeneratorDrawingAnEdge(), 1)
        negatives = sampler.sample(graph, [0, 1, 2], 2)
        self.assertEqual(negatives, [(0, 2), (1, 2)])

    def test_nothing_to_sample(self):
        cases = {
            "complete graph": (nx.complete_graph(4), [0, 1, 2, 3], 2),
            "one vertex": (nx.Graph(), [0], 2),
            "no vertex": (nx.Graph(), [], 2),
            "no negative requested": (nx.Graph(), [0, 1], 0),
        }
        for case, (graph, vertices, count) in cases.items():
            with self.subTest(case=case):
                negatives = new_sampler().sample(graph, vertices, count)
                self.assertEqual(negatives, [])

    def test_pairs_are_uniform(self):
        vertices = list(range(5))
        draws = 4000
        sampler = new_sampler()
        counts = Counter(
            canonical(sampler.sample(nx.Graph(), vertices, 1)[0])
            for _ in range(draws)
        )
        self.assertEqual(len(counts), 10)
        expected = draws / 10
        chi_square = sum(
            (c - expected) ** 2 / expected for c in counts.values()
        )
        self.assertLess(chi_square, 27.88)

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            UniformComplementSampler(np.random.default_rng(0), 0)
        with self.assertRaises(ValueError):
            new_sampler().sample(nx.Graph(), [0, 1], -1)


class TestDecoderScores(unittest.TestCase):
    labels = np.array([1, 1, 0, 0])

    def scores(self, probabilities):
        probabilities = np.array(probabilities)
        return decoder_scores_of(
            self.labels, probabilities, probabilities > 0.5
        )

    def test_perfect_inverted_and_constant_scores(self):
        self.assertEqual(self.scores([0.9, 0.8, 0.2, 0.1]).auc, 1.0)
        self.assertEqual(self.scores([0.9, 0.8, 0.2, 0.1]).f1, 1.0)
        self.assertEqual(self.scores([0.1, 0.2, 0.8, 0.9]).auc, 0.0)
        self.assertEqual(self.scores([0.4, 0.4, 0.4, 0.4]).auc, 0.5)
        self.assertEqual(self.scores([0.4, 0.4, 0.4, 0.4]).precision, 0.0)

    def test_one_class_has_undefined_threshold_free_scores(self):
        scores = decoder_scores_of(
            np.array([1, 1]), np.array([0.9, 0.2]), np.array([True, False])
        )
        self.assertTrue(math.isnan(scores.auc))
        self.assertTrue(math.isnan(scores.average_precision))
        self.assertEqual(scores.recall, 0.5)

    def test_scopes_keep_all_negatives(self):
        pairs = ScoredPairs(
            labels=[1, 1, 0, 0],
            probabilities=[0.9, 0.3, 0.6, 0.1],
            predicted=[True, False, True, False],
            scopes=["within", "cross", "within", "cross"],
        )
        scores = pairs.scores_by_scope()
        self.assertEqual(
            {
                scope: (s.positive_count, s.negative_count)
                for scope, s in scores.items()
            },
            {"all": (2, 2), "within": (1, 2), "cross": (1, 2)},
        )
        self.assertEqual(scores["within"].auc, 1.0)
        self.assertEqual(scores["cross"].auc, 0.5)

    def test_scopes_without_positives_are_omitted(self):
        pairs = ScoredPairs(
            labels=[1, 0],
            probabilities=[0.9, 0.1],
            predicted=[True, False],
            scopes=["within", "cross"],
        )
        self.assertEqual(set(pairs.scores_by_scope()), {"all", "within"})
        self.assertEqual(ScoredPairs().scores_by_scope(), {})

    def test_concatenation_keeps_order(self):
        first = ScoredPairs([1], [0.9], [True], ["within"])
        second = ScoredPairs([0], [0.2], [False], ["cross"])
        joined = ScoredPairs.concatenated([first, ScoredPairs(), second])
        self.assertEqual(joined.labels, [1, 0])
        self.assertEqual(joined.scopes, ["within", "cross"])
        self.assertEqual(len(joined), 2)


class TestTrainAndEvaluateDecoder(unittest.TestCase):
    positives = [(0, 1), (2, 3), (4, 5)]
    negatives = [(0, 5), (1, 4), (1, 5)]

    def test_threshold_scores_match_the_static_pipeline(self):
        embedding = snapshot_of(range(6))
        decoder = new_decoder()
        train_decoder(decoder, embedding, self.positives, self.negatives, SEED)
        partitions = frozen_vertex_partitions(ParityPartitioner(), range(6))
        scored = evaluate_decoder(
            embedding, decoder, self.positives, self.negatives, partitions
        )
        scores = scored.scores_by_scope()["all"]
        precision, recall, f1, accuracy = evaluate_predictions(
            embedding, decoder, self.positives, self.negatives
        )
        self.assertAlmostEqual(scores.precision, precision)
        self.assertAlmostEqual(scores.recall, recall)
        self.assertAlmostEqual(scores.f1, f1)
        self.assertAlmostEqual(scores.accuracy, accuracy)
        self.assertEqual(scored.labels, [1, 1, 1, 0, 0, 0])
        self.assertEqual(
            scored.scopes,
            ["cross", "cross", "cross", "cross", "cross", "within"],
        )

    def test_training_refits_the_decoder_once(self):
        decoder = new_decoder()
        losses = train_decoder(
            decoder, snapshot_of(range(6)), self.positives, self.negatives, SEED
        )
        self.assertEqual(decoder.refit_count, 1)
        self.assertEqual(len(losses), DECODER_EPOCHS + 1)

    def test_invalid_inputs_are_rejected(self):
        embedding = snapshot_of(range(6))
        other_dimension = EmbeddingSnapshot(
            list(range(6)), [np.ones(DIM + 1)] * 6, DIM + 1
        )
        invalid = {
            "no positives": lambda: train_decoder(
                new_decoder(), embedding, [], self.negatives, SEED
            ),
            "no negatives": lambda: evaluate_decoder(
                embedding, new_decoder(), self.positives, [], {}
            ),
            "dimension mismatch": lambda: train_decoder(
                new_decoder(),
                other_dimension,
                self.positives,
                self.negatives,
                SEED,
            ),
        }
        for case, run in invalid.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    run()


def stream_buffers():
    edges = list(nx.karate_club_graph().edges)
    np.random.default_rng(SEED).shuffle(edges)
    return list(buffered(events_of(edges), BUFFER_SIZE))


def run_stream(link_prediction, buffers):
    vertices = []
    for buffer_index, buffer in enumerate(buffers):
        for event in buffer:
            for vertex in (event.src, event.dest):
                if vertex not in vertices:
                    vertices.append(vertex)
        link_prediction.evaluate_buffer(
            buffer_index,
            buffer,
            list(vertices),
            [vector_of(v) for v in vertices],
            ParityPartitioner(),
        )


class RecordedCall:
    def __init__(self, name, positive_edges, negative_edges):
        self.name = name
        self.positives = {canonical(edge) for edge in positive_edges}
        self.negatives = {canonical(edge) for edge in negative_edges}


class TestPrequentialLinkPrediction(unittest.TestCase):
    def run_recorded_stream(self):
        calls = []

        def recording(name, function):
            def call(*args):
                calls.append((name, args))
                return function(*args)

            return call

        link_prediction = PrequentialLinkPrediction(
            new_decoder(), new_sampler(), 1, SEED, log_line=lambda line: None
        )
        with mock.patch.object(
            prequential,
            "train_decoder",
            recording("train", prequential.train_decoder),
        ), mock.patch.object(
            prequential,
            "evaluate_decoder",
            recording("test", prequential.evaluate_decoder),
        ):
            run_stream(link_prediction, stream_buffers())
        return link_prediction, calls

    def test_first_buffers_only_freeze_and_train(self):
        link_prediction, _ = self.run_recorded_stream()
        first, second, third = link_prediction.reports[:3]
        self.assertEqual(first.edge_counts["new_edge"], 0)
        self.assertEqual(first.scores, {})
        self.assertTrue(math.isnan(first.training_loss))
        self.assertEqual(second.scores, {})
        self.assertFalse(math.isnan(second.training_loss))
        self.assertIn("all", third.scores)

    def test_each_buffer_is_tested_before_it_trains(self):
        _, calls = self.run_recorded_stream()
        names = [name for name, _ in calls]
        self.assertEqual(names[0], "train")
        self.assertEqual(names[1:], ["test", "train"] * ((len(names) - 1) // 2))
        for (_, test_args), (_, train_args) in zip(
            calls[1::2], calls[2::2]
        ):
            self.assertEqual(test_args[2], train_args[2])
            self.assertEqual(test_args[3], train_args[3])

    def test_tested_positives_were_never_trained_on_nor_seen(self):
        _, calls = self.run_recorded_stream()
        trained = set()
        for name, args in calls:
            pairs = RecordedCall(name, args[2], args[3])
            if name == "test":
                self.assertEqual(pairs.positives & trained, set())
            else:
                trained |= pairs.positives

    def test_negatives_are_non_edges_of_the_current_graph(self):
        buffers = stream_buffers()
        link_prediction, calls = self.run_recorded_stream()
        seen = nx.Graph()
        tested_buffers = iter(
            args for name, args in calls if name == "train"
        )
        for report, buffer in zip(link_prediction.reports, buffers):
            seen.add_edges_from((event.src, event.dest) for event in buffer)
            if report.edge_counts["new_edge"] > 0:
                negatives = next(tested_buffers)[3]
                self.assertTrue(
                    all(
                        u != v and not seen.has_edge(u, v)
                        for u, v in negatives
                    )
                )
            else:
                self.assertEqual(report.negative_count, 0)

    def test_results_hold_every_buffer_and_pooled_scores(self):
        link_prediction, _ = self.run_recorded_stream()
        results = link_prediction.results()
        self.assertEqual(
            len(results[REPORTS_RESULT_NAME]), len(stream_buffers())
        )
        self.assertIn("all", results[POOLED_RESULT_NAME])
        self.assertEqual(
            sorted(results), sorted(link_prediction.result_names())
        )
        pooled = results[POOLED_RESULT_NAME]["all"]
        tested = sum(
            report.scores["all"].positive_count
            for report in link_prediction.reports
            if "all" in report.scores
        )
        self.assertEqual(pooled["positive_count"], tested)

    def test_one_line_is_logged_per_buffer(self):
        lines = []
        link_prediction = PrequentialLinkPrediction(
            new_decoder(), new_sampler(), 1, SEED, log_line=lines.append
        )
        run_stream(link_prediction, stream_buffers())
        self.assertEqual(len(lines), len(stream_buffers()))
        self.assertIn("not tested", lines[0])
        self.assertIn("all AUC", lines[-1])

    def test_same_seed_gives_the_same_reports(self):
        results = []
        for _ in range(2):
            link_prediction = PrequentialLinkPrediction(
                new_decoder(),
                new_sampler(),
                1,
                SEED,
                log_line=lambda line: None,
            )
            run_stream(link_prediction, stream_buffers())
            results.append(link_prediction.results())
        self.assertEqual(results[0], results[1])

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            PrequentialLinkPrediction(new_decoder(), new_sampler(), 0, SEED)
        link_prediction = PrequentialLinkPrediction(
            new_decoder(), new_sampler(), 1, SEED
        )
        with self.assertRaises(ValueError):
            link_prediction.evaluate_buffer(
                0, events_of([(0, 1)]), [0, 1], [np.ones(DIM + 1)] * 2,
                ParityPartitioner(),
            )


class TestDisabledLinkPrediction(unittest.TestCase):
    def test_does_nothing(self):
        link_prediction = DisabledLinkPrediction()
        run_stream(link_prediction, stream_buffers())
        self.assertEqual(link_prediction.results(), {})
        self.assertEqual(link_prediction.result_names(), [])


class TestDescriptionOfReport(unittest.TestCase):
    def test_untested_buffer(self):
        report = prequential.BufferLinkPredictionReport(
            buffer_index=0,
            edge_counts={novelty.value: 1 for novelty in EdgeNovelty},
            positive_counts={"within": 1, "cross": 0},
            negative_count=1,
            scores={},
            scored_pairs=ScoredPairs(),
            training_loss=0.5,
        )
        self.assertEqual(
            description_of_report(report),
            "Link prediction buffer 1: 1 new edges (1 within, 0 cross), "
            "1 negatives, skipped 1 with a new vertex, 1 repeated, "
            "1 self-loops; not tested",
        )


if __name__ == "__main__":
    unittest.main()
