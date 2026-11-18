import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    FrozenSet,
    Hashable,
    List,
    Mapping,
    Sequence,
    Tuple,
)

import networkx as nx
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from vertex_voyage.aggregation import VertexPartitioner
from vertex_voyage.seeding import derive_iteration_seed
from vertex_voyage.tasks.link_prediction import (
    EmbeddingSnapshot,
    IncrementalLogitsDecoder,
    LinkPredictionDataset,
    LinkPredictionSplit,
    predict_links,
)
from vertex_voyage.temporal import Event, EventType

logger = logging.getLogger(__name__)

Edge = Tuple[Hashable, Hashable]
PartitionsByVertex = Mapping[Hashable, FrozenSet]

UNDEFINED_SCORE = float("nan")
NO_ZERO_DIVISION_SCORE = 0.0
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
SMALLEST_VERTEX_COUNT_WITH_A_PAIR = 2
REPORTS_RESULT_NAME = "lp_reports"
POOLED_RESULT_NAME = "lp_pooled"


class EdgeNovelty(Enum):
    SELF_LOOP = "self_loop"
    NEW_VERTEX = "new_vertex"
    REPEATED_EDGE = "repeated_edge"
    NEW_EDGE = "new_edge"


class PartitionScope(Enum):
    WITHIN = "within"
    CROSS = "cross"


class ScoreScope(Enum):
    ALL = "all"
    WITHIN = "within"
    CROSS = "cross"


@dataclass(frozen=True)
class KnownStream:
    graph: nx.Graph
    buffer_graph: nx.Graph
    vertices: Container


NOVELTY_RULES: Tuple[
    Tuple[EdgeNovelty, Callable[[Hashable, Hashable, KnownStream], bool]],
    ...,
] = (
    (EdgeNovelty.SELF_LOOP, lambda u, v, known: u == v),
    (
        EdgeNovelty.NEW_VERTEX,
        lambda u, v, known: u not in known.vertices
        or v not in known.vertices,
    ),
    (
        EdgeNovelty.REPEATED_EDGE,
        lambda u, v, known: known.graph.has_edge(u, v)
        or known.buffer_graph.has_edge(u, v),
    ),
    (EdgeNovelty.NEW_EDGE, lambda u, v, known: True),
)


def novelty_of_edge(
    u: Hashable,
    v: Hashable,
    known: KnownStream,
) -> EdgeNovelty:
    return next(
        novelty for novelty, applies in NOVELTY_RULES if applies(u, v, known)
    )


def reject_removal_events(buffer: Sequence[Event]) -> None:
    removals = [event for event in buffer if event.type == EventType.REMOVE]
    if len(removals) > 0:
        logger.error(
            "Prequential link prediction supports only edge additions, "
            "the buffer removes %d edges, first: %r",
            len(removals),
            removals[0],
        )
        raise ValueError(
            "Prequential link prediction supports only edge additions"
        )
    else:
        logger.debug("Buffer of %d events only adds edges", len(buffer))


def edges_by_novelty(
    buffer: Sequence[Event],
    known_graph: nx.Graph,
    known_vertices: Container,
) -> Dict[EdgeNovelty, List[Edge]]:
    reject_removal_events(buffer)
    known = KnownStream(known_graph, nx.Graph(), known_vertices)
    edges: Dict[EdgeNovelty, List[Edge]] = {
        novelty: [] for novelty in EdgeNovelty
    }
    for event in buffer:
        novelty = novelty_of_edge(event.src, event.dest, known)
        edges[novelty].append((event.src, event.dest))
        known.buffer_graph.add_edge(event.src, event.dest)
    return edges


def frozen_vertex_partitions(
    partitioner: VertexPartitioner,
    vertices: Sequence[Hashable],
) -> Dict[Hashable, FrozenSet]:
    return {vertex: frozenset(partitioner.get(vertex)) for vertex in vertices}


def partition_scope_of(
    u: Hashable,
    v: Hashable,
    partitions_by_vertex: PartitionsByVertex,
) -> PartitionScope:
    shared_partitions = partitions_by_vertex.get(
        u, frozenset()
    ) & partitions_by_vertex.get(v, frozenset())
    if len(shared_partitions) > 0:
        return PartitionScope.WITHIN
    else:
        return PartitionScope.CROSS


def edge_counts_by_partition_scope(
    edges: Sequence[Edge],
    partitions_by_vertex: PartitionsByVertex,
) -> Dict[str, int]:
    scopes = [
        partition_scope_of(u, v, partitions_by_vertex) for u, v in edges
    ]
    return {scope.value: scopes.count(scope) for scope in PartitionScope}


def sampling_error(message: str) -> ValueError:
    logger.error(message)
    return ValueError(message)


class NegativeEdgeSampler(ABC):
    @abstractmethod
    def sample(
        self,
        graph: nx.Graph,
        vertices: Sequence[Hashable],
        count: int,
    ) -> List[Edge]:
        ...


class UniformComplementSampler(NegativeEdgeSampler):
    def __init__(
        self,
        generator: np.random.Generator,
        attempts_per_negative: int,
    ):
        if attempts_per_negative <= 0:
            raise sampling_error(
                "attempts_per_negative must be positive, got "
                f"{attempts_per_negative}"
            )
        else:
            logger.debug(
                "Uniform complement sampler with %d attempts per negative",
                attempts_per_negative,
            )
        self.generator = generator
        self.attempts_per_negative = attempts_per_negative

    def sample(
        self,
        graph: nx.Graph,
        vertices: Sequence[Hashable],
        count: int,
    ) -> List[Edge]:
        if count < 0:
            raise sampling_error(f"Cannot sample {count} negatives")
        else:
            logger.debug(
                "Sampling %d negatives among %d vertices", count, len(vertices)
            )
        drawn = self._draw_by_rejection(graph, vertices, count)
        if len(drawn) == count:
            return drawn
        else:
            logger.info(
                "Rejection sampling found %d of %d negatives, "
                "enumerating the complement",
                len(drawn),
                count,
            )
            return self._draw_from_enumerated_complement(
                graph, vertices, count
            )

    def _draw_by_rejection(
        self,
        graph: nx.Graph,
        vertices: Sequence[Hashable],
        count: int,
    ) -> List[Edge]:
        accepted: Dict[FrozenSet, Edge] = {}
        attempts_left = (
            self.attempts_per_negative * count
            if len(vertices) >= SMALLEST_VERTEX_COUNT_WITH_A_PAIR
            else 0
        )
        while len(accepted) < count and attempts_left > 0:
            attempts_left -= 1
            u_row, v_row = self.generator.integers(len(vertices), size=2)
            u, v = vertices[u_row], vertices[v_row]
            if u != v and not graph.has_edge(u, v):
                accepted.setdefault(frozenset((u, v)), (u, v))
            else:
                logger.debug("Rejected pair (%r, %r)", u, v)
        return list(accepted.values())

    def _draw_from_enumerated_complement(
        self,
        graph: nx.Graph,
        vertices: Sequence[Hashable],
        count: int,
    ) -> List[Edge]:
        complement = [
            (u, v)
            for u, v in itertools.combinations(vertices, 2)
            if not graph.has_edge(u, v)
        ]
        if len(complement) > count:
            chosen = self.generator.choice(
                len(complement), size=count, replace=False
            )
            return [complement[row] for row in chosen]
        elif len(complement) == count:
            return complement
        else:
            logger.warning(
                "The complement holds only %d pairs, fewer than the %d "
                "requested negatives",
                len(complement),
                count,
            )
            return complement


def labels_of(
    positive_edges: Sequence[Edge],
    negative_edges: Sequence[Edge],
) -> np.ndarray:
    return np.concatenate(
        [
            np.full(len(positive_edges), POSITIVE_LABEL),
            np.full(len(negative_edges), NEGATIVE_LABEL),
        ]
    )


def link_prediction_split_of(
    embedding: EmbeddingSnapshot,
    positive_edges: Sequence[Edge],
    negative_edges: Sequence[Edge],
) -> LinkPredictionSplit:
    pairs = list(positive_edges) + list(negative_edges)
    return LinkPredictionSplit(
        embedding.embed_nodes([u for u, _ in pairs]),
        embedding.embed_nodes([v for _, v in pairs]),
        labels_of(positive_edges, negative_edges).astype(float),
    )


def empty_split_of(dim: int) -> LinkPredictionSplit:
    return LinkPredictionSplit(
        np.zeros((0, dim)), np.zeros((0, dim)), np.zeros(0)
    )


def require_both_classes(
    positive_edges: Sequence[Edge],
    negative_edges: Sequence[Edge],
) -> None:
    if len(positive_edges) > 0 and len(negative_edges) > 0:
        logger.debug(
            "%d positive and %d negative pairs",
            len(positive_edges),
            len(negative_edges),
        )
    else:
        logger.error(
            "Need both classes, got %d positive and %d negative pairs",
            len(positive_edges),
            len(negative_edges),
        )
        raise ValueError("Need at least one positive and one negative pair")


def require_matching_dimension(
    embedding: EmbeddingSnapshot,
    decoder: IncrementalLogitsDecoder,
) -> None:
    if embedding.dim == decoder.vector_dim:
        logger.debug("Embedding and decoder have dimension %d", embedding.dim)
    else:
        logger.error(
            "Embedding dimension %d differs from decoder dimension %d",
            embedding.dim,
            decoder.vector_dim,
        )
        raise ValueError("Embedding and decoder dimensions differ")


def train_decoder(
    decoder: IncrementalLogitsDecoder,
    embedding: EmbeddingSnapshot,
    positive_edges: Sequence[Edge],
    negative_edges: Sequence[Edge],
    seed: int,
) -> List[float]:
    require_both_classes(positive_edges, negative_edges)
    require_matching_dimension(embedding, decoder)
    dataset = LinkPredictionDataset(
        train=link_prediction_split_of(
            embedding, positive_edges, negative_edges
        ),
        val=empty_split_of(embedding.dim),
    )
    train_losses, val_losses = decoder.refit(dataset, seed)
    logger.debug(
        "Decoder refit on %d pairs, no validation split (losses %s)",
        len(dataset.train.y),
        val_losses,
    )
    return train_losses


def threshold_free_scores_of(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[float, float]:
    positive_count = int(np.sum(labels == POSITIVE_LABEL))
    if 0 < positive_count < len(labels):
        return (
            float(roc_auc_score(labels, probabilities)),
            float(average_precision_score(labels, probabilities)),
        )
    else:
        logger.warning(
            "AUC and AP need both classes, got %d positives among %d pairs",
            positive_count,
            len(labels),
        )
        return UNDEFINED_SCORE, UNDEFINED_SCORE


@dataclass
class DecoderScores:
    positive_count: int
    negative_count: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    auc: float
    average_precision: float


def decoder_scores_of(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predicted: np.ndarray,
) -> DecoderScores:
    auc, average_precision = threshold_free_scores_of(labels, probabilities)
    return DecoderScores(
        positive_count=int(np.sum(labels == POSITIVE_LABEL)),
        negative_count=int(np.sum(labels == NEGATIVE_LABEL)),
        precision=float(
            precision_score(
                labels, predicted, zero_division=NO_ZERO_DIVISION_SCORE
            )
        ),
        recall=float(
            recall_score(
                labels, predicted, zero_division=NO_ZERO_DIVISION_SCORE
            )
        ),
        f1=float(
            f1_score(labels, predicted, zero_division=NO_ZERO_DIVISION_SCORE)
        ),
        accuracy=float(accuracy_score(labels, predicted)),
        auc=auc,
        average_precision=average_precision,
    )


POSITIVES_IN_SCOPE: Dict[ScoreScope, Callable[[np.ndarray], np.ndarray]] = {
    ScoreScope.ALL: lambda scopes: np.full(len(scopes), True),
    ScoreScope.WITHIN: lambda scopes: scopes == PartitionScope.WITHIN.value,
    ScoreScope.CROSS: lambda scopes: scopes == PartitionScope.CROSS.value,
}


@dataclass
class ScoredPairs:
    labels: List[int] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    predicted: List[bool] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)

    @staticmethod
    def concatenated(parts: Sequence["ScoredPairs"]) -> "ScoredPairs":
        return ScoredPairs(
            labels=[x for part in parts for x in part.labels],
            probabilities=[x for part in parts for x in part.probabilities],
            predicted=[x for part in parts for x in part.predicted],
            scopes=[x for part in parts for x in part.scopes],
        )

    def __len__(self) -> int:
        return len(self.labels)

    def scores_by_scope(self) -> Dict[str, DecoderScores]:
        labels = np.array(self.labels, dtype=int)
        probabilities = np.array(self.probabilities, dtype=float)
        predicted = np.array(self.predicted, dtype=bool)
        scopes = np.array(self.scopes, dtype=str)
        scores: Dict[str, DecoderScores] = {}
        for scope, positives_in_scope in POSITIVES_IN_SCOPE.items():
            kept = (labels == NEGATIVE_LABEL) | positives_in_scope(scopes)
            if np.any(labels[kept] == POSITIVE_LABEL):
                scores[scope.value] = decoder_scores_of(
                    labels[kept], probabilities[kept], predicted[kept]
                )
            else:
                logger.debug("No positive pair in scope %s", scope.value)
        return scores


def evaluate_decoder(
    embedding: EmbeddingSnapshot,
    decoder: IncrementalLogitsDecoder,
    positive_edges: Sequence[Edge],
    negative_edges: Sequence[Edge],
    partitions_by_vertex: PartitionsByVertex,
) -> ScoredPairs:
    require_both_classes(positive_edges, negative_edges)
    require_matching_dimension(embedding, decoder)
    pairs = list(positive_edges) + list(negative_edges)
    predictions = predict_links(pairs, embedding, decoder)
    return ScoredPairs(
        labels=labels_of(positive_edges, negative_edges).tolist(),
        probabilities=[float(probability) for _, probability in predictions],
        predicted=[bool(is_edge) for is_edge, _ in predictions],
        scopes=[
            partition_scope_of(u, v, partitions_by_vertex).value
            for u, v in pairs
        ],
    )


@dataclass
class BufferLinkPredictionReport:
    buffer_index: int
    edge_counts: Dict[str, int]
    positive_counts: Dict[str, int]
    negative_count: int
    scores: Dict[str, DecoderScores]
    scored_pairs: ScoredPairs
    training_loss: float


def edge_counts_of(
    edges: Mapping[EdgeNovelty, Sequence[Edge]],
) -> Dict[str, int]:
    return {novelty.value: len(edges[novelty]) for novelty in EdgeNovelty}


def description_of_scores(scores: Mapping[str, DecoderScores]) -> str:
    return ", ".join(
        f"{scope} AUC {score.auc:.4f} AP {score.average_precision:.4f} "
        f"F1 {score.f1:.4f}"
        for scope, score in scores.items()
    )


def description_of_report(report: BufferLinkPredictionReport) -> str:
    counts = (
        f"{report.edge_counts[EdgeNovelty.NEW_EDGE.value]} new edges "
        f"({report.positive_counts[PartitionScope.WITHIN.value]} within, "
        f"{report.positive_counts[PartitionScope.CROSS.value]} cross), "
        f"{report.negative_count} negatives, skipped "
        f"{report.edge_counts[EdgeNovelty.NEW_VERTEX.value]} with a new "
        f"vertex, {report.edge_counts[EdgeNovelty.REPEATED_EDGE.value]} "
        f"repeated, {report.edge_counts[EdgeNovelty.SELF_LOOP.value]} "
        "self-loops"
    )
    if len(report.scores) > 0:
        return (
            f"Link prediction buffer {report.buffer_index + 1}: {counts}; "
            f"{description_of_scores(report.scores)}"
        )
    else:
        return (
            f"Link prediction buffer {report.buffer_index + 1}: {counts}; "
            "not tested"
        )


class BufferLinkPrediction(ABC):
    @abstractmethod
    def evaluate_buffer(
        self,
        buffer_index: int,
        buffer: Sequence[Event],
        vertices: Sequence[Hashable],
        embeddings: Sequence,
        partitioner: VertexPartitioner,
    ) -> None:
        ...

    @abstractmethod
    def results(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def result_names(self) -> List[str]:
        ...


class DisabledLinkPrediction(BufferLinkPrediction):
    def evaluate_buffer(
        self,
        buffer_index: int,
        buffer: Sequence[Event],
        vertices: Sequence[Hashable],
        embeddings: Sequence,
        partitioner: VertexPartitioner,
    ) -> None:
        logger.debug("Link prediction is disabled, buffer %d", buffer_index)

    def results(self) -> Dict[str, Any]:
        return {}

    def result_names(self) -> List[str]:
        return []


class PrequentialLinkPrediction(BufferLinkPrediction):
    def __init__(
        self,
        decoder: IncrementalLogitsDecoder,
        sampler: NegativeEdgeSampler,
        negative_ratio: int,
        seed: int,
        log_line: Callable[[str], None] = logger.info,
    ):
        if negative_ratio <= 0:
            logger.error("negative_ratio must be positive: %d", negative_ratio)
            raise ValueError(
                f"negative_ratio must be positive, got {negative_ratio}"
            )
        else:
            logger.info(
                "Prequential link prediction with %d negatives per positive "
                "and seed %d",
                negative_ratio,
                seed,
            )
        self.decoder = decoder
        self.sampler = sampler
        self.negative_ratio = negative_ratio
        self.seed = seed
        self.log_line = log_line
        self.known_graph = nx.Graph()
        self.previous_embedding = EmbeddingSnapshot.empty(decoder.vector_dim)
        self.previous_partitions: Dict[Hashable, FrozenSet] = {}
        self.reports: List[BufferLinkPredictionReport] = []

    def evaluate_buffer(
        self,
        buffer_index: int,
        buffer: Sequence[Event],
        vertices: Sequence[Hashable],
        embeddings: Sequence,
        partitioner: VertexPartitioner,
    ) -> None:
        edges = edges_by_novelty(
            buffer, self.known_graph, self.previous_embedding
        )
        self.known_graph.add_edges_from(
            (event.src, event.dest) for event in buffer
        )
        positives = edges[EdgeNovelty.NEW_EDGE]
        negatives = self.sampler.sample(
            self.known_graph,
            self.previous_embedding.vertices(),
            self.negative_ratio * len(positives),
        )
        scored_pairs, training_loss = self._test_then_refit(
            buffer_index, positives, negatives
        )
        report = BufferLinkPredictionReport(
            buffer_index=buffer_index,
            edge_counts=edge_counts_of(edges),
            positive_counts=edge_counts_by_partition_scope(
                positives, self.previous_partitions
            ),
            negative_count=len(negatives),
            scores=scored_pairs.scores_by_scope(),
            scored_pairs=scored_pairs,
            training_loss=training_loss,
        )
        self.reports.append(report)
        self.log_line(description_of_report(report))
        self.previous_embedding = EmbeddingSnapshot(
            vertices, embeddings, self.decoder.vector_dim
        )
        self.previous_partitions = frozen_vertex_partitions(
            partitioner, vertices
        )

    def results(self) -> Dict[str, Any]:
        pooled = ScoredPairs.concatenated(
            [report.scored_pairs for report in self.reports]
        )
        return {
            REPORTS_RESULT_NAME: [asdict(report) for report in self.reports],
            POOLED_RESULT_NAME: {
                scope: asdict(scores)
                for scope, scores in pooled.scores_by_scope().items()
            },
        }

    def result_names(self) -> List[str]:
        return [REPORTS_RESULT_NAME, POOLED_RESULT_NAME]

    def _test_then_refit(
        self,
        buffer_index: int,
        positives: Sequence[Edge],
        negatives: Sequence[Edge],
    ) -> Tuple[ScoredPairs, float]:
        if len(positives) > 0 and len(negatives) > 0:
            scored_pairs = self._test_trained_decoder(positives, negatives)
            training_losses = train_decoder(
                self.decoder,
                self.previous_embedding,
                positives,
                negatives,
                derive_iteration_seed(self.seed, buffer_index),
            )
            return scored_pairs, training_losses[-1]
        else:
            logger.info(
                "Buffer %d has %d new edges and %d negatives, the decoder "
                "is neither tested nor trained",
                buffer_index,
                len(positives),
                len(negatives),
            )
            return ScoredPairs(), UNDEFINED_SCORE

    def _test_trained_decoder(
        self,
        positives: Sequence[Edge],
        negatives: Sequence[Edge],
    ) -> ScoredPairs:
        if self.decoder.refit_count > 0:
            return evaluate_decoder(
                self.previous_embedding,
                self.decoder,
                positives,
                negatives,
                self.previous_partitions,
            )
        else:
            logger.info("No trained decoder yet, the buffer only trains it")
            return ScoredPairs()
