
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN

from vertex_voyage.dynnode2vec import DynNode2Vec
from vertex_voyage.node2vec import Node2Vec
from vertex_voyage.temporal import EventType
from vertex_voyage.temporal_partitioning import (
    Partition,
    Event,
    EventSequence,
    TemporalGraphPartitioner,
)


class SemanticEmbeddingModel(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def update(self, buffer: list[Event]) -> None:
        pass

    @abstractmethod
    def embed_node(self, node: Any) -> np.ndarray:
        pass

    def embed_nodes(self, nodes: list[Any]) -> dict[Any, np.ndarray]:
        return {node: self.embed_node(node) for node in nodes}


class Node2VecSemanticEmbeddingModel(SemanticEmbeddingModel):
    """
    Stateless-over-time semantic model: each update re-fits on the buffer subgraph only.
    """

    def __init__(self, model: Optional[Node2Vec] = None, **node2vec_kwargs):
        self.model = model or Node2Vec(**node2vec_kwargs)
        self._known_nodes: set[Any] = set()

    @property
    def dim(self) -> int:
        return self.model.dim

    def update(self, buffer: list[Event]) -> None:
        g = nx.Graph()
        for event in buffer:
            if event.type == EventType.REMOVE:
                continue
            g.add_node(event.src)
            g.add_node(event.dest)
            g.add_edge(event.src, event.dest)
        self.model.fit(g, nodes=list(g.nodes()))
        self._known_nodes = set(g.nodes())

    def embed_node(self, node: Any) -> np.ndarray:
        if node not in self._known_nodes:
            return np.zeros(self.dim)
        return np.array(self.model.embed_node(node))


class DynNode2VecSemanticEmbeddingModel(SemanticEmbeddingModel):
    """
    Stateful semantic model: each update incrementally applies buffer events.
    """

    def __init__(self, model: Optional[DynNode2Vec] = None, **dynnode2vec_kwargs):
        self.model = model or DynNode2Vec(**dynnode2vec_kwargs)

    @property
    def dim(self) -> int:
        return self.model.dim

    def update(self, buffer: list[Event]) -> None:
        events = [event for event in buffer if event.type != EventType.REMOVE]
        if events:
            self.model.update(events)

    def embed_node(self, node: Any) -> np.ndarray:
        return np.array(self.model.embed_node(node))


class Metric(ABC):
    @property
    @abstractmethod
    def is_similarity(self) -> bool:
        pass

    @abstractmethod
    def score(self, left: np.ndarray, right: np.ndarray) -> float:
        pass

    def better(self, candidate: float, reference: float) -> bool:
        if self.is_similarity:
            return candidate > reference
        return candidate < reference


class DistanceMetric(Metric):
    @property
    def is_similarity(self) -> bool:
        return False

    def score(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.linalg.norm(left - right))


class CosineSimilarityMetric(Metric):
    @property
    def is_similarity(self) -> bool:
        return True

    def score(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))


class PartitionAssignment(ABC):
    @abstractmethod
    def assign(
        self,
        node_embeddings: dict[Any, np.ndarray],
        partitions: set[Partition],
        metric: Metric,
        partition_embedding_fn: Callable[[Partition], Optional[np.ndarray]],
    ) -> dict[Any, set[Partition]]:
        pass


class KNearestNeighborsAssignment(PartitionAssignment):
    def __init__(self, k: int = 1):
        self.k = max(1, k)

    def assign(
        self,
        node_embeddings: dict[Any, np.ndarray],
        partitions: set[Partition],
        metric: Metric,
        partition_embedding_fn: Callable[[Partition], Optional[np.ndarray]],
    ) -> dict[Any, set[Partition]]:
        if not partitions:
            return {node: set() for node in node_embeddings}

        ordered_partitions = sorted(partitions, key=lambda partition: partition.size())
        result: dict[Any, set[Partition]] = {}
        for node, node_embedding in node_embeddings.items():
            scored: list[tuple[float, Partition]] = []
            unscored: list[Partition] = []
            for partition in ordered_partitions:
                partition_embedding = partition_embedding_fn(partition)
                if partition_embedding is None:
                    unscored.append(partition)
                    continue
                scored.append((metric.score(node_embedding, partition_embedding), partition))

            if scored:
                scored.sort(key=lambda entry: entry[0], reverse=metric.is_similarity)
                assigned = [partition for _, partition in scored[: self.k]]
            else:
                assigned = []

            if len(assigned) < self.k:
                for partition in unscored:
                    if partition not in assigned:
                        assigned.append(partition)
                    if len(assigned) >= self.k:
                        break

            if not assigned:
                assigned = [ordered_partitions[0]]

            result[node] = set(assigned[: self.k])
        return result


class DBSCANAssignment(PartitionAssignment):
    def __init__(self, eps: float = 0.5, min_samples: int = 2, reassign_noise: bool = True):
        self.eps = eps
        self.min_samples = min_samples
        self.reassign_noise = reassign_noise

    def _to_distance(self, metric: Metric, left: np.ndarray, right: np.ndarray) -> float:
        value = metric.score(left, right)
        if metric.is_similarity:
            return max(0.0, 1.0 - value)
        return max(0.0, value)

    def assign(
        self,
        node_embeddings: dict[Any, np.ndarray],
        partitions: set[Partition],
        metric: Metric,
        partition_embedding_fn: Callable[[Partition], Optional[np.ndarray]],
    ) -> dict[Any, set[Partition]]:
        nodes = list(node_embeddings.keys())
        if not nodes:
            return {}
        if not partitions:
            return {node: set() for node in nodes}

        if len(nodes) == 1:
            fallback = min(partitions, key=lambda partition: partition.size())
            return {nodes[0]: {fallback}}

        distances = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
        for i in range(len(nodes)):
            distances[i, i] = 0.0
            for j in range(i + 1, len(nodes)):
                d = self._to_distance(metric, node_embeddings[nodes[i]], node_embeddings[nodes[j]])
                distances[i, j] = d
                distances[j, i] = d

        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed").fit_predict(distances)

        partition_list = sorted(partitions, key=lambda partition: partition.size())
        clusters = [label for label in sorted(set(labels)) if label != -1]
        clusters.sort(key=lambda label: int(np.sum(labels == label)), reverse=True)
        cluster_to_partition: dict[int, Partition] = {}
        for i, label in enumerate(clusters):
            cluster_to_partition[label] = partition_list[i % len(partition_list)]

        assignments: dict[Any, set[Partition]] = {}
        for idx, node in enumerate(nodes):
            label = int(labels[idx])
            if label != -1:
                assignments[node] = {cluster_to_partition[label]}

        if not self.reassign_noise:
            for idx, node in enumerate(nodes):
                if int(labels[idx]) == -1:
                    assignments[node] = set()
            return assignments

        knn_fallback = KNearestNeighborsAssignment(k=1)
        noise_nodes = [node for idx, node in enumerate(nodes) if int(labels[idx]) == -1]
        if noise_nodes:
            noise_embeddings = {node: node_embeddings[node] for node in noise_nodes}
            fallback_assignments = knn_fallback.assign(
                node_embeddings=noise_embeddings,
                partitions=partitions,
                metric=metric,
                partition_embedding_fn=partition_embedding_fn,
            )
            assignments.update(fallback_assignments)
        return assignments


class SemanticTemporalGraphPartitioner(TemporalGraphPartitioner):
    def __init__(
        self,
        partitions: set[Partition],
        embedding_model: SemanticEmbeddingModel,
        metric: Metric,
        assignment: PartitionAssignment,
    ):
        self.partitions = partitions
        self.embedding_model = embedding_model
        self.metric = metric
        self.assignment = assignment

    @staticmethod
    def _build_metric(metric: str) -> Metric:
        normalized = metric.lower().strip()
        if normalized in {"distance", "l2", "euclidean"}:
            return DistanceMetric()
        if normalized in {"cosine", "similarity", "cosine_similarity"}:
            return CosineSimilarityMetric()
        raise ValueError(f"Unsupported metric '{metric}'. Use 'distance' or 'cosine'.")

    @staticmethod
    def _build_assignment(
        assignment: str,
        *,
        k: int,
        eps: float,
        min_samples: int,
        reassign_noise: bool,
    ) -> PartitionAssignment:
        normalized = assignment.lower().strip()
        if normalized in {"knn", "k_nearest_neighbors", "k-nearest-neighbors"}:
            return KNearestNeighborsAssignment(k=k)
        if normalized in {"dbscan"}:
            return DBSCANAssignment(eps=eps, min_samples=min_samples, reassign_noise=reassign_noise)
        raise ValueError(f"Unsupported assignment '{assignment}'. Use 'knn' or 'dbscan'.")

    @classmethod
    def node2vec(
        cls,
        partitions: set[Partition],
        *,
        metric: str = "cosine",
        assignment: str = "knn",
        k: int = 1,
        eps: float = 0.5,
        min_samples: int = 2,
        reassign_noise: bool = True,
        model: Optional[Node2Vec] = None,
        **node2vec_kwargs,
    ) -> "SemanticTemporalGraphPartitioner":
        embedding_model = Node2VecSemanticEmbeddingModel(model=model, **node2vec_kwargs)
        metric_impl = cls._build_metric(metric)
        assignment_impl = cls._build_assignment(
            assignment,
            k=k,
            eps=eps,
            min_samples=min_samples,
            reassign_noise=reassign_noise,
        )
        return cls(partitions, embedding_model, metric_impl, assignment_impl)

    @classmethod
    def dynnode2vec(
        cls,
        partitions: set[Partition],
        *,
        metric: str = "distance",
        assignment: str = "dbscan",
        k: int = 1,
        eps: float = 0.5,
        min_samples: int = 2,
        reassign_noise: bool = True,
        model: Optional[DynNode2Vec] = None,
        **dynnode2vec_kwargs,
    ) -> "SemanticTemporalGraphPartitioner":
        embedding_model = DynNode2VecSemanticEmbeddingModel(model=model, **dynnode2vec_kwargs)
        metric_impl = cls._build_metric(metric)
        assignment_impl = cls._build_assignment(
            assignment,
            k=k,
            eps=eps,
            min_samples=min_samples,
            reassign_noise=reassign_noise,
        )
        return cls(partitions, embedding_model, metric_impl, assignment_impl)

    def _partition_embedding(self, partition: Partition) -> Optional[np.ndarray]:
        nodes = list(partition.graph().nodes())
        if not nodes:
            return None
        embeddings = [self.embedding_model.embed_node(node) for node in nodes]
        if not embeddings:
            return None
        stacked = np.stack(embeddings, axis=0)
        if stacked.size == 0:
            return None
        if np.allclose(stacked, 0):
            return None
        return np.mean(stacked, axis=0)

    def _set_vertex_partitions(self, vertex: Any, target_partitions: set[Partition]) -> None:
        current_partitions = self.get(vertex)
        for partition in current_partitions - target_partitions:
            partition.remove(vertex)
        for partition in target_partitions - current_partitions:
            partition.add(vertex)

    def push(self, buffer: EventSequence):
        events = list(buffer)
        if not events:
            return

        self.embedding_model.update(events)

        active_nodes = list({event.src for event in events} | {event.dest for event in events})
        node_embeddings = self.embedding_model.embed_nodes(active_nodes)
        assignments = self.assignment.assign(
            node_embeddings=node_embeddings,
            partitions=self.partitions,
            metric=self.metric,
            partition_embedding_fn=self._partition_embedding,
        )

        for node in active_nodes:
            target = assignments.get(node)
            if target is None or len(target) == 0:
                continue
            self._set_vertex_partitions(node, target)

        for event in events:
            if event.type == EventType.REMOVE:
                src_partitions = self.get(event.src)
                dest_partitions = self.get(event.dest)
                for partition in src_partitions.intersection(dest_partitions):
                    if partition.connects(event.src, event.dest):
                        partition.disconnect(event.src, event.dest)
                continue

            src_partitions = self.get(event.src)
            dest_partitions = self.get(event.dest)
            for partition in src_partitions.intersection(dest_partitions):
                partition.connect(event.src, event.dest)

    def get(self, vertex) -> set[Partition]:
        return {partition for partition in self.partitions if partition.has(vertex)}

