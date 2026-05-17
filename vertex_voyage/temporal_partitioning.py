
from abc import ABC, abstractmethod
from turtle import st
from typing import Any, Callable, Iterable, Optional, Set
from vertex_voyage.temporal import Event, EventSequence, FromIterable
from random import randint, random, choice, shuffle
import numpy as np 
import networkx as nx

import logging

logger = logging.getLogger(__name__)


class Partition:
    @abstractmethod
    def add(self, node):
        pass
    @abstractmethod
    def connect(self, node1, node2, directed=False):
        pass
    @abstractmethod
    def graph(self) -> nx.Graph:
        pass
    @abstractmethod
    def size(self) -> int:
        pass
    @abstractmethod
    def has(self, node) -> bool:
        pass
    @abstractmethod
    def connects(self, node1, node2, directed=False) -> bool:
        pass
    @abstractmethod
    def remove(self, node):
        pass
    @abstractmethod
    def disconnect(self, node1, node2, directed=False):
        pass
    
    def __add__(self, other):
        return SumPartition(self, other)
    
    def __sub__(self, other):
        return DiffPartition(self, other)

class SumPartition(Partition):
    def __init__(self, *partitions):
        self.partitions = partitions
    def add(self, node):
        for partition in self.partitions:
            partition.add(node)
    def connect(self, node1, node2, directed=False):
        for partition in self.partitions:
            partition.connect(node1, node2, directed)
    def graph(self) -> nx.Graph:
        g = nx.Graph()
        for partition in self.partitions:
            g = nx.compose(g, partition.graph())
        return g
    def size(self) -> int:
        return self.graph().number_of_nodes()
    def has(self, node) -> bool:
        return any(partition.has(node) for partition in self.partitions)
    def connects(self, node1, node2, directed=False) -> bool:
        return any(partition.connects(node1, node2, directed) for partition in self.partitions)
    def remove(self, node):
        for partition in self.partitions:
            partition.remove(node)
    def disconnect(self, node1, node2, directed=False):
        for partition in self.partitions:
            partition.disconnect(node1, node2, directed)

class DiffPartition(Partition):
    def __init__(self, partition1, partition2):
        self.partition1 = partition1
        self.partition2 = partition2
    def add(self, node):
        self.partition1.add(node)
    def connect(self, node1, node2, directed=False):
        self.partition1.connect(node1, node2, directed)
    def graph(self) -> nx.Graph:
        g1 = self.partition1.graph()
        g2 = self.partition2.graph()
        g = nx.difference(g1, g2)
        to_remove = [node for node in g.nodes() if not self.has(node)]
        g.remove_nodes_from(to_remove)
        return g
    def size(self) -> int:
        return self.graph().number_of_nodes()
    def has(self, node) -> bool:
        return self.partition1.has(node) and not self.partition2.has(node)
    def connects(self, node1, node2, directed=False) -> bool:
        return self.partition1.connects(node1, node2, directed) and not self.partition2.connects(node1, node2, directed)
    def remove(self, node):
        self.partition1.remove(node)
    def disconnect(self, node1, node2, directed=False):
        self.partition1.disconnect(node1, node2, directed)


class InMemoryPartition(Partition):
    def __init__(self, id):
        self.g = nx.Graph()
        self.id = id 
    def add(self, node):
        self.g.add_node(node)
    def connect(self, node1, node2, directed=False):
        self.g.add_edge(node1, node2)
    def graph(self) -> nx.Graph:
        return self.g.copy()
    def size(self) -> int:
        return self.g.number_of_nodes()
    def has(self, node) -> bool:
        return self.g.has_node(node)
    def connects(self, node1, node2, directed=False) -> bool:
        return self.g.has_edge(node1, node2)
    def remove(self, node):
        self.g.remove_node(node)
    def disconnect(self, node1, node2, directed=False):
        self.g.remove_edge(node1, node2)
    
    def __str__(self):
        return f"P{self.id}"
    def __repr__(self):
        return str(self)
    
    @staticmethod
    def empty(id=0):
        return InMemoryPartition(id)
    
    def __hash__(self) -> int:
        return hash(self.id)


class TemporalGraphPartitioner:
    @abstractmethod
    def push(self, batch: EventSequence):
        """
        Partitions an events into a partition number
        """
        pass

    @abstractmethod
    def get(self, vertex) -> Set[Partition]:
        """
        Returns the partition of a vertex
        """
        pass

    def get_partition_batches(self, batch: EventSequence) -> list[tuple[Partition, list[Event]]]:
        partition_batches = {}
        for event in batch:
            src_partitions = self.get(event.src)
            dest_partitions = self.get(event.dest)
            for partition in src_partitions.intersection(dest_partitions):
                partition_batches.setdefault(partition, []).append(event)
        return list(partition_batches.items())

    def get_distributed_embedding(self, models, nodes):
        results = []
        for node in nodes:
            node_partitions = self.get(node)
            if not node_partitions:
                logger.warning(f"Node {node} is not in any partition, returning zero embedding")
                results.append(np.zeros(models[next(iter(models))].dim))
            else:
                results.append(np.mean([models[partition].embed_node(node) for partition in node_partitions], axis=0))
                logger.debug(f"Node {node} is in partitions {[partition.id for partition in node_partitions]}, embedding dim: {results[-1].shape}")
        return results

class CatchAllPartitioner(TemporalGraphPartitioner):
    """
    Partition of vertex is the set of all partitions available.
    """
    def __init__(self, partitions: Set[Partition]):
        self.partitions = partitions

    def push(self, batch: EventSequence):
        for event in batch:
            for partition in self.partitions:
                partition.add(event.src)
                partition.add(event.dest)
                partition.connect(event.src, event.dest)
    
    def get(self, vertex) -> Set[Partition]:
        return self.partitions

class RandomPartitioner(TemporalGraphPartitioner):
    """
    Partition of vertex is a random partition.

    It takes distribution P(partition | vertex) as input, and samples a partition for each vertex according to the distribution.
    """
    def __init__(self, partitions: Set[Partition], distribution: Callable[[Any, Partition], float]):
        self.partitions = partitions
        self.distribution = distribution
    
    @staticmethod
    def uniform(partitions: set[Partition]):
        def uniform_distribution(vertex, partition):
            return 1 / len(partitions)
        return RandomPartitioner(partitions, uniform_distribution)
    
    @staticmethod
    def degree_based(partitions: Set[Partition]):
        def degree_based_distribution(vertex, partition):
            degree = sum(partition.has(neighbor) for neighbor in partition.graph().neighbors(vertex))
            total_degree = sum(sum(partition.has(neighbor) for neighbor in partition.graph().neighbors(vertex)) for partition in partitions)
            return degree / total_degree if total_degree > 0 else 1 / len(partitions)
        return RandomPartitioner(partitions, degree_based_distribution)
    
    @staticmethod
    def size_based(partitions: Set[Partition]):
        def size_based_distribution(vertex, partition):
            size = partition.size()
            total_size = sum(partition.size() for partition in partitions)
            return size / total_size if total_size > 0 else 1 / len(partitions)
        return RandomPartitioner(partitions, size_based_distribution)
    def push(self, batch: EventSequence):
        for event in batch:
            for vertex in [event.src, event.dest]:
                if any(partition.has(vertex) for partition in self.partitions):
                    continue
                partition = self.sample_partition(vertex)
                partition.add(vertex)
            for partition in self.partitions:
                if partition.has(event.src) and partition.has(event.dest):
                    partition.connect(event.src, event.dest)
    def sample_partition(self, vertex) -> Partition:
        partitions = list(self.partitions)
        probabilities = [self.distribution(vertex, partition) for partition in partitions]
        total = sum(probabilities)
        if total == 0:
            probabilities = [1 / len(partitions) for _ in partitions]
        else:
            probabilities = [p / total for p in probabilities]
        cdf = 0 
        for partition, p in zip(partitions, probabilities):
            cdf += p
            if random() < cdf:
                return partition

    def get(self, vertex) -> Set[Partition]:
        return {partition for partition in self.partitions if partition.has(vertex)}


class PartitionerProfile(TemporalGraphPartitioner):
    def __init__(self, partitioner: TemporalGraphPartitioner) -> None:
        """
        This is decorator for partitioner that allows us to profile the partitioner by keeping track of the:
        - number of events processed
        - number of batches processed
        - number of unique vertices seen
        - time taken to process events
        - partition sizes after processing every batch 
        - edge cut after processing every batch
        """
        self.partitioner = partitioner
        self.num_events = 0
        self.num_batches = 0
        self.unique_vertices = set()
        self.partition_sizes = []
        self.edge_cuts = []
        self.edge_cuts_percentage = []
        self.total_time = 0
        self.original_graph = nx.Graph()
        self.lost_edges_per_batch = []
        self.lost_edges_per_batch_percentage = []
    
    def push(self, batch: EventSequence):
        import time
        self.num_batches += 1
        # We need to do this to keep track of the original graph structure for calculating edge cuts, since the partitioner may not store all edges in the partitions
        batch = list(batch)
        self.original_graph.add_edges_from([(event.src, event.dest) for event in batch])
        start_time = time.time()
        self.partitioner.push(batch)
        self.total_time += time.time() - start_time
        partitions = self.partitioner.partitions if hasattr(self.partitioner, 'partitions') else set()
        if not partitions:
            raise ValueError("Partitioner must have a 'partitions' attribute to use PartitionerProfile")        
        for event in batch:
            self.num_events += 1
            self.unique_vertices.add(event.src)
            self.unique_vertices.add(event.dest)
        self.partition_sizes.append({partition: partition.size() for partition in partitions})
        edge_cut = self.calculate_edge_cut()
        self.edge_cuts.append(edge_cut)
        total_edges = self.original_graph.number_of_edges()
        self.edge_cuts_percentage.append(edge_cut / total_edges if total_edges > 0 else 0)
        
        _lost_edges = 0
        for event in batch:
            src_partitions = {partition for partition in partitions if partition.has(event.src)}
            dest_partitions = {partition for partition in partitions if partition.has(event.dest)}
            if src_partitions.isdisjoint(dest_partitions):
                _lost_edges += 1
        self.lost_edges_per_batch.append(_lost_edges)
        self.lost_edges_per_batch_percentage.append(_lost_edges / len(batch) if len(batch) > 0 else 0)
    
    def calculate_edge_cut(self) -> int:
        edge_cut = 0
        partitions = self.partitioner.partitions if hasattr(self.partitioner, 'partitions') else set()
        for node in self.unique_vertices:
            node_partitions = {partition for partition in partitions if partition.has(node)}
            for neighbor in self.original_graph.neighbors(node):
                neighbor_partitions = {partition for partition in partitions if partition.has(neighbor)}
                if node_partitions.isdisjoint(neighbor_partitions):
                    edge_cut += 1
        return edge_cut // 2 # each edge is counted twice
    
    def get(self, vertex) -> Set[Partition]:
        return self.partitioner.get(vertex)
    
    def print_profile(self):
        def p(msg):
            logger.info(msg)
            print(msg)
        p(f"Number of events processed: {self.num_events}")
        p(f"Number of batches processed: {self.num_batches}")
        p(f"Number of unique vertices seen: {len(self.unique_vertices)}")
        p(f"Total time taken to process events: {self.total_time:.2f} seconds")
        p(f"Partition sizes after processing every batch: {self.partition_sizes}")
        p(f"Edge cuts after processing every batch: {self.edge_cuts}")
        p(f"Edge cuts percentage after processing every batch: {self.edge_cuts_percentage}")
        p(f"Lost edges after processing every batch: {self.lost_edges_per_batch}")
        p(f"Lost edges percentage after processing every batch: {self.lost_edges_per_batch_percentage}")

class MostCommonNeighborPartitioner(TemporalGraphPartitioner):
    """
    Partition of vertex is the partition that contains the most neighbors of the vertex in the sampled events. The partitioner samples a subset of events according to a distribution P(event | vertex), and assigns the vertex to the partition that contains the most neighbors of the vertex in the sampled events. The partitioner also has a replication factor, which allows it to assign a vertex to multiple partitions if there are multiple partitions that contain a similar number of neighbors of the vertex in the sampled events. The partitioner also has a capacity penalty, which penalizes partitions that have more vertices than the average partition size, to encourage more balanced partitions.

    Formally, the score of a partition for a vertex is defined as:

    $$ S(P, v) = N(P, v) - \mu \cdot \max(0, |P| - \alpha \cdot (1 + \epsilon) \cdot \frac{1}{|P|} \sum_{P' \in P} |P'|) $$

    where $N(P, v)$ is the number of neighbors of vertex $v$ in partition $P$ in the sampled events, $\mu$ is the capacity penalty coefficient, $\alpha$ is the weight of the average partition size in the capacity penalty, $\epsilon$ is the imbalance tolerance, and $|P|$ is the size of partition $P$. The partitioner assigns the vertex to the top $k$ partitions with the highest scores, where $k$ is the replication factor. If there are multiple partitions with similar scores, the partitioner randomly assigns the vertex to some of those partitions until it reaches the replication factor.

    Also, if decay is not None, the partitioner will use an exponentially decaying weight for the neighbors in the sampled events, where the weight of a neighbor that was seen $t$ time steps ago is multiplied by $\exp(-\text{decay} \cdot t)$. This allows the partitioner to give more importance to recent neighbors when assigning vertices to partitions.

    Formally, the score of a partition for a vertex with decay is defined as:

    $$ S(P, v) = \sum_{u \in N(P, v)} \exp(-\text{decay} \cdot t(u)) - \mu \cdot \max(0, |P| - \alpha \cdot (1 + \epsilon) \cdot \frac{1}{|P|} \sum_{P' \in P} |P'|) $$

    
    """
    def __init__(
            self, 
            partitions: Set[Partition], 
            distribution: Callable[[Any, Event], float],
            replication_factor: int = 1,
            mu: float = 0,
            epsilon: float = 0.1,
            alpha: float = 1.0,
            decay: Optional[float] = None
        ):
        """
        distribution: P(event | vertex) is a function that takes an event and returns a probability of sampling that event. The partitioner will sample an event according to the distribution, and assign the vertex to the partition that contains the most neighbors of the vertex in the sampled event.
        """
        self.partitions = partitions
        self.distribution = distribution
        self.replication_factor = replication_factor
        self.randomly_assigned = set() # keep track of vertices that are randomly assigned to a partition due to not having enough neighbors in the sampled events
        self.mu = mu
        self.epsilon = epsilon
        self.alpha = alpha
        self.previous_capacity_estimate = 0
        self.decay = decay
        if decay is not None:
            self.full_graph = nx.Graph()
            self.time = 0
    def sample_events(self, batch: EventSequence) -> set[Event]:
        events = list(batch)
        probabilities = [
            self.distribution(event.src, event) + 
            self.distribution(event.dest, event) 
            for event in events
        ]
        sampled_events = []
        for event, p in zip(events, probabilities):
            if random() <= p:
                sampled_events.append(event)
        return set(sampled_events)

    def score_partition(self, vertex, partition, batch):
        if self.decay is None:
            neighbor_count = sum(1 for event in batch if (partition.has(event.src) and event.src != vertex) or (partition.has(event.dest) and event.dest != vertex))
        else:
            neighbor_count = 0 
            for event in batch:
                self.full_graph.add_edge(event.src, event.dest, timestamp=self.time)
            for u, v in self.full_graph.edges():
                if self.full_graph.has_edge(u, v):
                    timestamp = self.full_graph[u][v]['timestamp']
                    if partition.has(u) and u != vertex:
                        neighbor_count += np.exp(-self.decay * (self.time - timestamp))
                    if partition.has(v) and v != vertex:
                        neighbor_count += np.exp(-self.decay * (self.time - timestamp))
            self.time += 1
        average_capacity = self.alpha * (np.mean([partition.size() for partition in self.partitions]) if self.partitions else 0) * (1 + self.epsilon) + (1 - self.alpha) * self.previous_capacity_estimate
        self.previous_capacity_estimate = average_capacity
        capacity_penalty = self.mu * max(0, partition.size() - average_capacity)
        score = neighbor_count - capacity_penalty
        return score

    def push(self, batch: EventSequence):
        """
        Assign each vertex to the partition that contains the most neighbors of the vertex in the sampled events. Then connect the vertices in the same partition according to the events in the batch.
        """
        batch = list(batch)
        batch = self.sample_events(batch=batch)
        for event in batch:
            for vertex in [event.src, event.dest]:
                best_partitions = sorted(self.partitions, key=lambda partition: self.score_partition(vertex, partition, batch), reverse=True)[:self.replication_factor]
                for partition in best_partitions:
                    logger.debug(f"Assigning vertex {vertex} to partition {partition} based on score: {self.score_partition(vertex, partition, batch)}")
                    partition.add(vertex)
                shuffled_partitions = self.partitions.copy()
                shuffled_partitions = list(shuffled_partitions)
                shuffled_partitions = [partition for partition in shuffled_partitions if partition not in best_partitions]
                np.random.shuffle(shuffled_partitions)
                shuffled_partitions = shuffled_partitions[:self.replication_factor - len(best_partitions)]
                for partition in shuffled_partitions:
                    if vertex not in self.randomly_assigned:
                        logger.debug(f"Randomly assigning vertex {vertex} to partition {partition.id} due to replication factor")
                        partition.add(vertex)
                        self.randomly_assigned.add(vertex)
                for partition in self.partitions:
                    if partition not in best_partitions and partition not in shuffled_partitions and partition.has(vertex):
                        logger.debug(f"Removing vertex {vertex} from partition {partition} due to repartitioning")
                        partition.remove(vertex)
        for event in batch:
            for partition in self.partitions:
                if partition.has(event.src) and partition.has(event.dest):
                    partition.connect(event.src, event.dest)
    def get(self, vertex) -> Set[Partition]:
        return {partition for partition in self.partitions if partition.has(vertex)}
    
    @staticmethod
    def all_neighbors(
        partitions: Set[Partition], 
        replication_factor: int = 1,
        mu: float = 0,
        epsilon: float = 0.1,
        alpha: float = 1.0,
        decay: Optional[float] = None
    ):
        def all_neighbors_distribution(vertex, event):
            return 1.0
        return MostCommonNeighborPartitioner(partitions, all_neighbors_distribution, replication_factor, mu, epsilon, alpha, decay)
    
    @staticmethod
    def degree_based(
        partitions: Set[Partition], 
        replication_factor: int = 1,
        mu: float = 0,
        epsilon: float = 0.1,
        alpha: float = 1.0,
        decay: Optional[float] = None
    ):
        def degree_based_distribution(vertex, event):
            degree_src = sum(partition.has(neighbor) for partition in partitions for neighbor in partition.graph().neighbors(event.src))
            degree_dest = sum(partition.has(neighbor) for partition in partitions for neighbor in partition.graph().neighbors(event.dest))
            total_degree = degree_src + degree_dest
            if total_degree == 0:
                return 0.5
            return degree_src / total_degree
        return MostCommonNeighborPartitioner(partitions, degree_based_distribution, replication_factor, mu, epsilon, alpha, decay)
