
from abc import abstractmethod
from vertex_voyage.temporal import Event, EventSequence, FromIterable
from random import randint, random
import numpy as np 
class TemporalGraphPartitioner:
    @abstractmethod
    def partition(self, event: Event):
        """
        Partitions an event into a partition number
        """
        pass

    @abstractmethod
    def get_partition(self, vertex) -> int:
        """
        Returns the partition of a vertex
        """
        pass
    @abstractmethod
    def get_partition_num(self) -> int:
        """
        Returns the number of partitions
        """
        pass
    
    @abstractmethod
    def get_subgraph(self, partition_id: int) -> EventSequence:
        """
        Returns the subgraph of a partition
        """
        pass


def partition_temporal_graph(tg: EventSequence, partitioner: TemporalGraphPartitioner):
    """
    Partitions a temporal graph into partitions.

    :param tg: Temporal graph
    :param partitioner: Partitioner
    :return: List of partitions
    """
    partitions = dict()
    vertices = set()
    for event in tg:
        partitioner.partition(event)
        vertices.add(event.src)
        vertices.add(event.dest)
    for vertex in vertices:
        partition = partitioner.get_partition(vertex)
        if partition not in partitions:
            partitions[partition] = []
        partitions[partition].append(vertex)
    return partitions.values()

def get_most_common_partition(vertex, partition_map: dict, neighbor_map: dict):
    """
    Returns the most common partition of the neighbors of a vertex.
    :param vertex: Vertex
    :param partition_map: Partition map
    :param neighbor_map: Neighbor map
    :return: Most common partition
    """       
    freq = dict()
    if vertex not in neighbor_map:
        if vertex not in partition_map:
            return 0
        else:
            return partition_map[vertex]
    for neighbor in neighbor_map[vertex]:
        if neighbor in partition_map:
            partition = partition_map[neighbor]
            if partition not in freq:
                freq[partition] = 0
            freq[partition] += 1
    return max(freq, key=freq.get) if freq else partition_map[vertex]

class LabelPropagationTemporalGraphPartitioner(TemporalGraphPartitioner):
    def __init__(self, num_partitions: int, p: float = 0.5):
        """
        Initializes the LabelPropagationTemporalGraphPartitioner.
        :param num_partitions: Number of partitions
        :param p: Probability of reassigning a vertex to a random partition
        """
        self.num_partitions = num_partitions
        self.partition_map = dict()
        self.neighbor_map = dict()
        self.p = p
        self.subgraphs = [[] for _ in range(num_partitions)]

    def partition(self, event: Event):
        if event.src not in self.partition_map:
            self.partition_map[event.src] = randint(0, self.num_partitions - 1)
        if event.dest not in self.partition_map:
            self.partition_map[event.dest] = randint(0, self.num_partitions - 1)
        if event.src not in self.neighbor_map:
            self.neighbor_map[event.src] = set()
        if event.dest not in self.neighbor_map:
            self.neighbor_map[event.dest] = set()
        self.neighbor_map[event.src].add(event.dest)
        self.neighbor_map[event.dest].add(event.src)
        if random() < self.p:
            random_partition = randint(0, self.num_partitions - 1)
            self.partition_map[event.src] = random_partition
            self.partition_map[event.dest] = random_partition
            return 
        self.partition_map[event.src] = get_most_common_partition(event.src, self.partition_map, self.neighbor_map)
        self.partition_map[event.dest] = get_most_common_partition(event.dest, self.partition_map, self.neighbor_map)
        if self.partition_map[event.src] == self.partition_map[event.dest]:
            self.subgraphs[self.partition_map[event.src]].append(event)

    def get_partition(self, vertex) -> int:
        if vertex not in self.partition_map:
            return 0 
        return self.partition_map[vertex]
    def get_partition_num(self) -> int:
        return self.num_partitions

    def get_subgraph(self, partition_id: int) -> EventSequence:
        if partition_id < 0 or partition_id >= self.num_partitions:
            raise ValueError("Invalid partition ID")
        return FromIterable(self.subgraphs[partition_id])
    def __str__(self):
        return f"LabelPropagationTemporalGraphPartitioner(num_partitions={self.num_partitions})"
    def __repr__(self):
        return f"LabelPropagationTemporalGraphPartitioner(num_partitions={self.num_partitions})"

def edge_cut_matrix(tg: EventSequence, partitioner: TemporalGraphPartitioner):
    """
    Computes the edge cut matrix of a temporal graph.

    :param tg: Temporal graph
    :param partitioner: Partitioner
    :return: Edge cut matrix
    """
    matrix = np.zeros((partitioner.get_partition_num(), partitioner.get_partition_num()), dtype=int)
    neighbor_map = dict()
    old_partition = dict()
    for event in tg:
        if event.src not in old_partition:
            old_partition[event.src] = -1 
            neighbor_map[event.src] = set()
        if event.dest not in old_partition:
            old_partition[event.dest] = -1
            neighbor_map[event.dest] = set()
        partitioner.partition(event)
        if old_partition[event.src] != partitioner.get_partition(event.src):
            # source vertex changed its partition, update the matrix for its neighbors 
            for neighbor in neighbor_map[event.src]:
                if old_partition[neighbor] == old_partition[event.src]:
                    matrix[old_partition[event.src]][old_partition[neighbor]] -= 1
                matrix[partitioner.get_partition(event.src)][partitioner.get_partition(neighbor)] += 1
                if partitioner.get_partition(event.src) != partitioner.get_partition(neighbor):
                    matrix[partitioner.get_partition(neighbor)][partitioner.get_partition(event.src)] += 1
        if old_partition[event.dest] != partitioner.get_partition(event.dest):
            # destination vertex changed its partition, update the matrix for its neighbors
            for neighbor in neighbor_map[event.dest]:
                if old_partition[neighbor] == old_partition[event.dest]:
                    matrix[old_partition[event.dest]][old_partition[neighbor]] -= 1
                matrix[partitioner.get_partition(event.dest)][partitioner.get_partition(neighbor)] += 1
                if partitioner.get_partition(event.dest) != partitioner.get_partition(neighbor):
                    matrix[partitioner.get_partition(neighbor)][partitioner.get_partition(event.dest)] += 1
        old_partition[event.src] = partitioner.get_partition(event.src)
        old_partition[event.dest] = partitioner.get_partition(event.dest)
        matrix[old_partition[event.src]][old_partition[event.dest]] += 1
        if old_partition[event.src] != old_partition[event.dest]:
            matrix[old_partition[event.dest]][old_partition[event.src]] += 1
        # update the neighbor map
        neighbor_map[event.src].add(event.dest)
        neighbor_map[event.dest].add(event.src)
        
        yield matrix

def partition_sizes(tg: EventSequence, partitioner: TemporalGraphPartitioner):
    """
    Computes the partition sizes of a temporal graph.

    :param tg: Temporal graph
    :param partitioner: Partitioner
    :return: Partition sizes
    """
    partition_sizes = dict()
    vertices = set()
    for event in tg:
        partitioner.partition(event)
        vertices.add(event.src)
        vertices.add(event.dest)
        for vertex in vertices:
            partition = partitioner.get_partition(vertex)
            if partition not in partition_sizes:
                partition_sizes[partition] = 0
            partition_sizes[partition] += 1
        yield partition_sizes


class DummyPartitioner(TemporalGraphPartitioner):
    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions

    def partition(self, event: Event):
        pass 

    def get_partition(self, vertex):
        return vertex % self.num_partitions

    def get_partition_num(self):
        return self.num_partitions

    def get_subgraph(self, partition_id: int) -> EventSequence:
        return FromIterable([])

class WindowedLabelPropagationTemporalGraphPartitioner(TemporalGraphPartitioner):
    def __init__(self, num_partitions: int, window_size: int):
        """
        Initializes the WindowedLabelPropagationTemporalGraphPartitioner.
        :param num_partitions: Number of partitions
        :param window_size: Size of the window
        """
        self.num_partitions = num_partitions
        self.window_size = window_size
        self.partition_map = dict()
        self.neighbor_map = dict()
        self.window = [] 
        self.subgraphs = [[] for _ in range(num_partitions)]

    def partition_vertex(self, vertex):
        if vertex not in self.partition_map:
            if vertex in self.window and len(self.window) >= self.window_size:
                self.window.remove(vertex)
                self.partition_map[vertex] = get_most_common_partition(vertex, self.partition_map, self.neighbor_map)
            elif vertex not in self.window and len(self.window) < self.window_size:
                self.window.append(vertex)
                self.partition_map[vertex] = randint(0, self.num_partitions - 1)
            elif vertex not in self.window and len(self.window) >= self.window_size:
                # remove random element in window and add new element
                random_index = randint(0, len(self.window) - 1)
                self.partition_map[self.window[random_index]] = get_most_common_partition(self.window[random_index], self.partition_map, self.neighbor_map)
                self.window[random_index] = vertex
                self.partition_map[vertex] = randint(0, self.num_partitions - 1)
            else:
                # no need to update anything 
                pass 
        else:
            # vertex is already in the partition map
            if vertex in self.window and len(self.window) >= self.window_size:
                # vertex is in the window and the window is full
                self.window.remove(vertex)
                self.partition_map[vertex] = get_most_common_partition(vertex, self.partition_map, self.neighbor_map)

    
    def partition(self, event: Event):
        self.partition_vertex(event.src)
        self.partition_vertex(event.dest)
        if event.src not in self.neighbor_map:
            self.neighbor_map[event.src] = set()
        if event.dest not in self.neighbor_map:
            self.neighbor_map[event.dest] = set()
        self.neighbor_map[event.src].add(event.dest)
        self.neighbor_map[event.dest].add(event.src)
        if event.src not in self.partition_map:
            self.partition_map[event.src] = randint(0, self.num_partitions - 1)
        if event.dest not in self.partition_map:
            self.partition_map[event.dest] = randint(0, self.num_partitions - 1)
        if self.partition_map[event.src] == self.partition_map[event.dest]:
            self.subgraphs[self.partition_map[event.src]].append(event)


    def get_partition(self, vertex):
        if vertex not in self.partition_map:
            return 0
        return self.partition_map[vertex] 

    def get_partition_num(self):
        return self.num_partitions
    
    def get_subgraph(self, partition_id: int) -> EventSequence:
        if partition_id < 0 or partition_id >= self.num_partitions:
            raise ValueError("Invalid partition ID")
        return FromIterable(self.subgraphs[partition_id])
    
class CommonNeighborsPartitioner(TemporalGraphPartitioner):
    def __init__(self, num_partitions: int, threshold: int):
        """
        Initializes the WindowedLabelPropagationTemporalGraphPartitioner.
        :param num_partitions: Number of partitions
        :param threshold: Threshold for common neighbors
        """
        self.num_partitions = num_partitions
        self.threshold = threshold
        self.partition_map = dict()
        self.neighbor_map = dict()
        self.subgraphs = [[] for _ in range(num_partitions)]

    def partition_vertex(self, vertex):
        if vertex not in self.partition_map:
            self.partition_map[vertex] = randint(0, self.num_partitions - 1)
        else:
            if len(self.neighbor_map[vertex]) > self.threshold:
                # vertex has more than threshold neighbors, reassign to most common partition
                self.partition_map[vertex] = get_most_common_partition(vertex, self.partition_map, self.neighbor_map)
    
    def partition(self, event: Event):
        if event.src not in self.neighbor_map:
            self.neighbor_map[event.src] = set()
        if event.dest not in self.neighbor_map:
            self.neighbor_map[event.dest] = set()
        self.neighbor_map[event.src].add(event.dest)
        self.neighbor_map[event.dest].add(event.src)
        self.partition_vertex(event.src)
        self.partition_vertex(event.dest)
        if self.partition_map[event.src] == self.partition_map[event.dest]:
            self.subgraphs[self.partition_map[event.src]].append(event)
        

    def get_partition(self, vertex):
        if vertex not in self.partition_map:
            return 0
        return self.partition_map[vertex] 

    def get_partition_num(self):
        return self.num_partitions
    
    def get_subgraph(self, partition_id: int) -> EventSequence:
        if partition_id < 0 or partition_id >= self.num_partitions:
            raise ValueError("Invalid partition ID")
        return FromIterable(self.subgraphs[partition_id])
