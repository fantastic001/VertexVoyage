
from vertex_voyage.temporal import Event, EventSequence
from random import randint, random
import numpy as np 
class TemporalGraphPartitioner:
    def partition(self, event: Event):
        """
        Partitions an event into a partition number
        """
        pass

    def get_partition(self, vertex):
        """
        Returns the partition of a vertex
        """
        pass
    def get_partition_num(self):
        """
        Returns the number of partitions
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
            self.partition_map[event.src] = randint(0, self.num_partitions - 1)
            self.partition_map[event.dest] = randint(0, self.num_partitions - 1)
            return 
        freq = dict()
        for neighbor in self.neighbor_map[event.src]:
            if neighbor in self.partition_map:
                partition = self.partition_map[neighbor]
                if partition not in freq:
                    freq[partition] = 0
                freq[partition] += 1
        self.partition_map[event.src] = max(freq, key=freq.get) if freq else self.partition_map[event.src]
        freq = dict()
        for neighbor in self.neighbor_map[event.dest]:
            if neighbor in self.partition_map:
                partition = self.partition_map[neighbor]
                if partition not in freq:
                    freq[partition] = 0
                freq[partition] += 1
        self.partition_map[event.dest] = max(freq, key=freq.get) if freq else self.partition_map[event.dest]
    def get_partition(self, vertex):
        if vertex not in self.partition_map:
            return 0 
        return self.partition_map[vertex]
    def get_partition_num(self):
        return self.num_partitions
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

class DummyPartitioner(TemporalGraphPartitioner):
    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions

    def partition(self, event: Event):
        pass 

    def get_partition(self, vertex):
        return vertex % self.num_partitions

    def get_partition_num(self):
        return self.num_partitions


