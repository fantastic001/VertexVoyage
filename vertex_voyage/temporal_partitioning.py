
from vertex_voyage.temporal import Event, EventSequence
from random import randint

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
    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions
        self.partition_map = dict()
        self.neighbor_map = dict()

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
    