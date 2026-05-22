
import unittest

from vertex_voyage.temporal import Event, FromIterable
from vertex_voyage.temporal_partitioning import (
    InMemoryPartition,
    Partition,
    PartitionerProfile, 
    RandomPartitioner,
    CatchAllPartitioner,
    MostCommonNeighborPartitioner
)

class TestRandomPartitioner(unittest.TestCase):
    def test_random_partitioning(self):
        # Create some partitions
        partitions: set[Partition] = {InMemoryPartition.empty() for _ in range(3)}
        # Define a simple distribution function
        partitioner = RandomPartitioner.uniform(partitions)
        # Sample partitions for a set of vertices
        vertices = ['vertex1', 'vertex2', 'vertex3']
        # Create EventStream 
        events = FromIterable(iter([
            Event(src=x, dest=y, timestamp=0) for x,y in zip(vertices, vertices[1:])
        ]))
        partitioner.push(events)
        # Check that each vertex is assigned to a partition
        for vertex in vertices:
            assigned_partition = partitioner.get(vertex)
            self.assertTrue(assigned_partition)
        
        # Check that the assigned partitions are from the original set
        for vertex in vertices:
            assigned_partitions = partitioner.get(vertex)
            for partition in assigned_partitions:
                self.assertIn(partition, partitions)

class TestCatchAllPartitioner(unittest.TestCase):
    def test_catch_all_partitioning(self):
        # Create a single partition
        partitions: set[Partition] = {InMemoryPartition.empty() for _ in range(10)}
        # Create the catch-all partitioner
        partitioner = CatchAllPartitioner(partitions)
        # Sample partitions for a set of vertices
        vertices = ['vertex1', 'vertex2', 'vertex3']
        # Create EventStream 
        events = FromIterable(iter([
            Event(src=x, dest=y, timestamp=0) for x,y in zip(vertices, vertices[1:])
        ]))
        partitioner.push(events)
        # Check that each vertex is assigned to the same partition
        for vertex in vertices:
            assigned_partition = partitioner.get(vertex)
            self.assertEqual(assigned_partition, partitions)

class TestTemporalPartitioning(unittest.TestCase):
    def test_sum(self):
        # Create some partitions
        partitions: set[Partition] = {InMemoryPartition.empty() for _ in range(3)}
        # Define a simple distribution function
        partitioner = RandomPartitioner.uniform(partitions)
        vertices = ['vertex1', 'vertex2', 'vertex3']
        # Create EventStream
        events = FromIterable(iter([
            Event(src=x, dest=y, timestamp=0) for x,y in zip(vertices, vertices[1:])
        ]))
        partitioner.push(events)
        everything = [partitioner.get(vertex).pop() for vertex in vertices]
        s = sum((p for p in everything), InMemoryPartition.empty())
        self.assertEqual(s.size(), len(vertices))

    def test_diff(self):
        # Create some partitions
        partitions: set[Partition] = {InMemoryPartition.empty() for _ in range(3)}
        # Define a simple distribution function
        partitioner = RandomPartitioner.uniform(partitions)
        vertices = ['vertex1', 'vertex2', 'vertex3']
        # Create EventStream
        events = FromIterable(iter([
            Event(src=x, dest=y, timestamp=0) for x,y in zip(vertices, vertices[1:])
        ]))
        partitioner.push(events)
        everything = [partitioner.get(vertex).pop() for vertex in vertices]
        s = sum((p for p in everything), InMemoryPartition.empty())
        self.assertEqual((s - s).size(), 0)
    
    def test_partitioning_profile(self):
        # Create some partitions
        partitions: set[Partition] = {InMemoryPartition.empty() for _ in range(3)}
        # Define a simple distribution function
        partitioner = RandomPartitioner.uniform(partitions)
        profile = PartitionerProfile(partitioner)
        events = FromIterable(iter([
            Event(src=f"vertex{i}", dest=f"vertex{i+1}", timestamp=0) for i in range(10)
        ]))
        profile.push(events)
        profile.print_profile()
        self.assertEqual(profile.num_events, 10)
        self.assertEqual(profile.num_buffers, 1)
        self.assertEqual(len(profile.unique_vertices), 11)
        self.assertGreater(profile.total_time, 0)
        self.assertEqual(len(profile.partition_sizes), 1)
        self.assertEqual(len(profile.edge_cuts), 1)
        # Assert that sum of partition sizes equals number of unique vertices
        self.assertEqual(sum(profile.partition_sizes[0].values()), len(profile.unique_vertices))

class TestMostCommonPartitioner(unittest.TestCase):
    def test_most_common_partitioning(self):
        # Create some partitions
        partitions: set[Partition] = {InMemoryPartition.empty() for _ in range(3)}
        # Define a simple distribution function
        partitioner = MostCommonNeighborPartitioner.all_neighbors(partitions)
        vertices = ['vertex1', 'vertex2', 'vertex3']
        # Create EventStream
        events = FromIterable(iter([
            Event(src=x, dest=y, timestamp=0) for x,y in zip(vertices, vertices[1:])
        ]))
        partitioner.push(events)
        # Check that each vertex is assigned to a partition
        for vertex in vertices:
            assigned_partition = partitioner.get(vertex)
            self.assertTrue(assigned_partition)
        
        # Check that the assigned partitions are from the original set
        for vertex in vertices:
            assigned_partitions = partitioner.get(vertex)
            for partition in assigned_partitions:
                self.assertIn(partition, partitions)