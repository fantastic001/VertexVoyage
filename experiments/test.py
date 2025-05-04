
from vertex_voyage.benchmark import data_profile, CallCountingMock, SimplePatch
import numpy as np 
from vertex_voyage.benchmark_base import Benchmark

class SampleBenchmark(Benchmark):
    """
    A sample benchmark class for demonstration purposes.
    """
    NAME = "SampleBenchmark"

    def run(self, results_path):
        from vertex_voyage.temporal_partitioning import DummyPartitioner, edge_cut_matrix
        from vertex_voyage.temporal import FirstN, SBMSequence
        g = FirstN(SBMSequence([.5, .5], [[.5, .5], [.5, .5]]), 100)
        partitioner = DummyPartitioner(3)
        for matrix in edge_cut_matrix(g, partitioner):
            print(matrix)

    def display(self, results_path):
        # used for testing anyway, always call benchmark run
        self.run(results_path)
