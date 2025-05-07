
from vertex_voyage.temporal import to_nx_graph, SBMSequence, FirstN, ShuffledSequence
from vertex_voyage.benchmark_base import Benchmark
from vertex_voyage.temporal_partitioning import (
    LabelPropagationTemporalGraphPartitioner,
    edge_cut_matrix,
    partition_sizes,
    WindowedLabelPropagationTemporalGraphPartitioner,
    CommonNeighborsPartitioner
)
from vertex_voyage.partitioning import get_partition_average_balance
import matplotlib.pyplot as plt
import pandas as pd 
import os 
from experiments.utils import * 

class WindowedLPSBM(Benchmark):
    
    NAME = "Windowed Label Propagation with SBM over time"

    def run(self, results_folder):
        window_size = 1000
        df = benchmark_partitioner(
            self, 
            16, 
            WindowedLabelPropagationTemporalGraphPartitioner,
            window_size
        )
        df.to_csv(os.path.join(results_folder, "windowed_label_propagation.csv"), index=False)
    
    def display(self, results_folder):
        df = pd.read_csv(os.path.join(results_folder, "windowed_label_propagation.csv"))
        display_benchmark_results(df, "time", ["edge_cut", "balance"])


class WindowedLPSBM(Benchmark):
    
    NAME = "Windowed Label Propagation with SBM - window size"

    def run(self, results_folder):
        samples = range(100, 10000, 100)
        data = []
        for i, window_size in enumerate(samples):
            df_part = benchmark_partitioner(
                None, 
                16, 
                WindowedLabelPropagationTemporalGraphPartitioner,
                window_size,
                graph_generator = lambda: FirstN(ShuffledSequence(SBMSequence([1 / 5] * 5, [[0.5 if i == j else 0.1 for j in range(5)] for i in range(5)]), 100), 1000)
            )
            data.append({
                "window_size": window_size,
                "edge_cut": df_part[df_part.time == 900]["edge_cut"].mean(),
                "balance": df_part[df_part.time == 900]["balance"].mean()
            })
            self.report_progress(i+1, len(samples))
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_folder, "windowed_label_propagation.csv"), index=False)
    
    def display(self, results_folder):
        df = pd.read_csv(os.path.join(results_folder, "windowed_label_propagation.csv"))
        display_benchmark_results(df, "window_size", ["edge_cut", "balance"])

class CommonNeighborsBenchmark(Benchmark):
    
    NAME = "CN Partitioner SBM over time"

    def run(self, results_folder):
        threshold = 5
        df = benchmark_partitioner(
            self, 
            16, 
            CommonNeighborsPartitioner,
            threshold
        )
        df.to_csv(os.path.join(results_folder, "cn.csv"), index=False)
    
    def display(self, results_folder):
        df = pd.read_csv(os.path.join(results_folder, "cn.csv"))
        display_benchmark_results(df, "time", ["edge_cut", "balance"])

