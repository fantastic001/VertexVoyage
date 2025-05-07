
from vertex_voyage.temporal import to_nx_graph, SBMSequence, FirstN, ShuffledSequence
from vertex_voyage.benchmark_base import Benchmark
from vertex_voyage.temporal_partitioning import (
    LabelPropagationTemporalGraphPartitioner,
    edge_cut_matrix,
    partition_sizes,
    WindowedLabelPropagationTemporalGraphPartitioner
)
from vertex_voyage.partitioning import get_partition_average_balance
import matplotlib.pyplot as plt
import pandas as pd 
import os 

def benchmark_partitioner(benchmark, part_num, partitioner_class,  *args, graph_generator = None):
    data = [] 
    sbm_matrix = [] 
    p = 0.5
    community_count = 5 
    iterations = 50
    for i in range(community_count):
        sbm_matrix.append([p if i == j else (1 - p) / (community_count - 1) for j in range(community_count)])
    for k in range(iterations):
        if graph_generator:
            g = list(graph_generator())
        else:
            g = list(FirstN(ShuffledSequence(SBMSequence([1 / community_count] * community_count, sbm_matrix), 1000), 10000))
        partitioner = partitioner_class(part_num, *args)
        for i, matrix in enumerate(edge_cut_matrix(g, partitioner)):
            edge_cut = sum(matrix[a,b] for a in range(part_num) for b in range(part_num) if a > b)
            edge_cut = edge_cut / sum(matrix[a,b] for a in range(part_num) for b in range(part_num) if a >= b)
            data.append({
                "time": i,
                "edge_cut": edge_cut,
            })
        partitioner = partitioner_class(part_num, *args)
        for i, partition_sizes_ in enumerate(partition_sizes(g, partitioner)):
            data[i]["balance"] = get_partition_average_balance(partition_sizes_, part_num)
        if benchmark:
            benchmark.report_progress(k+1, iterations)
    return pd.DataFrame(data)

def display_benchmark_results(df, x_label, y_labels):
    plt.figure(figsize=(10, 6))
    x = df[x_label].unique()
    ys = [df.groupby(x_label)[y].mean() for y in y_labels]
    for y, label in zip(ys, y_labels):
        plt.plot(x, y, label=label)
    plt.xlabel(x_label)
    plt.grid(True)
    plt.legend()
    plt.show()

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