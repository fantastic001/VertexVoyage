
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

class WindowedLPSBM(Benchmark):
    
    NAME = "Windowed Label Propagation with SBM over time"

    def run(self, results_folder):
        data = [] 
        sbm_matrix = [] 
        part_num = 16
        p = 0.5
        community_count = 5 
        window_size = 5000
        iterations = 30
        for i in range(community_count):
            sbm_matrix.append([p if i == j else (1 - p) / (community_count - 1) for j in range(community_count)])
        for k in range(iterations):
            g = list(FirstN(ShuffledSequence(SBMSequence([1 / community_count] * community_count, sbm_matrix), 1000), 10000))
            partitioner = WindowedLabelPropagationTemporalGraphPartitioner(part_num, window_size)
            total_edges = len(g)
            for i, matrix in enumerate(edge_cut_matrix(g, partitioner)):
                edge_cut = sum(matrix[a,b] for a in range(part_num) for b in range(part_num) if a > b)
                edge_cut = edge_cut / sum(matrix[a,b] for a in range(part_num) for b in range(part_num) if a >= b)
                data.append({
                    "time": i,
                    "edge_cut": edge_cut,
                    "edge_cut_ratio": edge_cut,
                })
            partitioner = WindowedLabelPropagationTemporalGraphPartitioner(part_num, window_size)
            for i, partition_sizes_ in enumerate(partition_sizes(g, partitioner)):
                data[i]["balance"] = get_partition_average_balance(partition_sizes_, part_num)
            self.report_progress(k+1, iterations)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_folder, "windowed_label_propagation.csv"), index=False)
    
    def display(self, results_folder):
        df = pd.read_csv(os.path.join(results_folder, "windowed_label_propagation.csv"))
        plt.figure(figsize=(10, 6))
        x = df["time"].unique()
        y = df.groupby("time")["edge_cut"].mean()
        z = df.groupby("time")["balance"].mean()
        plt.plot(x, y, label="Edge Cut", color='blue')
        plt.plot(x, z, label="Balance", color='orange')
        plt.xlabel("Time")
        plt.ylabel("Edge Cut")
        plt.title("Label Propagation Edge Cut Over Time")
        plt.grid(True)
        plt.legend()
        plt.show()