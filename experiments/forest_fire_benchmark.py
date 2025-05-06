
from vertex_voyage.temporal import to_nx_graph, ForestFireEventSequence, FirstN, animate_graph
from vertex_voyage.benchmark_base import Benchmark
from vertex_voyage.temporal_partitioning import (
    LabelPropagationTemporalGraphPartitioner,
    edge_cut_matrix,
    partition_sizes
)
from vertex_voyage.partitioning import get_partition_average_balance
import matplotlib.pyplot as plt
import pandas as pd 
import os 

class ForestFireBenchmark(Benchmark):
    """
    Forest Fire Benchmark for evaluating temporal graph algorithms.
    """
    NAME = "Forest Fire Degree Distribution"

    def run(self, results_path):
        """
        Run the Forest Fire benchmark and save the results.
        """
        # Generate a forest fire event sequence
        event_sequence = FirstN(ForestFireEventSequence(.1), 1000)
        events = list(event_sequence)
        g = to_nx_graph(events)
        # Compute the degree distribution
        degree_sequence = [d for n, d in g.degree()]
        data = [] 
        for d in degree_sequence:
            data.append((d, 1))
        df = pd.DataFrame(data, columns=["degree", "count"])
        df = df.groupby("degree").sum().reset_index()
        df["degree"] = df["degree"].astype(int)
        df.to_csv(os.path.join(results_path, "forest_fire_degree_distribution.csv"), index=False)
    
    def display(self, results_path):
        """
        Display the results of the Forest Fire benchmark.
        """
        try:
            # Load the degree distribution data
            df = pd.read_csv(os.path.join(results_path, "forest_fire_degree_distribution.csv"))
            # Plot the degree distribution
            plt.figure(figsize=(10, 6))
            plt.bar(df["degree"], df["count"], width=0.8, color='blue')
            plt.xlabel("Degree")
            plt.ylabel("Count")
            plt.title("Forest Fire Degree Distribution")
            plt.grid(True)
            plt.show()

            
        except FileNotFoundError:
            self.run(results_path)
            self.display(results_path)

class LabelPropagationBenchmarkWithForestFire(Benchmark):
    """
    Label Propagation Benchmark for evaluating temporal graph algorithms.
    """
    NAME = "Label Propagation with Forest Fire"

    def run(self, results_path):
        """
        Run the Label Propagation benchmark and save the results.
        """
        # Generate a forest fire event sequence
        partition_num = 16
        random_assign = 0.5
        event_sequence = FirstN(ForestFireEventSequence(.1), 1000)
        events = list(event_sequence)
        data = [] 
        partitioner = LabelPropagationTemporalGraphPartitioner(partition_num, random_assign)
        for iteration in range(30):
            for t, matrix in enumerate(edge_cut_matrix(events, partitioner)):
                same_partition = sum(matrix[j][j] for j in range(matrix.shape[0]))
                edge_cut = (sum(sum(row) for row in matrix) - same_partition) / 2
                total_edges = same_partition + edge_cut
                if total_edges == 0:
                    edge_cut = 0
                else:
                    edge_cut = edge_cut / total_edges
                data.append({
                    "edge_cut": edge_cut,
                    "total_edges": total_edges,
                    "time": t,
                })
            partitioner = LabelPropagationTemporalGraphPartitioner(partition_num, random_assign)
            for i, partition_sizes_ in enumerate(partition_sizes(events, partitioner)):
                data[i]["balance"] = get_partition_average_balance(partition_sizes_, partition_num)
            self.report_progress(iteration, 30)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_path, "label_propagation_forest_fire.csv"), index=False)

    def display(self, results_path):
        """
        Display the results of the Label Propagation benchmark.
        """
        try:
            # Load the label propagation data
            df = pd.read_csv(os.path.join(results_path, "label_propagation_forest_fire.csv"))
            # Plot the edge cut over time
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

        except FileNotFoundError:
            self.run(results_path)
            self.display(results_path)
