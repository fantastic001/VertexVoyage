
from experiments.utils import is_full_benchmark
from vertex_voyage.benchmark_base import Benchmark
from vertex_voyage.temporal_partitioning import (
    LabelPropagationTemporalGraphPartitioner, 
    partition_temporal_graph,
    edge_cut_matrix
)
from vertex_voyage.temporal import SBMSequence, FirstN, ShuffledSequence
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import numpy as np 
from vertex_voyage.temporal import FileEventSequence

class TemporalLabelPropagationBasic(Benchmark):
    """
    A basic benchmark for the Label Propagation algorithm on temporal graphs.
    """

    NAME = "Temporal Label Propagation Partitioning Benchmark"

    def run(self, results_path):
        data = [] 
        repetitions = 30
        partition_candidates = [2, 4, 8, 16]
        for iteration in range(repetitions):
            for part_num in partition_candidates:
                for random_assignment_prob in np.arange(0, 1.1, 0.1):
                    graph = FirstN(ShuffledSequence(SBMSequence([.5, .5], [[.7, .3], [.3, .7]]), 100), 1000)
                    partitioner = LabelPropagationTemporalGraphPartitioner(part_num, random_assignment_prob)
                    partitions = partition_temporal_graph(graph, partitioner)
                    vertices = set()
                    for partition in partitions:
                        for vertex in partition:
                            vertices.add(vertex)
                    for i, partition in enumerate(partitions):
                        data.append({
                            "partition": i,
                            "part_num": part_num,
                            "size": len(partition),
                            "graph_size": len(vertices),
                            "expected_part_size": len(vertices) / part_num,
                            "random_assignment_prob": random_assignment_prob,
                        })
            self.report_progress(iteration+1, repetitions)
    
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_path, "results.csv"), index=False)
    
    def display(self, results_path):
        try:
            df = pd.read_csv(os.path.join(results_path, "results.csv"))
        except FileNotFoundError:
            self.run(results_path)
            return self.display(results_path)
        df = df[df["random_assignment_prob"] == 0.5]
        # plot average partition size / total graph size depending of the number of partitions 
        fig, ax = plt.subplots()
        balance = df["size"] / df["expected_part_size"]
        df["balance"] = balance
        df_grouped = df.groupby("part_num").mean().reset_index()
        df_grouped.plot.bar(x="part_num", y="balance", ax=ax)
        ax.set_xlabel("Number of Partitions")
        ax.set_ylabel("Average Partition Size")
        ax.set_title("Average Partition Size depending of the Number of Partitions")
        plt.show()
        df = pd.read_csv(os.path.join(results_path, "results.csv"))
        balance = df["size"] / df["expected_part_size"]
        df["balance"] = balance
        for part_num in df["part_num"].unique():
            df_part = df[df["part_num"] == part_num]
            x = [] 
            y = []
            for p in df_part["random_assignment_prob"].unique():
                x.append(p)
                y.append(df_part[df_part["random_assignment_prob"] == p]["balance"].mean())
            plt.plot(x, y, label=f"Number of Partitions: {part_num}")
        plt.legend()
        plt.xlabel("Random Assignment Probability")
        plt.ylabel("Average Partition Balance")
        plt.title("Average Partition Balance depending of the Random Assignment Probability")
        plt.show()



class TemporalLabelPropagationEdgeCut(Benchmark):
    """
    A benchmark for the Label Propagation algorithm on temporal graphs.
    It shows how edge cut progresses on SBM graph with shuffled events over time.
    """

    NAME = "Temporal Label Propagation Partitioning Benchmark - Edge Cut"

    def run(self, results_path):
        data = [] 
        precision = 5
        g = FirstN(ShuffledSequence(SBMSequence([.5, .5], [[.7, .3], [.3, .7]]), 100), 100000)
        events = list(g)
        for i, p in enumerate(np.linspace(0, 1, precision)):
            partitioner = LabelPropagationTemporalGraphPartitioner(16, p)
            for t, matrix in enumerate(edge_cut_matrix(events, partitioner)):
                same_partition = sum(matrix[j][j] for j in range(matrix.shape[0]))
                edge_cut = (sum(sum(row) for row in matrix) - same_partition) / 2
                total_edges = same_partition + edge_cut
                if total_edges == 0:
                    edge_cut = 0
                else:
                    edge_cut = edge_cut / total_edges
                data.append({
                    "Random Assignment Probability": p,
                    "Edge Cut": edge_cut,
                    "Partition number": matrix.shape[0],
                    "Time": t,
                })
            self.report_progress(i+1, precision)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_path, "results.csv"), index=False)
    
    def display(self, results_path):
        try:
            df = pd.read_csv(os.path.join(results_path, "results.csv"))
            for p in df["Random Assignment Probability"].unique():
                df_part = df[df["Random Assignment Probability"] == p]
                plt.plot(df_part["Time"], df_part["Edge Cut"], label=f"Random Assignment Probability: {p}")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Edge Cut")
            plt.title("Edge Cut over time")
            plt.show()
        except FileNotFoundError:
            self.run(results_path)
            return self.display(results_path)
        
class TemporalTwitchLabelPropagationBenchmark(Benchmark):
    """
    Twitch Label Propagation Benchmark
    """

    NAME = "Temporal Twitch Label Propagation Benchmark"

    def run(self, results_path):
        data = [] 
        if not is_full_benchmark():
            g = list(FirstN(FileEventSequence("data/twitch.txt"), 10000))
        else:
            g = list(FileEventSequence("data/twitch.txt"))
        partitioner = LabelPropagationTemporalGraphPartitioner(16, 0.5)
        for i, matrix in enumerate(edge_cut_matrix(g, partitioner)):
            same_partition = sum(matrix[j][j] for j in range(matrix.shape[0]))
            edge_cut = (sum(sum(row) for row in matrix) - same_partition) / 2
            total_edges = same_partition + edge_cut
            if total_edges == 0:
                edge_cut = 0
            else:
                edge_cut = edge_cut / total_edges
            data.append({
                "Edge Cut": edge_cut,
                "Partition number": matrix.shape[0],
                "Time": i,
            })
            self.report_progress(i+1, len(g))
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_path, "results.csv"), index=False)

    def display(self, results_path):
        try:
            df = pd.read_csv(os.path.join(results_path, "results.csv"))
            plt.plot(df["Time"], df["Edge Cut"])
            plt.xlabel("Time")
            plt.ylabel("Edge Cut")
            plt.title("Edge Cut over time")
            plt.show()
        except FileNotFoundError:
            self.run(results_path)
            return self.display(results_path)

class DifferentShuffleParams(Benchmark):
    """
    A benchmark for the Label Propagation algorithm on temporal graphs.
    It shows how edge cut progresses on SBM graph with shuffled events over time.
    """

    NAME = "Temporal Label Propagation Partitioning Benchmark - Different Shuffle Parameters"

    def run(self, results_path):
        data = [] 
        max_window_size = 1000
        for window_size in range(1,max_window_size+1):
            g = FirstN(ShuffledSequence(SBMSequence([.5, .5], [[.7, .3], [.3, .7]]), window_size), 10*window_size)
            partitioner = LabelPropagationTemporalGraphPartitioner(16, 0)
            for t, matrix in enumerate(edge_cut_matrix(g, partitioner)):
                same_partition = sum(matrix[j][j] for j in range(matrix.shape[0]))
                edge_cut = (sum(sum(row) for row in matrix) - same_partition) / 2
                total_edges = same_partition + edge_cut
                if total_edges == 0:
                    edge_cut = 0
                else:
                    edge_cut = edge_cut / total_edges
                data.append({
                    "Edge Cut": edge_cut,
                    "Partition number": matrix.shape[0],
                    "Time": t,
                    "Window Size": window_size,
                })
            self.report_progress(window_size, max_window_size)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_path, "results.csv"), index=False)

    def display(self, results_path):
        try:
            df = pd.read_csv(os.path.join(results_path, "results.csv"))
            # plot average edge cut depending of the window size 
            df.groupby("Window Size").mean().reset_index().plot(x="Window Size", y="Edge Cut")
            plt.xlabel("Window Size")
            plt.ylabel("Average Edge Cut")
            plt.title("Average Edge Cut depending of the Window Size")
            plt.show()
            
        except FileNotFoundError:
            self.run(results_path)
            return self.display(results_path)

class DifferentSBMParams(Benchmark):
    """
    A benchmark for the Label Propagation algorithm on temporal graphs.
    It shows how edge cut progresses based on different SBM parameters.
    """

    NAME = "Temporal Label Propagation Partitioning Benchmark - Different SBM Parameters"

    def run(self, results_path):
        data = [] 
        community_num = [2, 4, 8]
        same_community_prob = [0.7, 0.8, 0.9]
        for ci, communitties in enumerate(community_num):
            for prob in same_community_prob:
                comm_prob = [1 / communitties] * communitties
                connection_prob = [[prob if i == j else (1 - prob) / (communitties - 1) for j in range(communitties)] for i in range(communitties)]
                g = FirstN(ShuffledSequence(SBMSequence(comm_prob, connection_prob), 100), 1000)
                partitioner = LabelPropagationTemporalGraphPartitioner(16, 0)
                for t, matrix in enumerate(edge_cut_matrix(g, partitioner)):
                    same_partition = sum(matrix[j][j] for j in range(matrix.shape[0]))
                    edge_cut = (sum(sum(row) for row in matrix) - same_partition) / 2
                    total_edges = same_partition + edge_cut
                    if total_edges == 0:
                        edge_cut = 0
                    else:
                        edge_cut = edge_cut / total_edges
                    data.append({
                        "Edge Cut": edge_cut,
                        "Partition number": matrix.shape[0],
                        "Time": t,
                        "Community Number": communitties,
                        "Same Community Probability": prob,
                    })
            self.report_progress(ci+1, len(community_num))
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_path, "results.csv"), index=False)
    def display(self, results_path):
        try:
            df = pd.read_csv(os.path.join(results_path, "results.csv"))
            for comm_num in df["Community Number"].unique():
                df_part = df[df["Community Number"] == comm_num]
                df_part.groupby("Same Community Probability").mean().reset_index().plot.bar(x="Same Community Probability", y="Edge Cut", label=f"Community Number: {comm_num}")
            
            plt.title("Average Edge Cut depending of the Same Community Probability")
            plt.legend()
            plt.show()
        except FileNotFoundError:
            self.run(results_path)
            return self.display(results_path)