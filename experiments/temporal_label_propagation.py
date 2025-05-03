
from vertex_voyage.benchmark_base import Benchmark
from vertex_voyage.temporal_partitioning import LabelPropagationTemporalGraphPartitioner, partition_temporal_graph
from vertex_voyage.temporal import SBMSequence, FirstN, ShuffledSequence
import pandas as pd 
import os 
import matplotlib.pyplot as plt
class TemporalLabelPropagationBasic(Benchmark):
    """
    A basic benchmark for the Label Propagation algorithm on temporal graphs.
    """

    NAME = "Temporal Label Propagation Partitioning Benchmark"

    def run(self, results_path):
        data = [] 
        repetitions = 30
        partition_candidates = [2, 4, 8, 16]
        for _ in range(repetitions):
            for i, part_num in enumerate(partition_candidates):
                graph = FirstN(ShuffledSequence(SBMSequence([.5, .5], [[.7, .3], [.3, .7]]), 100), 1000)
                partitioner = LabelPropagationTemporalGraphPartitioner(part_num, .5)
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
                    })
                self.report_progress(i+1, len(partition_candidates) * repetitions)
    
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_path, "results.csv"), index=False)
    
    def display(self, results_path):
        df = pd.read_csv(os.path.join(results_path, "results.csv"))
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