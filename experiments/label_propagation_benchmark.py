
from vertex_voyage.benchmark_base import Benchmark
from vertex_voyage.partitioning import label_propagation_partitioner, evaluate_partition_sbm
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import networkx as nx 
import os 

class LabelPropagationBenchmark(Benchmark):
    """
    Label Propagation Benchmark
    """

    NAME = "Basic Label Propagation Benchmark"

    def run(self, results_path):
        graphs = [] 
        data = []
        total = 1000
        for size in range(1, total+1):
            for k in range(1,5):
                for p in np.arange(0.1, 0.5, 0.1):
                    sbm_matrix = np.zeros([k, k])
                    for i in range(k):
                        for j in range(k):
                            if i == j:
                                sbm_matrix[i][j] = 1 - p
                            else:
                                sbm_matrix[i][j] = p / (k - 1)
                    g = nx.stochastic_block_model([size] * k, sbm_matrix)
                    for partition_num in range(1, 5):
                        partitions = label_propagation_partitioner(g, partition_num)
                        intra, inter_loss = evaluate_partition_sbm(g, partitions)
                        data.append({
                            "size": size,
                            "k": k,
                            "p": p,
                            "intra_loss": intra,
                            "inter_loss": inter_loss
                        })
            self.report_progress(size, total)
        pd.DataFrame(data).to_csv(os.path.join(results_path, "label_propagation_benchmark.csv"))
    
    def display(self, results_path):
        data = pd.read_csv(os.path.join(results_path, "label_propagation_benchmark.csv"))
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for k in range(1, 5):
            subset = data[data["k"] == k]
            ax[0].scatter(subset["size"], subset["intra_loss"], label=f"k={k}")
            ax[1].scatter(subset["size"], subset["inter_loss"], label=f"k={k}")
        ax[0].set_title("Intra Loss")
        ax[1].set_title("Inter Loss")
        ax[0].legend()
        ax[1].legend()
        plt.show()