
from vertex_voyage.benchmark_base import Benchmark
from vertex_voyage.benchmark import data_profile,  display_data_profile
from vertex_voyage.partitioning import modified__lfm
import networkx as nx 
import pandas as pd 
import numpy as np
class LFMDataProfile(Benchmark):
    """
    This class performs a data profile on the LFM problem.
    It uses the Wild and More data profile method.
    """

    NAME = "LFM Data Profile"

    def run(self,results_path):
        def lfm_partitioner(graph):
            return modified__lfm(graph, 5, pm_k=1000)
        problems = []
        for l in range(10, 100, 20):
            for k in range(2, 10, 2):
                for p in np.arange(0.1, 1.1, 0.2):
                    for q in np.arange(.1, .5, 0.1):
                        problems.append(lambda: nx.planted_partition_graph(l, k, p, p*q))
        df = data_profile({
            "LFM": lfm_partitioner
        }, "vertex_voyage.partitioning.remove_vertex_from_community", problems, self)
        df.to_csv(f"{results_path}/results.csv", index=False)

    def display(self, results_path):
        df = pd.read_csv(f"{results_path}/results.csv")
        display_data_profile(df)