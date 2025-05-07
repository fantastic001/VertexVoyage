
from vertex_voyage.benchmark_base import Benchmark
from vertex_voyage.benchmark import data_profile,  display_data_profile, CallCountingMock
from vertex_voyage.partitioning import modified__lfm, Community
import networkx as nx 
import pandas as pd 
import numpy as np
import vertex_voyage.partitioning
import matplotlib.pyplot as plt
class LFMDataProfile(Benchmark):
    """
    This class performs a data profile on the LFM problem.
    It uses the Wild and More data profile method.
    """

    NAME = "LFM Data Profile"

    def run(self,results_path):
        def lfm_partitioner(graph):
            result =  modified__lfm(graph, 5, pm_k=1000)
            return result
        problems = []
        for l in range(10, 100, 20):
            for k in range(2, 10, 2):
                for p in np.arange(0.1, 1.1, 0.1):
                    for q in np.arange(.1, .5, 0.1):
                        problems.append(lambda: nx.planted_partition_graph(l, k, p, p*q))
        df = data_profile({
            "LFM": lfm_partitioner
        }, (vertex_voyage.partitioning, "remove_vertex_from_community"), problems, self)
        df.to_csv(f"{results_path}/results.csv", index=False)

    def display(self, results_path):
        df = pd.read_csv(f"{results_path}/results.csv")
        display_data_profile(df)

class LFMBalanceFactorBenchmark(Benchmark):
    """
    This benchmark measures the balance factor of the LFM partitioner
    """
    NAME = "LFM Balance Factor"
    
    def run(self, results_path):
        data = [] 
        ls = range(10, 40, 20)
        ks = range(2, 8, 2)
        ps = np.arange(0.1, 1.1, 0.1)
        qs = np.arange(.1, .5, 0.1)
        i = 0
        for iter in range(50):
            for l in ls:
                for k in ks:
                    for p in ps:
                        for q in qs:
                            i += 1
                            g = nx.planted_partition_graph(l, k, p, p*q)
                            partitions = modified__lfm(g, 16, pm_k=1000)
                            partitions = {i: len(p) for i, p in enumerate(partitions)}
                            balance = vertex_voyage.partitioning.get_partition_average_balance(partitions, 16)
                            data.append({
                                "l": l,
                                "k": k,
                                "p": p,
                                "q": q,
                                "balance": balance
                            })
                            self.report_progress(i, 50 * len(ls) * len(ks) * len(ps) * len(qs))
        df = pd.DataFrame(data)
        
        df.to_csv(f"{results_path}/results.csv", index=False)

    def display(self, results_path):
        try:
            df = pd.read_csv(f"{results_path}/results.csv")
            x = df["p"].unique()
            y = df.groupby("p")["balance"].mean()
            plt.plot(x, y)
            plt.xlabel("p")
            plt.ylabel("Balance Factor")
            plt.title("Balance Factor vs p")
            plt.grid(True)
            plt.show()
        except FileNotFoundError:
            self.run(results_path)
            self.display(results_path)