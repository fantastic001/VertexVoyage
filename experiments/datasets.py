
from vertex_voyage.temporal import FileEventSequence, FirstN
from experiments.utils import display_benchmark_results, benchmark_partitioner, is_full_benchmark
from vertex_voyage.benchmark_base import Benchmark
import pandas as pd 
import os 
import matplotlib.pyplot as plt 

from vertex_voyage.temporal_partitioning import (
    CommonNeighborsPartitioner,
    WindowedLabelPropagationTemporalGraphPartitioner
)

datasets = {
    "Twitch": lambda: FileEventSequence("data/twitch.txt"),
    "Wiki Talks": lambda: FileEventSequence("data/wiki-talks.txt"),
    "UK2002": lambda: FileEventSequence("data/uk2002.txt"),
    "Live Journal": lambda: FileEventSequence("data/LiveJournal.txt"),
    "SBM 10M": lambda: FileEventSequence("data/sbm_10M.txt"),
    "Cit-HepPh": lambda: FileEventSequence("data/cit_HepPh.txt"),
    "Cit-HepTh": lambda: FileEventSequence("data/cit_HepTh.txt"),
    "test": lambda: FileEventSequence("data/test.txt"),
}

def create_benchmark_class(partitioner_class, *args):
    class _Benchmark(Benchmark):
        NAME = "Benchmark for partitioner " + partitioner_class.__name__ + " on large datasets"

        def run(self, results_folder):
            N = 1000 if not is_full_benchmark() else None 
            data = [] 
            for name, generator in datasets.items():
                print("Dataset: " + name + " " * 70)
                df = benchmark_partitioner(
                    self,
                    16,
                    partitioner_class,
                    *args,
                    graph_generator=lambda: FirstN(generator(), N) if N else generator()
                )
                data.append({
                    "dataset": name,
                    "balance": df[df.time == N-1]["balance"].mean(),
                    "edge_cut": df[df.time == N-1]["edge_cut"].mean(),
                })
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(results_folder, "results.csv"))
        
        def display(self, results_folder):
            try:
                df = pd.read_csv(os.path.join(results_folder, "results.csv"))
                fig, ax = plt.subplots()
                df.plot(
                    x="dataset",
                    y=["balance", "edge_cut"],
                    kind="bar",
                    ax=ax,
                    title=self.NAME,
                    xlabel="Dataset",
                    ylabel="Value",
                )
                plt.grid(True)
                plt.show()
            except FileNotFoundError:
                self.run(results_folder)
                self.display(results_folder)
    return _Benchmark


CNBenchmark = create_benchmark_class(CommonNeighborsPartitioner, 10)
WinBenchmark = create_benchmark_class(
    WindowedLabelPropagationTemporalGraphPartitioner,
    1000
)

class DatasetBenchmark(Benchmark):
    NAME = "Dataset Benchmark"
    
    def run(self, results_folder):
        data = []
        for name, generator in datasets.items():
            print("Dataset: " + name + " " * 70)
            seq = generator()
            count = 0
            dg = {} 
            for event in seq:
                count += 1
                if event.src not in dg:
                    dg[event.src] = 0
                dg[event.src] += 1
            avg = sum(dg.values()) / len(dg) if len(dg) > 0 else 0
            var = sum((x - avg) ** 2 for x in dg.values()) / len(dg) if len(dg) > 0 else 0

            data.append({
                "dataset": name,
                "nodes": len(dg),
                "edges": count,
                "avg_degree": avg,
                "var_degree": var
            })
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(results_folder, "dataset_stats.csv"))
            
    
    def display(self, results_folder):
        try:
            df = pd.read_csv(os.path.join(results_folder, "dataset_stats.csv"))
            print(df.to_markdown())
        except FileNotFoundError:
            self.run(results_folder)
            self.display(results_folder)