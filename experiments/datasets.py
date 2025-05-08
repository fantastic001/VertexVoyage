
from vertex_voyage.temporal import FileEventSequence, FirstN
from experiments.utils import display_benchmark_results, benchmark_partitioner
from vertex_voyage.benchmark_base import Benchmark
import pandas as pd 
import os 
import matplotlib.pyplot as plt 

from vertex_voyage.temporal_partitioning import (
    CommonNeighborsPartitioner,
    WindowedLabelPropagationTemporalGraphPartitioner
)

datasets = {
    "SBM 10M": lambda: FileEventSequence("data/sbm_10M.txt"),
    "SBM 1B": lambda: FileEventSequence("data/sbm_1B.txt"),
    "Live Journal": lambda: FileEventSequence("data/LiveJournal.txt"),
    "UK2002": lambda: FileEventSequence("data/uk2002.txt"),
    "Twitch": lambda: FileEventSequence("data/twitch.txt"),
    "Wiki Talks": lambda: FileEventSequence("data/wiki-talks.txt")
}

def create_benchmark_class(partitioner_class, *args):
    class _Benchmark(Benchmark):
        NAME = "Benchmark for partitioner " + partitioner_class.__name__ + " on large datasets"

        def run(self, results_folder):
            N = 1000
            data = [] 
            for name, generator in datasets.items():
                print("Dataset: " + name + " " * 70)
                df = benchmark_partitioner(
                    self,
                    16,
                    partitioner_class,
                    *args,
                    graph_generator=lambda: FirstN(generator(), N)
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