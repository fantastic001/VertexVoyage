
from numpy import mod
from vertex_voyage import data
from vertex_voyage.temporal import Event, FirstN, Transform, to_nx_graph
from vertex_voyage.benchmark_base import Benchmark
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
from vertex_voyage.partitioning import (
    get_f1_reconstruction_score,
    get_partition_average_balance,
    get_node2vec_embedding,
    modified__lfm,
    random_partitioning,
    label_propagation_partitioner
)
from vertex_voyage.temporal_partitioning import (
    CommonNeighborsPartitioner,
    WindowedLabelPropagationTemporalGraphPartitioner
)
from vertex_voyage.node2vec import Node2Vec
from vertex_voyage.distger import DistGER
from experiments.datasets import datasets
from datetime import datetime
from experiments.utils import is_full_benchmark


def timeit(func, *args, **kwargs):
    start = datetime.now()
    result = func(*args, **kwargs)
    end = datetime.now()
    return result, (end - start).total_seconds()



class VertexEnumerator:
    def __init__(self):
        self.visited = set()
        self.index = {}
    
    def __call__(self, node):
        if node not in self.visited:
            self.visited.add(node)
            self.index[node] = len(self.visited) - 1
        return self.index[node]

def get_distger_embedding(dim,
                           min_walk_size,
                           max_walk_size,
                           n_walks,
                           window_size,
                           epochs,
                           p = .5, 
                           q = .5,
                           negative_sample_num = 50,
                           learning_rate = 0.01,
                           seed = None,
                           use_threads = True):
    from vertex_voyage.node2vec import Node2Vec
    def f(G):
        node2vec = DistGER(dim, min_walk_size, max_walk_size, n_walks, window_size, epochs, p, q, negative_sample_num, learning_rate, seed, use_threads)
        node2vec.fit(G)
        return {node: node2vec.embed_node(node) for node in G.nodes()}
    return f

def create_benchmark_class(emb_name, emb_gen, partitioner_class, *args, **kwargs):
    class _Benchmark(Benchmark):
        NAME = f"Performance comparison max {emb_name} vs partitioner {partitioner_class.__name__} on large datasets"

        def run(self, results_folder):
            N = 10 if not is_full_benchmark() else None
            data = [] 
            for name, generator in list(datasets.items()):
                print("Dataset: " + name + " " * 70)
                t = VertexEnumerator()
                if N:
                    g = FirstN(generator(), N)
                g = list(Transform(g, lambda x: Event(
                    src=t(int(x.src)),
                    dest=t(int(x.dest)),
                    timestamp=int(x.timestamp),
                    type=x.type,
                    attrs=x.attrs,
                )))
                g = to_nx_graph(g)
                partitions, partition_t = timeit(partitioner_class, g, 16, *args, **kwargs)
                balance = get_partition_average_balance({i: len(p) for i, p in enumerate(partitions)}, 16)
                print("Balance: ", balance)
                embeddings = [timeit(emb_gen, g.subgraph(p)) for p in partitions]
                emb_t = sum([t[1] for t in embeddings])
                max_emb_time = max([t[1] for t in embeddings])
                data.append({
                    "dataset": name,
                    "balance": balance,
                    "partition_time": partition_t,
                    "max_embedding_time": max_emb_time,
                    "embedding_time": emb_t,
                })
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(results_folder, "results.csv"))
        
        def display(self, results_folder):
            try:
                df = pd.read_csv(os.path.join(results_folder, "results.csv"))
                fig, ax = plt.subplots()
                df.plot(
                    x="dataset",
                    y=["partition_time", "max_embedding_time"],
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

Node2VecBenchmark = create_benchmark_class(
    "Node2Vec",
    get_node2vec_embedding(
        dim=128,
        epochs=1,
        p=0.5,
        q=0.5,
        learning_rate=0.01, 
        negative_sample_num=5,
        n_walks=100,
        seed=42,
        use_threads=True,
        walk_size=80,
        window_size=20,
    ),
    modified__lfm
)

DistGERBenchmark = create_benchmark_class(
    "DistGER",
    get_distger_embedding(
        dim=128,
        epochs=1,
        p=0.5,
        q=0.5,
        learning_rate=0.01, 
        negative_sample_num=5,
        n_walks=100,
        seed=42,
        use_threads=True,
        max_walk_size=80,
        min_walk_size=20,
        window_size=20,
    ),
    modified__lfm
)

RandomNode2VecBenchmark = create_benchmark_class(
    "Node2Vec",
    get_node2vec_embedding(
        dim=128,
        epochs=1,
        p=0.5,
        q=0.5,
        learning_rate=0.01, 
        negative_sample_num=5,
        n_walks=100,
        seed=42,
        use_threads=True,
        walk_size=80,
        window_size=20,
    ),
    random_partitioning
)

RandomDistGERBenchmark = create_benchmark_class(
    "DistGER",
    get_distger_embedding(
        dim=128,
        epochs=1,
        p=0.5,
        q=0.5,
        learning_rate=0.01, 
        negative_sample_num=5,
        n_walks=100,
        seed=42,
        use_threads=True,
        max_walk_size=80,
        min_walk_size=20,
        window_size=20,
    ),
    random_partitioning
)

LPANode2VecBenchmark = create_benchmark_class(
    "Node2Vec",
    get_node2vec_embedding(
        dim=128,
        epochs=1,
        p=0.5,
        q=0.5,
        learning_rate=0.01, 
        negative_sample_num=5,
        n_walks=100,
        seed=42,
        use_threads=True,
        walk_size=80,
        window_size=20,
    ),
    label_propagation_partitioner
)

LPADistGERBenchmark = create_benchmark_class(
    "DistGER",
    get_distger_embedding(
        dim=128,
        epochs=1,
        p=0.5,
        q=0.5,
        learning_rate=0.01, 
        negative_sample_num=5,
        n_walks=100,
        seed=42,
        use_threads=True,
        max_walk_size=80,
        min_walk_size=20,
        window_size=20,
    ),
    label_propagation_partitioner
)

def create_benchmark_class_for_partitioner(dataset, emb_name, emb_gen, partitioner, param, param_range, **kwargs):
    class _Benchmark(Benchmark):
        NAME = f"Performance comparison max {emb_name} vs {partitioner.__name__} on {dataset} with varying {param}"

        def run(self, results_folder):
            N = 10 if not is_full_benchmark() else None
            data = [] 
            generator = datasets[dataset]
            for i, p in enumerate(param_range):
                t = VertexEnumerator()
                if N:
                    g = FirstN(generator(), N)
                g = list(Transform(g, lambda x: Event(
                    src=t(int(x.src)),
                    dest=t(int(x.dest)),
                    timestamp=int(x.timestamp),
                    type=x.type,
                    attrs=x.attrs,
                )))
                g = to_nx_graph(g)
                kwargs[param] = p
                partitions, partition_t = timeit(partitioner, g, **kwargs)
                balance = get_partition_average_balance({i: len(p) for i, p in enumerate(partitions)}, kwargs["partition_count"])
                embeddings = [timeit(emb_gen, g.subgraph(p)) for p in partitions]
                emb_t = sum([t[1] for t in embeddings])
                max_emb_time = max([t[1] for t in embeddings])
                data.append({
                    f"{param}": p,
                    "balance": balance,
                    "partition_time": partition_t,
                    "max_embedding_time": max_emb_time,
                    "embedding_time": emb_t,
                })
                self.report_progress(i+1, len(param_range))
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(results_folder, "results.csv"))
        
        def display(self, results_folder):
            try:
                df = pd.read_csv(os.path.join(results_folder, "results.csv"))
                fig, ax = plt.subplots()
                df.plot(
                    x=param,
                    y=["partition_time", "max_embedding_time"],
                    kind="bar",
                    ax=ax,
                    title=self.NAME,
                    xlabel=param,
                    ylabel="Time (s)",
                )
                plt.grid(True)
                plt.show()
            except FileNotFoundError:
                self.run(results_folder)
                self.display(results_folder)
    return _Benchmark

lfm_threshold = create_benchmark_class_for_partitioner(
    "SBM 10M", 
    "Node2Vec", 
    get_node2vec_embedding(
        dim=128,
        epochs=1,
        p=0.5,
        q=0.5,
        learning_rate=0.01, 
        negative_sample_num=5,
        n_walks=100,
        seed=42,
        use_threads=True,
        walk_size=80,
        window_size=20,
    ),
    modified__lfm,
    "threshold",
    [0, .5, 1], 
    partition_count=16,
    pm_k=16,
    alpha=1
)

lfm_alpha = create_benchmark_class_for_partitioner(
    "SBM 10M", 
    "Node2Vec", 
    get_node2vec_embedding(
        dim=128,
        epochs=1,
        p=0.5,
        q=0.5,
        learning_rate=0.01, 
        negative_sample_num=5,
        n_walks=100,
        seed=42,
        use_threads=True,
        walk_size=80,
        window_size=20,
    ),
    modified__lfm,
    "alpha",
    [1, 1.5, 2], 
    partition_count=16,
    pm_k=16,
    alpha=1,
    threshold=0.5
)

lfm_k = create_benchmark_class_for_partitioner(
    "SBM 10M", 
    "Node2Vec", 
    get_node2vec_embedding(
        dim=128,
        epochs=1,
        p=0.5,
        q=0.5,
        learning_rate=0.01, 
        negative_sample_num=5,
        n_walks=100,
        seed=42,
        use_threads=True,
        walk_size=80,
        window_size=20,
    ),
    modified__lfm,
    "k",
    [1, 2, 4, 8, 16, 32, 64],
    partition_count=16,
    alpha=1
)

lfm_partition_count = create_benchmark_class_for_partitioner(
    "SBM 10M", 
    "Node2Vec", 
    get_node2vec_embedding(
        dim=128,
        epochs=1,
        p=0.5,
        q=0.5,
        learning_rate=0.01, 
        negative_sample_num=5,
        n_walks=100,
        seed=42,
        use_threads=True,
        walk_size=80,
        window_size=20,
    ),
    modified__lfm,
    "partition_count",
    [1, 2, 4, 8, 16, 32],
    pm_k=16,
    alpha=1,
    threshold=0.5
)

def create_benchmark_class_for_emb_with_lfm(dataset, emb_name, emb_gen, param, param_range, **kwargs):
    class _Benchmark(Benchmark):
        NAME = f"Performance comparison max {emb_name} vs LFM on {dataset} with varying {param}"

        def run(self, results_folder):
            N = 10 if not is_full_benchmark() else None
            data = [] 
            generator = datasets[dataset]
            for i, p in enumerate(param_range):
                t = VertexEnumerator()
                if N:
                    g = FirstN(generator(), N)
                g = list(Transform(g, lambda x: Event(
                    src=t(int(x.src)),
                    dest=t(int(x.dest)),
                    timestamp=int(x.timestamp),
                    type=x.type,
                    attrs=x.attrs,
                )))
                g = to_nx_graph(g)
                kwargs[param] = p
                partitions, partition_t = timeit(modified__lfm, g, 16, pm_k=16, alpha=1, threshold=0.5)
                embeddings = [timeit(emb_gen(**kwargs), g.subgraph(p)) for p in partitions]
                emb_t = sum([t[1] for t in embeddings])
                max_emb_time = max([t[1] for t in embeddings])
                data.append({
                    f"{param}": p,
                    "partition_time": partition_t,
                    "max_embedding_time": max_emb_time,
                    "embedding_time": emb_t,
                })
                self.report_progress(i+1, len(param_range))
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(results_folder, "results.csv"))
        
        def display(self, results_folder):
            try:
                df = pd.read_csv(os.path.join(results_folder, "results.csv"))
                fig, ax = plt.subplots()
                df.plot(
                    x=param,
                    y=["partition_time", "max_embedding_time"],
                    kind="bar",
                    ax=ax,
                    title=self.NAME,
                    xlabel=param,
                    ylabel="Time (s)",
                )
                plt.grid(True)
                plt.show()
            except FileNotFoundError:
                self.run(results_folder)
                self.display(results_folder)
    return _Benchmark

node2vec_walk_size = create_benchmark_class_for_emb_with_lfm(
    "SBM 10M", 
    "Node2Vec", 
    get_node2vec_embedding,
    "walk_size",
    list(range(80, 400, 20)),
    dim=128,
    epochs=1,
    p=0.5,
    q=0.5,
    learning_rate=0.01, 
    negative_sample_num=5,
    n_walks=100,
    seed=42,
    use_threads=True,
    window_size=20
)