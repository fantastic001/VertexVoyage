import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple
from vertex_voyage import node2vec
from vertex_voyage.node2vec import Node2Vec
import networkx as nx

class Walker:
    def __init__(self, id):
        self.id = id
        self.step = 0
        self.Hn = 0.0
        self.Sn = 1.0
        self.EnH = 0.0
        self.EnS = 1.0
        self.EnHS = 0.0
        self.EnHH = 0.0
        self.EnSS = 1.0
        self.N = 1.0
        self.R = 1.0
        self.last_vertex = None
        self.trace_flag = False

class Graph:
    def __init__(self, G: nx.Graph):
        # dict vertex -> List of tuples neighbor and common neighbors
        self.adjacency = {v: [(n, len(set(G.neighbors(v)) & set(G.neighbors(n)))) for n in G.neighbors(v)] for v in G.nodes()}
        self.out_degree = {v: len(neigh) for v, neigh in self.adjacency.items()}
    
    def random_neighbor(self, vertex: int):
        return random.choice(self.adjacency[vertex]) if self.adjacency[vertex] else None
    
    def is_remote(self, vertex: int):
        # Placeholder: define remote logic
        return False

def log2_safe(x):
    return math.log(x, 2) if x > 0 else float('-inf')

def iterate_stats(walker: Walker, fi: float, step_k=1.0):
    Sn = walker.Sn
    Hn = walker.Hn

    Snp1 = Sn + step_k
    t = log2_safe((Sn / Snp1)**Sn * (1 / Snp1)**step_k)
    if fi > 0:
        fj = fi + step_k
        t = log2_safe((Sn / Snp1)**Sn * (1 / Snp1)**step_k * (fj / fi)**fi * fj**step_k)
    
    Hnp1 = (Sn * Hn - t) / Snp1

    Np1 = walker.N + 1
    EnH = walker.EnH + (Hnp1 - walker.EnH) / Np1
    EnS = walker.EnS + (Snp1 - walker.EnS) / Np1
    EnHS = walker.EnHS + (Hnp1 * Snp1 - walker.EnHS) / Np1
    EnHH = walker.EnHH + (Hnp1**2 - walker.EnHH) / Np1
    EnSS = walker.EnSS + (Snp1**2 - walker.EnSS) / Np1

    D_H = EnHH - EnH**2
    D_S = EnSS - EnS**2
    if D_H > 0 and D_S > 0:
        R = (EnHS - EnH * EnS) / math.sqrt(D_H * D_S)
    else:
        R = 0.0

    # Update walker
    walker.Hn = Hnp1
    walker.Sn = Snp1
    walker.EnH = EnH
    walker.EnS = EnS
    walker.EnHS = EnHS
    walker.EnHH = EnHH
    walker.EnSS = EnSS
    walker.N = Np1
    walker.R = R
    walker.step += 1

def walk(graph: Graph, walker: Walker, threshold=0.999, min_length=20, max_length=100):
    walker_to_path_count = defaultdict(int)
    current_v = walker.last_vertex or walker.id  # use walker.id as initial position
    walker.step = 1
    while True:
        yield current_v
        if walker.step == max_length:
            break
        fi = walker_to_path_count[current_v]
        walker_to_path_count[current_v] += 1

        if walker.step > 0:
            iterate_stats(walker, fi)

        if walker.step > min_length and (walker.R**2 < threshold or walker.R < 0):
            break

        degree = graph.out_degree.get(current_v, 0)
        if degree == 0:
            break

        candidate = graph.random_neighbor(current_v)
        if not candidate:
            break

        dst, common_neighbors = candidate
        src_deg = graph.out_degree.get(current_v, 1)
        dst_deg = graph.out_degree.get(dst, 1)

        p_accept = (1 / (src_deg - common_neighbors)) * max(src_deg, dst_deg) / min(src_deg, dst_deg)
        p_norm = math.tanh(p_accept)
        p = random.random()

        if p_norm < p and walker.step > min_length:
            walker.trace_flag = True
            walker.last_vertex = current_v
            break  # emit trace

        if graph.is_remote(dst):
            break  # emit remote walker

        current_v = dst  # continue walking locally

    return walker


class DistGER(Node2Vec):
    """
    DistGER is a class that implements the DistGER algorithm for graph embedding.
    """
    def __init__(
        self,
        dim: int,
        min_walk_size: int,
        max_walk_size: int,
        n_walks: int,
        window_size: int,
        epochs: int = 10,
        p: float = 0.5,
        q: float = 0.5,
        negative_sample_num: int = 50,
        learning_rate: float = 0.01,
        threshold: float = 0.999,
        seed: int = None,
        use_threads: bool = True
    ):
        
        self.threshold = threshold
        self.min_walk_size = min_walk_size
        super().__init__(dim, max_walk_size, n_walks, window_size, epochs, p, q, negative_sample_num, learning_rate, seed, use_threads)

    def _random_walk(self, node):
        walker = Walker(node)
        graph = Graph(self.G)
        walk_ = list(self.nodes[n] for n in walk(graph, walker, threshold=self.threshold, min_length=self.min_walk_size, max_length=self.walk_size))
        return walk_
        