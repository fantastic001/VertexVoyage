
import networkx as nx 
import numpy as np 
import tensorflow as tf
import random 
from vertex_voyage.vv_graph import VVGraph
from vertex_voyage.word2vec import word2vec
import vertex_voyage.config as cfg 
import multiprocessing.pool as mpp
import vertex_voyage_native

class Node2Vec:

    def __init__(self, 
                 dim, 
                 walk_size, 
                 n_walks, 
                 window_size,
                 epochs=10, 
                 p = .5, 
                 q = .5,
                 negative_sample_num = 50,
                 learning_rate = 0.01,
                 seed = None,
                 use_threads = True
            ) -> None:
        self.dim = dim
        self.walk_size = walk_size 
        self.n_walks = n_walks  
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.p = p
        self.q = q
        self.negative_sample_num = negative_sample_num
        self.seed = seed
        self.use_threads = use_threads
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        else:
            np.random.seed()
            random.seed()
    
    def _weighted__get_next(self, prev, current):
        neighbors = self.node_to_neightbours_map[current]
        neighbors = list(set(neighbors))
        r = random.random()
        if neighbors == []:
            return current
        weights = np.zeros(len(neighbors), dtype=np.float32)
        for i, neighbor in enumerate(neighbors):
            if self.is_weighted:
                weight = self.G[current][neighbor].get('weight', 1)
            else:
                weight = 1
            if neighbor == prev:
                weights[i] = weight/self.p
            elif prev is not None and neighbor in self.node_to_neightbours_map[prev]:
                weights[i] = weight
            else:
                weights[i] = weight/self.q
        weights /= np.sum(weights)
        weights_cumsum = np.cumsum(weights)
        return neighbors[np.searchsorted(weights_cumsum, r)]
        # return np.random.choice(neighbors, p=weights / np.sum(weights))

    def _get_next(self, prev, current):
        # if self.is_weighted:
        #     return self._weighted__get_next(prev, current)
        # else:
            neightbors = self.node_to_neightbours_map[current]
            r = np.random.random()
            prev_neightbors = self.node_to_neightbours_map[prev] if prev is not None else []
            return vertex_voyage_native.get_next(r, prev_neightbors, neightbors, current, self.p, self.q, prev)

    def fit(self, G, nodes = None):
        self.G = G
        if isinstance(G, VVGraph):
            self.is_weighted = False 
        else:
            self.is_weighted = any('weight' in data for u, v, data in self.G.edges(data=True))
        self.node_to_neightbours_map = {node: list(self.G.neighbors(node)) for node in self.G.nodes}
        if nodes is None:
            nodes = list(G.nodes)
        self.g_nodes = nodes
        if not isinstance(G, VVGraph):
            self.nodes = {node: self._encode(node) for node in nodes}
        else:
            self.nodes = nodes 
        self.walks = self._random_walks()
        self.W = self._train() 
        W = np.zeros((len(nodes), self.dim))
        for i, node in enumerate(nodes):
            W[i] = self.embed_node(node)
        self.W = W
        return self.W

    def _encode(self, node):
        result = np.zeros(len(self.g_nodes))
        node_index = list(self.g_nodes).index(node)
        result[node_index] = 1
        return result

    def embed_node(self, node):
        try:
            if not isinstance(self.G, VVGraph):
                return self.W[list(self.nodes).index(node)]
            else:
                return self.W[node]
        except KeyError:
            return np.zeros(self.dim)
    
    def embed_nodes(self, nodes):
        return [self.embed_node(node) for node in nodes]

    def _random_walks(self):
        walks = []
        self.G: nx.Graph
        if self.G.number_of_nodes() == 0:
            return [] 
        if self.use_threads:
            starts = [np.random.choice(list(self.G.nodes)) for _ in range(self.n_walks)]
            with mpp.ThreadPool() as pool:
                walks = pool.map(self._random_walk, starts)
        else:
            for _ in range(self.n_walks):
                start = np.random.choice(list(self.G.nodes))
                walks.append(self._random_walk(start))
        return walks
    
    def _random_walk(self, node):
        if not isinstance(self.G, VVGraph):
            walk = [self.nodes[node]]
        else:
            walk = [node]
        current = node
        prev = None
        for _ in range(self.walk_size - 1):
            next_node = self._get_next(prev, current)
            if not isinstance(self.G, VVGraph):
                walk.append(self.nodes[next_node])
            else:
                walk.append(next_node)
            prev = current
            current = next_node
        return walk
    

    
    def _train(self):
        walks = [
            [n.argmax() if hasattr(n, "argmax") and n.ndim > 0 else n for n in walk] for walk in self.walks
        ]
        if len(walks) == 0:
            return np.zeros((len(self.nodes), self.dim))
        return word2vec(
            training_data=walks,
            vocab_size=len(self.g_nodes),
            embedding_dim=self.dim,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            window_size=self.window_size,
            num_ns=self.negative_sample_num,
            seed=self.seed
        )