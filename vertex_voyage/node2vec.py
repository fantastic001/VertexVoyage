
import networkx as nx 
import numpy as np 
import tensorflow as tf
import random 
from vertex_voyage.word2vec import word2vec
import vertex_voyage.config as cfg 

@cfg.pluggable
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
                 seed = None 
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
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        else:
            np.random.seed()
            random.seed()
    
    def _get_next(self, prev, current):
        neighbors = list(self.G.neighbors(current)) 
        neighbors = list(set(neighbors))
        if neighbors == []:
            return current
        weights = []
        for neighbor in neighbors:
            weight = self.G[current][neighbor].get('weight', 1)
            if neighbor == prev:
                weights.append(1/self.p)
            elif prev is not None and neighbor in self.G.neighbors(prev):
                weights.append(1)
            else:
                weights.append(1/self.q)
            weights[-1] *= weight
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()
        return np.random.choice(neighbors, p=weights)

    def fit(self, G, nodes = None):
        self.G = G
        if nodes is None:
            nodes = list(G.nodes())
        self.g_nodes = nodes
        self.nodes = {node: self._encode(node) for node in nodes}
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
            return self.W[list(self.nodes).index(node)]
        except KeyError:
            return np.zeros(self.dim)
    
    def embed_nodes(self, nodes):
        return [self.embed_node(node) for node in nodes]

    def _random_walks(self):
        walks = []
        for i in range(self.n_walks):
            node = np.random.choice(list(self.G.nodes()))
            walks.append(self._random_walk(node))
        return walks
    
    def _random_walk(self, node):
        walk = [self.nodes[node]]
        current = node
        prev = None
        for _ in range(self.walk_size - 1):
            next_node = self._get_next(prev, current)
            walk.append(self.nodes[next_node])
            prev = current
            current = next_node
        return walk
    

    
    def _train(self):
        walks = [
            [n.argmax() for n in walk] for walk in self.walks
        ]
        return word2vec(
            training_data=walks,
            vocab_size=len(self.G.nodes()),
            embedding_dim=self.dim,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            window_size=self.window_size,
            num_ns=self.negative_sample_num,
            seed=self.seed
        )