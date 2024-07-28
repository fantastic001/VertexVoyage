
import networkx as nx 
import numpy as np 
import tensorflow as tf
import random 
from vertex_voyage.word2vec import word2vec
class Node2Vec:

    def __init__(self, 
                 dim, 
                 walk_size, 
                 n_walks, 
                 window_size, 
                 epochs=10, 
                 p = .5, 
                 q = .5,
                 batch_size = None,
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
        self.batch_size = batch_size
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

    def fit(self, G):
        self.G = G
        self.nodes = {node: self._encode(node) for node in self.G.nodes()}
        self.walks = self._random_walks()
        self.W = self._train()
        return self.W

    def _encode(self, node):
        result = np.zeros(len(self.G.nodes()))
        node_index = list(self.G.nodes()).index(node)
        result[node_index] = 1
        return result

    def embed_node(self, node):
        return self.W[list(self.G.nodes()).index(node)]
    
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
        training_data = [] 
        node_indices = [node.argmax() for node in self.nodes.values()]
        for walk in self.walks:
            for i, node in enumerate(walk):
                node_index = node.argmax()
                for context in walk[max(0, i - self.window_size): min(len(walk), i + self.window_size)]:
                    if (node != context).any():
                        context_index = context.argmax()
                        training_data.append((node_index, context_index, 1))
                # choose window_size samples on random which are not in context range 
                non_context = list(set(node_indices) - set([node.argmax() for node in walk[max(0, i - self.window_size): min(len(walk), i + self.window_size)] ]))
                if len(non_context) > 0:
                    for _ in range(100*self.window_size):
                        sample = random.choice(non_context)
                        if sample != node_index:
                            training_data.append((node_index, sample, 0))
        return word2vec(training_data, len(self.G.nodes()), self.dim, self.learning_rate, self.epochs)