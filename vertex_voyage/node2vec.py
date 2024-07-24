
import networkx as nx 
import numpy as np 
import tensorflow as tf
import random 
class Node2Vec:

    def __init__(self, 
                 dim, 
                 walk_size, 
                 n_walks, 
                 window_size, 
                 epochs=10, 
                 p = .5, 
                 q = .5,
                 seed = None 
            ) -> None:
        self.dim = dim
        self.walk_size = walk_size 
        self.n_walks = n_walks  
        self.window_size = window_size
        self.epochs = epochs
        self.p = p
        self.q = q
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        else:
            np.random.seed()
            random.seed()
    
    def _get_next(self, prev, current):
        neighbors = list(self.G.neighbors(current))
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
        self.model = self._train()
        return self.model

    def _encode(self, node):
        result = np.zeros(len(self.G.nodes()))
        node_index = list(self.G.nodes()).index(node)
        result[node_index] = 1
        return result

    def embed_node(self, node):
        return self.model[list(self.G.nodes()).index(node)]
    
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
        x = []
        y = []
        for walk in self.walks:
            for i, node in enumerate(walk):
                for nb in walk[max(i - self.window_size, 0): i + self.window_size]:
                    if (nb != node).any():
                        x.append(node)
                        y.append(nb)  
        # use tensorflow and implement word2vec 
        x = np.array(x)
        y = np.array(y)
        print("x and y shape:")
        print(x.shape, y.shape)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=len(self.G.nodes()), output_dim=self.dim))
        model.add(tf.keras.layers.Dense(len(self.G.nodes()), activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x, y, epochs=self.epochs)
        # return weights of embedding layer 
        W, *_ = model.layers[0].get_weights()
        return W