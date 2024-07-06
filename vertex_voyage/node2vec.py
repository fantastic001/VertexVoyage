
import networkx as nx 
import numpy as np 
import tensorflow as tf

class Node2Vec:

    def __init__(self, dim, walk_size, n_walks, window_size, epochs=10) -> None:
        self.dim = dim
        self.walk_size = walk_size 
        self.n_walks = n_walks  
        self.window_size = window_size
        self.epochs = epochs
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
        return self.model[self.nodes[node]]
    
    def _random_walks(self):
        walks = []
        for i in range(self.n_walks):
            node = np.random.choice(list(self.G.nodes()))
            walks.append(self._random_walk(node))
        return walks
    
    def _random_walk(self, node):
        walk = [self.nodes[node]]
        for _ in range(self.walk_size):
            neighbors = list(self.G.neighbors(node))
            if neighbors:
                node = np.random.choice(neighbors)
                walk.append(self.nodes[node])
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