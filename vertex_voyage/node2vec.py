
import networkx as nx 
import numpy as np 
import tensorflow as tf
import random 
from vertex_voyage.vv_graph import VVGraph
from vertex_voyage.word2vec import word2vec
import vertex_voyage.config as cfg 
import multiprocessing.pool as mpp
import vertex_voyage_native

import logging 

logger = logging.getLogger(__name__)

class Node2Vec:

    def __init__(self, 
                 dim=128, 
                 walk_size=80, 
                 n_walks=10, 
                 window_size=10,
                 epochs=1, 
                 p = .5, 
                 q = .5,
                 negative_sample_num = 1,
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
        logger.info(f"Initialized Node2Vec with parameters: {self}")
    
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
        logger.info(f"Fitting Node2Vec on graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        if nodes is not None:
            logger.info(f"Using provided nodes for embedding: {len(nodes)} nodes.")
        if isinstance(G, VVGraph):
            self.is_weighted = False 
        else:
            self.is_weighted = any('weight' in data for u, v, data in self.G.edges(data=True))
        self.node_to_neightbours_map = {node: list(self.G.neighbors(node)) for node in self.G.nodes}
        if nodes is None:
            nodes = list(G.nodes)
        self.g_nodes = list(G.nodes)
        if not isinstance(self.G, VVGraph):
            self.nodes = {node: self._encode(node) for node in nodes}
        else:
            self.nodes = self.g_nodes
        self.walks = self._random_walks()
        logger.info(f"Generated {len(self.walks)} random walks.")
        self.W = self._train() 
        W = np.zeros((len(self.g_nodes), self.dim))
        for i, node in enumerate(self.g_nodes):
            W[i] = self.embed_node(node)
        self.W = W
        return self.W

    def _encode(self, node):
        try:
            return self.g_nodes.index(node)
        except ValueError:
            return None 

    def embed_node(self, node):
        node_idx = self._encode(node)
        if node_idx is None:
            return np.zeros(self.dim)
        return self.W[node_idx]
    
    def embed_nodes(self, nodes):
        return [self.embed_node(node) for node in nodes]

    def _random_walks(self):
        walks = []
        self.G: nx.Graph
        if self.G.number_of_nodes() == 0:
            return [] 
        if self.use_threads:
            starts = [n for _ in range(self.n_walks) for n in self.g_nodes]
            with mpp.ThreadPool() as pool:
                walks = pool.map(self._random_walk, starts)
        else:
            for _ in range(self.n_walks):
                for n in self.g_nodes:
                    start = n
                    walks.append(self._random_walk(start))
        return walks
    
    def _random_walk(self, node):
        logger.debug(f"Starting random walk from node: {node}")
        walk = [node]
        current = node
        prev = None
        for _ in range(self.walk_size - 1):
            next_node = self._get_next(prev, current)
            walk.append(next_node)
            prev = current
            current = next_node
        result = [self._encode(n) for n in walk]
        logger.debug(f"Completed random walk: {result}")
        return result

    
    def _train(self):
        from gensim.models import Word2Vec
        walks = [
            [n.argmax() if hasattr(n, "argmax") and n.ndim > 0 else n for n in walk] for walk in self.walks
        ]
        if len(walks) == 0:
            return np.zeros((len(self.nodes), self.dim))
        x = Word2Vec(
            sentences=walks,
            vector_size=self.dim,
            window=self.window_size,
            null_word=-1,
            sg=1,
            negative=self.negative_sample_num,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
            workers=cfg.get_config_int("workers", 6, "Number of workers during word2vec training") if self.use_threads else 1
        )
        logger.info(f"Trained Word2Vec model with {len(x.wv)} unique nodes in vocabulary.")
        for node in self.nodes:
            if node not in x.wv:
                logger.warning(f"Node {node} not found in Word2Vec vocabulary. This may indicate an issue with the random walks or encoding.")
            logger.debug(f"Node {node} has embedding: {x.wv[node] if node in x.wv else 'Not found'}")
        return x.wv

    def __repr__(self) -> str:
        return f"Node2Vec(dim={self.dim}, walk_size={self.walk_size}, n_walks={self.n_walks}, window_size={self.window_size}, epochs={self.epochs}, p={self.p}, q={self.q}, negative_sample_num={self.negative_sample_num}, learning_rate={self.learning_rate}, seed={self.seed}, use_threads={self.use_threads})"