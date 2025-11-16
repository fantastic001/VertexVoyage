

from vertex_voyage.node2vec import Node2Vec
from vertex_voyage.word2vec import word2vec
from vertex_voyage.temporal import Event 
from vertex_voyage.vv_graph import VVGraph

import networkx as nx

class DynNode2Vec(Node2Vec):
    def __init__(
        self,
        dim=128,
        walk_size=80,
        n_walks=10,
        window_size=10,
        epochs=1,
        p=0.5,
        q=0.5,
        negative_sample_num=1,
        learning_rate=0.01,
        seed=None,
        use_threads=True
    ) -> None:
        super().__init__(
            dim=dim,
            walk_size=walk_size,
            n_walks=n_walks,
            window_size=window_size,
            epochs=epochs,
            p=p,
            q=q,
            negative_sample_num=negative_sample_num,
            learning_rate=learning_rate,
            seed=seed,
            use_threads=use_threads
        )
        self.node_embeddings = {}
        self.node_to_neightbours_map = {}
        self.is_weighted = False
        self.G = nx.Graph()
        self.g_nodes = []  # list of nodes in the graph
        self.w2v_model: Word2Vec = None
    def update(self, event: Event):
        if not self.G.has_node(event.src):
            self.G.add_node(event.src)
            self.g_nodes.append(event.src)
        if not self.G.has_node(event.dest):
            self.G.add_node(event.dest)
            self.g_nodes.append(event.dest)
        self.G.add_edge(event.src, event.dest)
        self.node_to_neightbours_map.setdefault(event.src, list()).append(event.dest)
        self.node_to_neightbours_map.setdefault(event.dest, list()).append(event.src)
        # generate random walks for src and dest
        walks = [] 
        for _ in range(self.n_walks):
            walks.append(self._random_walk(event.src))
            walks.append(self._random_walk(event.dest))
        # train/update word2vec model
        if self.w2v_model is None:
            self.w2v_model = word2vec(
                embedding_dim=self.dim,
                vocab_size=len(self.g_nodes),
                training_data=walks,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                window_size=self.window_size,
                num_ns=self.negative_sample_num,
                seed=self.seed,
                epoch_callbacks=[]
            )
        else:
            self.w2v_model = word2vec(
                embedding_dim=self.dim,
                vocab_size=len(self.g_nodes),
                training_data=walks,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                window_size=self.window_size,
                num_ns=self.negative_sample_num,
                seed=self.seed,
                epoch_callbacks=[]
            ) 
        self.W = self.w2v_model
                

        
    