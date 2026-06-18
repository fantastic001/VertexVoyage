

import random
from venv import logger

from vertex_voyage.node2vec import Node2Vec
import vertex_voyage.word2vec as w2v
from vertex_voyage.temporal import Event 
from vertex_voyage.vv_graph import VVGraph

import networkx as nx
import vertex_voyage.config as cfg
import numpy as np

import logging

logger = logging.getLogger(__name__)
class DynNode2Vec(Node2Vec):
    def __init__(
        self,
        dim=128,
        walk_size=80,
        n_walks=10,
        window_size=10,
        epochs=10,
        p=0.5,
        q=0.5,
        negative_sample_num=5,
        learning_rate=0.01,
        seed=None,
        use_threads=True,
        retrain_threshold=30
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
            use_threads=use_threads,
        )
        self.node_embeddings = {}
        self.node_to_neightbours_map = {}
        self.is_weighted = False
        self.G = nx.Graph()
        self.g_nodes = []  # list of nodes in the graph
        self.node_to_idx = {}  # mapping from node to index in the embedding matrix
        self.w2v_model = None
        self.retrain_threshold = retrain_threshold
        logger.info(f"Initialized DynNode2Vec with retrain_threshold={self.retrain_threshold}")
    def update_model(self, walks, old_model=None):
        logger.info(f"Updating model with {len(walks)} walks. Old model: {'Yes' if old_model else 'No'}")
        from gensim.models import Word2Vec
        if old_model is not None:
            # continue training from old model
            if isinstance(old_model, w2v.Word2Vec):
                logger.info("Continuing training from custom Word2Vec model")
                self.W, model = w2v.word2vec(
                    embedding_dim=self.dim,
                    vocab_size=len(self.g_nodes),
                    training_data=walks,
                    learning_rate=self.learning_rate,
                    epochs=self.epochs,
                    window_size=self.window_size,
                    num_ns=self.negative_sample_num,
                    seed=self.seed,
                    epoch_callbacks=[],
                    old_model=old_model
                )
            else:
                logger.info("Continuing training from gensim Word2Vec model")
                old_model.build_vocab(walks, update=True)
                old_model.train(
                    corpus_iterable = walks,
                    total_examples=old_model.corpus_count,
                    epochs=self.epochs,
                    callbacks=[],
                )
                self.W = old_model.wv
                W = np.zeros((len(self.g_nodes), self.dim))
                for i, node in enumerate(self.g_nodes):
                    W[i] = self.embed_node(node)
                self.W = W
                model = old_model
        else:
            logger.info("Training new model from scratch")
            # train new model
            model = Word2Vec(
                vector_size=self.dim,
                window=self.window_size,
                min_count=0,
                sg=1,
                negative=self.negative_sample_num,
                alpha=self.learning_rate,
                epochs=self.epochs,
                seed=self.seed if self.seed is not None else random.randint(0, 10000),
                null_word=-1,
                shrink_windows=False,
                workers=cfg.get_config_int("dynnode2vec_workers", 6, "Number of workers during word2vec training for DynNode2Vec") if self.use_threads else 1
            )
            model.build_vocab(walks)
            model.train(
                corpus_iterable=walks,
                total_examples=model.corpus_count,
                epochs=model.epochs,
                callbacks=[],
            )
            self.W = model.wv
            W = np.zeros((len(self.g_nodes), self.dim))
            for i, node in enumerate(self.g_nodes):
                W[i] = self.embed_node(node)
            self.W = W
        return model
    def update(self, events: list[Event]):
        affected_nodes = {event.src for event in events} | {event.dest for event in events}
        for event in events:
            if not self.G.has_node(event.src):
                logger.debug(f"Adding new node {event.src} to the graph")
                self.G.add_node(event.src)
                self.g_nodes.append(event.src)
                self.node_to_idx[event.src] = len(self.g_nodes) - 1
            if self.w2v_model is not None and isinstance(self.w2v_model, w2v.Word2Vec):
                self.w2v_model = self.w2v_model.insert_weights(
                    [self.node_to_idx.get(event.src)]
                )
            if not self.G.has_node(event.dest):
                logger.debug(f"Adding new node {event.dest} to the graph")
                self.G.add_node(event.dest)
                self.g_nodes.append(event.dest)
                self.node_to_idx[event.dest] = len(self.g_nodes) - 1
                if self.w2v_model is not None and isinstance(self.w2v_model,    w2v.Word2Vec):
                    self.w2v_model = self.w2v_model.insert_weights(
                        [self.node_to_idx.get(event.dest)]
                    )
            self.G.add_edge(event.src, event.dest)
            self.node_to_neightbours_map.setdefault(event.src, list()).append(event.dest)
            self.node_to_neightbours_map.setdefault(event.dest, list()).append(event.src)
        logger.debug(f"Added {len(events)} edges to the graph")
        logger.debug(f"Current number of nodes: {len(self.G.nodes())}, edges: {len(self.G.edges())}")
        
        # generate random walks for src and dest
        if len(self.G.nodes()) < self.retrain_threshold:
            logger.debug(f"Graph has {len(self.G.nodes())} nodes, which is less than retrain_threshold={self.retrain_threshold}. Retraining model from scratch.")
            walks = self._random_walks()
            self.w2v_model = self.update_model(walks, None)
        else:
            logger.debug(f"Graph has {len(self.G.nodes())} nodes, which is greater than or equal to retrain_threshold={self.retrain_threshold}. Updating model with new walks for affected nodes.")
            walks = self._random_walks(affected_nodes)
            self.w2v_model = self.update_model(walks, self.w2v_model)