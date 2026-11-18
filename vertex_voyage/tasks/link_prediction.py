import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence, Type
import vertex_voyage.config as cfg

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)

# Model without explicit Sigmoid
class HadamardLogitsNet(nn.Module):
    def __init__(self, vector_dim, use_bias = True):
        super(HadamardLogitsNet, self).__init__()
        logger.info(f"Initializing HadamardLogitsNet with vector_dim={vector_dim}, use_bias={use_bias}")
        self.fc = nn.Linear(vector_dim, 1, bias=use_bias)
        
    def forward(self, u, v):
        hadamard = u * v
        return self.fc(hadamard) # Outputs "logits"

    criterion = nn.BCEWithLogitsLoss()

class QuadraticLogitsNet(nn.Module):
    def __init__(self, vector_dim, use_bias = True):
        super(QuadraticLogitsNet, self).__init__()
        logger.info(f"Initializing QuadraticLogitsNet with vector_dim={vector_dim}, use_bias={use_bias}")
        self.fc = nn.Linear(vector_dim, vector_dim, bias=use_bias)
        
    def forward(self, u, v):
        # x.T A y
        u_transformed = self.fc(u)
        logits = torch.sum(u_transformed * v, dim=1)
        return logits

    criterion = nn.BCEWithLogitsLoss()


class LinkPredictionModel(ABC):
    @abstractmethod
    def predict_proba(self, u_embedding, v_embedding) -> float:
        ...


class TorchLogitsLinkPredictionModel(LinkPredictionModel):
    def __init__(self, net: nn.Module):
        self.net = net

    def predict_proba(self, u_embedding, v_embedding) -> float:
        with torch.no_grad():
            logit = self.net(
                torch.tensor(u_embedding, dtype=torch.float32).unsqueeze(0),
                torch.tensor(v_embedding, dtype=torch.float32).unsqueeze(0),
            ).item()
        return 1 / (1 + np.exp(-logit))


def concatenate_embeddings(u_embeddings, v_embeddings):
    return np.concatenate((np.atleast_2d(u_embeddings), np.atleast_2d(v_embeddings)), axis=1)


class RandomForestLinkPredictionModel(LinkPredictionModel):
    def __init__(self, classifier: RandomForestClassifier):
        logger.info(f"Initializing RandomForestLinkPredictionModel with classifier: {classifier}")
        self.classifier = classifier

    def predict_proba(self, u_embedding, v_embedding) -> float:
        features = concatenate_embeddings(u_embedding, v_embedding)
        return self.classifier.predict_proba(features)[0][1]


@dataclass
class LinkPredictionSplit:
    u: np.ndarray
    v: np.ndarray
    y: np.ndarray


@dataclass
class LinkPredictionDataset:
    train: LinkPredictionSplit
    val: LinkPredictionSplit


class LinkPredictionModelTrainer(ABC):
    @abstractmethod
    def train(self, dataset: LinkPredictionDataset) -> "tuple[LinkPredictionModel, list, list]":
        ...


class TorchLogitsLinkPredictionModelTrainer(LinkPredictionModelTrainer):
    def __init__(self, net_class: Type[nn.Module], epochs=60, batch_size=32, learning_rate=0.1, use_bias=False):
        self.net_class = net_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_bias = use_bias

    def train(self, dataset: LinkPredictionDataset):
        net = self.net_class(vector_dim=dataset.train.u.shape[1], use_bias=self.use_bias)
        net, train_losses, val_losses = train_model(
            net,
            dataset.train.u, dataset.train.v, dataset.train.y,
            dataset.val.u, dataset.val.v, dataset.val.y,
            epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
        )
        return TorchLogitsLinkPredictionModel(net), train_losses, val_losses


class RandomForestLinkPredictionModelTrainer(LinkPredictionModelTrainer):
    def __init__(self, n_estimators=20, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def train(self, dataset: LinkPredictionDataset):
        classifier = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        classifier.fit(concatenate_embeddings(dataset.train.u, dataset.train.v), dataset.train.y)
        model = RandomForestLinkPredictionModel(classifier)
        train_losses = [self._compute_log_loss(model, dataset.train)]
        val_losses = [self._compute_log_loss(model, dataset.val)] if len(dataset.val.y) > 0 else list(train_losses)
        return model, train_losses, val_losses

    def _compute_log_loss(self, model: RandomForestLinkPredictionModel, split: LinkPredictionSplit) -> float:
        if cfg.get_config_bool(
            "lp_rdf_compute_log_loss", 
            False, 
            "Wether to compute log loss when using RDF."
            " It is a slow operation."
        ):
            probabilities = [model.predict_proba(u, v) for u, v in zip(split.u, split.v)]
            return log_loss(split.y, probabilities, labels=[0, 1])
        else:
            return 0


LINK_PREDICTION_MODEL_TRAINERS: "dict[str, Callable[[], LinkPredictionModelTrainer]]" = {
    "bilinear": lambda: TorchLogitsLinkPredictionModelTrainer(QuadraticLogitsNet),
    "hadamard": lambda: TorchLogitsLinkPredictionModelTrainer(HadamardLogitsNet),
    "random_forest": lambda: RandomForestLinkPredictionModelTrainer(),
}


LOSS_OF_EMPTY_SPLIT = float("nan")


def tensor_dataset_of(u, v, y) -> TensorDataset:
    return TensorDataset(
        torch.tensor(u, dtype=torch.float32),
        torch.tensor(v, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def batch_logits_of(model, u_batch, v_batch):
    return model(u_batch, v_batch).reshape(-1)


def mean_loss_of(model, loader: DataLoader) -> float:
    if len(loader) > 0:
        total_loss = 0.0
        with torch.no_grad():
            for u_batch, v_batch, y_batch in loader:
                logits = batch_logits_of(model, u_batch, v_batch)
                total_loss += model.criterion(logits, y_batch).item()
        return total_loss / len(loader)
    else:
        logger.debug("Empty split, its loss is undefined")
        return LOSS_OF_EMPTY_SPLIT


def descend_one_epoch(model, optimizer, loader: DataLoader) -> None:
    for u_batch, v_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = batch_logits_of(model, u_batch, v_batch)
        model.criterion(logits, y_batch).backward()
        optimizer.step()


def train_model(model, u_train, v_train, y_train, u_val, v_val, y_val, epochs=10, batch_size=32, learning_rate=0.1):
    if len(y_train) == 0:
        logger.error("Cannot train a link prediction model without pairs")
        raise ValueError("train_model needs at least one training pair")
    else:
        logger.debug(
            "Training link prediction model on %d pairs, validating on %d",
            len(y_train),
            len(y_val),
        )
    val_losses = []
    train_losses = []
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        tensor_dataset_of(u_train, v_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        tensor_dataset_of(u_val, v_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    for _ in range(epochs):
        model.eval()
        train_losses.append(mean_loss_of(model, train_loader))
        val_losses.append(mean_loss_of(model, val_loader))
        model.train()
        descend_one_epoch(model, optimizer, train_loader)
    model.eval()
    train_losses.append(mean_loss_of(model, train_loader))
    val_losses.append(mean_loss_of(model, val_loader))
    return model, train_losses, val_losses


INCREMENTAL_DECODER_NAMES = "bilinear, hadamard"
TORCH_RANDOM_STATE_OF_CPU_ONLY: list = []


def require_incremental_trainer(
    trainer: LinkPredictionModelTrainer,
) -> TorchLogitsLinkPredictionModelTrainer:
    if isinstance(trainer, TorchLogitsLinkPredictionModelTrainer):
        return trainer
    else:
        message = (
            f"{type(trainer).__name__} cannot continue training an existing "
            f"decoder, use one of: {INCREMENTAL_DECODER_NAMES}"
        )
        logger.error(message)
        raise ValueError(message)


class IncrementalLogitsDecoder(LinkPredictionModel):
    def __init__(
        self,
        trainer: LinkPredictionModelTrainer,
        vector_dim: int,
        seed: int,
    ):
        self.trainer = require_incremental_trainer(trainer)
        if vector_dim <= 0:
            logger.error("Decoder dimension must be positive: %d", vector_dim)
            raise ValueError(f"vector_dim must be positive, got {vector_dim}")
        else:
            logger.info(
                "Creating incremental %s decoder of dimension %d with seed %d",
                self.trainer.net_class.__name__,
                vector_dim,
                seed,
            )
        self.vector_dim = vector_dim
        self.refit_count = 0
        with torch.random.fork_rng(devices=TORCH_RANDOM_STATE_OF_CPU_ONLY):
            torch.manual_seed(seed)
            self.net = self.trainer.net_class(
                vector_dim=vector_dim,
                use_bias=self.trainer.use_bias,
            )
        self.model = TorchLogitsLinkPredictionModel(self.net)

    def refit(self, dataset: LinkPredictionDataset, seed: int):
        training_shape = np.shape(dataset.train.u)
        if len(training_shape) != 2 or training_shape[1] != self.vector_dim:
            logger.error(
                "Training vectors of shape %s do not match decoder "
                "dimension %d",
                training_shape,
                self.vector_dim,
            )
            raise ValueError("Training vectors do not match the decoder")
        else:
            logger.debug(
                "Refit %d of the decoder on %d pairs with seed %d",
                self.refit_count + 1,
                len(dataset.train.y),
                seed,
            )
        with torch.random.fork_rng(devices=TORCH_RANDOM_STATE_OF_CPU_ONLY):
            torch.manual_seed(seed)
            _, train_losses, val_losses = train_model(
                self.net,
                dataset.train.u, dataset.train.v, dataset.train.y,
                dataset.val.u, dataset.val.v, dataset.val.y,
                epochs=self.trainer.epochs,
                batch_size=self.trainer.batch_size,
                learning_rate=self.trainer.learning_rate,
            )
        self.refit_count += 1
        logger.debug("Decoder training losses: %s", train_losses)
        return train_losses, val_losses

    def predict_proba(self, u_embedding, v_embedding) -> float:
        return self.model.predict_proba(u_embedding, v_embedding)

def snapshot_error(message: str) -> ValueError:
    logger.error(message)
    return ValueError(message)


class EmbeddingSnapshot:
    def __init__(
        self,
        vertices: Sequence[Hashable],
        vectors: Sequence,
        dim: int,
    ):
        if dim <= 0:
            raise snapshot_error(
                f"Embedding dimension must be positive, got {dim}"
            )
        elif len(vertices) != len(vectors):
            raise snapshot_error(
                f"{len(vertices)} vertices but {len(vectors)} vectors"
            )
        else:
            logger.debug("Freezing embedding of %d vertices", len(vertices))
        matrix = np.array(vectors) if len(vectors) > 0 else np.zeros((0, dim))
        row_of_vertex = {vertex: row for row, vertex in enumerate(vertices)}
        if matrix.shape != (len(vertices), dim):
            raise snapshot_error(
                f"Vectors have shape {matrix.shape}, expected "
                f"{(len(vertices), dim)}"
            )
        elif len(row_of_vertex) != len(vertices):
            raise snapshot_error("A vertex appears more than once")
        else:
            matrix.setflags(write=False)
        self.dim = dim
        self._matrix = matrix
        self._row_of_vertex = row_of_vertex

    @staticmethod
    def empty(dim: int) -> "EmbeddingSnapshot":
        return EmbeddingSnapshot([], [], dim)

    @staticmethod
    def of(vectors_by_vertex: Mapping) -> "EmbeddingSnapshot":
        if len(vectors_by_vertex) > 0:
            vertices = list(vectors_by_vertex)
            vectors = [vectors_by_vertex[vertex] for vertex in vertices]
            return EmbeddingSnapshot(vertices, vectors, len(vectors[0]))
        else:
            raise snapshot_error(
                "Cannot infer the dimension of an empty embedding"
            )

    def embed_node(self, node):
        if node in self._row_of_vertex:
            return self._matrix[self._row_of_vertex[node]]
        else:
            logger.error("Vertex %r is not in the embedding snapshot", node)
            raise KeyError(node)

    def embed_nodes(self, nodes) -> np.ndarray:
        try:
            rows = [self._row_of_vertex[node] for node in nodes]
        except KeyError as missing:
            logger.error(
                "Vertex %r is not in the embedding snapshot", missing.args[0]
            )
            raise
        return self._matrix[rows]

    def vertices(self) -> list:
        return list(self._row_of_vertex)

    def __contains__(self, node) -> bool:
        return node in self._row_of_vertex

    def __len__(self) -> int:
        return len(self._row_of_vertex)


class DatasetGenerator:
    def __init__(self, graph, embedding_model):
        self.nodes = list(graph.nodes())
        self.embeddings = embedding_model.embed_nodes(self.nodes)
        self.embeddings = np.array(self.embeddings)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.edges = list(graph.edges())
        self.graph = graph

    def generate_data(self, positive_edges, negative_edges):
        u_data, v_data, y_data = [], [], []
        for u, v in positive_edges:
            u_data.append(self.embeddings[self.node_to_idx[u]])
            v_data.append(self.embeddings[self.node_to_idx[v]])
            y_data.append(1)  # Positive edge
        for u, v in negative_edges:
            u_data.append(self.embeddings[self.node_to_idx[u]])
            v_data.append(self.embeddings[self.node_to_idx[v]])
            y_data.append(0)  # Negative edge
        return np.array(u_data), np.array(v_data), np.array(y_data)
    def generate_train_val_data(self, val_ratio=0.2):
        random.shuffle(self.edges)
        split_idx = int(len(self.edges) * (1 - val_ratio))
        train_edges = self.edges[:split_idx]
        val_edges = self.edges[split_idx:]
        u_train, v_train, y_train = [], [], []
        negative_edges = []
        for _ in range(len(train_edges)):
            u_neg, v_neg = random.sample(self.nodes, 2)
            while self.graph.has_edge(u_neg, v_neg):
                u_neg, v_neg = random.sample(self.nodes, 2)
            negative_edges.append((u_neg, v_neg))
        u_train, v_train, y_train = self.generate_data(train_edges, negative_edges)
        u_val, v_val, y_val = [], [], []
        negative_val_edges = []
        for _ in range(len(val_edges)):
            u_neg, v_neg = random.sample(self.nodes, 2)
            while self.graph.has_edge(u_neg, v_neg):
                u_neg, v_neg = random.sample(self.nodes, 2)
            negative_val_edges.append((u_neg, v_neg))
        u_val, v_val, y_val = self.generate_data(val_edges, negative_val_edges)
        return np.array(u_train), np.array(v_train), np.array(y_train), np.array(u_val), np.array(v_val), np.array(y_val)

def train_on_static_graph(graph, embedding_model, val_ratio=0.2, cv_k=10, model_trainer: LinkPredictionModelTrainer = None):
    model_trainer = model_trainer or LINK_PREDICTION_MODEL_TRAINERS["bilinear"]()
    best_model = None
    best_train_losses = []
    best_val_losses = []
    dataset_generator = DatasetGenerator(graph, embedding_model)
    for _ in range(cv_k):
        u_train, v_train, y_train, u_val, v_val, y_val = dataset_generator.generate_train_val_data(val_ratio=val_ratio)
        dataset = LinkPredictionDataset(
            train=LinkPredictionSplit(np.array(u_train), np.array(v_train), np.array(y_train)),
            val=LinkPredictionSplit(np.array(u_val), np.array(v_val), np.array(y_val)),
        )
        model, train_losses, val_losses = model_trainer.train(dataset)
        if len(best_val_losses) == 0 or val_losses[-1] < best_val_losses[-1]:
            best_model = model
            best_train_losses = train_losses
            best_val_losses = val_losses
    return best_model, best_train_losses, best_val_losses

def predict_links(node_pairs, embedding_model, link_prediction_model: LinkPredictionModel):
    """
    Predicts the existence of edges for given node pairs using the trained link prediction model and node embeddings.
    Args:
        node_pairs (list of tuples): List of node pairs (u, v) for which to predict edge existence.
        embedding_model: The model used to generate node embeddings.
        link_prediction_model: The trained LinkPredictionModel that takes node embeddings as input and outputs a probability.
    Returns:
        List of tuples: Each tuple contains (is_edge, probability) where is_edge is a boolean indicating predicted edge existence and probability is the confidence score for that prediction.
    """
    predictions = []
    for u, v in node_pairs:
        u_emb = embedding_model.embed_node(u)
        v_emb = embedding_model.embed_node(v)
        prob = link_prediction_model.predict_proba(u_emb, v_emb)
        is_edge = prob > 0.5
        predictions.append((is_edge, prob))
    return predictions

def generate_embedding_dict(graph, embedding_model):
    """
    Generates a dictionary mapping each node to its embedding vector using the provided embedding model.
    Args:
        embedding_model: The model used to generate node embeddings.
        graph: The graph containing the nodes for which embeddings are to be generated.
    Returns:
        dict: A dictionary where keys are node identifiers and values are their corresponding embedding vectors.
    """
    embedding_dict = {}
    for node in graph.nodes():
        embedding_dict[node] = embedding_model.embed_node(node)
    return embedding_dict

def ensemble_predict_links(node_pairs, embedding_dicts, link_prediction_models):
    """
    Predicts edge existence for given node pairs using an ensemble of embedding models and link prediction models.
    Args:
        node_pairs (list of tuples): List of node pairs (u, v) for which to predict edge existence.
        embedding_dicts (list): List of dictionaries containing node embeddings.

        link_prediction_models (list): List of trained link prediction models corresponding to each embedding model.
    Returns:
        List of tuples: Each tuple contains (is_edge, probability) where is_edge is a boolean indicating predicted edge existence and probability is the average confidence score across all models for that prediction.
    """
    predictions = []
    for u, v in node_pairs:
        probs = []
        u_emb = []
        v_emb = []
        for embedding_dict in embedding_dicts:
            if u in embedding_dict:
                u_emb.append(embedding_dict[u])
            if v in embedding_dict:
                v_emb.append(embedding_dict[v])
        if len(u_emb) == 0 or len(v_emb) == 0:
            predictions.append((False, 0.0))  # No embeddings available, predict no edge
            continue
        u_emb = np.mean(u_emb, axis=0)  # Average embeddings if multiple models provide them
        v_emb = np.mean(v_emb, axis=0)
        for link_prediction_model in link_prediction_models:
            probs.append(link_prediction_model.predict_proba(u_emb, v_emb))
        avg_prob = np.mean(probs)
        is_edge = avg_prob > 0.5
        predictions.append((is_edge, avg_prob))
    return predictions


def evaluate_predictions(embedding_model, model, positive_edges, negative_edges):
    """
    Evaluates the performance of link prediction by calculating precision, recall, F1 score, and accuracy.
    Args:
        model: The link prediction model to be evaluated.
        positive_edges (set): Set of true positive edges in the graph for comparison.
        negative_edges (set): Set of true negative edges in the graph for comparison.
    Returns:
        tuple: precision, recall, F1 score, and accuracy of the predictions.
    """
    TP = FP = TN = FN = 0
    test_edges = list(positive_edges) + list(negative_edges)
    predictions = predict_links(test_edges, embedding_model, model)
    for (u, v), (is_edge, prob) in zip(test_edges, predictions):
        if is_edge and (u, v) in positive_edges:
            TP += 1
        elif is_edge and (u, v) in negative_edges:
            FP += 1
        elif not is_edge and (u, v) in negative_edges:
            TN += 1
        elif not is_edge and (u, v) in positive_edges:
            FN += 1
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN > 0 else 0.0
    return precision, recall, f1_score, accuracy

class Ranks:
    def __init__(self, ranks = None):
        self.ranks = ranks if ranks is not None else []
    
    def mean_rank(self):
        return np.mean(self.ranks)
    def mrr(self):
        return np.mean([1.0 / rank for rank in self.ranks])
    def hits_at_k(self, k):
        return np.mean([1.0 if rank <= k else 0.0 for rank in self.ranks])
    def add_rank(self, rank):
        self.ranks.append(rank)
    def add_scores(self, scores):
        # target score is the first one
        target_score = scores[0]
        rank = 1 + sum(1 for score in scores[1:] if score >= target_score)
        self.add_rank(rank)

def heart_benchmark(embedding_model, model, g, positive_edges, ns=500, ps=1000):
    """
    Evaluates link prediction performance using the HEART benchmark.

    This function computes the ranks of true positive edges among a set of negative samples generated from the graph. It randomly selects a subset of positive edges for evaluation and generates negative samples by pairing nodes that do not have an edge in the graph. The predicted probabilities from the link prediction model are used to rank the true positive edge among these negative samples.

    Negative samples are generated by randomly selecting nodes from the graph and ensuring that they do not have an edge with the source node of the positive edge being evaluated. The function returns a Ranks object containing the ranks of the true positive edges.

    Args:
        embedding_model: The model used to generate node embeddings.
        model: The trained link prediction model that takes node embeddings as input and outputs logits.
        g: The graph from which positive and negative edges are derived.
        positive_edges (list of tuples): List of true positive edges in the graph for evaluation.
        ns (int): Number of negative samples to generate for each positive edge.
        ps (int): Number of positive edges to sample for evaluation.
    Note:
        g should not contain the positive edges being evaluated, as they are used for testing. The function generates negative samples by randomly pairing nodes that do not have an edge in the graph, and then ranks the true positive edge among these negative samples based on the predicted probabilities from the link prediction model.
    """
    assert ns > 0, "Number of negative samples (ns) must be greater than 0."
    assert ps > 0, "Number of positive samples (ps) must be greater than 0."
    assert len(positive_edges) > 0, "There must be at least one positive edge for evaluation."
    assert len(g.nodes()) > 0, "Graph must contain nodes for evaluation."
    assert len(g.edges()) > 0, "Graph must contain edges for evaluation."
    assert not any(g.has_edge(x, y) for x, y in positive_edges), "Graph should not contain the positive edges being evaluated."
    ranks = Ranks()
    # Limit to ps positive edges for evaluation
    sample = random.sample(positive_edges, min(len(positive_edges), ps))
    for x,y in sample:
        # Lets save g.has_edge method in order to avoid Python attribute lookup
        has_edge = g.has_edge
        # Generate negative samples
        negative_samples = []
        while len(negative_samples) < ns:
            u = random.choice(list(g.nodes()))
            # We do not check inside positive edges for performance reasons, 
            # but we check against the graph's edges to ensure we are 
            # generating true negative samples.
            # Probability that it is inside positive edges is low, and even if 
            # it is, it will just make the evaluation slightly more difficult, 
            # which is acceptable for this metric.
            if not has_edge(x, u):
                negative_samples.append((x, u))
        scores = predict_links([(x, y)] + negative_samples, embedding_model, model)
        scores = [prob for _, prob in scores]  # Extract probabilities
        ranks.add_scores(scores)
    return ranks