import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random 

# Model without explicit Sigmoid
class HadamardLogitsNet(nn.Module):
    def __init__(self, vector_dim, use_bias = True):
        super(HadamardLogitsNet, self).__init__()
        self.fc = nn.Linear(vector_dim, 1, bias=use_bias)
        
    def forward(self, u, v):
        hadamard = u * v
        return self.fc(hadamard) # Outputs "logits"

    criterion = nn.BCEWithLogitsLoss()

class QuadraticLogitsNet(nn.Module):
    def __init__(self, vector_dim, use_bias = True):
        super(QuadraticLogitsNet, self).__init__()
        self.fc = nn.Linear(vector_dim, vector_dim, bias=use_bias)
        
    def forward(self, u, v):
        # x.T A y
        u_transformed = self.fc(u)
        logits = torch.sum(u_transformed * v, dim=1)
        return logits

    criterion = nn.BCEWithLogitsLoss()

def train_model(model, u_train, v_train, y_train, u_val, v_val, y_val, epochs=10, batch_size=32, learning_rate=0.1):
    val_losses = [] 
    train_losses = []
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_dataset = TensorDataset(torch.tensor(u_train, dtype=torch.float32), 
                                  torch.tensor(v_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(
            torch.tensor(u_val, dtype=torch.float32), 
            torch.tensor(v_val, dtype=torch.float32), 
            torch.tensor(y_val, dtype=torch.float32)), 
        batch_size=batch_size,
        shuffle=False
    )
    for epoch in range(epochs):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for u_batch, v_batch, y_batch in train_loader:
                logits = model(u_batch, v_batch).squeeze()
                loss = model.criterion(logits, y_batch)
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
        # Validation
        with torch.no_grad():
            val_loss = 0
            for u_batch, v_batch, y_batch in val_loader:
                logits = model(u_batch, v_batch).squeeze()
                loss = model.criterion(logits, y_batch)
                current_val_loss = loss.item()
                val_loss += current_val_loss
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
        # Update
        model.train()
        for u_batch, v_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(u_batch, v_batch).squeeze()
            loss = model.criterion(logits, y_batch)
            current_loss = loss.item()
            loss.backward()
            optimizer.step()
            total_loss += current_loss
    # Calculate one final validation loss after training
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for u_batch, v_batch, y_batch in train_loader:
            logits = model(u_batch, v_batch).squeeze()
            loss = model.criterion(logits, y_batch)
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
    # Validation
    with torch.no_grad():
        val_loss = 0
        for u_batch, v_batch, y_batch in val_loader:
            logits = model(u_batch, v_batch).squeeze()
            loss = model.criterion(logits, y_batch)
            current_val_loss = loss.item()
            val_loss += current_val_loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
    return model, train_losses, val_losses


def train_on_static_graph(graph, embedding_model, val_ratio=0.2, epochs=60, batch_size=32, learning_rate=0.1, use_bias=False, cv_k=10, model_class = QuadraticLogitsNet):
    nodes = list(graph.nodes())
    embeddings = embedding_model.embed_nodes(nodes)
    embeddings = np.array(embeddings)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edges = list(graph.edges())
    best_model = None
    best_train_losses = []
    best_val_losses = []
    for _ in range(cv_k):
        random.shuffle(edges)
        split_idx = int(len(edges) * (1 - val_ratio))
        train_edges = edges[:split_idx]
        val_edges = edges[split_idx:]
        u_train, v_train, y_train = [], [], []
        for u, v in train_edges:
            u_train.append(embeddings[node_to_idx[u]])
            v_train.append(embeddings[node_to_idx[v]])
            y_train.append(1)  # Positive edge
        for _ in range(len(train_edges)):
            u_neg, v_neg = random.sample(nodes, 2)
            while graph.has_edge(u_neg, v_neg):
                u_neg, v_neg = random.sample(nodes, 2)
            u_train.append(embeddings[node_to_idx[u_neg]])
            v_train.append(embeddings[node_to_idx[v_neg]])
            y_train.append(0)  # Negative edge
        u_val, v_val, y_val = [], [], []
        for u, v in val_edges:
            u_val.append(embeddings[node_to_idx[u]])
            v_val.append(embeddings[node_to_idx[v]])
            y_val.append(1)  # Positive edge
        for _ in range(len(val_edges)):
            u_neg, v_neg = random.sample(nodes, 2)
            while graph.has_edge(u_neg, v_neg):
                u_neg, v_neg = random.sample(nodes, 2)
            u_val.append(embeddings[node_to_idx[u_neg]])
            v_val.append(embeddings[node_to_idx[v_neg]])
            y_val.append(0)  # Negative edge
        model, train_losses, val_losses = train_model(model_class(
                vector_dim=embeddings.shape[1],
                use_bias=use_bias
            ), 
            np.array(u_train), np.array(v_train), np.array(y_train), 
            np.array(u_val), np.array(v_val), np.array(y_val), 
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
        )
        if len(best_val_losses) == 0 or val_losses[-1] < best_val_losses[-1]:
            best_model = model
            best_train_losses = train_losses
            best_val_losses = val_losses
    return best_model, best_train_losses, best_val_losses

def predict_links(node_pairs, embedding_model, link_prediction_model):
    """
    Predicts the existence of edges for given node pairs using the trained link prediction model and node embeddings.
    Args:
        node_pairs (list of tuples): List of node pairs (u, v) for which to predict edge existence.
        embedding_model: The model used to generate node embeddings.
        link_prediction_model: The trained link prediction model that takes node embeddings as input and outputs logits.
    Returns:
        List of tuples: Each tuple contains (is_edge, probability) where is_edge is a boolean indicating predicted edge existence and probability is the confidence score for that prediction.
    """
    predictions = []
    for u, v in node_pairs:
        u_emb = embedding_model.embed_node(u)
        v_emb = embedding_model.embed_node(v)
        with torch.no_grad():
            logit = link_prediction_model(
                torch.tensor(u_emb, dtype=torch.float32).unsqueeze(0), 
                torch.tensor(v_emb, dtype=torch.float32).unsqueeze(0)
            ).item()
            prob = 1 / (1 + np.exp(-logit))  # Sigmoid
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
        for i in range(len(link_prediction_models)):
            with torch.no_grad():
                logit = link_prediction_models[i](
                    torch.tensor(u_emb, dtype=torch.float32).unsqueeze(0), 
                    torch.tensor(v_emb, dtype=torch.float32).unsqueeze(0)
                ).item()
                prob = 1 / (1 + np.exp(-logit))  # Sigmoid
                probs.append(prob)
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

def hits_k_score(embedding_model, model, g, positive_edges, k=3, ns=500):
    """
    Evaluates link prediction performance using the Hits@K metric, which measures the proportion of true positive edges that are ranked in the top K predictions among a set of negative samples.

    Args:
        embedding_model: The model used to generate node embeddings.
        model: The link prediction model to be evaluated.
        g: The graph containing the nodes and edges.
        positive_edges (set): Set of true positive edges for testing.
        k (int): The number of top predictions to consider for Hits@K.
        ns (int): The number of negative samples to generate for each positive edge.

    Returns:
        float: The Hits@K score, representing the proportion of true positive edges that are ranked in the top K predictions.
    
    Note:
        g should not contain the positive edges being evaluated, as they are used for testing. The function generates negative samples by randomly pairing nodes that do not have an edge in the graph, and then ranks the true positive edge among these negative samples based on the predicted probabilities from the link prediction model.
    """
    hits = 0
    for x,y in positive_edges:
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
        topk = np.argpartition(scores, -k)[-k:]  # Get indices of top K scores
        if 0 in topk:  # Check if the true positive edge (index 0) is in the top K predictions
            hits += 1
    return hits / len(positive_edges) if positive_edges else 0.0
