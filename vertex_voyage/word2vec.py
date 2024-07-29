
import numpy as np



# Sigmoid function for binary classification
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def word2vec(training_data, vocab_size, embedding_dim, learning_rate, epochs):
    """
    Word2Vec implementation using a simple neural network with one hidden layer.

    Parameters:
    training_data: list of tuples (center_word, context_word, label)
    vocab_size: size of the vocabulary
    embedding_dim: dimensionality of word embeddings
    learning_rate: learning rate for gradient descent
    epochs: number of training epochs
    """
    # Initialize word vectors
    W_input = np.random.rand(vocab_size, embedding_dim)
    W_output = np.random.rand(vocab_size, embedding_dim)

    # Training loop
    for epoch in range(epochs):
        prev_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")
        progress = 0
        total_loss = 0
        for center_word, context_word, label in training_data:
            progress += 1
            # Forward pass
            h = W_input[center_word]  # Input word vector
            u = np.dot(W_output[context_word], h)  # Output layer input

            # Sigmoid activation for binary classification
            y_pred = sigmoid(u)

            # Loss calculation
            loss = -label * np.log(y_pred) - (1 - label) * np.log(1 - y_pred)
            total_loss += loss if not np.isnan(loss) and not np.isinf(loss) else 0
            avg_loss = total_loss / progress
            p = int(progress / len(training_data) * 40)
            print("Progress: " + "=" * p + " " * (40 - p) + " Loss: %f" % avg_loss, end="\r")

            # Backpropagation
            grad_u = y_pred - label 
            grad_W_output = np.outer(grad_u, h)
            grad_h = np.dot(W_output[context_word].T, grad_u) 

            # Update weights using gradient descent
            W_output[context_word] -= learning_rate * grad_u * h
            W_input[center_word] -= learning_rate * grad_h
            prev_loss = loss
        print()

    # Extract word vectors from W_input
    word_vectors = W_input
    return word_vectors