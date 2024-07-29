
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots
  
def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in sequences:

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=num_ns,
          seed=seed
    )

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

def word2vec(training_data, vocab_size, embedding_dim, learning_rate, epochs, window_size, num_ns, seed):
    """
    Word2Vec implementation using a simple neural network with one hidden layer.

    Parameters:
    training_data: list of tuples (center_word, context_word, label)
    vocab_size: size of the vocabulary
    embedding_dim: dimensionality of word embeddings
    learning_rate: learning rate for gradient descent
    epochs: number of training epochs
    """
    walk_size = len(training_data[0])
    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=walk_size
    )
    training_data = [[str(word) for word in walk] for walk in training_data]
    # flatten training data 
    training_data = [word for walk in training_data for word in walk]
    vectorize_layer.adapt(training_data)
    text_vector_ds = vectorize_layer(training_data)
    vocab = vectorize_layer.get_vocabulary()
    vocab_size = len(vocab)

    word2vec = Word2Vec(vocab_size, embedding_dim)
    word2vec.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    targets, contexts, labels = generate_training_data(
        text_vector_ds, window_size=window_size, num_ns=num_ns, vocab_size=vocab_size, seed=seed)
    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    print(dataset)
    # Train model
    word2vec.fit(dataset, epochs=epochs)
    return word2vec.target_embedding.get_weights()[0]