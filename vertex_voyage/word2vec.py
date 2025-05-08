
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, preprocessing
import random 

def skipgrams(
    sequence,
    vocabulary_size,
    window_size=4,
    negative_samples=1.0,
    shuffle=True,
    categorical=False,
    sampling_table=None,
    seed=None,
):
    """Generates skipgram word pairs.

    DEPRECATED.

    This function transforms a sequence of word indexes (list of integers)
    into tuples of words of the form:

    - (word, word in the same window), with label 1 (positive samples).
    - (word, random word from the vocabulary), with label 0 (negative samples).

    Read more about Skipgram in this gnomic paper by Mikolov et al.:
    [Efficient Estimation of Word Representations in
    Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

    Args:
        sequence: A word sequence (sentence), encoded as a list
            of word indices (integers). If using a `sampling_table`,
            word indices are expected to match the rank
            of the words in a reference dataset (e.g. 10 would encode
            the 10-th most frequently occurring token).
            Note that index 0 is expected to be a non-word and will be skipped.
        vocabulary_size: Int, maximum possible word index + 1
        window_size: Int, size of sampling windows (technically half-window).
            The window of a word `w_i` will be
            `[i - window_size, i + window_size+1]`.
        negative_samples: Float >= 0. 0 for no negative (i.e. random) samples.
            1 for same number as positive samples.
        shuffle: Whether to shuffle the word couples before returning them.
        categorical: bool. if False, labels will be
            integers (eg. `[0, 1, 1 .. ]`),
            if `True`, labels will be categorical, e.g.
            `[[1,0],[0,1],[0,1] .. ]`.
        sampling_table: 1D array of size `vocabulary_size` where the entry i
            encodes the probability to sample a word of rank i.
        seed: Random seed.

    Returns:
        couples, labels: where `couples` are int pairs and
            `labels` are either 0 or 1.

    Note:
        By convention, index 0 in the vocabulary is
        a non-word and will be skipped.
    """
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0, 1])
                else:
                    labels.append(1)

    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [
            [words[i % len(words)], random.randint(1, vocabulary_size - 1)]
            for i in range(num_negative_samples)
        ]
        if categorical:
            labels += [[1, 0]] * num_negative_samples
        else:
            labels += [0] * num_negative_samples

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels

def generate_skip_grams(sequences, window_size, num_ns, vocab_size, seed):
    targets, contexts, labels = [], [], []
    for sequence in sequences:
      positive_skip_grams, _ = skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          window_size=window_size,
          negative_samples=0
      )
      if not positive_skip_grams:
            continue
      for target, context_word in positive_skip_grams:
        context_class = tf.expand_dims(
            tf.constant([context_word], dtype="int64"), 1)
        # context_class = tf.reshape(tf.constant(context_class, dtype="int64"), (1, 1))
        if vocab_size < 10 or num_ns > vocab_size:
            negative_sampling_candidates = [] 
        else:
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,  # class that should be sampled as 'positive'
                num_true=1,  # each positive skip-gram has 1 positive context class
                num_sampled=num_ns,  # number of negative context words to sample
                unique=True,  # all the negative samples should be unique
                range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
                seed=seed,  # seed for reproducibility
                name="negative_sampling"  # name of this operation
            )
        context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
        label = tf.constant([1] + [0]*len(negative_sampling_candidates), dtype="int64")
        targets.append(target)
        contexts.append(context)
        labels.append(label)
    if not targets:
        return None
    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)
    targets = tf.constant(targets, dtype=tf.int64)
    contexts = tf.constant(contexts, dtype=tf.int64)
    labels = tf.constant(labels, dtype=tf.int64)
    buffer_size = int(0.1 * len(targets))
    batch_size = min(1024, buffer_size // 2)
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size=10000, seed=seed).batch(1024, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

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
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots

def train_word2vec_model(training_data, vocab_size, embedding_dim, learning_rate, epochs, epoch_callbacks):
    """
    Train a Word2Vec model using the skip-gram approach.

    Parameters:
    training_data: tf.data.Dataset object containing (target, context) pairs
    vocab_size: size of the vocabulary
    embedding_dim: dimensionality of word embeddings
    learning_rate: learning rate for optimizer
    epochs: number of training epochs

    Returns:
    model: trained Word2Vec model
    """
    model = Word2Vec(vocab_size, embedding_dim)
    loss_fn = loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss_fn)
    
    model.fit(training_data, epochs=epochs, verbose=0, callbacks=[CustomCallback(epoch_callbacks)])
    
    return model




def __legacy_word2vec(training_data, vocab_size, embedding_dim, learning_rate, epochs, window_size, num_ns, seed):
    """
    DEPRECATED 

    Word2Vec implementation using a simple neural network with one hidden layer.

    Parameters:
    training_data: list of tuples (center_word, context_word, label)
    vocab_size: size of the vocabulary
    embedding_dim: dimensionality of word embeddings
    learning_rate: learning rate for gradient descent
    epochs: number of training epochs
    """
    import gensim
    m = gensim.models.Word2Vec(
       training_data, 
       vector_size=embedding_dim, 
       window=window_size, 
       negative=num_ns, 
       min_count=1, 
       sg=1,
       alpha=learning_rate,
       max_vocab_size=None,
       epochs=epochs,
       seed=seed  
    )
    return m.wv

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_callbacks):
        super(CustomCallback, self).__init__()
        self.epoch_callbacks = epoch_callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.epoch_callbacks:
            callback(epoch, logs, self.model)

def word2vec(training_data, vocab_size, embedding_dim, learning_rate, epochs, window_size, num_ns, seed = None, epoch_callbacks = None):
    """
    Word2Vec implementation using a simple neural network with one hidden layer.

    Parameters:
    training_data: list of tuples (center_word, context_word, label)
    vocab_size: size of the vocabulary
    embedding_dim: dimensionality of word embeddings
    learning_rate: learning rate for gradient descent
    epochs: number of training epochs
    """
    if epoch_callbacks is None:
        epoch_callbacks = []
    
    
    if seed is None:
        seed = random.randint(0, 10e6)
    walks = training_data
    if min(min(walk) for walk in walks) == 0:
        walks = [[w + 1 for w in walk] for walk in walks]
    training = generate_skip_grams(walks, window_size, num_ns, vocab_size+1, seed)
    # if training is None, then return zero weights
    if training is None:
        return np.zeros((vocab_size+1, embedding_dim))
    model = train_word2vec_model(training, vocab_size+1, embedding_dim, learning_rate, epochs, epoch_callbacks)
    weights = model.get_layer('w2v_embedding').get_weights()[0]
    return weights[1:]  # Skip the first row (non-word)

if __name__ == "__main__":
  from vertex_voyage.node2vec import Node2Vec
  import networkx as nx
  G = nx.erdos_renyi_graph(100, 0.05)
  G = nx.relabel_nodes(G, {i: i for i in G.nodes()})
  n = Node2Vec(dim=64, walk_size=10, n_walks=5, p=0.5, q=1, negative_sample_num=5, window_size=5)
  n.nodes = list(G.nodes())
  n.node_to_neightbours_map = {i: list(G.neighbors(i)) for i in G.nodes()}
  n.is_weighted = False
  n.G = G
  walks = n._random_walks()
  vocab_size = len(n.nodes)
  w = word2vec(
      walks,
      vocab_size,
      embedding_dim=64,
      learning_rate=0.01,
      epochs=10,
      window_size=5,
      num_ns=5,
  )