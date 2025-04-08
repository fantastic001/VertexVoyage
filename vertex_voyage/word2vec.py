
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, preprocessing
import random 
SEED = 42

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
      for target, context_word in positive_skip_grams:
        context_class = tf.expand_dims(
            tf.constant([context_word], dtype="int64"), 1)
        # context_class = tf.reshape(tf.constant(context_class, dtype="int64"), (1, 1))
        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,  # class that should be sampled as 'positive'
          num_true=1,  # each positive skip-gram has 1 positive context class
          num_sampled=num_ns,  # number of negative context words to sample
          unique=True,  # all the negative samples should be unique
          range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
          seed=SEED,  # seed for reproducibility
          name="negative_sampling"  # name of this operation
        )
        context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
        label = tf.constant([1] + [0]*num_ns, dtype="int64")
        targets.append(target)
        contexts.append(context)
        labels.append(label)
    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)
    print("targets shape", targets.shape)
    print("contexts shape", contexts.shape)
    print("labels shape", labels.shape)
    targets = tf.constant(targets, dtype=tf.int64)
    contexts = tf.constant(contexts, dtype=tf.int64)
    labels = tf.constant(labels, dtype=tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size=10000, seed=seed)
    return dataset




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


def generate_training(sequences, window_size, num_ns, vocab_size, seed):
  result = [] 
  for seq in sequences:
    training = generate_skip_grams(seq, window_size, num_ns, vocab_size, seed)
  return result

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
  training = generate_skip_grams(walks, 5, 5, vocab_size, SEED)
  # print(len(training))
  # for i in training:
  #   print(i)