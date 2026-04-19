from random import seed

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import logging

# logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


if __name__ == "__main__":
    sentences = [
            ["cat", "say", "meow"], 
            ["dog", "say", "woof"],
            ["cat", "chase", "mouse"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["dog", "cat"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "dog"],
            ["cat", "like", "dog"],
            ["dog", "like", "cat"],
            ["what", "does", "the", "fox", "say"],
            ["the", "fox", "say", "what"],
            ["the", "dog", "say", "what"],
            ["the", "cat", "say", "what"],
            ["lion", "say", "roar"],
            ["tiger", "say", "grrr"],
            ["lion", "chase", "tiger"],
            ["tiger", "chase", "lion"],
            ["the", "lion", "chase", "the", "tiger"],
            ["the", "tiger", "chase", "the", "lion"],
            ["lion", "like", "tiger"],
            ["tiger", "like", "lion"],
            ["the", "lion", "like", "the", "tiger"],
            ["the", "tiger", "like", "the", "lion"],
            ["the", "cat", "like", "the", "dog"],
            ["the", "dog", "like", "the", "cat"],
            ["the", "lion"],
            ["the", "tiger"],
            ["the", "fox"],
            ["the", "mouse"],
            ["the", "cat"],
            ["the", "dog"],
            ["cat", "like", "dog"],
            ["cat", "like", "dog"],
            ["cat", "like", "dog"],
            ["cat", "like", "dog"],
            ["cat", "like", "dog"],



        ]
    training = [sentences[0]]
    model = Word2Vec(training, min_count=1, vector_size=2)
    # Visualize the word vectors
    vocab = set(word for sentence in training for word in sentence)
    for i in range(1, len(sentences)):
        training = [sentences[i]]
        for sent in training:
            for word in sent:
                if word not in vocab:
                    vocab.add(word)
        model.build_vocab(sentences[:i+1], update=True)
        # update the model with new sentences
        model.train(
            training, 
            total_words=len(vocab), 
            epochs=model.epochs
        )
        words = list(model.wv.index_to_key)
        vectors = [model.wv[word] for word in words]
        print("Cat: ", model.wv["cat"])
        print("Dog: ", model.wv["dog"])
        print()
        # plt.scatter(*zip(*vectors))
        # for word, vector in zip(words, vectors):
        #     plt.annotate(word, vector)
        # plt.title("Word2Vec Visualization")
        # plt.xlabel("Dimension 1")
        # plt.ylabel("Dimension 2")
        # plt.grid()
        # plt.show()
    cat_dog_similarities = [] 
    cat_meow_similarities = []
    cat_dog_distance_ranks = []
    total_vocabulary = set(word for sentence in sentences for word in sentence)
    for i in range(2, len(sentences)):
        training = sentences[:i]
        model = Word2Vec(
            sentences=training, 
            min_count=0,
            vector_size=10,
            null_word=None,
            window=5,
            sg=1,
            epochs=10,
            alpha=0.025,
        )
        print(f"{i}: Top 5 similar to cat: ", model.wv.most_similar("cat", topn=5))
        cat_dog_similarities.append(model.wv.similarity("cat", "dog"))
        cat_meow_similarities.append(model.wv.similarity("cat", "meow"))
        distances = [] 
        for word in total_vocabulary:
            if word in model.wv and "cat" in model.wv:
                distance = model.wv.distance("cat", word)
                distances.append((word, distance))
        distances.sort(key=lambda x: x[1])
        cat_dog_distance_ranks.append(next((rank for rank, (word, _) in enumerate(distances) if word == "dog"), None))
    plt.plot(range(2, len(sentences)), cat_dog_similarities, label="cat-dog")
    plt.plot(range(2, len(sentences)), cat_meow_similarities, label="cat-meow")
    plt.ylim(-1, 1)
    plt.title("Similarity over time")
    plt.xlabel("Number of sentences")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid()
    plt.show()
    plt.title("Distance rank of dog to cat over time")
    plt.xlabel("Number of sentences")
    plt.ylabel("Distance rank")
    plt.bar(range(2, len(sentences)), cat_dog_distance_ranks)
    plt.grid()
    plt.show()



    # now do this with retraining 
    cat_dog_similarities = [] 
    cat_meow_similarities = []
    cat_dog_distance_ranks = []
    say_meow_similarities = []
    say_meow_distance_ranks = []
    total_vocabulary = set(word for sentence in sentences for word in sentence)
    model = Word2Vec(
        min_count=0,
        vector_size=10,
        null_word=None,
        window=5,
        sg=1,
        epochs=10,
        alpha=0.025,
    )
    for i in range(0, len(sentences)):
        # This is nice test because in dynnode2vec we usually have a lot of sentences and only a few new ones at each step, so we can see how the model updates with new data without forgetting old data. 
        # In this scenario, we assume that sentences will be similar to one 
        # already seen by the model, so we expect the model to be able to learn from the new sentences without forgetting the old ones.
        # samples = sentences[i]
        # training = [
        #     t for t in sentences[:i+1] if any(word in t for word in samples)
        # ]
        training = [sentences[i]]
        # in dynnode2vec - training is newly generated walks from affected nodes (delta nodes = added nodes + nodes on edge ends)
        model.build_vocab(training, update= i > 0)
        model.train(
            corpus_iterable=training,
            total_examples=model.corpus_count,
            # total_words=len(total_vocabulary),
            epochs=model.epochs,
        )
        
        if "dog" not in model.wv or "cat" not in model.wv:
            cat_dog_similarities.append(0)
            cat_meow_similarities.append(0)
            say_meow_similarities.append(0)
            cat_dog_distance_ranks.append(0)
            say_meow_distance_ranks.append(0)
            continue

        print(f"{i}: Top 5 similar to cat: ", model.wv.most_similar("cat", topn=5))
        cat_dog_similarities.append(model.wv.similarity("cat", "dog"))
        cat_meow_similarities.append(model.wv.similarity("cat", "meow"))
        say_meow_similarities.append(model.wv.similarity("say", "meow"))
        distances = []
        say_distances = []
        for word in total_vocabulary:
            if word in model.wv and "cat" in model.wv:
                distance = model.wv.distance("cat", word)
                distances.append((word, distance))
            if word in model.wv and "say" in model.wv:
                distance = model.wv.distance("say", word)
                say_distances.append((word, distance))
        distances.sort(key=lambda x: x[1])
        say_distances.sort(key=lambda x: x[1])
        cat_dog_distance_ranks.append(next((rank for rank, (word, _) in enumerate(distances) if word == "dog"), None))
        say_meow_distance_ranks.append(next((rank for rank, (word, _) in enumerate(say_distances) if word == "meow"), None))
    plt.plot(range(0, len(sentences)), cat_dog_similarities, label="cat-dog")
    plt.plot(range(0, len(sentences)), cat_meow_similarities, label="cat-meow")
    plt.plot(range(0, len(sentences)), say_meow_similarities, label="say-meow")
    plt.ylim(-1, 1)
    plt.title("Similarity over time")
    plt.xlabel("Number of sentences")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid()
    plt.show()
    plt.title("Distance rank of dog to cat over time")
    plt.xlabel("Number of sentences")
    plt.ylabel("Distance rank")
    plt.bar(range(0, len(sentences)), cat_dog_distance_ranks)
    plt.grid()
    plt.show()
    plt.title("Distance rank of meow to say over time")
    plt.xlabel("Number of sentences")
    plt.ylabel("Distance rank")
    say_counts = [sum(1 for sentence in sentences[:i] for word in sentence if word == "say") for i in range(2, len(sentences))]
    intensities = [count / max(say_counts) for count in say_counts]
    plt.bar(range(0, len(sentences)), say_meow_distance_ranks, color=[(intensity, 1-intensity, 0, 1) for intensity in intensities])
    # plt.bar(range(2, len(sentences)), say_meow_distance_ranks)
    plt.grid()
    plt.show()



    print("Total vocabulary size: ", len(total_vocabulary))