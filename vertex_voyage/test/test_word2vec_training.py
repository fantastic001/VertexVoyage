import json
import os
import tempfile
import unittest
from unittest import mock

from vertex_voyage.word2vec_training import (
    RANDOM_INITIALISED_WORD2VEC,
    TRAINED_WORD2VEC,
    RandomInitialisedWord2Vec,
    TrainedWord2Vec,
    configured_word2vec_training,
    training_arguments_of_fresh_model,
    training_arguments_of_updated_model,
    word2vec_training_names,
    word2vec_training_of,
)

EPOCHS = 3
CONFIG_ENVVAR = "VERTEX_VOYAGE_CONFIG"
TRAINING_ENVVAR = "VERTEX_VOYAGE_WORD2VEC_TRAINING"
WALKS = [[0, 1, 2], [2, 1, 0]]


class RecordingWord2Vec:
    def __init__(self):
        self.corpus_count = len(WALKS)
        self.corpus_total_words = sum(len(walk) for walk in WALKS)
        self.epochs = EPOCHS
        self.alpha = 0.025
        self.min_alpha = 0.0001
        self.compute_loss = False
        self.wv = {}
        self.training_calls = []

    def train(self, corpus_iterable, **arguments):
        self.training_calls.append(
            dict(arguments, walks=list(corpus_iterable))
        )


def empty_config_environment() -> dict:
    handle, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(handle, "w") as config_file:
        json.dump({}, config_file)
    return {CONFIG_ENVVAR: path}


class TestTrainingArguments(unittest.TestCase):
    def setUp(self):
        self.model = RecordingWord2Vec()

    def test_fresh_model_is_trained_as_gensim_would(self):
        self.assertEqual(
            training_arguments_of_fresh_model(self.model),
            {
                "total_examples": len(WALKS),
                "total_words": self.model.corpus_total_words,
                "epochs": EPOCHS,
                "start_alpha": self.model.alpha,
                "end_alpha": self.model.min_alpha,
                "compute_loss": False,
                "callbacks": [],
            },
        )

    def test_updated_model_is_trained_for_the_given_epochs(self):
        self.assertEqual(
            training_arguments_of_updated_model(self.model, 2),
            {
                "total_examples": len(WALKS),
                "epochs": 2,
                "callbacks": [],
            },
        )


class TestTrainedWord2Vec(unittest.TestCase):
    def test_training_forwards_the_walks_and_every_argument(self):
        model = RecordingWord2Vec()
        arguments = training_arguments_of_fresh_model(model)
        TrainedWord2Vec().train(model, WALKS, arguments)
        self.assertEqual(
            model.training_calls,
            [dict(arguments, walks=WALKS)],
        )


class TestRandomInitialisedWord2Vec(unittest.TestCase):
    def test_model_is_left_untrained(self):
        model = RecordingWord2Vec()
        RandomInitialisedWord2Vec().train(
            model,
            WALKS,
            training_arguments_of_fresh_model(model),
        )
        self.assertEqual(model.training_calls, [])


class TestWord2VecTrainingRegistry(unittest.TestCase):
    def test_every_name_creates_its_training(self):
        self.assertEqual(
            word2vec_training_names(),
            [RANDOM_INITIALISED_WORD2VEC, TRAINED_WORD2VEC],
        )
        self.assertIsInstance(
            word2vec_training_of(TRAINED_WORD2VEC),
            TrainedWord2Vec,
        )
        self.assertIsInstance(
            word2vec_training_of(RANDOM_INITIALISED_WORD2VEC),
            RandomInitialisedWord2Vec,
        )

    def test_unknown_training_raises(self):
        with self.assertRaises(ValueError):
            word2vec_training_of("missing")

    def test_configured_training_defaults_to_training(self):
        with mock.patch.dict(
            os.environ,
            empty_config_environment(),
            clear=True,
        ):
            self.assertIsInstance(
                configured_word2vec_training(),
                TrainedWord2Vec,
            )

    def test_configured_training_is_read_from_the_environment(self):
        environment = empty_config_environment()
        environment[TRAINING_ENVVAR] = RANDOM_INITIALISED_WORD2VEC
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertIsInstance(
                configured_word2vec_training(),
                RandomInitialisedWord2Vec,
            )


if __name__ == "__main__":
    unittest.main()
