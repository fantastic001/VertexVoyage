import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Sequence

from vertex_voyage.config import get_config_str

logger = logging.getLogger(__name__)

TRAINED_WORD2VEC = "train"
RANDOM_INITIALISED_WORD2VEC = "random_init_only"
NO_CALLBACKS: list = []


def training_arguments_of_fresh_model(model: Any) -> dict:
    return {
        "total_examples": model.corpus_count,
        "total_words": model.corpus_total_words,
        "epochs": model.epochs,
        "start_alpha": model.alpha,
        "end_alpha": model.min_alpha,
        "compute_loss": model.compute_loss,
        "callbacks": NO_CALLBACKS,
    }


def training_arguments_of_updated_model(model: Any, epochs: int) -> dict:
    return {
        "total_examples": model.corpus_count,
        "epochs": epochs,
        "callbacks": NO_CALLBACKS,
    }


class Word2VecTraining(ABC):
    @abstractmethod
    def train(
        self,
        model: Any,
        walks: Sequence[Sequence[Any]],
        arguments: Mapping[str, Any],
    ) -> None:
        ...


class TrainedWord2Vec(Word2VecTraining):
    def train(
        self,
        model: Any,
        walks: Sequence[Sequence[Any]],
        arguments: Mapping[str, Any],
    ) -> None:
        logger.info(
            "Training word2vec on %d walks for %s epochs",
            len(walks),
            arguments.get("epochs"),
        )
        model.train(corpus_iterable=walks, **arguments)


class RandomInitialisedWord2Vec(Word2VecTraining):
    def train(
        self,
        model: Any,
        walks: Sequence[Sequence[Any]],
        arguments: Mapping[str, Any],
    ) -> None:
        logger.info(
            "Skipping word2vec training of %d walks, keeping the %d randomly "
            "initialised vectors",
            len(walks),
            len(model.wv),
        )


WORD2VEC_TRAINING_OF_NAME: Dict[str, Callable[[], Word2VecTraining]] = {
    TRAINED_WORD2VEC: TrainedWord2Vec,
    RANDOM_INITIALISED_WORD2VEC: RandomInitialisedWord2Vec,
}


def word2vec_training_names() -> List[str]:
    return sorted(WORD2VEC_TRAINING_OF_NAME)


def word2vec_training_of(name: str) -> Word2VecTraining:
    if name in WORD2VEC_TRAINING_OF_NAME:
        return WORD2VEC_TRAINING_OF_NAME[name]()
    else:
        logger.error("Unknown word2vec training %r", name)
        raise ValueError(
            f"Unknown word2vec training {name!r}, expected one of: "
            f"{', '.join(word2vec_training_names())}"
        )


def configured_word2vec_training() -> Word2VecTraining:
    return word2vec_training_of(
        get_config_str(
            "word2vec_training",
            TRAINED_WORD2VEC,
            "Whether the word2vec model of a partition is trained on its "
            f"walks or kept at its random initialisation: "
            f"{', '.join(word2vec_training_names())}",
        )
    )
