import logging
import random
from collections.abc import Callable

import numpy as np

from vertex_voyage.config import get_config_int

logger = logging.getLogger("Seeding")

DEFAULT_SEED = get_config_int("default_seed", 42, "Seed used by seeded commands when none is given")
MAX_SEED_EXCLUSIVE = 2 ** 32


def _seed_builtin_random(seed: int) -> None:
    random.seed(seed)


def _seed_numpy(seed: int) -> None:
    np.random.seed(seed)


def _seed_torch(seed: int) -> None:
    try:
        import torch
    except ImportError:
        logger.info("torch is not installed, skipping torch seeding")
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _seed_tensorflow(seed: int) -> None:
    try:
        import tensorflow as tf
    except ImportError:
        logger.info("tensorflow is not installed, skipping tensorflow seeding")
        return
    tf.random.set_seed(seed)


SEEDERS: tuple[Callable[[int], None], ...] = (
    _seed_builtin_random,
    _seed_numpy,
    _seed_torch,
    _seed_tensorflow,
)


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or not 0 <= seed < MAX_SEED_EXCLUSIVE:
        logger.error(f"Invalid seed {seed!r}, expected an integer in [0, {MAX_SEED_EXCLUSIVE})")
        raise ValueError(f"Invalid seed {seed!r}, expected an integer in [0, {MAX_SEED_EXCLUSIVE}).")


def seed_shared_random_state(seed: int) -> None:
    _validate_seed(seed)
    logger.info(f"Seeding shared random state with seed {seed}")
    for seeder in SEEDERS:
        seeder(seed)
    logger.info(f"Shared random state seeded with seed {seed}")
