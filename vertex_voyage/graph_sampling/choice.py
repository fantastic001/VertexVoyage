from typing import Sequence, TypeVar

import numpy as np

from vertex_voyage.graph_sampling.interface import Rng, sampling_error

Item = TypeVar("Item")

NO_WEIGHT = 0.0


def choose_uniformly(rng: Rng, items: Sequence[Item]) -> Item:
    if len(items) > 0:
        return items[int(rng.integers(len(items)))]
    else:
        raise sampling_error("Cannot choose from an empty sequence")


def choose_weighted(
    rng: Rng,
    items: Sequence[Item],
    weights: Sequence[float],
) -> Item:
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    total_weight = float(weight_array.sum())
    has_matching_lengths = len(items) == len(weight_array)
    if has_matching_lengths and np.isfinite(total_weight) and total_weight > 0:
        try:
            index = rng.choice(len(items), p=weight_array / total_weight)
        except ValueError as error:
            message = f"Invalid sampling weights: {error}"
            raise sampling_error(message) from error
        else:
            return items[int(index)]
    else:
        raise sampling_error(
            f"Cannot draw from {len(items)} items with "
            f"{len(weight_array)} weights totalling {total_weight!r}"
        )
