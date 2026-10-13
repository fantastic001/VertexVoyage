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


class CumulativeDistribution:
    def __init__(self, weights: Sequence[float]):
        weight_array = np.asarray(weights, dtype=float).reshape(-1)
        self._require_valid_weights(weight_array)
        self._positive_indices = np.flatnonzero(weight_array)
        self._cumulative_weights = np.cumsum(weight_array)

    @staticmethod
    def _require_valid_weights(weight_array: np.ndarray) -> None:
        all_finite = bool(np.all(np.isfinite(weight_array)))
        if all_finite and bool(np.all(weight_array >= NO_WEIGHT)):
            return
        else:
            raise sampling_error(
                "Sampling weights must be finite and non-negative"
            )

    @property
    def total_weight(self) -> float:
        if len(self._positive_indices) > 0:
            return float(self._cumulative_weights[-1])
        else:
            return NO_WEIGHT

    def draw_index(self, rng: Rng) -> int:
        if self.total_weight > NO_WEIGHT:
            threshold = rng.random() * self.total_weight
            index = int(
                np.searchsorted(
                    self._cumulative_weights,
                    threshold,
                    side="right",
                )
            )
            return min(index, int(self._positive_indices[-1]))
        else:
            raise sampling_error(
                "Cannot draw from a distribution with zero total weight"
            )


def choose_weighted(
    rng: Rng,
    items: Sequence[Item],
    weights: Sequence[float],
) -> Item:
    if len(items) == len(weights):
        return items[CumulativeDistribution(weights).draw_index(rng)]
    else:
        raise sampling_error(
            f"Got {len(items)} items but {len(weights)} weights"
        )
