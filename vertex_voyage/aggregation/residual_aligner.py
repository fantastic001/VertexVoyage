from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch

from vertex_voyage.aggregation.embeddings import unit_vector_or_none
from vertex_voyage.aggregation.interface import Embedding

DISTANCE_EPSILON = 1e-12
NO_PENALTY = 0.0
LEFT = 0
RIGHT = 1
ATTRACTION_TERM = "attraction"
REPULSION_TERM = "repulsion"
RANKING_TERM = "ranking"
LOSS_NOT_COMPUTED = "loss not computed"


@dataclass(frozen=True)
class LossTerm:
    name: str
    mean: torch.Tensor
    pair_count: int

    def mean_value(self) -> float:
        return float(self.mean.detach())

    def summed_value(self) -> float:
        return self.mean_value() * self.pair_count

    def report(self) -> str:
        return "%s mean %f sum %f over %d pairs" % (
            self.name,
            self.mean_value(),
            self.summed_value(),
            self.pair_count,
        )


LossTerms = Tuple[LossTerm, ...]
AlignmentLoss = Callable[[torch.Tensor, torch.Tensor, float], LossTerms]


def mean_loss_of(terms: LossTerms) -> torch.Tensor:
    return torch.stack([term.mean for term in terms]).sum()


def summed_loss_of(terms: LossTerms) -> float:
    return sum(term.summed_value() for term in terms)


def loss_report_of(terms: LossTerms) -> str:
    if len(terms) > 0:
        return "loss %f, summed loss %f (%s)" % (
            float(mean_loss_of(terms).detach()),
            summed_loss_of(terms),
            "; ".join(term.report() for term in terms),
        )
    else:
        return LOSS_NOT_COMPUTED


class ResidualAligner(torch.nn.Module):
    def __init__(
        self,
        partition_count: int,
        dimension: int,
        hidden_width: int,
    ) -> None:
        super().__init__()
        self.partition_count = partition_count
        self.dimension = dimension
        self.hidden_width = hidden_width
        self.first = torch.nn.ModuleList(
            torch.nn.Linear(dimension, hidden_width)
            for _ in range(partition_count)
        )
        self.second = torch.nn.Linear(hidden_width, dimension)

    def forward(
        self,
        vectors: torch.Tensor,
        partitions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = vectors.new_zeros((len(vectors), self.hidden_width))
        for partition, layer in enumerate(self.first):
            rows = partitions == partition
            hidden[rows] = layer(vectors[rows])
        moved = vectors + self.second(torch.relu(hidden))
        return torch.nn.functional.normalize(moved, dim=-1)

    def has_shape(self, partition_count: int, dimension: int) -> bool:
        same_partitions = self.partition_count == partition_count
        return same_partitions and self.dimension == dimension


def pair_distance(out: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    difference = out[pairs[:, LEFT]] - out[pairs[:, RIGHT]]
    return torch.sqrt((difference ** 2).sum(dim=1) + DISTANCE_EPSILON)


def random_rows(out: torch.Tensor, count: int) -> torch.Tensor:
    return torch.randint(len(out), (count,))


def contrastive_loss(
    out: torch.Tensor,
    links: torch.Tensor,
    margin: float,
) -> LossTerms:
    negatives = torch.stack(
        [random_rows(out, len(links)), random_rows(out, len(links))],
        dim=1,
    )
    hinge = torch.clamp(margin - pair_distance(out, negatives), min=NO_PENALTY)
    return (
        LossTerm(
            ATTRACTION_TERM,
            (pair_distance(out, links) ** 2).mean(),
            len(links),
        ),
        LossTerm(REPULSION_TERM, (hinge ** 2).mean(), len(negatives)),
    )


def triplet_loss(
    out: torch.Tensor,
    links: torch.Tensor,
    margin: float,
) -> LossTerms:
    both = torch.cat([links, links.flip(dims=(1,))])
    negatives = torch.stack(
        [both[:, LEFT], random_rows(out, len(both))],
        dim=1,
    )
    gap = pair_distance(out, both) - pair_distance(out, negatives)
    return (
        LossTerm(
            RANKING_TERM,
            torch.clamp(gap + margin, min=NO_PENALTY).mean(),
            len(both),
        ),
    )


@dataclass(frozen=True)
class AlignerWeights:
    first: Tuple[Tuple[np.ndarray, np.ndarray], ...]
    second: Tuple[np.ndarray, np.ndarray]

    def align(self, partition: int, unit: Embedding) -> Embedding:
        first_weight, first_bias = self.first[partition]
        second_weight, second_bias = self.second
        hidden = np.maximum(first_weight @ unit + first_bias, NO_PENALTY)
        moved = unit + second_weight @ hidden + second_bias
        aligned = unit_vector_or_none(moved)
        if aligned is None:
            return unit
        else:
            return aligned


def as_numpy(parameter: torch.Tensor) -> np.ndarray:
    return parameter.detach().numpy().astype(float)


def weights_of(model: ResidualAligner) -> AlignerWeights:
    return AlignerWeights(
        first=tuple(
            (as_numpy(layer.weight), as_numpy(layer.bias))
            for layer in model.first
        ),
        second=(as_numpy(model.second.weight), as_numpy(model.second.bias)),
    )
