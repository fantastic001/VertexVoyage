from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from vertex_voyage.aggregation.interface import Embedding

NO_PENALTY = 0.0
LEFT = 0
RIGHT = 1
ATTRACTION_TERM = "attraction"
REPULSION_TERM = "repulsion"
RANKING_TERM = "ranking"
LOSS_NOT_COMPUTED = "loss not computed"
UNWEIGHTED = 1.0


@dataclass(frozen=True)
class LossTerm:
    name: str
    mean: torch.Tensor
    pair_count: int
    weight: float = UNWEIGHTED

    def weighted_mean(self) -> torch.Tensor:
        return self.mean * self.weight

    def mean_value(self) -> float:
        return float(self.mean.detach())

    def summed_value(self) -> float:
        return self.mean_value() * self.pair_count

    def report(self) -> str:
        unweighted_report = "%s mean %f sum %f over %d pairs" % (
            self.name,
            self.mean_value(),
            self.summed_value(),
            self.pair_count,
        )
        if self.weight == UNWEIGHTED:
            return unweighted_report
        else:
            return "%s weight %f" % (unweighted_report, self.weight)


LossTerms = Tuple[LossTerm, ...]
AlignmentLoss = Callable[[torch.Tensor, torch.Tensor, float], LossTerms]


def mean_loss_of(terms: LossTerms) -> torch.Tensor:
    return torch.stack([term.weighted_mean() for term in terms]).sum()


def loss_report_of(terms: LossTerms) -> str:
    if len(terms) > 0:
        return "loss %f, summed loss %f (%s)" % (
            float(mean_loss_of(terms).detach()),
            sum(term.summed_value() for term in terms),
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

    def align(self, partition: int, unit: Embedding) -> Embedding:
        with torch.no_grad():
            aligned = self(
                torch.as_tensor(unit, dtype=torch.float32).reshape(1, -1),
                torch.tensor([partition]),
            )
        return aligned[0].numpy().astype(float)


def pair_distance(out: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pairwise_distance(
        out[pairs[:, LEFT]],
        out[pairs[:, RIGHT]],
    )


def contrastive_loss(
    out: torch.Tensor,
    links: torch.Tensor,
    margin: float,
) -> LossTerms:
    negatives = torch.randint(len(out), (len(links), 2))
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
    return (
        LossTerm(
            RANKING_TERM,
            torch.nn.functional.triplet_margin_loss(
                out[both[:, LEFT]],
                out[both[:, RIGHT]],
                out[torch.randint(len(out), (len(both),))],
                margin=margin,
            ),
            len(both),
        ),
    )
