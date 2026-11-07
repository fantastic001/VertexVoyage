import dataclasses
from abc import ABC, abstractmethod
from typing import Callable, Dict

import torch

from vertex_voyage.aggregation.interface import aggregation_error, logger
from vertex_voyage.aggregation.residual_aligner import (
    LEFT,
    RIGHT,
    AlignmentLoss,
    LossTerms,
)

UNIFORM_LINK_WEIGHTING = "uniform"
INVERSE_CROSS_FRACTION_LINK_WEIGHTING = "inverse_cross_fraction"
WITHIN_SCOPE = "within"
CROSS_SCOPE = "cross"


def cross_partition_mask_of(
    partitions: torch.Tensor,
    links: torch.Tensor,
) -> torch.Tensor:
    return partitions[links[:, LEFT]] != partitions[links[:, RIGHT]]


def cross_partition_fraction_of(
    partitions: torch.Tensor,
    links: torch.Tensor,
) -> float:
    if len(links) > 0:
        return float(cross_partition_mask_of(partitions, links).float().mean())
    else:
        return 0.0


def terms_of_scope(
    terms: LossTerms,
    scope: str,
    weight: float,
) -> LossTerms:
    return tuple(
        dataclasses.replace(
            term,
            name="%s %s" % (term.name, scope),
            weight=term.weight * weight,
        )
        for term in terms
    )


class AlignmentObjective(ABC):
    def __init__(self, loss: AlignmentLoss, margin: float):
        self.loss = loss
        self.margin = margin

    @abstractmethod
    def terms_of(
        self,
        out: torch.Tensor,
        partitions: torch.Tensor,
        links: torch.Tensor,
    ) -> LossTerms:
        ...


class UniformObjective(AlignmentObjective):
    def terms_of(
        self,
        out: torch.Tensor,
        partitions: torch.Tensor,
        links: torch.Tensor,
    ) -> LossTerms:
        return self.loss(out, links, self.margin)


class InverseCrossFractionObjective(AlignmentObjective):
    def terms_of(
        self,
        out: torch.Tensor,
        partitions: torch.Tensor,
        links: torch.Tensor,
    ) -> LossTerms:
        cross = cross_partition_mask_of(partitions, links)
        cross_count = int(cross.sum())
        if 0 < cross_count < len(links):
            return self._weighted_terms_of_scopes(out, links, cross)
        else:
            logger.debug(
                "%d of %d links cross partitions, weighting skipped",
                cross_count,
                len(links),
            )
            return self.loss(out, links, self.margin)

    def _weighted_terms_of_scopes(
        self,
        out: torch.Tensor,
        links: torch.Tensor,
        cross: torch.Tensor,
    ) -> LossTerms:
        link_count = len(links)
        cross_count = int(cross.sum())
        cross_fraction = cross_count / link_count
        within_weight = link_count - cross_count
        cross_weight = cross_count / cross_fraction
        total_weight = within_weight + cross_weight
        return terms_of_scope(
            self.loss(out, links[~cross], self.margin),
            WITHIN_SCOPE,
            within_weight / total_weight,
        ) + terms_of_scope(
            self.loss(out, links[cross], self.margin),
            CROSS_SCOPE,
            cross_weight / total_weight,
        )


OBJECTIVE_OF_LINK_WEIGHTING: Dict[
    str,
    Callable[[AlignmentLoss, float], AlignmentObjective],
] = {
    UNIFORM_LINK_WEIGHTING: UniformObjective,
    INVERSE_CROSS_FRACTION_LINK_WEIGHTING: InverseCrossFractionObjective,
}


def objective_of(
    link_weighting: str,
    loss: AlignmentLoss,
    margin: float,
) -> AlignmentObjective:
    if link_weighting in OBJECTIVE_OF_LINK_WEIGHTING:
        return OBJECTIVE_OF_LINK_WEIGHTING[link_weighting](loss, margin)
    else:
        raise aggregation_error(
            f"Unknown link weighting {link_weighting!r}, expected one of: "
            f"{', '.join(sorted(OBJECTIVE_OF_LINK_WEIGHTING))}"
        )
