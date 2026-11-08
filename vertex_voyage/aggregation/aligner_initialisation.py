from abc import ABC, abstractmethod
from typing import Callable, Dict, Sequence

import numpy as np
import torch

from vertex_voyage.aggregation.interface import aggregation_error, logger
from vertex_voyage.aggregation.procrustes import (
    ProcrustesSettings,
    fitted_rotations_of,
)
from vertex_voyage.aggregation.replica_links import ReplicaLink, ReplicaTable
from vertex_voyage.aggregation.residual_aligner import ResidualAligner

RANDOM_INITIALISATION = "random"
IDENTITY_INITIALISATION = "identity"
PROCRUSTES_INITIALISATION = "procrustes"
SIGNED_COPIES = 2


class AlignerInitialisation(ABC):
    @abstractmethod
    def initialise(
        self,
        model: ResidualAligner,
        table: ReplicaTable,
        links: Sequence[ReplicaLink],
    ) -> None:
        ...


class RandomInitialisation(AlignerInitialisation):
    def initialise(
        self,
        model: ResidualAligner,
        table: ReplicaTable,
        links: Sequence[ReplicaLink],
    ) -> None:
        logger.info("Aligner keeps its random initialisation")


def zero_residual_of(model: ResidualAligner) -> None:
    with torch.no_grad():
        model.second.weight.zero_()
        model.second.bias.zero_()


class IdentityInitialisation(AlignerInitialisation):
    def initialise(
        self,
        model: ResidualAligner,
        table: ReplicaTable,
        links: Sequence[ReplicaLink],
    ) -> None:
        zero_residual_of(model)
        logger.info("Aligner initialised as the identity")


def require_room_for_linear_maps(model: ResidualAligner) -> None:
    if model.hidden_width >= SIGNED_COPIES * model.dimension:
        return
    else:
        raise aggregation_error(
            f"Aligner hidden width {model.hidden_width} cannot hold linear "
            f"maps of dimension {model.dimension}, it needs at least "
            f"{SIGNED_COPIES * model.dimension}"
        )


def load_rotations_into(
    model: ResidualAligner,
    rotations: Sequence[np.ndarray],
) -> None:
    require_room_for_linear_maps(model)
    if len(rotations) == model.partition_count:
        dimension = model.dimension
        identity = torch.eye(dimension)
        with torch.no_grad():
            for layer, rotation in zip(model.first, rotations):
                offset = torch.as_tensor(rotation, dtype=torch.float32)
                offset = (offset - identity).T
                layer.weight.zero_()
                layer.bias.zero_()
                layer.weight[:dimension] = offset
                layer.weight[dimension:SIGNED_COPIES * dimension] = -offset
            model.second.weight.zero_()
            model.second.bias.zero_()
            model.second.weight[:, :dimension] = identity
            model.second.weight[:, dimension:SIGNED_COPIES * dimension] = (
                -identity
            )
    else:
        raise aggregation_error(
            f"Got {len(rotations)} rotations for an aligner of "
            f"{model.partition_count} partitions"
        )


class ProcrustesInitialisation(AlignerInitialisation):
    def __init__(self, sweeps: int):
        if sweeps > 0:
            self._sweeps = sweeps
        else:
            raise aggregation_error(
                f"Procrustes sweeps must be positive, got {sweeps!r}"
            )

    def initialise(
        self,
        model: ResidualAligner,
        table: ReplicaTable,
        links: Sequence[ReplicaLink],
    ) -> None:
        rotations = fitted_rotations_of(table, links, self._sweeps)
        if rotations is not None:
            load_rotations_into(model, rotations)
            logger.info(
                "Aligner initialised with Procrustes rotations of %d "
                "partitions",
                len(rotations),
            )
        else:
            logger.warning(
                "No anchors for Procrustes, aligner initialised as identity"
            )
            zero_residual_of(model)


INITIALISATION_OF_NAME: Dict[str, Callable[[], AlignerInitialisation]] = {
    RANDOM_INITIALISATION: RandomInitialisation,
    IDENTITY_INITIALISATION: IdentityInitialisation,
    PROCRUSTES_INITIALISATION: lambda: ProcrustesInitialisation(
        ProcrustesSettings().sweeps
    ),
}


def aligner_initialisation_of(name: str) -> AlignerInitialisation:
    if name in INITIALISATION_OF_NAME:
        return INITIALISATION_OF_NAME[name]()
    else:
        raise aggregation_error(
            f"Unknown aligner initialisation {name!r}, expected one of: "
            f"{', '.join(sorted(INITIALISATION_OF_NAME))}"
        )
