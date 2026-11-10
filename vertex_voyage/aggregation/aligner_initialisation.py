from abc import ABC, abstractmethod
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch

from vertex_voyage.aggregation.interface import aggregation_error, logger
from vertex_voyage.aggregation.procrustes import (
    ProcrustesSettings,
    fitted_rotations_of,
)
from vertex_voyage.aggregation.replica_links import ReplicaLink, ReplicaTable
from vertex_voyage.aggregation.residual_aligner import ResidualAligner
from vertex_voyage.config import get_config_float

RANDOM_INITIALISATION = "random"
IDENTITY_INITIALISATION = "identity"
PROCRUSTES_INITIALISATION = "procrustes"
SIGNED_COPIES = 2
IDENTITY_ROTATION_TOLERANCE = get_config_float(
    "alignment_identity_rotation_tolerance",
    1e-9,
    "Largest entry of Q - I for which a fitted Procrustes rotation Q counts "
    "as the identity and gets cancelling random hidden units instead",
)

HiddenRows = Tuple[torch.Tensor, torch.Tensor]


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


def is_identity_offset(offset: torch.Tensor) -> bool:
    return float(offset.abs().max()) <= IDENTITY_ROTATION_TOLERANCE


def signed_offset_rows(offset: torch.Tensor) -> HiddenRows:
    return (
        torch.cat([offset, -offset]),
        torch.zeros(SIGNED_COPIES * len(offset)),
    )


def cancelling_random_rows(
    layer: torch.nn.Linear,
    dimension: int,
) -> HiddenRows:
    weight = layer.weight[:dimension].clone()
    bias = layer.bias[:dimension].clone()
    return torch.cat([weight, weight]), torch.cat([bias, bias])


def hidden_rows_of_rotation(
    layer: torch.nn.Linear,
    rotation: np.ndarray,
    dimension: int,
) -> HiddenRows:
    offset = torch.as_tensor(rotation, dtype=torch.float32)
    offset = (offset - torch.eye(dimension)).T
    if not is_identity_offset(offset):
        return signed_offset_rows(offset)
    else:
        return cancelling_random_rows(layer, dimension)


def load_rotation_into_layer(
    layer: torch.nn.Linear,
    rotation: np.ndarray,
    dimension: int,
) -> None:
    weight, bias = hidden_rows_of_rotation(layer, rotation, dimension)
    layer.weight[:SIGNED_COPIES * dimension] = weight
    layer.bias[:SIGNED_COPIES * dimension] = bias


def load_signed_identity_into(layer: torch.nn.Linear, dimension: int) -> None:
    identity = torch.eye(dimension)
    layer.weight.zero_()
    layer.bias.zero_()
    layer.weight[:, :dimension] = identity
    layer.weight[:, dimension:SIGNED_COPIES * dimension] = -identity


def identity_rotation_count(
    rotations: Sequence[np.ndarray],
    dimension: int,
) -> int:
    identity = np.eye(dimension)
    return sum(
        is_identity_offset(torch.as_tensor(rotation - identity))
        for rotation in rotations
    )


def load_rotations_into(
    model: ResidualAligner,
    rotations: Sequence[np.ndarray],
) -> None:
    require_room_for_linear_maps(model)
    if len(rotations) == model.partition_count:
        dimension = model.dimension
        with torch.no_grad():
            for layer, rotation in zip(model.first, rotations):
                load_rotation_into_layer(layer, rotation, dimension)
            load_signed_identity_into(model.second, dimension)
        logger.info(
            "%d of %d Procrustes rotations are the identity and start "
            "with cancelling random hidden units",
            identity_rotation_count(rotations, dimension),
            len(rotations),
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
