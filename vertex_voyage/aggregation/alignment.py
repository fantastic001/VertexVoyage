from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from vertex_voyage.aggregation.interface import (
    Embedding,
    PartitionKey,
    V,
    aggregation_error,
    logger,
)
from vertex_voyage.aggregation.replica_alignment import (
    DEFAULT_LINK_MEMORY,
    ReplicaAlignmentAggregator,
)
from vertex_voyage.aggregation.replica_links import (
    PartitionIndex,
    ReplicaTable,
    RowPair,
)
from vertex_voyage.aggregation.residual_aligner import (
    AlignerWeights,
    AlignmentLoss,
    ResidualAligner,
    contrastive_loss,
    triplet_loss,
    weights_of,
)
from vertex_voyage.config import get_config_float, get_config_int
from vertex_voyage.graph_sampling import GraphSampler, SamplingError
from vertex_voyage.seeding import MAX_SEED_EXCLUSIVE

CONTRASTIVE_MARGIN = get_config_float(
    "alignment_contrastive_margin",
    1.0,
    "Distance below which the contrastive loss repels negative replicas",
)
TRIPLET_MARGIN = get_config_float(
    "alignment_triplet_margin",
    0.5,
    "Distance by which the triplet loss keeps negatives behind positives",
)
LARGEST_USEFUL_MARGIN = 2.0


@dataclass(frozen=True)
class AlignmentSettings:
    hidden_width_factor: int = get_config_int(
        "alignment_hidden_width_factor",
        2,
        "Hidden width of the aligner as a multiple of the embedding size",
    )
    train_steps: int = get_config_int(
        "alignment_train_steps",
        100,
        "Optimiser steps the aligner takes after every buffer",
    )
    learning_rate: float = get_config_float(
        "alignment_learning_rate",
        0.01,
        "Learning rate of the aligner optimiser",
    )
    link_memory: int = DEFAULT_LINK_MEMORY
    negative_pool: int = get_config_int(
        "alignment_negative_pool",
        256,
        "Replicas drawn from the sampler as extra negatives per buffer",
    )


def require_useful_margin(margin: float) -> None:
    if 0 < margin <= LARGEST_USEFUL_MARGIN:
        return
    else:
        raise aggregation_error(
            f"Alignment margin must be in (0, {LARGEST_USEFUL_MARGIN}], "
            f"got {margin!r}"
        )


class AlignmentAggregator(ReplicaAlignmentAggregator[V, PartitionKey]):
    def __init__(
        self,
        loss: AlignmentLoss,
        margin: float,
        settings: Optional[AlignmentSettings] = None,
        seed: Optional[int] = None,
    ):
        require_useful_margin(margin)
        self._loss = loss
        self._margin = margin
        self._settings = AlignmentSettings() if settings is None else settings
        self._rng = np.random.default_rng(
            np.random.randint(MAX_SEED_EXCLUSIVE) if seed is None else seed
        )
        super().__init__(self._settings.link_memory)
        self._model: Optional[ResidualAligner] = None
        self._optimiser: Optional[torch.optim.Optimizer] = None
        self._weights: Optional[AlignerWeights] = None

    def _fit_alignment(
        self,
        table: ReplicaTable,
        sampler: GraphSampler,
        partition_index: PartitionIndex,
    ) -> None:
        row_pairs = table.rows_of_links(self._links)
        self._add_sampled_negative_replicas(table, sampler, partition_index)
        if len(row_pairs) > 0:
            self._train_aligner_on_links(table, row_pairs)
        else:
            logger.debug("No embedded replica links yet, aligner not trained")

    def _add_sampled_negative_replicas(
        self,
        table: ReplicaTable,
        sampler: GraphSampler,
        partition_index: PartitionIndex,
    ) -> None:
        try:
            for _ in range(self._settings.negative_pool):
                vertex = sampler.node(self._rng)
                partitions = sorted(partition_index.indexes_holding(vertex))
                self._add_replica_in_random_partition(
                    table,
                    vertex,
                    partitions,
                )
        except SamplingError as error:
            logger.debug("Sampler gave no further negatives: %s", error)

    def _add_replica_in_random_partition(
        self,
        table: ReplicaTable,
        vertex: V,
        partitions: List[int],
    ) -> None:
        if len(partitions) > 0:
            chosen = partitions[int(self._rng.integers(len(partitions)))]
            row = table.row_of((chosen, vertex))
            logger.debug("Negative replica %r got row %r", vertex, row)
        else:
            logger.debug("Sampled vertex %r is in no partition", vertex)

    def _train_aligner_on_links(
        self,
        table: ReplicaTable,
        row_pairs: List[RowPair],
    ) -> None:
        vectors = torch.from_numpy(table.vectors())
        partitions = torch.from_numpy(table.partitions())
        links = torch.tensor(row_pairs, dtype=torch.long)
        model = self._aligner_of_shape(
            table.partition_count(),
            vectors.shape[1],
        )
        loss_value = float("nan")
        for _ in range(self._settings.train_steps):
            self._optimiser.zero_grad()
            loss = self._loss(model(vectors, partitions), links, self._margin)
            loss.backward()
            self._optimiser.step()
            loss_value = float(loss.detach())
        self._weights = weights_of(model)
        logger.info(
            "Aligner trained on %d links over %d replicas, loss %f",
            len(row_pairs),
            len(table),
            loss_value,
        )

    def _aligner_of_shape(
        self,
        partition_count: int,
        dimension: int,
    ) -> ResidualAligner:
        if self._model is not None and self._model.has_shape(
            partition_count,
            dimension,
        ):
            return self._model
        else:
            logger.info(
                "Creating aligner for %d partitions of dimension %d",
                partition_count,
                dimension,
            )
            self._model = ResidualAligner(
                partition_count,
                dimension,
                self._settings.hidden_width_factor * dimension,
            )
            self._optimiser = torch.optim.Adam(
                self._model.parameters(),
                lr=self._settings.learning_rate,
            )
            return self._model

    def _aligned_replica(self, partition: int, unit: Embedding) -> Embedding:
        if self._weights is None:
            return unit
        elif partition < len(self._weights.first):
            return self._weights.align(partition, unit)
        else:
            return unit


class ContrastiveAlignmentAggregator(AlignmentAggregator[V, PartitionKey]):
    def __init__(
        self,
        settings: Optional[AlignmentSettings] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(contrastive_loss, CONTRASTIVE_MARGIN, settings, seed)


class TripletAlignmentAggregator(AlignmentAggregator[V, PartitionKey]):
    def __init__(
        self,
        settings: Optional[AlignmentSettings] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(triplet_loss, TRIPLET_MARGIN, settings, seed)
