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
    DEFAULT_OUTPUT_SCALE,
    REPLICA_NORM_OUTPUT_SCALE,
    UNIT_OUTPUT_SCALE,
    resolved_output_scale,
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
    LossTerms,
    ResidualAligner,
    contrastive_loss,
    loss_report_of,
    mean_loss_of,
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
EPOCH_LOSS_LOG_INTERVAL = get_config_int(
    "alignment_epoch_loss_log_interval",
    1,
    "Aligner epochs between two loss log lines, 0 silences them",
)
FIRST_EPOCH = 1


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
    output_scale: str = DEFAULT_OUTPUT_SCALE
    negative_pool: int = get_config_int(
        "alignment_negative_pool",
        256,
        "Replicas drawn from the sampler as extra negatives per buffer",
    )


def is_logged_epoch(epoch: int) -> bool:
    if EPOCH_LOSS_LOG_INTERVAL > 0:
        return epoch % EPOCH_LOSS_LOG_INTERVAL == 0
    else:
        return False


def log_loss_of_epoch(epoch: int, epoch_count: int, terms: LossTerms) -> None:
    if is_logged_epoch(epoch):
        logger.info(
            "Aligner epoch %d/%d, %s",
            epoch,
            epoch_count,
            loss_report_of(terms),
        )
    else:
        logger.debug("Aligner epoch %d/%d not logged", epoch, epoch_count)


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
        output_scale: str = REPLICA_NORM_OUTPUT_SCALE,
    ):
        require_useful_margin(margin)
        self._loss = loss
        self._margin = margin
        self._settings = AlignmentSettings() if settings is None else settings
        self._rng = np.random.default_rng(
            np.random.randint(MAX_SEED_EXCLUSIVE) if seed is None else seed
        )
        super().__init__(
            self._settings.link_memory,
            resolved_output_scale(self._settings.output_scale, output_scale),
        )
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
        terms: LossTerms = ()
        epoch_count = self._settings.train_steps
        for epoch in range(FIRST_EPOCH, epoch_count + FIRST_EPOCH):
            self._optimiser.zero_grad()
            terms = self._loss(model(vectors, partitions), links, self._margin)
            mean_loss_of(terms).backward()
            self._optimiser.step()
            log_loss_of_epoch(epoch, epoch_count, terms)
        self._weights = weights_of(model)
        logger.info(
            "Aligner trained on %d links over %d replicas, %s",
            len(row_pairs),
            len(table),
            loss_report_of(terms),
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
        super().__init__(
            contrastive_loss,
            CONTRASTIVE_MARGIN,
            settings,
            seed,
            REPLICA_NORM_OUTPUT_SCALE,
        )


class TripletAlignmentAggregator(AlignmentAggregator[V, PartitionKey]):
    def __init__(
        self,
        settings: Optional[AlignmentSettings] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            triplet_loss,
            TRIPLET_MARGIN,
            settings,
            seed,
            UNIT_OUTPUT_SCALE,
        )
