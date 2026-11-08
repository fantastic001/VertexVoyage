from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from vertex_voyage.aggregation.aligner_configuration import (
    AlignerConfiguration,
)
from vertex_voyage.aggregation.aligner_initialisation import (
    RANDOM_INITIALISATION,
    aligner_initialisation_of,
)
from vertex_voyage.aggregation.aligner_training import (
    VALIDATION_MODE_OF_FLAG,
    AlignerTrainer,
    EpochOutcome,
    TrainingHistory,
    ValidationMode,
    link_splitter_of,
    row_pair_tensor_of,
)
from vertex_voyage.aggregation.alignment_objective import (
    INVERSE_CROSS_FRACTION_LINK_WEIGHTING,
    cross_partition_fraction_of,
    objective_of,
)
from vertex_voyage.aggregation.interface import (
    Embedding,
    PartitionKey,
    V,
    aggregation_error,
    logger,
)
from vertex_voyage.aggregation.replica_alignment import (
    REPLICA_NORM_OUTPUT_SCALE,
    UNIT_OUTPUT_SCALE,
    ReplicaAlignmentAggregator,
    ReplicaAlignmentSettings,
)
from vertex_voyage.aggregation.replica_links import (
    PartitionIndex,
    ReplicaLink,
    ReplicaTable,
    RowPair,
)
from vertex_voyage.aggregation.residual_aligner import (
    AlignmentLoss,
    ResidualAligner,
    contrastive_loss,
    triplet_loss,
)
from vertex_voyage.config import (
    get_config_bool,
    get_config_float,
    get_config_int,
    get_config_str,
)
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
class AlignmentSettings(ReplicaAlignmentSettings):
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
    negative_pool: int = get_config_int(
        "alignment_negative_pool",
        256,
        "Replicas drawn from the sampler as extra negatives per buffer",
    )
    validation: ValidationMode = VALIDATION_MODE_OF_FLAG[
        get_config_bool(
            "alignment_validation",
            False,
            "Hold out edges as validation links and log their loss per epoch",
        )
    ]
    validation_fraction: float = get_config_float(
        "alignment_validation_fraction",
        0.2,
        "Fraction of edges held out for validation when it is enabled",
    )
    vanishing_gradient_norm: float = get_config_float(
        "alignment_vanishing_gradient_norm",
        1e-7,
        "Gradient norm below which the aligner gradient counts as vanishing",
    )
    link_weighting: str = get_config_str(
        "alignment_link_weighting",
        INVERSE_CROSS_FRACTION_LINK_WEIGHTING,
        "Weighting of positive links: uniform, or inverse_cross_fraction "
        "(cross-partition links weighted by 1/delta, delta being their share)",
    )
    initialisation: str = get_config_str(
        "alignment_initialisation",
        RANDOM_INITIALISATION,
        "Starting parameters of a new aligner: random, identity (zero "
        "residual) or procrustes (per-partition rotations fitted on anchors)",
    )


def log_epoch_outcome(
    epoch: int,
    epoch_count: int,
    outcome: EpochOutcome,
) -> None:
    logging_enabled = EPOCH_LOSS_LOG_INTERVAL > 0
    if logging_enabled and epoch % EPOCH_LOSS_LOG_INTERVAL == 0:
        logger.info(
            "Aligner epoch %d/%d, %s",
            epoch,
            epoch_count,
            outcome.report(),
        )
        log_vanishing_gradient_of_epoch(epoch, outcome)
    else:
        logger.debug("Aligner epoch %d/%d not logged", epoch, epoch_count)


def log_vanishing_gradient_of_epoch(epoch: int, outcome: EpochOutcome) -> None:
    if outcome.gradients.total_vanishes():
        logger.warning(
            "Vanishing aligner gradient at epoch %d, below %e: %s",
            epoch,
            outcome.gradients.vanishing_norm,
            ", ".join(outcome.gradients.vanishing_names()),
        )
    else:
        logger.debug("Aligner gradient at epoch %d is not vanishing", epoch)


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
        self._settings = AlignmentSettings() if settings is None else settings
        self._objective = objective_of(
            self._settings.link_weighting,
            loss,
            margin,
        )
        self._initialisation = aligner_initialisation_of(
            self._settings.initialisation
        )
        self._splitter = link_splitter_of(
            self._settings.validation,
            self._settings.validation_fraction,
        )
        if seed is None:
            self._seed = int(np.random.randint(MAX_SEED_EXCLUSIVE))
        else:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        super().__init__(self._settings, output_scale)
        logger.info("Aligner configuration: %s", self.configuration().report())
        self._model: Optional[ResidualAligner] = None
        self._optimiser: Optional[torch.optim.Optimizer] = None

    def configuration(self) -> AlignerConfiguration:
        return AlignerConfiguration(
            self._objective,
            self._output_scale_name,
            self._seed,
            self._settings,
            EPOCH_LOSS_LOG_INTERVAL,
        )

    def _fit_alignment(
        self,
        table: ReplicaTable,
        sampler: GraphSampler,
        partition_index: PartitionIndex,
    ) -> None:
        training_links, validation_links = self._splitter.split(self._links)
        training_pairs = table.rows_of_links(training_links)
        validation_pairs = table.rows_of_links(validation_links)
        self._add_sampled_negative_replicas(table, sampler, partition_index)
        if len(training_pairs) > 0:
            self._train_aligner_on_links(
                table,
                training_links,
                training_pairs,
                validation_pairs,
            )
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
            chosen = int(self._rng.choice(partitions))
            row = table.row_of((chosen, vertex))
            logger.debug("Negative replica %r got row %r", vertex, row)
        else:
            logger.debug("Sampled vertex %r is in no partition", vertex)

    def _train_aligner_on_links(
        self,
        table: ReplicaTable,
        training_replica_links: List[ReplicaLink],
        training_pairs: List[RowPair],
        validation_pairs: List[RowPair],
    ) -> None:
        vectors = torch.from_numpy(table.vectors())
        partitions = torch.from_numpy(table.partitions())
        training_links = row_pair_tensor_of(training_pairs)
        validation_links = row_pair_tensor_of(validation_pairs)
        logger.info(
            "Aligner link weighting %s, cross-partition fraction %f",
            self._settings.link_weighting,
            cross_partition_fraction_of(partitions, training_links),
        )
        trainer = AlignerTrainer(
            self._aligner_for_table(table, training_replica_links),
            self._optimiser,
            self._objective,
            self._settings.vanishing_gradient_norm,
        )
        history = TrainingHistory()
        epoch_count = self._settings.train_steps
        for epoch in range(FIRST_EPOCH, epoch_count + FIRST_EPOCH):
            outcome = trainer.run_epoch(
                vectors,
                partitions,
                training_links,
                validation_links,
            )
            history.record(outcome)
            log_epoch_outcome(epoch, epoch_count, outcome)
        logger.info(
            "Aligner trained on %d links, validated on %d, "
            "over %d replicas, %s",
            len(training_pairs),
            len(validation_pairs),
            len(table),
            history.report(),
        )
        history.log_vanishing_summary()

    def _aligner_for_table(
        self,
        table: ReplicaTable,
        links: List[ReplicaLink],
    ) -> ResidualAligner:
        partition_count = table.partition_count()
        dimension = table.vectors().shape[1]
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
            self._initialisation.initialise(self._model, table, links)
            self._optimiser = torch.optim.Adam(
                self._model.parameters(),
                lr=self._settings.learning_rate,
            )
            return self._model

    def _aligned_replica(self, partition: int, unit: Embedding) -> Embedding:
        if self._model is not None and partition < self._model.partition_count:
            return self._model.align(partition, unit)
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
