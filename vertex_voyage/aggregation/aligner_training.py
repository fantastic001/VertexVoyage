import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import torch

from vertex_voyage.aggregation.alignment_objective import AlignmentObjective
from vertex_voyage.aggregation.interface import aggregation_error, logger
from vertex_voyage.aggregation.replica_links import ReplicaLink
from vertex_voyage.aggregation.residual_aligner import (
    LossTerm,
    LossTerms,
    loss_report_of,
)

TOTAL_GRADIENT_NAME = "total"
VANISHING_MARKER = " (vanishing)"
HASH_RANGE = 2 ** 32
VERTEX_OF_REPLICA = 1
ROW_PAIR_WIDTH = 2
NO_VANISHING_EPOCHS = 0

LinkSplit = Tuple[List[ReplicaLink], List[ReplicaLink]]
Parameters = Sequence[torch.nn.Parameter]
Gradients = Tuple[torch.Tensor, ...]


class ValidationMode(Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"


VALIDATION_MODE_OF_FLAG: Dict[bool, ValidationMode] = {
    True: ValidationMode.ENABLED,
    False: ValidationMode.DISABLED,
}


class LinkSplitter(ABC):
    @abstractmethod
    def split(self, links: Iterable[ReplicaLink]) -> LinkSplit:
        ...


class TrainingOnlySplitter(LinkSplitter):
    def split(self, links: Iterable[ReplicaLink]) -> LinkSplit:
        return list(links), []


class EdgeHashSplitter(LinkSplitter):
    def __init__(self, validation_fraction: float):
        if 0 < validation_fraction < 1:
            self._validation_fraction = validation_fraction
        else:
            raise aggregation_error(
                "Validation fraction must be in (0, 1), "
                f"got {validation_fraction!r}"
            )

    def split(self, links: Iterable[ReplicaLink]) -> LinkSplit:
        training: List[ReplicaLink] = []
        validation: List[ReplicaLink] = []
        for link in links:
            if self._holds_out_edge_of(link):
                validation.append(link)
            else:
                training.append(link)
        logger.debug(
            "Split links into %d training and %d validation",
            len(training),
            len(validation),
        )
        return training, validation

    def _holds_out_edge_of(self, link: ReplicaLink) -> bool:
        left, right = link
        edge = (left[VERTEX_OF_REPLICA], right[VERTEX_OF_REPLICA])
        edge_hash = zlib.crc32(repr(edge).encode())
        return edge_hash / HASH_RANGE < self._validation_fraction


SPLITTER_OF_MODE: Dict[ValidationMode, Callable[[float], LinkSplitter]] = {
    ValidationMode.DISABLED: lambda fraction: TrainingOnlySplitter(),
    ValidationMode.ENABLED: EdgeHashSplitter,
}


def link_splitter_of(mode: ValidationMode, fraction: float) -> LinkSplitter:
    if mode in SPLITTER_OF_MODE:
        return SPLITTER_OF_MODE[mode](fraction)
    else:
        raise aggregation_error(f"Unknown validation mode {mode!r}")


def row_pair_tensor_of(row_pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(row_pairs, dtype=torch.long).reshape(
        -1,
        ROW_PAIR_WIDTH,
    )


def norm_of_gradients(gradients: Gradients) -> float:
    if len(gradients) > 0:
        return float(
            torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g) for g in gradients])
            )
        )
    else:
        return 0.0


@dataclass(frozen=True)
class TermGradient:
    name: str
    gradients: Gradients

    def norm(self) -> float:
        return norm_of_gradients(self.gradients)


def term_gradient_of(term: LossTerm, parameters: Parameters) -> TermGradient:
    weighted_mean = term.weighted_mean()
    if weighted_mean.requires_grad:
        gradients = torch.autograd.grad(
            weighted_mean,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
    else:
        gradients = tuple(None for _ in parameters)
    return TermGradient(
        term.name,
        tuple(
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, gradients)
        ),
    )


class GradientReport:
    def __init__(
        self,
        term_gradients: Tuple[TermGradient, ...],
        vanishing_norm: float,
    ):
        self.term_gradients = term_gradients
        self.vanishing_norm = vanishing_norm
        self.total = tuple(
            torch.stack(per_parameter).sum(dim=0)
            for per_parameter in zip(
                *(term.gradients for term in term_gradients)
            )
        )

    def total_norm(self) -> float:
        return norm_of_gradients(self.total)

    def norms(self) -> Dict[str, float]:
        norms = {term.name: term.norm() for term in self.term_gradients}
        norms[TOTAL_GRADIENT_NAME] = self.total_norm()
        return norms

    def total_vanishes(self) -> bool:
        return self.total_norm() < self.vanishing_norm

    def vanishing_names(self) -> List[str]:
        return [
            name
            for name, norm in self.norms().items()
            if norm < self.vanishing_norm
        ]

    def report(self) -> str:
        vanishing = set(self.vanishing_names())
        return "gradient norm %s" % "; ".join(
            "%s %e%s" % (
                name,
                norm,
                VANISHING_MARKER if name in vanishing else "",
            )
            for name, norm in self.norms().items()
        )

    def apply_to(self, parameters: Parameters) -> None:
        for parameter, gradient in zip(parameters, self.total):
            parameter.grad = gradient.detach()


@dataclass(frozen=True)
class EpochOutcome:
    terms: LossTerms
    gradients: GradientReport
    validation_terms: LossTerms

    def report(self) -> str:
        return "%s, %s, validation %s" % (
            loss_report_of(self.terms),
            self.gradients.report(),
            loss_report_of(self.validation_terms),
        )


class AlignerTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        objective: AlignmentObjective,
        vanishing_norm: float,
    ):
        self._model = model
        self._optimiser = optimiser
        self._objective = objective
        self._vanishing_norm = vanishing_norm
        self._parameters = list(model.parameters())

    def run_epoch(
        self,
        vectors: torch.Tensor,
        partitions: torch.Tensor,
        training_links: torch.Tensor,
        validation_links: torch.Tensor,
    ) -> EpochOutcome:
        if len(training_links) > 0:
            terms, gradients = self._descend_on_links(
                vectors,
                partitions,
                training_links,
            )
            return EpochOutcome(
                terms,
                gradients,
                self.validation_terms_of(
                    vectors,
                    partitions,
                    validation_links,
                ),
            )
        else:
            raise aggregation_error("Aligner epoch needs training links")

    def _descend_on_links(
        self,
        vectors: torch.Tensor,
        partitions: torch.Tensor,
        links: torch.Tensor,
    ) -> Tuple[LossTerms, GradientReport]:
        self._optimiser.zero_grad()
        terms = self._objective.terms_of(
            self._model(vectors, partitions),
            partitions,
            links,
        )
        gradients = GradientReport(
            tuple(term_gradient_of(term, self._parameters) for term in terms),
            self._vanishing_norm,
        )
        gradients.apply_to(self._parameters)
        self._optimiser.step()
        return terms, gradients

    def validation_terms_of(
        self,
        vectors: torch.Tensor,
        partitions: torch.Tensor,
        links: torch.Tensor,
    ) -> LossTerms:
        if len(links) > 0:
            with torch.no_grad():
                return self._objective.terms_of(
                    self._model(vectors, partitions),
                    partitions,
                    links,
                )
        else:
            logger.debug("No validation links, validation loss skipped")
            return ()


class TrainingHistory:
    def __init__(self) -> None:
        self.epoch_count = 0
        self.vanishing_epoch_count = NO_VANISHING_EPOCHS
        self.last_outcome: Optional[EpochOutcome] = None

    def record(self, outcome: EpochOutcome) -> None:
        self.epoch_count += 1
        self.last_outcome = outcome
        if outcome.gradients.total_vanishes():
            self.vanishing_epoch_count += 1
        else:
            logger.debug("Gradient norm %e", outcome.gradients.total_norm())

    def log_vanishing_summary(self) -> None:
        if self.vanishing_epoch_count > NO_VANISHING_EPOCHS:
            logger.warning(
                "Vanishing total gradient in %d of %d aligner epochs",
                self.vanishing_epoch_count,
                self.epoch_count,
            )
        else:
            logger.debug(
                "No vanishing gradient over %d epochs",
                self.epoch_count,
            )

    def report(self) -> str:
        if self.last_outcome is not None:
            return self.last_outcome.report()
        else:
            return loss_report_of(())

