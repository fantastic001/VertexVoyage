import logging
from typing import Callable, Sequence, Tuple, TypeVar

from vertex_voyage.graph_sampling.interface import (
    AdjacencyGraph,
    Edge,
    GraphSampler,
    NoCandidatesError,
    Rng,
    SamplingError,
    V,
    logger,
    sampling_error,
)
from vertex_voyage.graph_sampling.uniform import UniformSampler

Drawn = TypeVar("Drawn")


class RoundRobinSampler(GraphSampler[V]):
    _failure_log_levels = {
        SamplingError: logging.ERROR,
        NoCandidatesError: logging.DEBUG,
    }

    def __init__(self, samplers: Sequence[GraphSampler]):
        if len(samplers) > 0:
            self._samplers: Tuple[GraphSampler, ...] = tuple(samplers)
        else:
            raise sampling_error("Round robin needs at least one sampler")
        self._cursor = 0

    def _draw_from_next_feasible_sampler(
        self,
        draw: Callable[[GraphSampler], Drawn],
    ) -> Drawn:
        sampler_count = len(self._samplers)
        failure_type = SamplingError
        for offset in range(sampler_count):
            index = (self._cursor + offset) % sampler_count
            try:
                drawn = draw(self._samplers[index])
            except NoCandidatesError as error:
                failure_type = NoCandidatesError
                logger.debug("Sampler %d has no candidates: %s", index, error)
            except SamplingError as error:
                logger.debug("Sampler %d skipped: %s", index, error)
            else:
                self._cursor = (index + 1) % sampler_count
                return drawn
        message = "No sampler in the rotation can serve the draw"
        logger.log(self._failure_log_levels[failure_type], message)
        raise failure_type(message)

    def node(self, rng: Rng) -> V:
        return self._draw_from_next_feasible_sampler(
            lambda sampler: sampler.node(rng)
        )

    def neighbor(self, rng: Rng, node: V) -> V:
        return self._draw_from_next_feasible_sampler(
            lambda sampler: sampler.neighbor(rng, node)
        )

    def non_neighbor(self, rng: Rng, node: V) -> V:
        return self._draw_from_next_feasible_sampler(
            lambda sampler: sampler.non_neighbor(rng, node)
        )

    def edge(self, rng: Rng) -> Edge:
        return self._draw_from_next_feasible_sampler(
            lambda sampler: sampler.edge(rng)
        )

    def non_edge(self, rng: Rng) -> Edge:
        return self._draw_from_next_feasible_sampler(
            lambda sampler: sampler.non_edge(rng)
        )


class RoundRobinUniform(RoundRobinSampler[V]):
    def __init__(self, *graphs: AdjacencyGraph):
        super().__init__([UniformSampler(graph) for graph in graphs])
