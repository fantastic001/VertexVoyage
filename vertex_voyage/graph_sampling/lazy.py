from typing import Callable, List, Optional

from vertex_voyage.graph_sampling.interface import (
    Edge,
    GraphSampler,
    Rng,
    V,
    logger,
    sampling_error,
)

SamplerFactory = Callable[[], GraphSampler]


class LazySampler(GraphSampler[V]):
    def __init__(self, build_sampler: SamplerFactory):
        if callable(build_sampler):
            self._build_sampler = build_sampler
        else:
            raise sampling_error(
                f"Sampler factory {build_sampler!r} is not callable"
            )
        self._built_sampler: Optional[GraphSampler] = None

    def _sampler(self) -> GraphSampler:
        if self._built_sampler is None:
            logger.debug("Building lazily created sampler on first use")
            self._built_sampler = self._build_sampler()
        else:
            pass
        return self._built_sampler

    def node(self, rng: Rng) -> V:
        return self._sampler().node(rng)

    def neighbor(self, rng: Rng, node: V) -> V:
        return self._sampler().neighbor(rng, node)

    def non_neighbor(self, rng: Rng, node: V) -> V:
        return self._sampler().non_neighbor(rng, node)

    def edge(self, rng: Rng) -> Edge:
        return self._sampler().edge(rng)

    def non_edge(self, rng: Rng) -> Edge:
        return self._sampler().non_edge(rng)

    def walk(self, rng: Rng, start: V, length: int) -> List[V]:
        return self._sampler().walk(rng, start, length)
