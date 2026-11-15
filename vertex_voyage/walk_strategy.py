import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Protocol, Sequence

from vertex_voyage.config import get_config_str

logger = logging.getLogger(__name__)

NODE2VEC_WALK = "node2vec"
RANDOM_VERTEX_SEQUENCE_WALK = "random_vertex_sequence"
REPEAT_START_VERTEX_WALK = "repeat_start_vertex"
SHORTEST_WALK_SIZE = 1


class WalkContext(Protocol):
    def vertices(self) -> Sequence[Any]:
        ...

    def next_vertex(self, previous: Any, current: Any) -> Any:
        ...


def require_walk_size(walk_size: int) -> None:
    if walk_size >= SHORTEST_WALK_SIZE:
        return
    else:
        logger.error("Walk size %r is below %d", walk_size, SHORTEST_WALK_SIZE)
        raise ValueError(
            f"Walk size must be at least {SHORTEST_WALK_SIZE}, "
            f"got {walk_size!r}"
        )


class WalkStrategy(ABC):
    @abstractmethod
    def walk_from(
        self,
        start: Any,
        walk_size: int,
        context: WalkContext,
    ) -> List[Any]:
        ...


class Node2VecWalk(WalkStrategy):
    def walk_from(
        self,
        start: Any,
        walk_size: int,
        context: WalkContext,
    ) -> List[Any]:
        require_walk_size(walk_size)
        walk = [start]
        previous = None
        current = start
        for _ in range(walk_size - 1):
            following = context.next_vertex(previous, current)
            walk.append(following)
            previous = current
            current = following
        return walk


class RandomVertexSequenceWalk(WalkStrategy):
    def walk_from(
        self,
        start: Any,
        walk_size: int,
        context: WalkContext,
    ) -> List[Any]:
        require_walk_size(walk_size)
        vertices = list(context.vertices())
        if len(vertices) > 0:
            return [start] + [
                random.choice(vertices) for _ in range(walk_size - 1)
            ]
        else:
            return [start] * walk_size


class RepeatStartVertexWalk(WalkStrategy):
    def walk_from(
        self,
        start: Any,
        walk_size: int,
        context: WalkContext,
    ) -> List[Any]:
        require_walk_size(walk_size)
        return [start] * walk_size


WALK_STRATEGY_OF_NAME: Dict[str, Callable[[], WalkStrategy]] = {
    NODE2VEC_WALK: Node2VecWalk,
    RANDOM_VERTEX_SEQUENCE_WALK: RandomVertexSequenceWalk,
    REPEAT_START_VERTEX_WALK: RepeatStartVertexWalk,
}


def walk_strategy_names() -> List[str]:
    return sorted(WALK_STRATEGY_OF_NAME)


def walk_strategy_of(name: str) -> WalkStrategy:
    if name in WALK_STRATEGY_OF_NAME:
        return WALK_STRATEGY_OF_NAME[name]()
    else:
        logger.error("Unknown walk strategy %r", name)
        raise ValueError(
            f"Unknown walk strategy {name!r}, expected one of: "
            f"{', '.join(walk_strategy_names())}"
        )


def configured_walk_strategy() -> WalkStrategy:
    return walk_strategy_of(
        get_config_str(
            "walk_strategy",
            NODE2VEC_WALK,
            "How a random walk of a partition is generated: "
            f"{', '.join(walk_strategy_names())}",
        )
    )
