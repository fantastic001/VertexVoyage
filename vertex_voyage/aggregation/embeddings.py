from typing import Optional

import numpy as np

from vertex_voyage.aggregation.interface import (
    Embedding,
    PartitionModels,
    V,
    aggregation_error,
    logger,
)

NORM_FLOOR = 1e-12


def embedding_dimension_of(embedding_model: PartitionModels) -> int:
    if len(embedding_model) > 0:
        return next(iter(embedding_model.values())).dim
    else:
        raise aggregation_error(
            "Cannot aggregate without any embedding model"
        )


def zero_embedding_of_unplaced_vertex(
    embedding_model: PartitionModels,
    vertex: V,
) -> Embedding:
    logger.warning(
        "Vertex %r is in no partition, returning zero embedding",
        vertex,
    )
    return np.zeros(embedding_dimension_of(embedding_model))


def unit_vector_or_none(vector: Embedding) -> Optional[Embedding]:
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm > NORM_FLOOR:
        return values / norm
    else:
        return None
