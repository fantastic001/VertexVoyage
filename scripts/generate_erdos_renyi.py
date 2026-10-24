#!/usr/bin/env python3
import argparse
import logging
import sys
from typing import Iterable, TextIO

import networkx as nx

logger = logging.getLogger(__name__)

EVENT_FIELD_SEPARATOR = " "
MINIMUM_NODE_COUNT = 1
MINIMUM_PROBABILITY = 0.0
MAXIMUM_PROBABILITY = 1.0
EXIT_CODE_BROKEN_PIPE = 1


def parse_node_count(value: str) -> int:
    node_count = int(value)
    if node_count >= MINIMUM_NODE_COUNT:
        return node_count
    else:
        raise argparse.ArgumentTypeError(
            f"number of nodes must be >= {MINIMUM_NODE_COUNT}, got {value}"
        )


def parse_probability(value: str) -> float:
    probability = float(value)
    if MINIMUM_PROBABILITY <= probability <= MAXIMUM_PROBABILITY:
        return probability
    else:
        raise argparse.ArgumentTypeError(
            f"probability must be in [{MINIMUM_PROBABILITY}, "
            f"{MAXIMUM_PROBABILITY}], got {value}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an Erdos-Renyi G(n, p) graph as a space separated "
            "event sequence (src dest timestamp) readable by "
            "FileEventSequence"
        )
    )
    parser.add_argument("nodes", type=parse_node_count)
    parser.add_argument("probability", type=parse_probability)
    return parser.parse_args()


def generate_erdos_renyi_edges(
    node_count: int,
    probability: float,
) -> Iterable[tuple[int, int]]:
    assert node_count >= MINIMUM_NODE_COUNT
    assert MINIMUM_PROBABILITY <= probability <= MAXIMUM_PROBABILITY
    graph = nx.fast_gnp_random_graph(node_count, probability)
    return graph.edges()


def write_edges_as_event_sequence(
    edges: Iterable[tuple[int, int]],
    output: TextIO,
) -> int:
    written_event_count = 0
    for timestamp, (src, dest) in enumerate(edges):
        output.write(
            EVENT_FIELD_SEPARATOR.join(map(str, (src, dest, timestamp)))
            + "\n"
        )
        written_event_count += 1
    return written_event_count


def main() -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    arguments = parse_arguments()
    logger.info(
        f"Generating G(n={arguments.nodes}, p={arguments.probability})"
    )
    edges = generate_erdos_renyi_edges(
        arguments.nodes,
        arguments.probability,
    )
    try:
        written_event_count = write_edges_as_event_sequence(
            edges,
            sys.stdout,
        )
    except BrokenPipeError:
        logger.error("Output pipe closed before all edges were written")
        return EXIT_CODE_BROKEN_PIPE
    else:
        logger.info(f"Wrote {written_event_count} edge events")
        return 0


if __name__ == "__main__":
    sys.exit(main())
