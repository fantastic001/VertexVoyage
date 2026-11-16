import csv
import logging
import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np

from experiments.ablation.grid import Cell, datasets_of
from experiments.ablation.jobs import (
    FINAL_ITERATION_FILE,
    NO_LIMIT,
    is_completed,
    job_of,
)
from experiments.ablation.variants import Variant

logger = logging.getLogger(__name__)

MISSING_SCORE = "—"
SCORE_COLUMNS = [
    "variant",
    "dataset",
    "partitions",
    "replication_factor",
    "seed",
    "f1",
    "directory",
]


def final_f1_of(directory: str) -> Optional[float]:
    if is_completed(directory):
        with open(os.path.join(directory, FINAL_ITERATION_FILE), "rb") as f:
            return float(pickle.load(f)[-1])
    else:
        return None


def score_rows_of(
    variant: Variant,
    cells: Sequence[Cell],
    seeds: Sequence[int],
) -> List[dict]:
    rows = []
    for cell in cells:
        for seed in seeds:
            directory = job_of(variant, cell, seed, NO_LIMIT).directory
            rows.append(
                {
                    "variant": variant.name,
                    "dataset": cell.dataset,
                    "partitions": cell.partitions,
                    "replication_factor": cell.replication_factor,
                    "seed": seed,
                    "f1": final_f1_of(directory),
                    "directory": directory,
                }
            )
    return rows


def scores_by_cell_of(rows: Sequence[dict]) -> Dict[Cell, List[float]]:
    scores: Dict[Cell, List[float]] = {}
    for row in rows:
        cell = Cell(
            row["dataset"],
            row["partitions"],
            row["replication_factor"],
        )
        scores.setdefault(cell, [])
        if row["f1"] is not None:
            scores[cell].append(row["f1"])
        else:
            continue
    return scores


def summary_of(scores: Sequence[float], expected: int) -> str:
    if scores:
        return (
            f"{np.mean(scores):.3f} ± {np.std(scores):.3f} "
            f"(n={len(scores)}/{expected})"
        )
    else:
        return MISSING_SCORE


def table_of(
    scores: Dict[Cell, List[float]],
    replication_factor: int,
    expected: int,
) -> str:
    cells = [c for c in scores if c.replication_factor == replication_factor]
    partitions = sorted({cell.partitions for cell in cells})
    lines = [
        "| Dataset | " + " | ".join(f"P={p}" for p in partitions) + " |",
        "|---|" + "---|" * len(partitions),
    ]
    for dataset in datasets_of(cells):
        summaries = [
            summary_of(
                scores.get(Cell(dataset, p, replication_factor), []),
                expected,
            )
            for p in partitions
        ]
        lines.append(f"| {dataset} | " + " | ".join(summaries) + " |")
    return "\n".join(lines)


def report_of(
    variant: Variant,
    cells: Sequence[Cell],
    seeds: Sequence[int],
) -> str:
    scores = scores_by_cell_of(score_rows_of(variant, cells, seeds))
    sections = [f"# {variant.name}: {variant.description}"]
    for factor in sorted({cell.replication_factor for cell in scores}):
        table = table_of(scores, factor, len(seeds))
        sections.append(f"RF={factor}\n\n{table}")
    return "\n\n".join(sections)


def write_scores_csv(
    variant: Variant,
    cells: Sequence[Cell],
    seeds: Sequence[int],
    path: str,
) -> None:
    rows = score_rows_of(variant, cells, seeds)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d scores of %s to %s", len(rows), variant.name, path)


def status_of(
    variant: Variant,
    cells: Sequence[Cell],
    seeds: Sequence[int],
) -> str:
    scores = scores_by_cell_of(score_rows_of(variant, cells, seeds))
    lines = [
        f"{cell.dataset} P={cell.partitions} RF={cell.replication_factor}: "
        f"{len(cell_scores)}/{len(seeds)} seeds completed"
        for cell, cell_scores in scores.items()
    ]
    completed = sum(len(cell_scores) for cell_scores in scores.values())
    lines.append(f"Total: {completed}/{len(scores) * len(seeds)}")
    return "\n".join(lines)
