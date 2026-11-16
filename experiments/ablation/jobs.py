import json
import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

from experiments.ablation.grid import Cell, restricted_cells
from experiments.ablation.settings import BASE_SEED, OUTPUT_DIRECTORY
from experiments.ablation.variants import BASE_FLAGS, Flags, Variant

logger = logging.getLogger(__name__)

NO_LIMIT = -1
FINAL_ITERATION_FILE = "iteration_f1s_0.pkl"


class JobSelection(Enum):
    ALL = "all"
    PENDING = "pending"


@dataclass(frozen=True)
class Job:
    variant: str
    dataset: str
    partitions: int
    replication_factor: int
    seed: int
    directory: str
    argv: Tuple[str, ...]
    env: Mapping[str, str]


def seeds_of(count: int) -> List[int]:
    if count > 0:
        return list(range(BASE_SEED, BASE_SEED + count))
    else:
        logger.error("Seed count must be positive, got %d", count)
        raise ValueError(f"Seed count must be positive, got {count}")


def cell_flags_of(cell: Cell, seed: int, limit: int) -> Flags:
    limit_flags: Flags = {"limit": str(limit)} if limit > 0 else {}
    return {
        "name": cell.dataset,
        "partitions": str(cell.partitions),
        "replication-factor": str(cell.replication_factor),
        "seed": str(seed),
        **limit_flags,
    }


def effective_flags_of(
    variant: Variant,
    cell: Cell,
    seed: int,
    limit: int,
) -> Flags:
    flags: Flags = {
        **BASE_FLAGS,
        **cell_flags_of(cell, seed, limit),
        **variant.flags,
        **variant.dataset_flags(cell.dataset),
    }
    return {
        name: value
        for name, value in flags.items()
        if name not in variant.removed_flags
    }


def argv_of(flags: Flags) -> List[str]:
    argv: List[str] = []
    for name, value in flags.items():
        argv.extend([f"--{name}"] if value is None else [f"--{name}", value])
    return argv


def directory_of(variant: Variant, flags: Flags) -> str:
    return os.path.join(
        OUTPUT_DIRECTORY,
        variant.name,
        str(flags["name"]).replace(" ", "_"),
        f"P{flags['partitions']}_RF{flags['replication-factor']}",
        f"seed{flags['seed']}",
    )


def job_of(variant: Variant, cell: Cell, seed: int, limit: int) -> Job:
    flags = effective_flags_of(variant, cell, seed, limit)
    directory = directory_of(variant, flags)
    flags["checkpoint"] = directory
    return Job(
        variant=variant.name,
        dataset=cell.dataset,
        partitions=int(str(flags["partitions"])),
        replication_factor=cell.replication_factor,
        seed=seed,
        directory=directory,
        argv=tuple(argv_of(flags)),
        env=dict(variant.env),
    )


def jobs_of_variant(
    variant: Variant,
    datasets: Sequence[str],
    seeds: Sequence[int],
    limit: int,
) -> List[Job]:
    jobs_by_directory: Dict[str, Job] = {}
    for cell in restricted_cells(variant.cells, datasets):
        for seed in seeds:
            job = job_of(variant, cell, seed, limit)
            jobs_by_directory.setdefault(job.directory, job)
    return list(jobs_by_directory.values())


def is_completed(directory: str) -> bool:
    return os.path.exists(os.path.join(directory, FINAL_ITERATION_FILE))


JOB_FILTER_OF_SELECTION: Dict[JobSelection, Callable[[Job], bool]] = {
    JobSelection.ALL: lambda job: True,
    JobSelection.PENDING: lambda job: not is_completed(job.directory),
}


def selected_jobs(jobs: Sequence[Job], selection: JobSelection) -> List[Job]:
    return [job for job in jobs if JOB_FILTER_OF_SELECTION[selection](job)]


def write_jobs(path: str, jobs: Sequence[Job]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as jobs_file:
        for job in jobs:
            jobs_file.write(json.dumps(asdict(job)) + "\n")
    logger.info("Wrote %d jobs to %s", len(jobs), path)


def job_from_record(record: dict) -> Job:
    return Job(**{**record, "argv": tuple(record["argv"])})


def read_job(path: str, index: int) -> Job:
    with open(path) as jobs_file:
        lines = jobs_file.read().splitlines()
    if 0 <= index < len(lines):
        return job_from_record(json.loads(lines[index]))
    else:
        logger.error(
            "Job index %d is outside the %d jobs of %s",
            index,
            len(lines),
            path,
        )
        raise IndexError(f"Job index {index} outside 0..{len(lines) - 1}")
