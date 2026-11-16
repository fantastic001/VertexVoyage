import argparse
import logging
import os
import sys
from typing import Callable, Dict, List

from experiments.ablation.collect import report_of, status_of, write_scores_csv
from experiments.ablation.grid import Cell, datasets_of, restricted_cells
from experiments.ablation.jobs import (
    NO_LIMIT,
    Job,
    JobSelection,
    jobs_of_variant,
    read_job,
    seeds_of,
    selected_jobs,
    write_jobs,
)
from experiments.ablation.runner import FAILURE, SUCCESS, run_job, run_jobs
from experiments.ablation.settings import OUTPUT_DIRECTORY, SEED_COUNT
from experiments.ablation.variants import VARIANTS, variant_of

logger = logging.getLogger("experiments.ablation")

SCORES_FILE = "scores.csv"


def jobs_of_arguments(arguments: argparse.Namespace) -> List[Job]:
    jobs = jobs_of_variant(
        variant_of(arguments.variant),
        arguments.datasets,
        seeds_of(arguments.seeds),
        arguments.limit,
    )
    return selected_jobs(jobs, JobSelection(arguments.selection))


def list_variants(arguments: argparse.Namespace) -> int:
    for name, variant in VARIANTS.items():
        print(f"{name:32} {variant.description}")
    return SUCCESS


def list_datasets(arguments: argparse.Namespace) -> int:
    print("\n".join(datasets_of(variant_of(arguments.variant).cells)))
    return SUCCESS


def show_jobs(arguments: argparse.Namespace) -> int:
    for job in jobs_of_arguments(arguments):
        environment = " ".join(f"{k}={v}" for k, v in job.env.items())
        print(f"{job.directory}\n  {environment} {' '.join(job.argv)}")
    return SUCCESS


def plan_jobs(arguments: argparse.Namespace) -> int:
    jobs = jobs_of_arguments(arguments)
    write_jobs(arguments.out, jobs)
    print(len(jobs))
    return SUCCESS


def run_indexed_job(arguments: argparse.Namespace) -> int:
    return run_job(read_job(arguments.jobs, arguments.index))


def run_local_jobs(arguments: argparse.Namespace) -> int:
    return run_jobs(jobs_of_arguments(arguments))


def cells_of_arguments(arguments: argparse.Namespace) -> List[Cell]:
    return restricted_cells(
        variant_of(arguments.variant).cells,
        arguments.datasets,
    )


def print_status(arguments: argparse.Namespace) -> int:
    print(
        status_of(
            variant_of(arguments.variant),
            cells_of_arguments(arguments),
            seeds_of(arguments.seeds),
        )
    )
    return SUCCESS


def collect_scores(arguments: argparse.Namespace) -> int:
    variant = variant_of(arguments.variant)
    cells = cells_of_arguments(arguments)
    seeds = seeds_of(arguments.seeds)
    write_scores_csv(
        variant,
        cells,
        seeds,
        os.path.join(OUTPUT_DIRECTORY, variant.name, SCORES_FILE),
    )
    print(report_of(variant, cells, seeds))
    return SUCCESS


def add_variant_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("variant", choices=list(VARIANTS))


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    add_variant_argument(parser)
    parser.add_argument("--seeds", type=int, default=SEED_COUNT)
    parser.add_argument("--datasets", nargs="*", default=[])


def add_job_arguments(parser: argparse.ArgumentParser) -> None:
    add_selection_arguments(parser)
    parser.add_argument("--limit", type=int, default=NO_LIMIT)
    parser.add_argument(
        "--selection",
        choices=[selection.value for selection in JobSelection],
        default=JobSelection.ALL.value,
    )


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--index", type=int, required=True)


def add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    add_job_arguments(parser)
    parser.add_argument("--out", required=True)


def add_local_arguments(parser: argparse.ArgumentParser) -> None:
    add_job_arguments(parser)
    parser.set_defaults(selection=JobSelection.PENDING.value)


def add_no_arguments(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults()


COMMANDS: Dict[str, tuple] = {
    "list": (list_variants, add_no_arguments, "List ablation variants"),
    "datasets": (list_datasets, add_variant_argument, "Datasets of a variant"),
    "show": (show_jobs, add_job_arguments, "Print jobs without running"),
    "plan": (plan_jobs, add_plan_arguments, "Write jobs file for SLURM"),
    "run": (run_indexed_job, add_run_arguments, "Run one job of jobs file"),
    "local": (run_local_jobs, add_local_arguments, "Run jobs one by one"),
    "status": (print_status, add_selection_arguments, "Completed seeds"),
    "collect": (collect_scores, add_selection_arguments, "Tables and CSV"),
}


def parser_of(commands: Dict[str, tuple]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.ablation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (handler, add_arguments, description) in commands.items():
        subparser = subparsers.add_parser(name, help=description)
        add_arguments(subparser)
        subparser.set_defaults(handler=handler)
    return parser


def main(argv: List[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    arguments = parser_of(COMMANDS).parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = arguments.handler
    try:
        return handler(arguments)
    except (ValueError, IndexError, OSError) as error:
        logger.error("%s failed: %s", arguments.command, error)
        return FAILURE


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
