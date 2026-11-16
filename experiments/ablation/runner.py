import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from typing import Dict, List, Sequence

from experiments.ablation.jobs import Job
from experiments.ablation.settings import REPOSITORY_ROOT

logger = logging.getLogger(__name__)

PLUGINS = "experiments.*"
JOB_RECORD_FILE = "ablation.json"
OUTPUT_LOG_FILE = "output.log"
VERTEX_VOYAGE_LOG_FILE = "vv.log"
UNKNOWN_COMMIT = "unknown"
SUCCESS = 0
FAILURE = 1


def git_commit_of(repository: os.PathLike) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == SUCCESS:
        return completed.stdout.strip()
    else:
        logger.error("Cannot read git commit: %s", completed.stderr.strip())
        return UNKNOWN_COMMIT


def job_environment_of(job: Job) -> Dict[str, str]:
    return {
        **os.environ,
        **job.env,
        "VERTEX_VOYAGE_PLUGINS": PLUGINS,
        "VERTEX_VOYAGE_LOG_FILE": os.path.join(
            job.directory,
            VERTEX_VOYAGE_LOG_FILE,
        ),
    }


def command_of(job: Job) -> List[str]:
    return [sys.executable, "-m", "vertex_voyage", "temporal_test", *job.argv]


def write_job_record(job: Job) -> None:
    record = {**asdict(job), "commit": git_commit_of(REPOSITORY_ROOT)}
    with open(os.path.join(job.directory, JOB_RECORD_FILE), "w") as output:
        json.dump(record, output, indent=2)


def run_job(job: Job) -> int:
    os.makedirs(job.directory, exist_ok=True)
    command = command_of(job)
    logger.info("Running %s with env %s: %s", job.directory, job.env, command)
    with open(os.path.join(job.directory, OUTPUT_LOG_FILE), "a") as output:
        write_job_record(job)
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=job_environment_of(job),
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode == SUCCESS:
        logger.info("Job %s finished", job.directory)
    else:
        logger.error(
            "Job %s failed with exit code %d, see %s",
            job.directory,
            completed.returncode,
            os.path.join(job.directory, OUTPUT_LOG_FILE),
        )
    return SUCCESS if completed.returncode == SUCCESS else FAILURE


def run_jobs(jobs: Sequence[Job]) -> int:
    failed = [job.directory for job in jobs if run_job(job) != SUCCESS]
    if not failed:
        logger.info("All %d jobs finished", len(jobs))
        return SUCCESS
    else:
        logger.error("%d of %d jobs failed: %s", len(failed), len(jobs), failed)
        return FAILURE
