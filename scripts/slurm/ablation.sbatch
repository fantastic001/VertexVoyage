#!/bin/bash
#SBATCH --job-name=vv-ablation
#SBATCH --export=ALL
set -euo pipefail

readonly JOBS_FILE="${1:?usage: sbatch ablation.sbatch JOBS_FILE}"
readonly CONDA_ROOT="${ABLATION_CONDA_ROOT:-/cm/shared/apps/miniconda3}"
readonly CONDA_ENV="${ABLATION_CONDA_ENV:-myenv}"
readonly TASK_INDEX="${SLURM_ARRAY_TASK_ID:?must run as a SLURM array task}"

activate_conda() {
    module load miniconda3
    local conda_setup
    if conda_setup="$("${CONDA_ROOT}/bin/conda" shell.bash hook 2>/dev/null)"
    then
        eval "${conda_setup}"
    elif [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
        . "${CONDA_ROOT}/etc/profile.d/conda.sh"
    else
        export PATH="${CONDA_ROOT}/bin:${PATH}"
    fi
    set +u
    conda activate "${CONDA_ENV}"
    set -u
    hash -r
}

if [ ! -f "${JOBS_FILE}" ]; then
    echo "Jobs file ${JOBS_FILE} does not exist" >&2
    exit 1
else
    echo "Task ${TASK_INDEX} of ${JOBS_FILE} on $(hostname)"
fi

activate_conda
srun python -m experiments.ablation run \
    --jobs "${JOBS_FILE}" \
    --index "${TASK_INDEX}"
