#!/bin/bash
set -euo pipefail

readonly SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd "${SCRIPT_DIRECTORY}/../.." && pwd)"
readonly SBATCH_SCRIPT="${SCRIPT_DIRECTORY}/ablation.sbatch"
readonly PYTHON="${PYTHON:-python}"
readonly OUTPUT_ROOT="${VERTEX_VOYAGE_ABLATION_OUTPUT_ROOT:-ablation_runs}"
readonly MAX_PARALLEL="${ABLATION_MAX_PARALLEL:-20}"
readonly SEEDS="${VERTEX_VOYAGE_ABLATION_SEED_COUNT:-10}"

usage() {
    cat >&2 <<USAGE
Usage: $(basename "$0") COMMAND [ARGUMENTS]

Commands:
  list                                   list ablation variants
  datasets VARIANT                       datasets of a variant
  show VARIANT [DATASET ...]             print jobs without submitting
  submit VARIANT [DATASET ...] [--dry-run]
                                         submit pending jobs of a variant,
                                         one SLURM array per dataset
  status VARIANT [DATASET ...]           completed seeds per cell
  collect VARIANT [DATASET ...]          result tables and scores.csv

Environment:
  ABLATION_CONDA_ENV, ABLATION_CONDA_ROOT conda environment on the nodes
  ABLATION_MAX_PARALLEL                   concurrent tasks per array (20)
  ABLATION_TIME, ABLATION_MEM, ABLATION_CPUS
                                          override per-dataset resources
  ABLATION_SBATCH_ARGS                    extra sbatch options, e.g.
                                          "--partition=all --account=x"
  VERTEX_VOYAGE_ABLATION_OUTPUT_ROOT      output directory (ablation_runs)
  VERTEX_VOYAGE_ABLATION_SEED_COUNT       seeds per cell (10)
  VERTEX_VOYAGE_ABLATION_BASE_SEED        first seed (42)
USAGE
}

die() {
    echo "Error: $*" >&2
    exit 1
}

ablation() {
    (
        cd "${REPOSITORY_ROOT}"
        "${PYTHON}" -m experiments.ablation "$@"
    )
}

resources_of() {
    local dataset="$1"
    local time memory cpus
    case "${dataset}" in
        CITESEER) time="02:00:00"; memory="8G";  cpus="4" ;;
        DBLP)     time="08:00:00"; memory="16G"; cpus="4" ;;
        AstroPh)  time="12:00:00"; memory="32G"; cpus="4" ;;
        Enron)    time="24:00:00"; memory="32G"; cpus="4" ;;
        *)        time="24:00:00"; memory="32G"; cpus="4" ;;
    esac
    echo "--time=${ABLATION_TIME:-${time}}" \
        "--mem=${ABLATION_MEM:-${memory}}" \
        "--cpus-per-task=${ABLATION_CPUS:-${cpus}}"
}

require_variant() {
    local variant="${1:-}"
    if [ -n "${variant}" ]; then
        ablation datasets "${variant}" > /dev/null \
            || die "unknown variant '${variant}', see: $0 list"
    else
        usage
        die "missing VARIANT"
    fi
}

jobs_file_of() {
    local variant="$1" dataset="$2" mode="$3"
    local stamp
    local name
    stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    name="${dataset// /_}-${stamp}.jsonl"
    if [ "${mode}" = "submit" ]; then
        echo "${REPOSITORY_ROOT}/${OUTPUT_ROOT}/${variant}/jobs/${name}"
    else
        echo "$(mktemp -d)/${name}"
    fi
}

submit_dataset() {
    local variant="$1" dataset="$2" mode="$3"
    local jobs_file count log_directory
    jobs_file="$(jobs_file_of "${variant}" "${dataset}" "${mode}")"
    count="$(ablation plan "${variant}" --datasets "${dataset}" \
        --seeds "${SEEDS}" --selection pending --out "${jobs_file}")"
    if [ "${count}" -gt 0 ]; then
        log_directory="${REPOSITORY_ROOT}/${OUTPUT_ROOT}/${variant}/slurm"
        local -a command=(
            sbatch
            --job-name="vv-${variant}-${dataset// /_}"
            --array="0-$((count - 1))%${MAX_PARALLEL}"
            --chdir="${REPOSITORY_ROOT}"
            --output="${log_directory}/%A_%a.out"
            --error="${log_directory}/%A_%a.err"
        )
        read -r -a resources <<< "$(resources_of "${dataset}")"
        read -r -a extra <<< "${ABLATION_SBATCH_ARGS:-}"
        command+=("${resources[@]}" ${extra[@]+"${extra[@]}"})
        command+=("${SBATCH_SCRIPT}" "${jobs_file}")
        echo "${dataset}: ${count} pending jobs in ${jobs_file}"
        if [ "${mode}" = "submit" ]; then
            mkdir -p "${log_directory}"
            "${command[@]}"
        else
            echo "  dry run: ${command[*]}"
        fi
    else
        echo "${dataset}: nothing pending"
    fi
}

submit_variant() {
    local variant="$1"
    shift
    local mode="submit"
    local -a datasets=()
    local argument
    for argument in "$@"; do
        if [ "${argument}" = "--dry-run" ]; then
            mode="dry-run"
        else
            datasets+=("${argument}")
        fi
    done
    if [ "${#datasets[@]}" -eq 0 ]; then
        while IFS= read -r argument; do
            datasets+=("${argument}")
        done < <(ablation datasets "${variant}")
    else
        ablation show "${variant}" --datasets "${datasets[@]}" > /dev/null \
            || die "datasets ${datasets[*]} are not part of ${variant}"
    fi
    for argument in "${datasets[@]}"; do
        submit_dataset "${variant}" "${argument}" "${mode}"
    done
}

with_datasets() {
    local command="$1" variant="$2"
    shift 2
    if [ "$#" -gt 0 ]; then
        ablation "${command}" "${variant}" --seeds "${SEEDS}" --datasets "$@"
    else
        ablation "${command}" "${variant}" --seeds "${SEEDS}"
    fi
}

main() {
    local command="${1:-}"
    if [ "$#" -gt 0 ]; then
        shift
    else
        usage
        die "missing COMMAND"
    fi
    case "${command}" in
        list)
            ablation list ;;
        datasets)
            require_variant "${1:-}"
            ablation datasets "$1" ;;
        show|status|collect)
            require_variant "${1:-}"
            with_datasets "${command}" "$@" ;;
        submit)
            require_variant "${1:-}"
            submit_variant "$@" ;;
        -h|--help|help)
            usage ;;
        *)
            usage
            die "unknown command '${command}'" ;;
    esac
}

main "$@"
