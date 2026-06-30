#!/bin/bash 

export VERTEX_VOYAGE_PLUGINS="experiments.*"
# Enable job control
set -m

CANCELLED=0
OSTYPE=$(uname)
kill_all_jobs() {
    echo "Killing all background jobs..."
    while read -r job; do
        # On OSX, we use pkill to get the process group ID and kill the entire group, which is necessary to ensure that all child processes are also killed. On Linux, we can directly kill the job.
        if [[ "$OSTYPE" == "Darwin"* ]]; then
            echo "Killing job $job with pkill (OSX)"
            pkill -P "$job" 2>/dev/null || true
        else
            echo "Killing job $job with kill (Linux)"
            kill -9 "$job" 2>/dev/null || true
        fi
        sleep 1 # Give some time for the job to be killed before checking again
    done < <(jobs -p)
    # unblock wait loop 
    echo >&3

    CANCELLED=1
}

# Kill all jobs if script is interrupted
trap kill_all_jobs SIGINT SIGTERM

# python -m vertex_voyage.benchmark 

# This is needed to wait for the first background job to finish before proceeding, without waiting for all jobs. We use a named pipe (FIFO) to achieve this. This is portable and works in bash without relying on specific job control features.
tmpdir=$(mktemp -d)
pipe=$(basename "$tmpdir")-pipe
rm -rf "$tmpdir/$pipe" # Remove file placeholder to create actual FIFO
mkfifo "$tmpdir/$pipe"
exec 3<>"$tmpdir/$pipe"
rm "$tmpdir/$pipe" # Clean from disk immediately; descriptor 3 keeps it open in memory

if [ "$DEBUG" = "1" ]; then
    echo "Running in DEBUG mode"
    set -x
fi
MAX_JOBS=${MAX_JOBS:-1}

for parts in 1 2 4 8; do 
for RF in 1 3; do
    if [ $RF -lt $parts ]; then
        continue
    fi
    for dataset in CITESEER AstroPh DBLP AS-Oregon Enron; do 
        if [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; then
            echo "Maximum number of concurrent jobs ($MAX_JOBS) reached. Waiting for a job to finish..."
            read <&3 # Wait for a job to signal completion
        fi
        if [ $CANCELLED -eq 1 ]; then
            echo "Experiment cancelled. Exiting."
            kill_all_jobs
            exit 1
        fi
        echo "----------------------------------------"
        echo "Running experiment on dataset $dataset with P=$parts and RF=$RF"
        CP_DIR="temporal_runs/$dataset-${parts}_${RF}/"
        mkdir -p $CP_DIR
        export VERTEX_VOYAGE_LOG_FILE="${CP_DIR}vv.log"
        export VERTEX_VOYAGE_F1_COMPUTATION_THRESHOLD=10000
        (python -m vertex_voyage  temporal_test \
            --name $dataset \
            --long-run \
            --track-seen \
            --iterations 2 \
            --use-dataset-params \
            --buffer-size 1000 \
            --replication-factor $RF \
            --partitioner neighbors.all \
            --mu 1 --alpha 1 --partitions $parts \
            --checkpoint ${CP_DIR} \
            > ${CP_DIR}output.log 2>&1; echo >&3) &
        echo "----------------------------------------"
    done
done
done

wait
echo "All experiments completed."
exec 3>&-