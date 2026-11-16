# Benchmarks

 

To run particular benchmark from experiments folder, run:

    python -m vertex_voyage.benchmark BENCHMARK_NAME 

To list all benchmarks:

    python -m vertex_voyage.benchmark --list 

By default, we run benchmarks on small datasets for testing purposes. To run full benchmarks:

    export VERTEX_VOYAGE_FULL_BENCHMARK=1 
    python -m vertex_voyage.benchmark BENCHMARK_NAME 

Or, to run on cluster:

    export VERTEX_VOYAGE_FULL_BENCHMARK=1 
    python -m vertex_voyage.benchmark --all --no-display 


## Running benchmarks with SLURM 

below is example of slurm script which can be used:

First we need to generate arguments for jobs which is file `args.txt` which contains benchmark names. Every name is on a separate line.

```bash
python -m vertex_voyage.benchmark --list | grep -E "^ - " | sed 's/^ - //' > args.txt
```

Then we can use this script:

```bash 
#!/bin/bash
#SBATCH --job-name=py-array
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-32%10
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --export=ALL
# #SBATCH --gres=gpu:1                # uncomment if you need a GPU

echo "Task $SLURM_ARRAY_TASK_ID args: $ARG_LINE"
set -euo pipefail
mkdir -p logs


module load miniconda3
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/cm/shared/apps/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cm/shared/apps/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/cm/shared/apps/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/cm/shared/apps/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


echo activating
conda activate myenv 2>&1

hash -r

export VERTEX_VOYAGE_FULL_BENCHMARK=0
ARG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" args.txt)


srun python -m vertex_voyage.benchmark --no-display "$ARG_LINE"

```

