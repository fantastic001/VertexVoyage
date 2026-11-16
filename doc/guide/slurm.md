# Running on a SLURM cluster

This page covers generic SLURM usage. The ablation study has its own
submission tooling, described in
[`experiments/ablation/README.md`](../../experiments/ablation/README.md#running-on-the-cluster).
Benchmark arrays are described in [benchmarks.md](benchmarks.md).

## Dependency installation

Ensure you have the dependencies installed with conda:

```bash
module load miniconda3
conda create -n myenv
conda activate myenv
pip install -r requirements.txt
```

See [install.md](install.md) for compiling the Rust extension.

## Data

Datasets are expected under `data/`. Their sources are listed in
[datasets.md](../datasets.md), and [adding-a-dataset.md](adding-a-dataset.md)
describes how a new dataset is registered.

## Running tests

Check `test.sbatch` and change it accordingly. Then check `args.txt` and
specify which command-line arguments you want to test. After this, run it:

```bash
sbatch test.sbatch
```

## Checking the queue

```bash
squeue -u "$USER"
```
