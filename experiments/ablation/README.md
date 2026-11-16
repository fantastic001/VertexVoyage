# Ablation study runner

Runs the variants of the
ablation study
with `vv temporal_test`, one variant at a time, on a SLURM cluster or locally.

## Terms

- **Variant**: one row of the variant table in `variants.py`. It changes one
  setting of the full pipeline (`baseline`) through CLI flags, environment
  variables or both. `python -m experiments.ablation list` prints all of them.
- **Cell**: a triple (dataset, $P$, RF) of `grid.py`. Every variant runs
  RF = 1 with $P \in \{1, 2, 4\}$ on CITESEER, AstroPh, DBLP and Enron, and
  RF = 3 with $P = 4$ on CITESEER and AstroPh (all four datasets for
  `random_partitioning`).
- **Job**: one `temporal_test` run of a variant on a cell with one seed and
  `--iterations 1`. Each cell has $n$ jobs with seeds
  $s_0, s_0 + 1, \dots, s_0 + n - 1$ (default $s_0 = 42$, $n = 10$).
- **Score** of a cell: mean ± standard deviation (`np.std`) of the final F1 of
  its completed jobs, with `n=k/n` showing how many of them completed.

The full pipeline is

```
temporal_test --long-run --use-dataset-params --ordering bfs --track-seen
  --buffer-size 1000 --partitioner-name neighbors.all --mu 1 --alpha 1
  --aggregator triplet --iterations 1 --skip-full-graph-baseline
  --name D --partitions P --replication-factor RF --seed S --checkpoint DIR
```

## Output

Every job writes to its own checkpoint directory

```
<output root>/<variant>/<dataset>/P<p>_RF<rf>/seed<s>/
  metadata.json  iteration_f1s_0.pkl  ...   PersistedRun checkpoint
  ablation.json                             variant, env, argv, git commit
  output.log  vv.log                        process output and vertex voyage log
```

A job counts as completed when `iteration_f1s_0.pkl` exists. Completed jobs
are never submitted again, so resubmitting a variant only runs what failed or
timed out. Jobs whose flags coincide share a directory: `no_partitioning`
runs `--partitions 1` once per dataset and RF, and its $P = 2$ and $P = 4$
columns repeat that score.

## Quick test (about 10 minutes)

Runs two variants on 300 CITESEER events with 2 seeds into a separate
output root, so nothing touches real results. Measured on a laptop: about
3–4 minutes per variant.

```bash
export VERTEX_VOYAGE_ABLATION_OUTPUT_ROOT=ablation_quick_test
export VERTEX_VOYAGE_ABLATION_SEED_COUNT=2

python -m experiments.ablation list
python -m experiments.ablation show baseline --datasets CITESEER --limit 300
python -m experiments.ablation local baseline --datasets CITESEER --limit 300
python -m experiments.ablation local random_vertex_sequence \
    --datasets CITESEER --limit 300
python -m experiments.ablation status baseline --datasets CITESEER
python -m experiments.ablation collect baseline --datasets CITESEER
python -m experiments.ablation collect random_vertex_sequence \
    --datasets CITESEER
```

Expected:

- `local` ends with `All 8 jobs finished` (4 cells × 2 seeds). Running it
  again ends with `All 0 jobs finished`, because completed jobs are skipped.
- `collect` prints an RF=1 and an RF=3 table with `n=2/2` in every CITESEER
  cell and writes `ablation_quick_test/<variant>/scores.csv`.
- `ablation_quick_test/random_vertex_sequence/CITESEER/P1_RF1/seed42/ablation.json`
  holds `"VERTEX_VOYAGE_WALK_STRATEGY": "random_vertex_sequence"`.

Then check the SLURM submission without submitting anything:

```bash
scripts/slurm/submit_ablation.sh submit baseline CITESEER --dry-run
```

With the quick-test variables still set it prints `CITESEER: nothing pending`,
because the quick test completed every CITESEER job. After
`unset VERTEX_VOYAGE_ABLATION_OUTPUT_ROOT VERTEX_VOYAGE_ABLATION_SEED_COUNT`
it prints `CITESEER: 40 pending jobs in ...` and the
`sbatch --array=0-39%20 ...` command it would run.

Remove `ablation_quick_test/` afterwards.

## Running on the cluster

On the login node, activate the conda environment of the project (the one
named by `ABLATION_CONDA_ENV`, default `myenv`, as in the project README)
from the repository root, then run one variant at a time:

```bash
scripts/slurm/submit_ablation.sh list
scripts/slurm/submit_ablation.sh submit baseline
scripts/slurm/submit_ablation.sh status baseline
scripts/slurm/submit_ablation.sh collect baseline
scripts/slurm/submit_ablation.sh submit one_epoch CITESEER AstroPh
```

`submit VARIANT [DATASET ...]` writes the pending jobs of each dataset to
`<output root>/<variant>/jobs/<dataset>-<UTC time>-<pid>.jsonl` and submits
one SLURM array per dataset. Jobs files are never overwritten, so a running
array keeps reading the lines it was submitted with. Each array task runs
`python -m experiments.ablation run --jobs FILE --index $SLURM_ARRAY_TASK_ID`
through `scripts/slurm/ablation.sbatch`; its SLURM logs go to
`<output root>/<variant>/slurm/`.

| Variable | Default | Meaning |
|---|---|---|
| `ABLATION_CONDA_ENV` | `myenv` | conda environment activated on the nodes |
| `ABLATION_CONDA_ROOT` | `/cm/shared/apps/miniconda3` | conda installation |
| `ABLATION_MAX_PARALLEL` | `20` | concurrent tasks per array |
| `ABLATION_TIME`, `ABLATION_MEM`, `ABLATION_CPUS` | per dataset | override the resources of `resources_of` |
| `ABLATION_SBATCH_ARGS` | empty | extra options, e.g. `--partition=all --account=x` |
| `PYTHON` | `python` | interpreter used on the login node |
| `VERTEX_VOYAGE_ABLATION_OUTPUT_ROOT` | `ablation_runs` | output root, relative to the repository |
| `VERTEX_VOYAGE_ABLATION_SEED_COUNT` | `10` | seeds per cell |
| `VERTEX_VOYAGE_ABLATION_BASE_SEED` | `42` | first seed |
| `VERTEX_VOYAGE_ABLATION_BUFFER_SIZE` | `1000` | buffer size of the full pipeline |

Default resources per task, a first guess to adjust after the first runs:

| Dataset | Time | Memory | CPUs |
|---|---|---|---|
| CITESEER | 2 h | 8 G | 4 |
| DBLP | 8 h | 16 G | 4 |
| AstroPh | 12 h | 32 G | 4 |
| Enron | 24 h | 32 G | 4 |

`unit_buffer` processes one buffer per event (about 198 000 on AstroPh and
368 000 on Enron), so submit those datasets with a larger time limit, e.g.
`ABLATION_TIME=72:00:00 scripts/slurm/submit_ablation.sh submit unit_buffer Enron`.
