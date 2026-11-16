# Seeding and reproducibility

## How runs are seeded

`--seed S` on `vv test` and `vv temporal_test` seeds every framework through
[`vertex_voyage/seeding.py`](../../vertex_voyage/seeding.py):

- `seed_shared_random_state(S)` seeds Python's `random`, NumPy, PyTorch and
  TensorFlow (whichever are installed). $S$ must satisfy $0 \le S < 2^{32}$.
- `derive_iteration_seed(S, i)` gives iteration $i$ of a multi-iteration run
  its own seed, `SeedSequence([S, i]).generate_state(1)[0]`. Iterations are
  independent, and the same $(S, i)$ always gives the same stream.
- `DEFAULT_SEED` (config key `default_seed`, environment variable
  `VERTEX_VOYAGE_DEFAULT_SEED`) is used when `--seed` is not given.

## Seed protocol for reported results

Results recorded from 2026-11 onwards follow the protocol of the
[ablation runner](../../experiments/ablation/README.md): every cell
(dataset, $P$, $RF$) is run as $n = 10$ independent jobs with
`--iterations 1` and seeds $42, 43, \dots, 51$. Tables report
$\bar F \pm \sigma$ with the population standard deviation (`np.std`) and
the number of completed runs, `n=k/10`. Older results (E02–E06) were
recorded without fixed seeds. [ROADMAP.md](../../ROADMAP.md) lists them
for re-running.

## Known non-determinism

The alignment aggregator is not bit-reproducible at a fixed seed: two Enron
runs with the same seed ended at different losses (see
[E10](../../results/E10-partition-alignment/README.md#open-issues)).
Candidate causes are unseeded negative sampling and nondeterministic
threaded kernels.

## Pattern: independent streams per pipeline step

The pattern below spawns one child seed per pipeline step, so that changing
one step does not change the random stream of the others.


```python

import random
import numpy as np
import sklearn.model_selection
import sklearn.ensemble
import torch

# 1. Initialize the master controller
MASTER_SEED = 42
master_ss = np.random.SeedSequence(MASTER_SEED)

# 2. Spawn 4 completely independent child sequences (one per pipeline step)
# This guarantees that changing step 3 won't change the random stream for steps 1, 2, or 4!
child_seeds = master_ss.spawn(4)

# ==========================================
# STEP 1: Native Python Algorithms
# ==========================================
# Extract an integer from the first child seed
python_seed = int(child_seeds[0].generate_state(1)[0])
random.seed(python_seed)
my_list = [1, 2, 3, 4, 5]
random.shuffle(my_list) # Isolated native python randomness


# ==========================================
# STEP 2: NumPy Data Processing
# ==========================================
# Pass the second child seed directly into default_rng
numpy_rng = np.random.default_rng(child_seeds[1])
noise = numpy_rng.normal(0, 1, size=(100, 10))


# ==========================================
# STEP 3: Scikit-Learn (Data Splitting / ML)
# ==========================================
# Scikit-Learn wants an integer for 'random_state'. Pull an integer from the third child.
sklearn_seed = int(child_seeds[2].generate_state(1)[0])

# Pass it to your splitter or estimators
splitter = sklearn.model_selection.TrainTestSplit(random_state=sklearn_seed)
rf_model = sklearn.ensemble.RandomForestClassifier(random_state=sklearn_seed)


# ==========================================
# STEP 4: PyTorch (Deep Learning)
# ==========================================
# PyTorch Generators require a 64-bit unsigned integer seed.
pytorch_seed = int(child_seeds[3].generate_state(1)[0])

# Create an independent PyTorch Generator object
torch_gen = torch.Generator()
torch_gen.manual_seed(pytorch_seed)

# Pass it to your PyTorch components (e.g., DataLoaders or Initializers)
# loader = DataLoader(dataset, batch_size=32, generator=torch_gen)
# x = torch.rand((5, 5), generator=torch_gen)

```

**PyTorch DataLoaders**: Each background worker process will clone the generator's state, leading to identical data shuffling across different processes. To fix this, always use PyTorch's worker_init_fn to uniquely seed each worker process.