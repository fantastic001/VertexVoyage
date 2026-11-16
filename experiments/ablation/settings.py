import os
from pathlib import Path

from vertex_voyage.config import get_config_int, get_config_str

OUTPUT_ROOT = get_config_str(
    "ablation_output_root",
    "ablation_runs",
    "Directory holding one checkpoint directory per ablation job",
)
SEED_COUNT = get_config_int(
    "ablation_seed_count",
    10,
    "Number of seeds, each run as its own job, per ablation cell",
)
BASE_SEED = get_config_int(
    "ablation_base_seed",
    42,
    "First seed of an ablation cell; the others follow consecutively",
)
BASE_BUFFER_SIZE = get_config_int(
    "ablation_buffer_size",
    1000,
    "Buffer size of the full pipeline in the ablation study",
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = os.path.join(REPOSITORY_ROOT, OUTPUT_ROOT)
