
from vertex_voyage.temporal import to_nx_graph, SBMSequence, FirstN, ShuffledSequence
from vertex_voyage.partitioning import get_partition_average_balance
import matplotlib.pyplot as plt
import pandas as pd 
from vertex_voyage.config import get_config_bool

def is_full_benchmark():
    """
    Check if the full benchmark is enabled in the configuration.
    """
    return get_config_bool("full_benchmark", False, "Flag to enable full benchmark mode.")
