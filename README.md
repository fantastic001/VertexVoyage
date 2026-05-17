
# Introduction 

VertexVoyage is distribued implementation of node2vec algorithm for graph embedding. 

## Installation 

To install VertexVoyage using pip, you can follow these steps:

1. Open your terminal or command prompt.

2. Make sure you have pip installed. You can check by running the command `pip --version`. If pip is not installed, you can install it by following the instructions at https://pip.pypa.io/en/stable/installing/.

3. Once you have pip installed, you can use the following command to install VertexVoyage:

   ```
   pip install vertex_voyage
   ```

   This command will download and install the latest version of VertexVoyage from the Python Package Index (PyPI).

   Note: Make sure you have an active internet connection for pip to download the package.

To install VertexVoyage using the setup.py file, you can follow these steps:

1. Download the VertexVoyage source code from the repository or obtain the setup.py file.

2. Open your terminal or command prompt and navigate to the directory where the setup.py file is located.

3. Run the following command to install VertexVoyage:

   ```
   python setup.py install
   ```

   This command will execute the setup.py file and install VertexVoyage on your system.

   Note: Make sure you have Python installed and added to your system's PATH environment variable.

Both methods will install VertexVoyage and its dependencies on your system, allowing you to use it in your cluster.

## Setting up ZK client 

Export `ZK_HOSTS` envvar 

    export ZK_HOSTS=localhost:2181,zoo1:2181,zoo2:2181

Run ZK test 

    vertex_voyage zk 

# Running ZK instance 

You can run inside Docker container 

    SERVER_NUM=1 ./zk/run_server.sh 1 

Where 1 iz ZK id

You can also run ensemble. 


# Starting VertexVoyage 

please take a look at `docker-compose.yml` file inside `docker/` directory to see how to run VertexVoyage cluster. 

You can create pipeline which you can use to communicate specific requests to cluster. Take a look at few samples inside `pipelines/` directory. 

to run specific pipeline, run:

    python -m client execute --pipeline pipelines/zachary.yml  --results-folder results/

Please run `python -m client --help` for full list of possible commands. 

## Using XML-RPC in VertexVoyage

VertexVoyage utilizes XML-RPC for communication between the client and the VertexVoyage nodes. The main method available for communication is the `execute` method. This method allows you to invoke specific functions in the VertexVoyage node by providing the `method_name` parameter and any additional keyword arguments based on the selected functionality.

Here is an example of how to use the `execute` method in Python:
```py
import xmlrpc.client

# Replace 'leader_ip' with the actual IP address of the leader node
leader_ip = '192.168.0.1'

# Create an XML-RPC proxy to communicate with the leader node
proxy = xmlrpc.client.ServerProxy(f"http://{leader_ip}:8000/")

# Call the create_graph method on the leader node
graph_name = "my_graph"
result = proxy.execute("create_graph", name=graph_name)

# Print the result
print(result)
```

Methods are documented in the following table:


| **Method**            | **Parameters**                                                                                                                                           | **Description**                                                                                              |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `create_graph`        | `name: str`                                                                                                                                               | Creates a new empty graph with the given name.                                                              |
| `add_vertex`          | `graph_name: str`, `vertex_name: str`                                                                                                                     | Adds a vertex with the specified name to the graph.                                                         |
| `add_edge`            | `graph_name: str`, `vertex1: str`, `vertex2: str`                                                                                                         | Adds an edge between two vertices in the specified graph.                                                   |
| `partition_graph`     | `graph_name: str`                                                                                                                                         | Partitions the graph into subgraphs and calculates its corruptibility. Returns the partitions and metrics.  |
| `get_partition`       | `graph_name: str`, `partition_num: int`                                                                                                                   | Retrieves information about a specific partition of the graph.                                              |
| `get_embedding`       | `graph_name: str`, `dim: int=128`, `epochs: int=10`, `learning_rate: float=0.01`, `n_walks: int=10`, `negative_sample_num: int=1`, `p: float=1`, `q: float=1`, `window_size: int=10`, `walk_size: int=10` | Generates node embeddings for the graph using the Node2Vec algorithm with specified parameters.            |
| `get_vertices`        | `graph_name: str`                                                                                                                                         | Retrieves a list of vertices in the graph.                                                                  |
| `get_edges`           | `graph_name: str`                                                                                                                                         | Retrieves a list of edges in the graph.                                                                     |
| `import_karate_club`  | `name: str`                                                                                                                                               | Imports the Karate Club graph dataset and creates a graph with the specified name.                          |
| `get_leader`          | None                                                                                                                                                      | Retrieves information about the leader node in the system.                                                  |
| `zk`                  | None                                                                                                                                                      | Provides information about the ZooKeeper nodes, leader, and the current node.                               |
| `process`             | `graph_name: str`, `dim: int=128`, `epochs: int=10`, `learning_rate: float=0.01`, `n_walks: int=10`, `negative_sample_num: int=1`, `p: float=1`, `q: float=1`, `window_size: int=10`, `walk_size: int=10` | Processes the graph by computing and normalizing embeddings using parallel Node2Vec computations.           |
| `import_gml`          | `url: str`, `graph_name: str`                                                                                                                             | Imports a graph in GML format from a specified URL.                                                         |
| `download_gml`        | `dataset_name: str`, `source: str="snap"`                                                                                                                 | Downloads a GML file for a dataset from the specified source ("snap", "konect", or "network_repository").   |
| `generate_graph`      | `graph_name: str`, `n: int`, `p: float`, `q: float`, `c: int`                                                                                             | Generates a planted partition graph with specified parameters and saves it.                                 |
| `list`                | None                                                                                                                                                      | Lists all the available graphs stored.                                                                      |
| `upload_gml`          | `graph_name: str`, `data: bytes`, `append: bool=False`                                                                                                    | Uploads or appends data to a GML file for the specified graph.                                              | 


# Configuration 

In order to get configuration parameters you can modify, execute:

    python scripts/detect_function_calls.py  vertex_voyage/ g
et_config_ --exclude get_config_location

Every configuration parameter can be changed in configuration file and overwritten by using 
environment variable. 

For instance, `port` can be specified in configuration file and can be modified using `VERTEX_VOYAGE_PORT` environment variable. 

Configuration file is specified using `VERTEX_VOYAGE_CONFIG` and default location is at `~/.vertex_voyage/config.json`


# Running Vertex Voyage without ZooKeeper

To run Vertex Voyage without ZooKeeper, you need to ensure that the configuration does not require ZooKeeper for its operations. Follow these steps:

1. **Modify Configuration**: Update the configuration file to disable ZooKeeper dependencies. You can specify the configuration file using the `VERTEX_VOYAGE_CONFIG` environment variable. The default location is `~/.vertex_voyage/config.json`.

2. **Set Environment Variables**: Ensure that any necessary configuration parameters are set via environment variables. For example, you can set the port using the `VERTEX_VOYAGE_PORT` environment variable. In order not to use ZK, set `NO_ZK=0` in your environment.

3. **Run Vertex Voyage**: Execute the Vertex Voyage application as usual. If you are running it inside a SLURM cluster, Vertex Voyage will automatically detect the SLURM environment and configure itself accordingly.


Example command to run Vertex Voyage:
```sh
VERTEX_VOYAGE_CONFIG=~/.vertex_voyage/config.json VERTEX_VOYAGE_PORT=8080 vertex_voyage
```

# Running benchmarks 

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

# Partitioning methods

## Most common neighbor partitioner

Partition of vertex is the partition that contains the most neighbors of the vertex in the sampled events. The partitioner samples a subset of events according to a distribution P(event | vertex), and assigns the vertex to the partition that contains the most neighbors of the vertex in the sampled events. The partitioner also has a replication factor, which allows it to assign a vertex to multiple partitions if there are multiple partitions that contain a similar number of neighbors of the vertex in the sampled events. The partitioner also has a capacity penalty, which penalizes partitions that have more vertices than the average partition size, to encourage more balanced partitions.

Formally, the score of a partition for a vertex is defined as:

$$ S(P, v) = N(P, v) - \mu \cdot \max(0, |P| - \alpha \cdot (1 + \epsilon) \cdot \frac{1}{|P|} \sum_{P' \in P} |P'|) $$

where $N(P, v)$ is the number of neighbors of vertex $v$ in partition $P$ in the sampled events, $\mu$ is the capacity penalty coefficient, $\alpha$ is the weight of the average partition size in the capacity penalty, $\epsilon$ is the imbalance tolerance, and $|P|$ is the size of partition $P$. The partitioner assigns the vertex to the top $k$ partitions with the highest scores, where $k$ is the replication factor. If there are multiple partitions with similar scores, the partitioner randomly assigns the vertex to some of those partitions until it reaches the replication factor.

Also, if decay is not None, the partitioner will use an exponentially decaying weight for the neighbors in the sampled events, where the weight of a neighbor that was seen $t$ time steps ago is multiplied by $\exp(-\text{decay} \cdot t)$. This allows the partitioner to give more importance to recent neighbors when assigning vertices to partitions.

Formally, the score of a partition for a vertex with decay is defined as:

$$ S(P, v) = \sum_{u \in N(P, v)} \exp(-\text{decay} \cdot t(u)) - \mu \cdot \max(0, |P| - \alpha \cdot (1 + \epsilon) \cdot \frac{1}{|P|} \sum_{P' \in P} |P'|) $$

# Running experiments 

To run experiment for partitioning and embedding on a specific dataset, run:

    vv test --name Cit-HepPh --partitions 1  --break-early  --use-dataset-params --long-run  --epochs 10

For temporal experiments, run:

    vv temporal_test --name CITESEER  --long-run --iterations 10  --use-dataset-params  --batch-size 50 --partitions 5 --replication-factor 2 --partitioner neighbors.all  --mu 0.5

## Test command flags

 Runs a test of the embedding quality for a given dataset and partitioning parameters. It loads the dataset, partitions it using the specified method, computes embeddings for each partition using the specified algorithm, and then evaluates the quality of the embeddings by reconstructing the graph and computing the F1 score against the original graph. It also computes a global embedding by averaging the partition embeddings and evaluates its F1 score as well.

Parameters:

| Parameter | Description |
|-----------|-------------|
| `--name` |  The name of the dataset to use. |
| `--partitions` | The number of partitions to create. |
| `--alpha` | The alpha parameter for the partitioning algorithm (if applicable). |
| `--threshold` | The threshold parameter for the partitioning algorithm (if applicable). |
| `--break-early` | If True, breaks after the first combination of p and q parameters is tested. |
| `--skip-global` | If True, skips the global F1 score computation. |
| `--dim` | The dimensionality of the embeddings. |
| `--default-p` | If > 0, uses this value for p in the embedding algorithm instead of testing multiple values. |
| `--default-q` | If > 0, uses this value for q in the embedding algorithm instead of testing multiple values. |
| `--epochs` | The number of epochs to train the embedding model for. |
| `--long-run` | If True, uses more walks and larger walk sizes for the embedding algorithm, which may lead to better embeddings but takes longer to compute. |
| `--use-dataset-params` | If True, overrides the parameters with dataset-specific parameters from the dataset_params dictionary if they are available. |
| `--use-lpa` | If True, uses label propagation for partitioning instead of the default partitioning algorithm. |
| `--algorithm` | The embedding algorithm to use (e.g., "node2vec", "distger", "dynnode2vec"). |

## Temporal test command flags

Runs a temporal test of the embedding quality for a given dataset and partitioning parameters. It loads the dataset as a stream of events, partitions it using the specified method, computes embeddings for each partition using the specified algorithm, and then evaluates the quality of the embeddings by reconstructing the graph and computing the F1 score against the original graph after each batch of events is processed. It also computes a global embedding by averaging the partition embeddings and evaluates its F1 score as well.

Parameters:

| Parameter | Description |
|-----------|-------------|
| `--name` | The name of the dataset to use. |
| `--partitions` | The number of partitions to create. |
| `--partitioner-name` | The name of the partitioning algorithm to use (e.g., "random", "random.degree", "neighbors.all", "neighbors.degree"). |
| `--dim` | The dimensionality of the embeddings. |
| `--default-p` | If > 0, uses this value for p in the embedding algorithm instead of testing multiple values. |
| `--default-q` | If > 0, uses this value for q in the embedding algorithm instead of testing multiple values. |
| `--epochs` | The number of epochs to train the embedding model for after each batch of events. |
| `--long-run` | If True, uses more walks and larger walk sizes for the embedding algorithm, which may lead to better embeddings but takes longer to compute. |
| `--use-dataset-params` | If True, overrides the parameters with dataset-specific parameters from the dataset_params dictionary if they are available. |
| `--use-lpa` | If True, uses label propagation for partitioning instead of the default partitioning algorithm (not implemented in this method). |
| `--algorithm` | The embedding algorithm to use (e.g., "dynnode2vec"). |
| `--track-seen` | If True, processes events in an order that prioritizes events connected to already seen nodes, which simulates a more realistic temporal scenario where new events are more likely to involve nodes that have already been observed. |
| `--iterations` | The number of times to repeat the entire process for averaging results. |
| `--limit` | If > 0, limits the number of events to process from the dataset for quicker testing. |
| `--batch-size` | The number of events to process in each batch before updating the embeddings and evaluating the F1 score. |

# Development 

## Conda setup

Create environment

```
conda create myenv -c conda-forge rust python=3.9 
```

Activate conda and install dependencies:

```
conda activate myenv
pip install -r requirements.txt
```

Compile Rust code with `maturin`

```
cd vertex_voyage_native
maturin develop
```

# Experimental results

## Dynnode2vec with batched event processing 

Example of run:

```sh
vv temporal_test --name CITESEER  --long-run --track-seen --iterations 10  --use-dataset-params  --batch-size 100
```

| Dataset | Batch size | Average F1 score | Baseline |
|---------|------------|------------------|----------|
| CITESEER | 100 | 57.82% | 40.75% |
