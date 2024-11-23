
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