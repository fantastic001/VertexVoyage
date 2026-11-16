# Running a VertexVoyage cluster

## Setting up ZK client 

Export `ZK_HOSTS` envvar 

    export ZK_HOSTS=localhost:2181,zoo1:2181,zoo2:2181

Run ZK test 

    vertex_voyage zk 

## Running ZK instance 

You can run inside Docker container 

    SERVER_NUM=1 ./zk/run_server.sh 1 

Where 1 iz ZK id

You can also run ensemble. 


## Starting VertexVoyage 

please take a look at `docker-compose.yml` file inside `docker/` directory to see how to run VertexVoyage cluster. 

You can create pipeline which you can use to communicate specific requests to cluster. Take a look at few samples inside `pipelines/` directory. 

to run specific pipeline, run:

    python -m client execute --pipeline pipelines/zachary.yml  --results-folder pipeline_results/

Please run `python -m client --help` for full list of possible commands. 


## Configuration 

In order to get configuration parameters you can modify, execute:

    python scripts/detect_function_calls.py  vertex_voyage/ g
et_config_ --exclude get_config_location

Every configuration parameter can be changed in configuration file and overwritten by using 
environment variable. 

For instance, `port` can be specified in configuration file and can be modified using `VERTEX_VOYAGE_PORT` environment variable. 

Configuration file is specified using `VERTEX_VOYAGE_CONFIG` and default location is at `~/.vertex_voyage/config.json`


## Running Vertex Voyage without ZooKeeper

To run Vertex Voyage without ZooKeeper, you need to ensure that the configuration does not require ZooKeeper for its operations. Follow these steps:

1. **Modify Configuration**: Update the configuration file to disable ZooKeeper dependencies. You can specify the configuration file using the `VERTEX_VOYAGE_CONFIG` environment variable. The default location is `~/.vertex_voyage/config.json`.

2. **Set Environment Variables**: Ensure that any necessary configuration parameters are set via environment variables. For example, you can set the port using the `VERTEX_VOYAGE_PORT` environment variable. In order not to use ZK, set `NO_ZK=0` in your environment.

3. **Run Vertex Voyage**: Execute the Vertex Voyage application as usual. If you are running it inside a SLURM cluster, Vertex Voyage will automatically detect the SLURM environment and configure itself accordingly.


Example command to run Vertex Voyage:
```sh
VERTEX_VOYAGE_CONFIG=~/.vertex_voyage/config.json VERTEX_VOYAGE_PORT=8080 vertex_voyage
```

