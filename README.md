
# Introduction 

VertexVoyage is distribued implementation of node2vec algorithm for graph embedding. 

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

