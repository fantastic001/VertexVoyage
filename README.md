
# Introduction 

## Setting up ZK client 

Export `ZK_HOSTS` envvar 

    export ZK_HOSTS=localhost:2181,zoo1:2181,zoo2:2181

Run ZK test 

    vertex_voyage zk 

# Running ZK istance 

You can run inside Docker container 

    SERVER_NUM=1 ./zk/run_server.sh 1 

Where 1 iz ZK id

You can also run ensemble. 

