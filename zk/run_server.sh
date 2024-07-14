#!/bin/bash 

SERVER_NUM=3

THIS_DIR=$(dirname $(readlink -f $0))

ID=${1:-1}

DOCKER_IMAGE_NAME="zookeeper"
DOCKER_HOSTNAME=zoo$ID
DOCKER_CONTAINER_NAME=zoo$ID

CONFIG_FILE="$THIS_DIR/zoo.cfg"




ZOO_MY_ID=$ID
ZOO_SERVERS=""

for i in $(seq 1 $SERVER_NUM); do
    ZOO_SERVERS="$ZOO_SERVERS server.$i=zoo$i:2888:3888;2181"
done 

CLIENT_PORT=$((2180 + $ID))
PEER_PORT=$((2887 + $ID))
ELECTION_PORT=$((3887 + $ID))

network="zoo"
links="--net $network"

# create network if it does not exist
docker network inspect $network > /dev/null 2>&1 || docker network create $network


docker run --rm -it  \
    -p $CLIENT_PORT:2181 \
    -p $PEER_PORT:2888 \
    -p $ELECTION_PORT:3888 \
    -v $CONFIG_FILE:/conf/zoo.cfg \
    $links \
    -e ZOO_MY_ID=$ZOO_MY_ID \
    -e ZOO_SERVERS="$ZOO_SERVERS" \
    --hostname $DOCKER_HOSTNAME \
    --name $DOCKER_CONTAINER_NAME $DOCKER_IMAGE_NAME 

