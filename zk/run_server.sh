#!/bin/bash 

DOCKER_IMAGE_NAME="zookeeper"
DOCKER_CONTAINER_NAME="zookeeper-$(date +%s)"

docker run --rm -it  \
    -p 2181:2181 \
    -p 2888:2888 \
    -p 3888:3888 \
    --name $DOCKER_CONTAINER_NAME $DOCKER_IMAGE_NAME