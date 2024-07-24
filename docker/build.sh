#!/bin/bash


# exit on error
set -e

THIS_DIR=$(readlink -f $(dirname $0))
APP_DIR=$(dirname $THIS_DIR)

# Get the app name from the app directory
APP_NAME=$(basename $APP_DIR)

# image tag is lowercase of app name
IMAGE_TAG=$(echo $APP_NAME | tr '[:upper:]' '[:lower:]')

# Build the Docker image
docker build \
    -t $IMAGE_TAG \
    -f $APP_DIR/docker/Dockerfile \
    --build-arg APP_DIR=$APP_DIR \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    --network=host \
    $APP_DIR 

