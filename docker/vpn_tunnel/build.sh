#!/bin/bash 

THIS_DIR=$(readlink -f $(dirname $0))

is_arm() {
    if [ "$(uname -m)" == "arm64" ]; then
        return 0
    else
        return 1
    fi
}

INSTALL_DIR=$THIS_DIR/install

if is_arm; then 
    export DOCKER_DEFAULT_PLATFORM=linux/amd64
fi 


docker build -t anyconnect -f $THIS_DIR/Dockerfile $THIS_DIR