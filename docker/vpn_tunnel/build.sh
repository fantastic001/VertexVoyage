#!/bin/bash 

THIS_DIR=$(readlink -f $(dirname $0))

INSTALL_DIR=$THIS_DIR/install

docker build -t anyconnect -f $THIS_DIR/Dockerfile $THIS_DIR