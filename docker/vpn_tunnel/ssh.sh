#!/bin/bash 

THIS_DIR=$(readlink -f $(dirname $0))

# ssh -L 0.0.0.0:6969:10.68.6.58:22 -o ProxyCommand="socat STDIO UNIX:. /sockets/ssh.sock" stefan.nozinic.pmfns@10.68.6.58

sshpass -p "$SSH_PASSWORD" ssh -L 0.0.0.0:9292:$SERVER_IP:$SERVER_PORT -o ProxyCommand="socat STDIO UNIX:$THIS_DIR/sockets/ssh.sock" $SSH_USER@$SERVER_IP