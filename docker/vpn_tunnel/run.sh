#!/bin/sh

is_inside_container() {
    cat /proc/1/cmdline | grep -q "run.sh"
}

VPN_GATEWAY=${VPN_GATEWAY:-"vpn-gateway"}
VPN_USER=${VPN_USER:-"vpn-user"}
VPN_PASSWORD=${VPN_PASSWORD:-"vpn-password"}
SSH_UER=${SSH_USER:-"ssh-user"}


ANYCONNECT_PATH="/opt/cisco/anyconnect/bin/vpn"
ANYCONNECT_AGENT_PATH="/opt/cisco/anyconnect/bin/vpnagentd"
connect_to_vpn() {
    echo "Starting AnyConnect agent"
    $ANYCONNECT_AGENT_PATH
    echo "Connecting to VPN $VPN_GATEWAY"
    $ANYCONNECT_PATH -s connect $VPN_GATEWAY <<EOF
$VPN_USER
$VPN_PASSWORD
y
EOF

    test $? -eq 0 && echo "Connected to VPN" || echo "Failed to connect to VPN"
    cat /opt/cisco/anyconnect*.log
}


SERVER_IP=${SERVER_IP:-"10.68.6.58"}
SERVER_PORT=${SERVER_PORT:-"22"}
LOCAL_PORT=${LOCAL_PORT:-"22"}

ssh_tunnel() {
    REMOTE_SERVER=$1
    REMOTE_PORT=$2
    LOCAL_PORT=$3
    USER=$4
    SSH_PASSWORD=$5

    echo "Creating forward SSH tunnel to $REMOTE_SERVER:$REMOTE_PORT"
    # do not check host keys 
    if [ -z "$SSH_PASSWORD" ]; then
        echo "SSH_PASSWORD is not set"
        return 1
    fi
    if [ -z "$USER" ]; then
        echo "USER is not set"
        return 1
    fi
    sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -N \
        -L 127.0.0.1:$LOCAL_PORT:$REMOTE_SERVER:$REMOTE_PORT \
        $USER@$REMOTE_SERVER &
    
    # Create the Unix socket
    socat UNIX-LISTEN:/tmp/sockets/ssh.sock,fork TCP:127.0.0.1:$LOCAL_PORT &
    wait
}

if is_inside_container; then 
    echo "Running inside container"
    cp /etc/resolv.conf /etc/resolv.conf.bak
    umount /etc/resolv.conf
    cp /etc/resolv.conf.bak /etc/resolv.conf
    connect_to_vpn
    # no argument - tunnel ssh 
    if [ $# -eq 0 ]; then
        ssh_tunnel $SERVER_IP $SERVER_PORT $LOCAL_PORT $SSH_USER $SSH_PASSWORD
    else
        $@
    fi
else
    echo "Not running inside container"
    echo "Publishing container port $LOCAL_PORT to host port 6969"
    docker run \
        --rm \
        -e VPN_GATEWAY=$VPN_GATEWAY \
        -e VPN_USER=$VPN_USER \
        -e VPN_PASSWORD=$VPN_PASSWORD \
        -e SERVER_IP=$SERVER_IP \
        -e SERVER_PORT=$SERVER_PORT \
        -e LOCAL_PORT=$LOCAL_PORT \
        -e SSH_USER=$SSH_USER \
        -e SSH_PASSWORD=$SSH_PASSWORD \
        --cap-add=SYS_ADMIN \
        --name vpn-tunnel \
        -p 6969:22 \
        --privileged \
        -v ./sockets:/tmp/sockets/ \
        -it anyconnect $@
fi