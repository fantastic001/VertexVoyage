
# VPN and AI platform 

build container:

    ./build.sh 

Create config file like in example `config/config_example.sh`

Source config file

    . config/config_ai.sh

Then run it

    ./run.sh 

and then connect to platform 

    . config/config_ai.sh
    sudo -E ./ssh.sh 

Now you can use port `9292` on localhost to establish SSH connection 

## Login directly from the container (macOS)

On macOS the Docker VM breaks socket forwarding, so SSH runs inside the
container instead:

    ./login.sh                 # interactive shell on the cluster
    ./login.sh nvidia-smi      # run a single remote command

The image is built on first use. The script reads `config/config_ai.sh`
(override with `CONFIG_FILE=...`), starts the `vpn-login` container, runs
`vpn-connect` inside it (a no-op when already connected, otherwise it starts
the Cisco agent and connects with retries), then SSHs to
`$SSH_USER@$SERVER_IP`. The container keeps running, so the next login reuses
the VPN session and reconnects it if it dropped. To stop it when the session
ends, set `CONTAINER_ON_EXIT=stop`.

To (re)connect the VPN by hand inside the container:

    docker exec -it vpn-login vpn-connect      # from the host
    vpn-connect                                # from a shell in the container
