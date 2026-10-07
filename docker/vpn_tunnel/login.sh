#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE="${CONFIG_FILE:-${THIS_DIR}/config/config_ai.sh}"
IMAGE_NAME="${IMAGE_NAME:-anyconnect}"
CONTAINER_NAME="${CONTAINER_NAME:-vpn-login}"
VPN_CONNECT_SCRIPT="${THIS_DIR}/vpn_connect.sh"
VPN_CONNECT_COMMAND=/usr/local/bin/vpn-connect
CONTAINER_ON_EXIT="${CONTAINER_ON_EXIT:-keep}"
SERVER_WAIT_TIMEOUT_SECONDS="${SERVER_WAIT_TIMEOUT_SECONDS:-180}"
SERVER_PROBE_TIMEOUT_SECONDS="${SERVER_PROBE_TIMEOUT_SECONDS:-5}"
SERVER_PROBE_INTERVAL_SECONDS="${SERVER_PROBE_INTERVAL_SECONDS:-3}"
SSH_CONNECT_TIMEOUT_SECONDS="${SSH_CONNECT_TIMEOUT_SECONDS:-20}"
SSH_ALIVE_INTERVAL_SECONDS="${SSH_ALIVE_INTERVAL_SECONDS:-30}"
SSH_ALIVE_COUNT_MAX="${SSH_ALIVE_COUNT_MAX:-4}"

REQUIRED_CONFIG_VARIABLES=(
    VPN_GATEWAY
    VPN_USER
    VPN_PASSWORD
    SERVER_IP
    SERVER_PORT
    SSH_USER
    SSH_PASSWORD
)

KEEP_ALIVE_COMMAND=(sleep infinity)

log_info() {
    echo "[login] $*" >&2
}

log_error() {
    echo "[login] ERROR: $*" >&2
}

load_config() {
    if [ -f "${CONFIG_FILE}" ]; then
        log_info "Loading config ${CONFIG_FILE}"
        # shellcheck disable=SC1090
        . "${CONFIG_FILE}"
    else
        log_info "Config ${CONFIG_FILE} not found, using environment"
    fi
}

validate_config() {
    local missing=()
    local name
    for name in "${REQUIRED_CONFIG_VARIABLES[@]}"; do
        if [ -z "${!name:-}" ]; then
            missing+=("${name}")
        fi
    done
    if [ "${#missing[@]}" -ne 0 ]; then
        log_error "Missing config variables: ${missing[*]}"
        log_error "See ${THIS_DIR}/config/config_example.sh"
        exit 1
    fi
    case "${CONTAINER_ON_EXIT}" in
        keep | stop) ;;
        *)
            log_error "CONTAINER_ON_EXIT must be 'keep' or 'stop'"
            exit 1
            ;;
    esac
}

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        log_error "docker is not installed"
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        log_error "docker daemon is not running"
        exit 1
    fi
}

ensure_image_built() {
    if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        log_info "Image ${IMAGE_NAME} already built"
    else
        log_info "Image ${IMAGE_NAME} not found, building"
        "${THIS_DIR}/build.sh"
    fi
}

is_container_running() {
    local state
    state="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" \
        2>/dev/null || true)"
    [ "${state}" = "true" ]
}

start_vpn_container() {
    log_info "Starting container ${CONTAINER_NAME}"
    docker run \
        --detach \
        --rm \
        --name "${CONTAINER_NAME}" \
        --cap-add=SYS_ADMIN \
        --privileged \
        -e VPN_GATEWAY \
        -e VPN_USER \
        -e VPN_PASSWORD \
        "${IMAGE_NAME}" \
        "${KEEP_ALIVE_COMMAND[@]}" >/dev/null
}

ensure_container_running() {
    if is_container_running; then
        log_info "Reusing running container ${CONTAINER_NAME}"
    else
        start_vpn_container
    fi
}

install_vpn_connect_script() {
    docker cp "${VPN_CONNECT_SCRIPT}" \
        "${CONTAINER_NAME}:${VPN_CONNECT_COMMAND}" >/dev/null
    docker exec "${CONTAINER_NAME}" chmod +x "${VPN_CONNECT_COMMAND}"
}

connect_vpn() {
    log_info "Ensuring VPN is connected"
    if docker exec \
        -e VPN_GATEWAY \
        -e VPN_USER \
        -e VPN_PASSWORD \
        "${CONTAINER_NAME}" \
        "${VPN_CONNECT_COMMAND}"; then
        log_info "VPN is up"
    else
        log_error "VPN connection failed"
        exit 1
    fi
}

is_server_reachable() {
    docker exec \
        -e SERVER_IP \
        -e SERVER_PORT \
        "${CONTAINER_NAME}" \
        timeout "${SERVER_PROBE_TIMEOUT_SECONDS}" \
        bash -c 'exec 3<>"/dev/tcp/${SERVER_IP}/${SERVER_PORT}"' \
        2>/dev/null
}

wait_for_server_reachable() {
    local deadline=$((SECONDS + SERVER_WAIT_TIMEOUT_SECONDS))
    log_info "Waiting for ${SERVER_IP}:${SERVER_PORT} to answer"
    until is_server_reachable; do
        if [ "${SECONDS}" -ge "${deadline}" ]; then
            log_error "${SERVER_IP}:${SERVER_PORT} unreachable through VPN"
            log_error "Try: docker restart ${CONTAINER_NAME} && $0"
            exit 1
        fi
        sleep "${SERVER_PROBE_INTERVAL_SECONDS}"
    done
    log_info "${SERVER_IP}:${SERVER_PORT} is reachable"
}

is_interactive_terminal() {
    [ -t 0 ] && [ -t 1 ]
}

ssh_into_cluster() {
    local exec_terminal_flags=(-i)
    local ssh_terminal_flags=(-T)
    if is_interactive_terminal; then
        exec_terminal_flags=(-i -t -e "TERM=${TERM:-xterm-256color}")
        ssh_terminal_flags=(-t)
    else
        log_info "No terminal attached, running without a remote TTY"
    fi
    log_info "Connecting to ${SSH_USER}@${SERVER_IP}:${SERVER_PORT}"
    SSHPASS="${SSH_PASSWORD}" docker exec \
        "${exec_terminal_flags[@]}" \
        -e SSHPASS \
        "${CONTAINER_NAME}" \
        sshpass -e ssh \
        "${ssh_terminal_flags[@]}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR \
        -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_SECONDS}" \
        -o ServerAliveInterval="${SSH_ALIVE_INTERVAL_SECONDS}" \
        -o ServerAliveCountMax="${SSH_ALIVE_COUNT_MAX}" \
        -p "${SERVER_PORT}" \
        "${SSH_USER}@${SERVER_IP}" \
        "$@"
}

stop_container_if_requested() {
    if [ "${CONTAINER_ON_EXIT}" = "stop" ]; then
        log_info "Stopping container ${CONTAINER_NAME}"
        docker stop "${CONTAINER_NAME}" >/dev/null || true
    else
        log_info "Container ${CONTAINER_NAME} left running for reuse"
    fi
}

main() {
    load_config
    validate_config
    export VPN_GATEWAY VPN_USER VPN_PASSWORD
    require_docker
    ensure_image_built
    ensure_container_running
    install_vpn_connect_script
    connect_vpn
    wait_for_server_reachable
    local status=0
    ssh_into_cluster "$@" || status=$?
    stop_container_if_requested
    exit "${status}"
}

main "$@"
