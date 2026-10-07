#!/usr/bin/env bash
set -euo pipefail

CISCO_CLIENT_ROOTS=(
    /opt/cisco/secureclient
    /opt/cisco/anyconnect
)
VPN_CONNECT_ATTEMPTS="${VPN_CONNECT_ATTEMPTS:-3}"
VPN_UPGRADE_POLL_SECONDS="${VPN_UPGRADE_POLL_SECONDS:-5}"
VPN_AGENT_START_SECONDS="${VPN_AGENT_START_SECONDS:-3}"
CONNECTED_STATE="Connected"
RESOLV_CONF=/etc/resolv.conf

log_info() {
    echo "[vpn-connect] $*" >&2
}

log_error() {
    echo "[vpn-connect] ERROR: $*" >&2
}

validate_credentials() {
    local missing=()
    local name
    for name in VPN_GATEWAY VPN_USER VPN_PASSWORD; do
        if [ -z "${!name:-}" ]; then
            missing+=("${name}")
        fi
    done
    if [ "${#missing[@]}" -ne 0 ]; then
        log_error "Missing environment variables: ${missing[*]}"
        exit 1
    fi
}

find_cisco_client_root() {
    local root
    for root in "${CISCO_CLIENT_ROOTS[@]}"; do
        if [ -x "${root}/bin/vpnagentd" ]; then
            echo "${root}"
            return 0
        fi
    done
    log_error "No Cisco client found in ${CISCO_CLIENT_ROOTS[*]}"
    return 1
}

read_vpn_state() {
    local root
    root="$(find_cisco_client_root)"
    "${root}/bin/vpn" -s state </dev/null 2>/dev/null \
        | sed -n 's/.*state: *//p' \
        | tail -n 1
}

is_vpn_connected() {
    [ "$(read_vpn_state)" = "${CONNECTED_STATE}" ]
}

make_resolv_conf_writable() {
    if grep -q " ${RESOLV_CONF} " /proc/mounts; then
        log_info "Detaching Docker-managed ${RESOLV_CONF}"
        cp "${RESOLV_CONF}" "${RESOLV_CONF}.bak"
        umount "${RESOLV_CONF}"
        cp "${RESOLV_CONF}.bak" "${RESOLV_CONF}"
    else
        log_info "${RESOLV_CONF} already writable"
    fi
}

ensure_vpn_agent_running() {
    local root
    root="$(find_cisco_client_root)"
    if pgrep -x vpnagentd >/dev/null; then
        log_info "VPN agent already running"
    else
        log_info "Starting VPN agent ${root}/bin/vpnagentd"
        "${root}/bin/vpnagentd"
        sleep "${VPN_AGENT_START_SECONDS}"
    fi
}

wait_for_client_upgrade() {
    while pgrep -f vpndownloader >/dev/null; do
        log_info "Cisco client upgrade in progress, waiting"
        sleep "${VPN_UPGRADE_POLL_SECONDS}"
    done
}

request_vpn_connection() {
    local root
    root="$(find_cisco_client_root)"
    log_info "Connecting to ${VPN_GATEWAY} as ${VPN_USER}"
    printf '%s\n%s\n' "${VPN_USER}" "${VPN_PASSWORD}" \
        | "${root}/bin/vpn" -s connect "${VPN_GATEWAY}" || true
}

connect_with_retries() {
    local attempt
    for attempt in $(seq 1 "${VPN_CONNECT_ATTEMPTS}"); do
        log_info "Attempt ${attempt}/${VPN_CONNECT_ATTEMPTS}"
        ensure_vpn_agent_running
        request_vpn_connection
        wait_for_client_upgrade
        if is_vpn_connected; then
            log_info "VPN connected"
            return 0
        else
            log_info "VPN state: $(read_vpn_state)"
        fi
    done
    log_error "VPN not connected after ${VPN_CONNECT_ATTEMPTS} attempts"
    return 1
}

main() {
    validate_credentials
    if is_vpn_connected; then
        log_info "VPN already connected"
    else
        make_resolv_conf_writable
        connect_with_retries
    fi
}

main "$@"
