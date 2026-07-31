#!/usr/bin/env bash
# LEGACY/DEBUG ONLY: persistent SSH tunnel to the WM server.
#
# Production clients must use 192.168.123.240 directly over g1 TELEOP. No
# production launcher calls this script. Starting it requires an explicit
# opt-in so a loopback tunnel cannot silently replace the wired contract.

set -Eeuo pipefail

WM_LOCAL_HOST="${WM_LOCAL_HOST:-127.0.0.1}"
WM_LOCAL_PORT="${WM_LOCAL_PORT:-8016}"
WM_REMOTE_HOST="${WM_REMOTE_HOST:-127.0.0.1}"
WM_REMOTE_PORT="${WM_REMOTE_PORT:-8016}"
WM_SSH_TARGET="${WM_SSH_TARGET:-hongyi@192.168.123.240}"
WM_READY_TIMEOUT="${WM_READY_TIMEOUT:-180}"
CONTROL_SOCKET="${WM_TUNNEL_CONTROL_SOCKET:-${TMPDIR:-/tmp}/psi-wm-tunnel-${UID}.sock}"
READY_URL="http://${WM_LOCAL_HOST}:${WM_LOCAL_PORT}/ready"

usage() {
    cat <<EOF
Usage: $(basename "$0") {start|status|stop}

LEGACY DEBUG PATH ONLY. Production uses direct 192.168.123.240:8016.
To start intentionally, export PSIX_ENABLE_LEGACY_WM_TUNNEL=1 first.

  start   Authenticate once and keep the WM tunnel running in the background.
  status  Show whether the SSH master and WM /ready endpoint are healthy.
  stop    Stop the tunnel created by this script.

Tunnel: ${WM_LOCAL_HOST}:${WM_LOCAL_PORT} -> \
${WM_SSH_TARGET}:${WM_REMOTE_HOST}:${WM_REMOTE_PORT}
EOF
}

http_ok() {
    curl --noproxy '*' -fsS --max-time 2 "$READY_URL" 2>/dev/null \
        | python3 -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("ready") is True else 1)' \
        >/dev/null 2>&1
}

master_ok() {
    ssh -S "$CONTROL_SOCKET" -O check "$WM_SSH_TARGET" >/dev/null 2>&1
}

wait_ready() {
    local deadline=$((SECONDS + WM_READY_TIMEOUT))
    while ((SECONDS < deadline)); do
        if http_ok; then
            echo "[wm-tunnel] WM ready: $READY_URL"
            return 0
        fi
        sleep 1
    done
    echo "[wm-tunnel] ERROR: WM /ready unavailable after ${WM_READY_TIMEOUT}s" >&2
    echo "[wm-tunnel] SSH tunnel may still be running; check the WM server." >&2
    return 1
}

start_tunnel() {
    if [[ "${PSIX_ENABLE_LEGACY_WM_TUNNEL:-0}" != "1" ]]; then
        echo "[wm-tunnel] REFUSED: legacy tunnel is disabled by default." >&2
        echo "[wm-tunnel] Production command: ./g1_teleop_network.sh check" >&2
        echo "[wm-tunnel] Debug-only opt-in: PSIX_ENABLE_LEGACY_WM_TUNNEL=1 $0 start" >&2
        return 2
    fi
    if http_ok; then
        echo "[wm-tunnel] WM endpoint is already ready: $READY_URL"
        if master_ok; then
            echo "[wm-tunnel] managed SSH master: running"
        else
            echo "[wm-tunnel] endpoint belongs to another existing tunnel/process"
        fi
        return 0
    fi

    if master_ok; then
        echo "[wm-tunnel] managed SSH master already running; waiting for WM"
        wait_ready
        return
    fi

    # Remove only this script's stale control socket.  Never touch an unknown
    # listener or kill a tunnel owned by another terminal/process.
    [[ ! -e "$CONTROL_SOCKET" ]] || rm -f -- "$CONTROL_SOCKET"

    local listener
    listener="$(ss -H -ltnp "sport = :$WM_LOCAL_PORT")"
    if [[ -n "$listener" ]]; then
        echo "[wm-tunnel] ERROR: local port $WM_LOCAL_PORT is already occupied:" >&2
        echo "$listener" >&2
        return 1
    fi

    echo "[wm-tunnel] opening persistent tunnel: ${WM_LOCAL_HOST}:${WM_LOCAL_PORT} " \
         "-> ${WM_SSH_TARGET}:${WM_REMOTE_HOST}:${WM_REMOTE_PORT}"
    echo "[wm-tunnel] Enter the WM-server SSH password once; the tunnel then stays up."
    ssh -M -S "$CONTROL_SOCKET" -fN \
        -L "${WM_LOCAL_HOST}:${WM_LOCAL_PORT}:${WM_REMOTE_HOST}:${WM_REMOTE_PORT}" \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o ConnectTimeout=10 \
        -o StrictHostKeyChecking=accept-new \
        "$WM_SSH_TARGET"
    wait_ready
}

show_status() {
    local rc=0
    if master_ok; then
        echo "[wm-tunnel] managed SSH master: running"
    else
        echo "[wm-tunnel] managed SSH master: not running"
        rc=1
    fi
    if http_ok; then
        echo "[wm-tunnel] WM endpoint: ready ($READY_URL)"
        curl --noproxy '*' -fsS --max-time 2 "$READY_URL"
        echo
        rc=0
    else
        echo "[wm-tunnel] WM endpoint: unavailable ($READY_URL)"
    fi
    return "$rc"
}

stop_tunnel() {
    if master_ok; then
        ssh -S "$CONTROL_SOCKET" -O exit "$WM_SSH_TARGET" >/dev/null
        echo "[wm-tunnel] stopped"
    else
        echo "[wm-tunnel] no managed tunnel is running"
    fi
    [[ ! -e "$CONTROL_SOCKET" ]] || rm -f -- "$CONTROL_SOCKET"
}

case "${1:-}" in
    start) start_tunnel ;;
    status) show_status ;;
    stop) stop_tunnel ;;
    -h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
