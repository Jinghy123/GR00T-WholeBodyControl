#!/usr/bin/env bash
#
# SSH into the G1 board.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reference episodes used by --episode. Default is the ZED-mini pick-cloth
# rollout set; --ref-dir points at any other directory of .mp4 references
# (e.g. data/0709_mp4_refs/.../observation.images.egocentric).
REF_DIR="${REF_DIR:-$SCRIPT_DIR/zed_recordings/40k_clothes}"

# --- Arguments ------------------------------------------------------------------
# --episode <N>         overlay reference episode N from REF_DIR (see --list)
# --ref-dir <dir>       directory the --episode index resolves against
# --overlay <file.mp4>  overlay this exact file; wins over --episode
# --list                print the episodes available in REF_DIR and exit
#
# The overlay blends the video's first frame over the live viewer window, so the
# camera can be lined up with how a recorded episode started.
OVERLAY=""
EPISODE=""
LIST_ONLY=0
# Ego view only by default. The stereo pane is a second window fed by the reply's
# slot 1; without --show-stereo the viewer does not even decode it.
SHOW_STEREO=0
# --no-viewer brings up only the camera server and holds it, for running this in
# tmux on a headless host (the viewer needs an X display and aborts without one).
# The viewer is then started by hand on a machine that has a display.
NO_VIEWER=0
usage() {
    cat <<EOF
Usage: $(basename "$0") [--episode N | --overlay FILE] [--ref-dir DIR] [--stereo]
                        [--no-viewer] [--list]

  --episode N     overlay reference episode N from REF_DIR
  --ref-dir DIR   where --episode looks (default: $REF_DIR)
  --overlay FILE  overlay an exact .mp4; takes precedence over --episode
  --stereo        also open the stereo window (default: ego only)
  --no-viewer     server only: don't start the viewer on this host, just print
                  the command to run it elsewhere (for headless / tmux use)
  --list          list the episodes available in REF_DIR and exit

With no overlay argument the viewer starts clean (no reference frame).
EOF
}
while [ $# -gt 0 ]; do
    case "$1" in
        --episode)   EPISODE="${2:?--episode needs an index}"; shift 2 ;;
        --episode=*) EPISODE="${1#--episode=}"; shift ;;
        --ref-dir)   REF_DIR="${2:?--ref-dir needs a directory}"; shift 2 ;;
        --ref-dir=*) REF_DIR="${1#--ref-dir=}"; shift ;;
        --overlay)   OVERLAY="${2:?--overlay needs a file}"; shift 2 ;;
        --overlay=*) OVERLAY="${1#--overlay=}"; shift ;;
        --stereo|--show-stereo) SHOW_STEREO=1; shift ;;
        --no-viewer|--server-only|--headless) NO_VIEWER=1; shift ;;
        --list|--list-episodes) LIST_ONLY=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# All .mp4 in REF_DIR, sorted. Empty array instead of an error when absent, so
# --episode can report the situation itself.
list_refs() {
    [ -d "$REF_DIR" ] || return 0
    find "$REF_DIR" -maxdepth 1 -name '*.mp4' -printf '%f\n' 2>/dev/null | sort
}

if [ "$LIST_ONLY" = "1" ]; then
    echo "Reference dir: $REF_DIR"
    if [ ! -d "$REF_DIR" ]; then
        echo "  (directory does not exist)"
        exit 1
    fi
    refs="$(list_refs)"
    if [ -z "$refs" ]; then
        echo "  (no .mp4 files)"
        exit 1
    fi
    echo "$refs" | nl -ba -w4 -s'  '
    exit 0
fi

# Resolve --episode to a file. Names are tried first so an index means the same
# episode regardless of what else sits in the directory; the positional fallback
# covers reference sets that use some other convention, and always announces
# what it picked rather than resolving silently.
if [ -n "$EPISODE" ]; then
    if [ -n "$OVERLAY" ]; then
        echo "Note: --overlay given, ignoring --episode $EPISODE." >&2
    else
        case "$EPISODE" in
            ''|*[!0-9]*) echo "Error: --episode must be a non-negative integer, got '$EPISODE'" >&2; exit 1 ;;
        esac
        [ -d "$REF_DIR" ] || { echo "Error: reference dir not found: $REF_DIR" >&2; exit 1; }
        for cand in \
            "$(printf 'ego_%03d.mp4' "$EPISODE")" \
            "$(printf 'episode_%06d.mp4' "$EPISODE")" \
            "$(printf 'ego_%d.mp4' "$EPISODE")" \
            "$(printf 'episode_%d.mp4' "$EPISODE")"; do
            if [ -f "$REF_DIR/$cand" ]; then
                OVERLAY="$REF_DIR/$cand"
                break
            fi
        done
        if [ -z "$OVERLAY" ]; then
            nth="$(list_refs | sed -n "${EPISODE}p")"
            if [ -n "$nth" ]; then
                OVERLAY="$REF_DIR/$nth"
                echo "No ego_/episode_ file for index $EPISODE; using the ${EPISODE}th .mp4: $nth"
            else
                echo "Error: no reference episode $EPISODE in $REF_DIR" >&2
                echo "Available (use --list for the full set):" >&2
                list_refs | head -20 >&2
                exit 1
            fi
        fi
        echo "Reference episode $EPISODE -> $OVERLAY"
    fi
fi

if [ -n "$OVERLAY" ] && [ ! -f "$OVERLAY" ]; then
    echo "Error: overlay file not found: $OVERLAY" >&2
    exit 1
fi

USERNAME="unitree"
IP="192.168.123.164"
PASSWORD="123"
CAMERA_ID="f682"   # USB id substring that identifies the camera
ROS_SETUP="/opt/ros/noetic/setup.bash"   # ROS env to source on the board (noetic)
ZMQ_BIND_PORT="5558"   # port realsense_server binds on (--zmq-bind)
POSE_ZMQ_PORT="5570"   # host pose stream port (--pose-zmq)
VIEWER_PORT="5559"     # port the host viewer connects to (--port)
VIEWER_ENV="${VIEWER_ENV:-sonic}"   # conda env for the host viewer (needs cv2/zmq/numpy)
HOST_CONDA="${HOST_CONDA:-$HOME/miniforge3}"   # conda installation path on this host
BOARD_CONDA="/home/unitree/miniforge3"   # conda installation path on the G1 board
BOARD_CONDA_ENV="sonic"   # conda env to activate on the G1 board
OVERLAY_ALPHA="0.5"       # overlay opacity 0..1 (higher = overlay more visible)
OVERLAY_SWAP="1"          # 1 = swap overlay B<->R channels (false color) for contrast

# IPv4 of this host on the board's network (source address used to reach the board).
HOST_IP="$(ip -4 route get "$IP" 2>/dev/null | grep -oP 'src \K\S+')"
if [ -z "$HOST_IP" ]; then
    echo "Error: could not determine host IPv4 to reach $IP." >&2
    exit 1
fi

# Check that the board is reachable before attempting to connect.
if ! ping -c 1 -W 2 "$IP" >/dev/null 2>&1; then
    echo "Error: $IP is not reachable." >&2
    exit 1
fi

# Check that the camera is mounted by looking for its USB id in lsusb.
# A non-interactive ssh command skips the ROS prompt in .bashrc, so this is safe.
# Loop until the camera shows up, prompting the user to re-plug it each time.
until sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
        "${USERNAME}@${IP}" "lsusb | grep -qi '$CAMERA_ID'"; do
    echo "Error: camera ($CAMERA_ID) not detected via lsusb." >&2
    read -r -p "Please re-plug the camera, then press Enter to retry... "
done
echo "Camera ($CAMERA_ID) detected."

# Start the camera server on the board, detached (nohup + background) so this
# script can go on to launch the viewer. We source conda and ROS directly instead
# of relying on an interactive shell, which avoids the .bashrc ROS prompt entirely.
# sudo -S reads the password from stdin. The last echo returns the server PID.
echo "Starting camera server on $IP ..."
SERVER_PID="$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
        "${USERNAME}@${IP}" bash -s <<EOF | tail -1
echo "$PASSWORD" | sudo -S chmod 777 /dev/ttyUSB0 2>/dev/null
export LD_PRELOAD=/lib/aarch64-linux-gnu/libffi.so.7
source "$ROS_SETUP"
# Source conda on the board: prefer the configured path, then fall back to
# common install locations (miniforge3 / miniconda3 / anaconda3).
for conda_root in "$BOARD_CONDA" ~/miniforge3 ~/miniconda3 ~/anaconda3; do
    if [ -f "\$conda_root/etc/profile.d/conda.sh" ]; then
        source "\$conda_root/etc/profile.d/conda.sh"
        break
    fi
done
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found on the board (checked $BOARD_CONDA and common paths)." >&2
    exit 1
fi
conda activate "$BOARD_CONDA_ENV"
cd ~/GR00T-WholeBodyControl
nohup python realsense_server.py \
    --zed-only \
    --zmq-bind tcp://0.0.0.0:$ZMQ_BIND_PORT \
    --enable-neck-motor \
    --pose-zmq tcp://$HOST_IP:$POSE_ZMQ_PORT \
    > ~/realsense_server.log 2>&1 &
echo \$!
EOF
)" || true

if ! [[ "$SERVER_PID" =~ ^[0-9]+$ ]]; then
    echo "Error: failed to start camera server (no PID returned)." >&2
    exit 1
fi
echo "Camera server started (pid $SERVER_PID, log: ~/realsense_server.log)."

# Stop the viewer and the remote server when this script exits.
VIEWER_PID=""
cleanup() {
    echo
    if [ -n "$VIEWER_PID" ]; then
        kill "$VIEWER_PID" 2>/dev/null || true
    fi
    echo "Stopping camera server on $IP (pid $SERVER_PID) ..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
        "${USERNAME}@${IP}" "kill $SERVER_PID 2>/dev/null" || true
}
trap cleanup EXIT

# Wait for the server's viewer port to start accepting connections.
echo "Waiting for server port $VIEWER_PORT ..."
for _ in $(seq 1 30); do
    if timeout 1 bash -c ">/dev/tcp/$IP/$VIEWER_PORT" 2>/dev/null; then
        break
    fi
    sleep 1
done

# The neck motor thread binds its state PUB on 5560 once it started OK. If it
# isn't up, neck control (and the r+Enter reset below) silently does nothing —
# fail loudly with the server log instead.
NECK_STATE_PORT="5560"
NECK_OK=0
for _ in $(seq 1 10); do
    if timeout 1 bash -c ">/dev/tcp/$IP/$NECK_STATE_PORT" 2>/dev/null; then
        NECK_OK=1
        break
    fi
    sleep 1
done
if [ "$NECK_OK" != "1" ]; then
    echo "WARNING: neck state PUB ($IP:$NECK_STATE_PORT) is not up — the neck" >&2
    echo "motor thread failed to start; camera streams but the neck won't move." >&2
    echo "--- server log tail ---" >&2
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
        "${USERNAME}@${IP}" "tail -20 ~/realsense_server.log" >&2 || true
    echo "-----------------------" >&2
fi

# Build the viewer command line. With --no-viewer it is only printed, so the
# viewer can be started by hand on a machine that has a display. The host conda
# env is activated either way: reset_neck.py below runs from it too.
cd "$(dirname "$0")"
source "$HOST_CONDA/etc/profile.d/conda.sh"
conda activate "$VIEWER_ENV"
VIEWER_ARGS=(--server "$IP" --port "$VIEWER_PORT")
if [ "$SHOW_STEREO" = "1" ]; then
    VIEWER_ARGS+=(--show-stereo)
fi
if [ -n "$OVERLAY" ]; then
    echo "Overlay: first frame of $OVERLAY (alpha $OVERLAY_ALPHA)"
    VIEWER_ARGS+=(--overlay "$OVERLAY" --overlay-alpha "$OVERLAY_ALPHA")
    if [ "$OVERLAY_SWAP" = "1" ]; then
        VIEWER_ARGS+=(--overlay-swap-channels)
    fi
fi

if [ "$NO_VIEWER" = "1" ]; then
    echo
    echo "Server only (--no-viewer): no viewer started on this host."
    echo "To watch the stream, run this on a machine with a display, in the repo"
    echo "and in the '$VIEWER_ENV' env:"
    printf '    python test_viewer.py'
    printf ' %q' "${VIEWER_ARGS[@]}"
    echo
else
    echo "Starting viewer (conda env: $VIEWER_ENV) ..."
    python test_viewer.py "${VIEWER_ARGS[@]}" &
    VIEWER_PID=$!
fi

# Neck reset from this terminal, same UX as auto_deploy_neck_realsense.sh: type
# r+Enter to ease the neck back to (0, 0). reset_neck.py binds the pose PUB
# port itself, which is free here because this script runs no pose publisher
# (if one is running, reset_neck.py fails with a clear message and we go on).
echo
if [ "$NO_VIEWER" = "1" ]; then
    echo "Camera server up on $IP:$VIEWER_PORT (pid $SERVER_PID)."
else
    echo "Viewer running (pid $VIEWER_PID)."
fi
echo "Type r+Enter to reset the neck to (0, 0); q+Enter or Ctrl+C to quit."

server_alive() {
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        "${USERNAME}@${IP}" "kill -0 $SERVER_PID" 2>/dev/null
}

# Stay up as long as what we babysit lives: the viewer normally, the remote
# server with --no-viewer — polled over ssh every $HEALTH_EVERY ticks so a
# crashed server ends the session instead of leaving a terminal that looks fine.
HEALTH_EVERY=30
ticks=0
while :; do
    if [ "$NO_VIEWER" = "1" ]; then
        ticks=$((ticks + 1))
        if [ "$((ticks % HEALTH_EVERY))" = "0" ] && ! server_alive; then
            sleep 2   # retry once: a single ssh hiccup is not a dead server
            if ! server_alive; then
                echo "Camera server (pid $SERVER_PID) died on $IP." >&2
                echo "--- server log tail ---" >&2
                sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
                    "${USERNAME}@${IP}" "tail -20 ~/realsense_server.log" >&2 || true
                echo "-----------------------" >&2
                break
            fi
        fi
    else
        kill -0 "$VIEWER_PID" 2>/dev/null || break
    fi
    # No tty (nohup, piped stdin): read hits EOF at once and would spin, so tick
    # on sleep instead and skip the r/q commands there is nobody to type.
    if [ ! -t 0 ]; then
        sleep 1
        continue
    fi
    if read -r -t 1 line; then
        case "${line,,}" in
            r)
                echo "Resetting neck to (0, 0) ..."
                python reset_neck.py --neck-pub-port "$POSE_ZMQ_PORT" \
                    --neck-state-zmq "tcp://$IP:$NECK_STATE_PORT" || true
                ;;
            q|quit|exit)
                break
                ;;
        esac
    fi
done
