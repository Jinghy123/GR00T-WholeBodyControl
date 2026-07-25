#!/usr/bin/env bash
#
# SSH into the G1 board.

set -euo pipefail

# --- Arguments ------------------------------------------------------------------
# --overlay <file.mp4>  blend the video's first frame over the live viewer window
#                       (e.g. to line the camera up with a recorded episode)
OVERLAY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --overlay)   OVERLAY="${2:?--overlay needs a file}"; shift 2 ;;
        --overlay=*) OVERLAY="${1#--overlay=}"; shift ;;
        *) echo "Unknown argument: $1 (supported: --overlay <file.mp4>)" >&2; exit 1 ;;
    esac
done
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
VIEWER_ENV="psi_deploy"   # conda env for the host viewer
HOST_CONDA="$HOME/miniconda3"   # conda installation path on this host
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

# Start the client image viewer on the host. It runs in the background so this
# terminal can take neck-reset commands.
cd "$(dirname "$0")"
source "$HOST_CONDA/etc/profile.d/conda.sh"
conda activate "$VIEWER_ENV"
echo "Starting viewer (conda env: $VIEWER_ENV) ..."
VIEWER_ARGS=(--server "$IP" --port "$VIEWER_PORT" --show-stereo)
if [ -n "$OVERLAY" ]; then
    echo "Overlaying first frame of: $OVERLAY (alpha $OVERLAY_ALPHA)"
    VIEWER_ARGS+=(--overlay "$OVERLAY" --overlay-alpha "$OVERLAY_ALPHA")
    if [ "$OVERLAY_SWAP" = "1" ]; then
        VIEWER_ARGS+=(--overlay-swap-channels)
    fi
fi
python test_viewer.py "${VIEWER_ARGS[@]}" &
VIEWER_PID=$!

# Neck reset from this terminal, same UX as auto_deploy_neck_realsense.sh: type
# r+Enter to ease the neck back to (0, 0). reset_neck.py binds the pose PUB
# port itself, which is free here because this script runs no pose publisher
# (if one is running, reset_neck.py fails with a clear message and we go on).
echo
echo "Viewer running (pid $VIEWER_PID)."
echo "Type r+Enter to reset the neck to (0, 0); q+Enter or Ctrl+C to quit."
while kill -0 "$VIEWER_PID" 2>/dev/null; do
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
