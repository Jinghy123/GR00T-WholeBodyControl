#!/usr/bin/env bash
#
# SSH into the G1 board, start the RealSense native camera server (with neck
# motor + host pose stream), then run the live viewer on this host (SUB to the
# server's viewer PUB port). Auto-provisions the conda envs on both the board
# (offline) and this host if they don't exist yet.

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
CAMERA_MATCH="RealSense"        # lsusb substring identifying the RealSense camera
CONDA_ENV="realsense"           # conda env for the camera server on the board
CONDA_SH="~/miniforge3/etc/profile.d/conda.sh"   # conda profile script on the board
REMOTE_DIR="GR00T-WholeBodyControl"   # repo dir (relative to the board's HOME)
REMOTE_LOG="~/realsense_native_server.log"
POSE_ZMQ_PORT="5570"            # host pose stream port (--pose-zmq)
VIEWER_PORT="5559"              # server's viewer PUB port the host viewer subscribes to
VIEWER_ENV="realsense"          # conda env for the host image viewer
VIEWER_SCALE="2"                # viewer window scale (2 = twice as wide and tall)
OVERLAY_ALPHA="0.5"             # overlay opacity 0..1 (higher = overlay more visible)
OVERLAY_SWAP="1"                # 1 = swap overlay B<->R channels (false color) for contrast
HOST_CONDA="$HOME/miniconda3"   # conda installation path on this host

# --- Offline provisioning of the board's `realsense` conda env -----------------
# The board has no internet, so the env is created from its local conda pkg cache
# and pip wheels shipped from this host. pyrealsense2 is NOT pip-installable here
# (the PyPI aarch64 wheel enumerates 0 devices on this Jetson/Tegra board); the
# working librealsense build is copied from an existing board env instead.
# dynamixel-sdk (+ pyserial) is needed for --enable-neck-motor.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYREALSENSE_SRC_ENV="ruohai"     # board env holding the working pyrealsense2 build to copy
WHEELS_DIR="$SCRIPT_DIR/realsense_wheels_aarch64"   # host cache of aarch64 wheels
REQ_FILE="$SCRIPT_DIR/requirements-realsense-server.txt"
CLIENT_REQ_FILE="$SCRIPT_DIR/requirements-realsense-client.txt"   # host client pip deps
REMOTE_WHEELS="~/rs_wheels_aarch64"   # where wheels are staged on the board
PY_VER="3.10"
PY_PKGS=(numpy==2.2.6 opencv-python==4.13.0.92 pyzmq==27.1.0
         dynamixel-sdk==3.7.31 pyserial==3.5)   # pip deps (must match wheels)

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

# Check that the camera is mounted by looking for it in lsusb. A non-interactive
# ssh command skips the ROS prompt in .bashrc, so this is safe. Loop until the
# camera shows up, prompting the user to re-plug it each time.
until sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
        "${USERNAME}@${IP}" "lsusb | grep -qi '$CAMERA_MATCH'"; do
    echo "Error: RealSense camera ($CAMERA_MATCH) not detected via lsusb." >&2
    read -r -p "Please re-plug the camera, then press Enter to retry... "
done
echo "RealSense camera ($CAMERA_MATCH) detected."

# --- Ensure the board's `realsense` conda env exists (offline provisioning) -----
# Checked by importing every runtime dep, not just by env existence: an env left
# behind by auto_deploy_neckless.sh exists but lacks dynamixel-sdk, and the neck
# thread silently degrades to camera-only without it.
board_env_ok() {
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${IP}" \
        "source $CONDA_SH; conda activate $CONDA_ENV 2>/dev/null && \
         PYTHONNOUSERSITE=1 python -c 'import cv2, numpy, zmq, dynamixel_sdk, serial, pyrealsense2' 2>/dev/null"
}

ensure_board_env() {
    if board_env_ok; then
        echo "Board conda env '$CONDA_ENV' OK (all imports present)."
        return 0
    fi
    echo "Board conda env '$CONDA_ENV' missing or incomplete — provisioning (board stays offline)..."

    # 1. Make sure the host has the aarch64 wheels (download once; needs host
    #    internet the first time, then cached in $WHEELS_DIR for offline reuse).
    mkdir -p "$WHEELS_DIR"
    if [ "$(ls -1 "$WHEELS_DIR"/*.whl 2>/dev/null | wc -l)" -lt "${#PY_PKGS[@]}" ]; then
        echo "Downloading aarch64 wheels on the host -> $WHEELS_DIR ..."
        "$HOST_CONDA/bin/python" -m pip download --only-binary=:all: \
            --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 \
            --python-version 310 --abi cp310 -d "$WHEELS_DIR" "${PY_PKGS[@]}"
    else
        echo "Using cached aarch64 wheels in $WHEELS_DIR."
    fi

    # 2. Ship the wheels + requirements to the board.
    echo "Copying wheels to the board ..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${IP}" \
        "mkdir -p $REMOTE_WHEELS"
    sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no \
        "$WHEELS_DIR"/*.whl "$REQ_FILE" "${USERNAME}@${IP}:$REMOTE_WHEELS/" >/dev/null

    # 3. Create the env offline if missing (from the board's conda pkg cache),
    #    install/top-up the wheels, and copy the working pyrealsense2 build from
    #    $PYREALSENSE_SRC_ENV if it isn't importable yet. Idempotent, so it also
    #    repairs an incomplete env (e.g. one made by auto_deploy_neckless.sh).
    echo "Creating/repairing env + installing deps on the board ..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${IP}" bash -s <<EOF
set -e
source $CONDA_SH
if ! conda env list | grep -qE "/envs/$CONDA_ENV\\\$"; then
    conda create -n $CONDA_ENV --offline python=$PY_VER pip -y
fi
conda activate $CONDA_ENV
export PYTHONNOUSERSITE=1
# --ignore-installed so packages land in the env even if ~/.local has copies.
pip install --no-index --find-links $REMOTE_WHEELS --ignore-installed ${PY_PKGS[*]}

# pyrealsense2: copy the working (platform-specific) build from the source env.
if ! python -c "import pyrealsense2" 2>/dev/null; then
    SP="\$(conda info --base)/envs"
    SRC="\$SP/$PYREALSENSE_SRC_ENV/lib/python$PY_VER/site-packages"
    DST="\$SP/$CONDA_ENV/lib/python$PY_VER/site-packages"
    if [ ! -d "\$SRC/pyrealsense2" ]; then
        echo "ERROR: source env '$PYREALSENSE_SRC_ENV' has no pyrealsense2 to copy (\$SRC)." >&2
        exit 1
    fi
    rm -rf "\$DST/pyrealsense2" "\$DST"/pyrealsense2-*.dist-info
    cp -a "\$SRC/pyrealsense2" "\$DST/"
    cp -a "\$SRC"/pyrealsense2-*.dist-info "\$DST/" 2>/dev/null || true
fi

PYTHONNOUSERSITE=1 python -c "import cv2, numpy, zmq, dynamixel_sdk, serial, pyrealsense2 as rs; print('[provision] realsense env OK — devices=', len(rs.context().query_devices()))"
EOF

    if ! board_env_ok; then
        echo "Error: failed to provision board conda env '$CONDA_ENV'." >&2
        exit 1
    fi
    echo "Board conda env '$CONDA_ENV' provisioned."
}

# --- Ensure the host's `realsense` conda env exists (host has internet) ---------
# Assumes host conda is already sourced.
ensure_host_env() {
    if conda env list | grep -qE "/envs/$VIEWER_ENV\$"; then
        echo "Host conda env '$VIEWER_ENV' already exists."
        return 0
    fi
    echo "Host conda env '$VIEWER_ENV' not found — creating and installing client deps..."
    conda create -n "$VIEWER_ENV" -c conda-forge "python=$PY_VER" -y
    # --ignore-installed so deps land in the env even if ~/.local has copies.
    conda run -n "$VIEWER_ENV" python -m pip install --ignore-installed -r "$CLIENT_REQ_FILE"
    echo "Host conda env '$VIEWER_ENV' provisioned."
}

ensure_board_env

# Ensure the host viewer env exists too (host has internet, so plain pip install).
source "$HOST_CONDA/etc/profile.d/conda.sh"
ensure_host_env

# PROVISION_ONLY=1 stops here (useful to just create/verify the envs).
if [ "${PROVISION_ONLY:-0}" = "1" ]; then
    echo "PROVISION_ONLY set — envs are ready, exiting before server bring-up."
    exit 0
fi

# Start the camera server on the board, detached (nohup + background) so this
# script can go on to launch the viewer. We source conda directly instead of
# relying on an interactive shell, which avoids the .bashrc ROS prompt entirely.
# sudo -S reads the password from stdin. The last echo returns the server PID.
echo "Starting RealSense native server on $IP ..."
SERVER_PID="$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
        "${USERNAME}@${IP}" bash -s <<EOF | tail -1
echo "$PASSWORD" | sudo -S chmod 777 /dev/ttyUSB0 2>/dev/null
echo "$PASSWORD" | sudo -S killall -9 videohub_pc4 2>/dev/null
source $CONDA_SH
conda activate $CONDA_ENV
export PYTHONNOUSERSITE=1   # ignore ~/.local so the realsense env's deps win
cd ~/$REMOTE_DIR
nohup python -u realsense_native_server.py \
    --no-ir \
    --enable-neck-motor \
    --pose-zmq tcp://$HOST_IP:$POSE_ZMQ_PORT \
    > $REMOTE_LOG 2>&1 &
echo \$!
EOF
)" || true

if ! [[ "$SERVER_PID" =~ ^[0-9]+$ ]]; then
    echo "Error: failed to start RealSense native server (no PID returned)." >&2
    exit 1
fi
echo "RealSense native server started (pid $SERVER_PID, log: $REMOTE_LOG)."

# Stop the viewer and the remote server when this script exits.
VIEWER_PID=""
cleanup() {
    echo
    if [ -n "$VIEWER_PID" ]; then
        kill "$VIEWER_PID" 2>/dev/null || true
    fi
    echo "Stopping RealSense native server on $IP (pid $SERVER_PID) ..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
        "${USERNAME}@${IP}" "kill $SERVER_PID 2>/dev/null" || true
}
trap cleanup EXIT

# Wait for the server's viewer PUB port to start accepting connections.
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
        "${USERNAME}@${IP}" "tail -20 $REMOTE_LOG" >&2 || true
    echo "-----------------------" >&2
fi

# Start the live viewer on the host (SUB mode watches the viewer PUB without
# stealing frames from the recording/inference REP socket). Host conda was
# sourced (and the env ensured) above. The viewer runs in the background so
# this terminal can take neck-reset commands.
cd "$SCRIPT_DIR"
conda activate "$VIEWER_ENV"
export PYTHONNOUSERSITE=1   # ignore ~/.local so the realsense env's deps win
echo "Starting viewer (conda env: $VIEWER_ENV) ..."
VIEWER_ARGS=(--server "$IP" --port "$VIEWER_PORT" --sub --scale "$VIEWER_SCALE")
if [ -n "$OVERLAY" ]; then
    echo "Overlaying first frame of: $OVERLAY (alpha $OVERLAY_ALPHA)"
    VIEWER_ARGS+=(--overlay "$OVERLAY" --overlay-alpha "$OVERLAY_ALPHA")
    if [ "$OVERLAY_SWAP" = "1" ]; then
        VIEWER_ARGS+=(--overlay-swap-channels)
    fi
fi
python realsense_viewer.py "${VIEWER_ARGS[@]}" &
VIEWER_PID=$!

# Neck reset from this terminal, same UX as pico_manus_thread_server: type
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
                    --neck-state-zmq "tcp://$IP:5560" || true
                ;;
            q|quit|exit)
                break
                ;;
        esac
    fi
done
