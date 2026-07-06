#!/usr/bin/env bash
#
# Auto bring-up of the RealSense camera server on the G1 board (neckless variant).
#
# SSHes into the board, verifies the RealSense is plugged in, then launches
# realsense_native_server.py detached. The server is stopped again when this
# script exits (Ctrl+C).

set -euo pipefail

USERNAME="unitree"
IP="192.168.123.164"
PASSWORD="123"
CAMERA_MATCH="RealSense"        # lsusb substring identifying the RealSense camera
CONDA_ENV="realsense"          # conda env for the camera server on the board
CONDA_SH="~/miniforge3/etc/profile.d/conda.sh"   # conda profile script on the board
REMOTE_DIR="GR00T-WholeBodyControl"   # repo dir (relative to the board's HOME)
REMOTE_LOG="~/realsense_native_server.log"
VIEWER_ENV="realsense"           # conda env for the host image viewer
HOST_CONDA="$HOME/miniconda3"    # conda installation path on this host

# --- Offline provisioning of the board's `realsense` conda env -----------------
# The board has no internet, so the env is created from its local conda pkg cache
# and pip wheels shipped from this host. pyrealsense2 is NOT pip-installable here
# (the PyPI aarch64 wheel enumerates 0 devices on this Jetson/Tegra board); the
# working librealsense build is copied from an existing board env instead.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYREALSENSE_SRC_ENV="ruohai"     # board env holding the working pyrealsense2 build to copy
WHEELS_DIR="$SCRIPT_DIR/realsense_wheels_aarch64"   # host cache of aarch64 wheels
REQ_FILE="$SCRIPT_DIR/requirements-realsense-server.txt"
CLIENT_REQ_FILE="$SCRIPT_DIR/requirements-realsense-client.txt"   # host client pip deps
REMOTE_WHEELS="~/rs_wheels_aarch64"   # where wheels are staged on the board
PY_VER="3.10"
PY_PKGS=(numpy==2.2.6 opencv-python==4.13.0.92 pyzmq==27.1.0)   # pip deps (must match wheels)

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
board_env_exists() {
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${IP}" \
        "source $CONDA_SH; conda env list | grep -qE \"/envs/$CONDA_ENV\\\$\""
}

ensure_board_env() {
    if board_env_exists; then
        echo "Board conda env '$CONDA_ENV' already exists."
        return 0
    fi
    echo "Board conda env '$CONDA_ENV' not found — provisioning (board stays offline)..."

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

    # 3. Create the env offline (from the board's conda pkg cache), install the
    #    wheels, and copy the working pyrealsense2 build from $PYREALSENSE_SRC_ENV.
    echo "Creating env + installing deps on the board ..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "${USERNAME}@${IP}" bash -s <<EOF
set -e
source $CONDA_SH
conda create -n $CONDA_ENV --offline python=$PY_VER pip -y
conda activate $CONDA_ENV
export PYTHONNOUSERSITE=1
# --ignore-installed so packages land in the env even if ~/.local has copies.
pip install --no-index --find-links $REMOTE_WHEELS --ignore-installed ${PY_PKGS[*]}

# pyrealsense2: copy the working (platform-specific) build from the source env.
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

PYTHONNOUSERSITE=1 python -c "import cv2, numpy, zmq, pyrealsense2 as rs; print('[provision] realsense env OK — devices=', len(rs.context().query_devices()))"
EOF

    if ! board_env_exists; then
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
# script can go on. We source conda directly instead of relying on an interactive
# shell, which avoids the .bashrc ROS prompt entirely. The last echo returns the
# server PID.
echo "Starting RealSense native server on $IP ..."
SERVER_PID="$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
        "${USERNAME}@${IP}" bash -s <<EOF | tail -1
source $CONDA_SH
conda activate $CONDA_ENV
export PYTHONNOUSERSITE=1   # ignore ~/.local so the realsense env's deps win
cd ~/$REMOTE_DIR
nohup python realsense_native_server.py --no-depth > $REMOTE_LOG 2>&1 &
echo \$!
EOF
)" || true

if ! [[ "$SERVER_PID" =~ ^[0-9]+$ ]]; then
    echo "Error: failed to start RealSense native server (no PID returned)." >&2
    exit 1
fi
echo "RealSense native server started (pid $SERVER_PID, log: $REMOTE_LOG)."

# Stop the remote server when this script exits (e.g. on Ctrl+C).
cleanup() {
    echo
    echo "Stopping RealSense native server on $IP (pid $SERVER_PID) ..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
        "${USERNAME}@${IP}" "kill $SERVER_PID 2>/dev/null" || true
}
trap cleanup EXIT

# Give the server a moment to initialize the camera before the viewer subscribes.
sleep 3

# Launch the image viewer on the host to test the camera connection. Runs in the
# foreground; closing it (Ctrl+C) tears the server down via the cleanup trap above.
# Host conda was sourced (and the env ensured) above.
cd "$SCRIPT_DIR"
conda activate "$VIEWER_ENV"
export PYTHONNOUSERSITE=1   # ignore ~/.local so the realsense env's deps win
echo "Starting image viewer (conda env: $VIEWER_ENV) ..."
python realsense_viewer.py --server "$IP" --sub
