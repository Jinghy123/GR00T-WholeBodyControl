#!/usr/bin/env bash
#
# SSH into the G1 board.

set -euo pipefail

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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sonic
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

# Stop the remote server when this script exits (e.g. when the viewer is closed).
cleanup() {
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

# Start the client image viewer on the host.
cd "$(dirname "$0")"
source "$HOST_CONDA/etc/profile.d/conda.sh"
conda activate "$VIEWER_ENV"
echo "Starting viewer (conda env: $VIEWER_ENV) ..."
python test_viewer.py --server "$IP" --port "$VIEWER_PORT" --show-stereo
