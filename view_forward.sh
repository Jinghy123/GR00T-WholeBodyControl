#!/usr/bin/env bash
# Run this on the REMOTE computer to watch the G1 ZED viewer stream.
#
# realsense_server.py on the robot publishes the live ZED frames on a ZMQ PUB
# at <G1>:5559 (separate from the 5558 recording socket). This script opens an
# SSH tunnel through the desktop (the machine wired to the G1), then runs
# test_viewer.py locally against the tunneled port. Only JPEG bytes cross the
# wire; decoding/imshow happens on this machine.
#
# Requirements on THIS (remote) machine:
#   - python with: opencv-python (cv2), numpy, pyzmq
#   - a copy of test_viewer.py (this repo) next to this script
#
# Usage:
#   ./view_forward.sh user@desktop-ip            # desktop = host wired to the G1
#   G1_HOST=192.168.123.164 PORT=5559 ./view_forward.sh user@desktop-ip
set -euo pipefail

DESKTOP_SSH="${1:-}"
G1_HOST="${G1_HOST:-192.168.123.164}"   # realsense_server.py PUB host (the robot)
PORT="${PORT:-5559}"                    # viewer PUB port
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIEWER="${HERE}/test_viewer.py"

if [[ -z "${DESKTOP_SSH}" ]]; then
  echo "usage: $0 user@desktop-ip   (the desktop machine that is wired to the G1)"
  exit 1
fi
if [[ ! -f "${VIEWER}" ]]; then
  echo "error: test_viewer.py not found at ${VIEWER} (copy this repo's test_viewer.py here)"
  exit 1
fi

# 1) SSH tunnel  localhost:PORT -> (desktop) -> G1:PORT, managed via a control socket
CTRL="$(mktemp -u /tmp/sonic-view-XXXXXX.ctl)"
echo "[view] opening tunnel 127.0.0.1:${PORT} -> ${G1_HOST}:${PORT} via ${DESKTOP_SSH}"
ssh -fN -M -S "${CTRL}" -o ExitOnForwardFailure=yes \
    -L "127.0.0.1:${PORT}:${G1_HOST}:${PORT}" "${DESKTOP_SSH}"

cleanup() {
  echo "[view] closing tunnel"
  ssh -S "${CTRL}" -O exit "${DESKTOP_SSH}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 2) run the viewer against the local end of the tunnel
echo "[view] starting viewer (q or ESC to quit)"
python "${VIEWER}" --server 127.0.0.1 --port "${PORT}" --show-stereo
