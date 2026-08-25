#!/usr/bin/env bash
# Launch the whole sonic stack in ONE tmux window, 4 tiled panes (2x2), so you
# see all programs at once. Each pane runs an interactive shell, so Ctrl-C only
# stops that program (the shell/ssh stays alive for restart).
#
# Usage:
#   ./start_sonic.sh           # create session, start all panes, attach
#   tmux kill-session -t sonic # tear everything down
set -euo pipefail

SESSION="sonic"
G1_PASS="123"                       # g1 login + sudo password
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # repo root (this script's dir)

# Read-only fail-closed preflight before any tmux pane is created. Activate the
# profile explicitly beforehand with ./g1_teleop_network.sh activate.
"$REPO_DIR/g1_teleop_network.sh" check

# Send a command line into pane $1 of the main window (types it + Enter).
send() { tmux send-keys -t "${SESSION}:main.$1" "$2" C-m; }

# ── (re)create the session with 4 tiled panes ─────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[start_sonic] session '${SESSION}' already exists."
  echo "[start_sonic] attach: tmux attach -t ${SESSION}   |   stop: tmux kill-session -t ${SESSION}"
  exit 1
fi
tmux new-session -d -s "${SESSION}" -n main   # pane 0
tmux split-window -t "${SESSION}:main"        # pane 1
tmux split-window -t "${SESSION}:main"        # pane 2
tmux split-window -t "${SESSION}:main"        # pane 3
tmux select-layout -t "${SESSION}:main" tiled
tmux set-option -t "${SESSION}" mouse on               # click to focus a pane, drag borders, scroll
tmux set-option -t "${SESSION}" pane-border-status top 2>/dev/null || true

# ── Pane 0: G1 onboard — camera + neck server ─────────────────────────────
tmux select-pane -t "${SESSION}:main.0" -T "g1 (camera+neck)"
send 0 "sshpass -p ${G1_PASS} ssh -t -o StrictHostKeyChecking=accept-new g1"
sleep 2
send 0 "echo ${G1_PASS} | sudo -S chmod 777 /dev/ttyUSB0"
send 0 "export LD_PRELOAD=/lib/aarch64-linux-gnu/libffi.so.7"
send 0 "conda activate sonic"
send 0 "cd ~/GR00T-WholeBodyControl"
send 0 "python realsense_server.py \
    --zed-only \
    --zmq-bind tcp://0.0.0.0:5558 \
    --enable-pico --pico-ip 192.168.0.241 \
    --enable-neck-motor \
    --pose-zmq tcp://192.168.123.222:5570"

# ── Pane 1: Desktop — WBC deploy (gear_sonic_deploy) ──────────────────────
tmux select-pane -t "${SESSION}:main.1" -T "deploy (WBC)"
send 1 "cd ${REPO_DIR}/gear_sonic_deploy"
send 1 "source scripts/setup_env.sh"
send 1 "./deploy.sh --input-type zmq enp4s0"

# ── Pane 2: Desktop — body/hand/neck server (SlimeVR + Manus) ──────────────
tmux select-pane -t "${SESSION}:main.2" -T "manus (SlimeVR)"
send 2 "cd ${REPO_DIR}"
send 2 "export GR00T_ROOT=\"\$PWD\""
send 2 "export PYTHONPATH=\"\$GR00T_ROOT/external_dependencies/gmr_shim:\$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:\$PYTHONPATH\""
send 2 "export LD_LIBRARY_PATH=\"\$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:\$LD_LIBRARY_PATH\""
send 2 "export PYTHONPATH=\$PWD/GMR:\$PYTHONPATH"
send 2 "source .venv_teleop/bin/activate"
send 2 "python gear_sonic/scripts/slimevr_manus_thread_server.py"

# ── Pane 3: Desktop — data recorder (neck cmd + state) ─────────────────────
tmux select-pane -t "${SESSION}:main.3" -T "record (data)"
send 3 "cd ${REPO_DIR}"
send 3 "source .venv_teleop/bin/activate"
send 3 "python g1_data_server.py \
    --neck-zmq       tcp://localhost:5570 \
    --neck-state-zmq tcp://192.168.123.164:5560"

echo "[start_sonic] session '${SESSION}' started: 4 tiled panes (g1 / deploy / manus / record)."
echo "[start_sonic] attaching now (Ctrl-b o to move between panes, Ctrl-b d to detach, 'tmux kill-session -t ${SESSION}' to stop all)."
tmux attach -t "${SESSION}"
