#!/bin/bash
# Replay a record_sonic.py episode in the MuJoCo sim, end to end.
#
#   usage: sim_replay/replay_in_mujoco.sh [episode.pkl]
#
# Brings up three processes:
#   1) run_sim_loop_dropctl.py   MuJoCo sim (.venv_sim), DDS + LowState on lo
#   2) run_wbc_deploy.sh         WBC, SUB tokens/commands :5556, state PUB :5557
#   3) start_control.py          start command only, so the policy is holding the
#                                robot BEFORE its feet touch the ground
#   4) replay_sonic.py           BIND :5556 (tokens) + :5570 (neck)
#
# The ordering in 3/4 is not cosmetic: dropping the robot before control starts
# gives you a limp robot that falls and resets, and dropping it after the tokens
# start gives you a release transient mid-episode.
#
# replay_sonic.py runs with --no-stop so WBC control survives the last tick;
# without it the stop command ends control and the robot collapses on the spot.
#
# The viewer window is screen-recorded for the length of the episode into
# $LOG_DIR/<episode>.mp4 (VIDEO=<path> to change it, NO_VIDEO=1 to skip).
#
# When the episode ends the WBC and the sim (which owns the MuJoCo viewer window)
# are both shut down. Set KEEP_SIM=1 to leave them running instead.
set -u

REPO="$(cd "$(dirname "$(dirname "$(readlink -f "$0")")")" && pwd)"
SR="$REPO/sim_replay"
EPISODE="${1:-$REPO/recordings/run1.pkl}"
LOG_DIR="${LOG_DIR:-/tmp/g1_sim_replay}"
DROP_FLAG="${DROP_FLAG:-/tmp/g1_sim_drop}"
export DROP_FLAG
mkdir -p "$LOG_DIR"

# The startup stages take ~60 s and print nothing of their own (deploy/sim output
# goes to $LOG_DIR), so each one drives a progress bar. `_wait_bar <label> <secs>
# <test-cmd>` polls test-cmd once a second and returns as soon as it succeeds,
# giving up after <secs>; <secs> also sets the bar's full width, so a stage that
# finishes early just stops short. On a non-tty it degrades to a line every 10 s.
_wait_bar() {
  local label="$1" tot="$2" test_cmd="$3" w=30 i=0 filled empty
  while [ "$i" -lt "$tot" ]; do
    eval "$test_cmd" && break
    sleep 1
    i=$((i + 1))
    if [ -t 2 ]; then
      local f=$((i * w / tot))
      printf -v filled '%*s' "$f" ''
      printf -v empty '%*s' "$((w - f))" ''
      printf '\r[%s] |%s%s| %2ds/%ds' \
          "$label" "${filled// /#}" "${empty// /.}" "$i" "$tot" >&2
    elif [ $((i % 10)) -eq 0 ]; then
      echo "[$label] ${i}s/${tot}s" >&2
    fi
  done
  eval "$test_cmd" && local ok=0 || local ok=1
  [ -t 2 ] && printf '\r[%s] %s after %ds%*s\n' \
      "$label" "$([ $ok -eq 0 ] && echo done || echo TIMED OUT)" "$i" 20 '' >&2
  return $ok
}

# --- viewer recording --------------------------------------------------------
# The MuJoCo viewer is an ordinary X11 window, so the episode is captured with
# ffmpeg's x11grab bound to that window's id - not to a screen region, which
# would break the moment the window is moved. It is a screen grab either way:
# anything covering the viewer covers the recording too. NO_VIDEO=1 skips it,
# VIDEO=<path> picks the file (default $LOG_DIR/<episode>.mp4).
VIDEO="${VIDEO:-$LOG_DIR/$(basename "${EPISODE%.pkl}").mp4}"
FFPID=""

_start_recording() {
  [ "${NO_VIDEO:-0}" = "1" ] && return 0
  command -v ffmpeg >/dev/null || { echo "[record] no ffmpeg, not recording"; return 0; }
  command -v xwininfo >/dev/null || { echo "[record] no xwininfo, not recording"; return 0; }
  [ -n "${DISPLAY:-}" ] || { echo "[record] no DISPLAY, not recording"; return 0; }
  local wid
  wid=$(xwininfo -root -tree 2>/dev/null | grep -m1 '"MuJoCo' | awk '{print $1}')
  [ -n "$wid" ] || { echo "[record] no MuJoCo window found, not recording"; return 0; }
  # pad to even dimensions: yuv420p rejects an odd-sized window
  ffmpeg -nostdin -loglevel warning -y -f x11grab -framerate 30 -window_id "$wid" \
      -i "$DISPLAY" -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
      -c:v libx264 -preset veryfast -pix_fmt yuv420p "$VIDEO" \
      > "$LOG_DIR/record.log" 2>&1 &
  FFPID=$!
  sleep 1
  kill -0 "$FFPID" 2>/dev/null || {
    echo "[record] ffmpeg exited immediately, see $LOG_DIR/record.log"; FFPID=""; return 0; }
  echo "[record] capturing viewer window $wid -> $VIDEO"
}

# SIGINT, not SIGKILL: ffmpeg has to write the mp4 index on the way out or the
# file is unplayable. Also runs from the EXIT trap, so a Ctrl-C still saves.
_stop_recording() {
  [ -n "$FFPID" ] || return 0
  kill -INT "$FFPID" 2>/dev/null
  wait "$FFPID" 2>/dev/null || true
  FFPID=""
  [ -s "$VIDEO" ] && echo "[record] saved $VIDEO ($(du -h "$VIDEO" | cut -f1))"
  return 0
}
trap _stop_recording EXIT INT TERM

# 1) sim (reuse a running one)
if ! pgrep -f "run_sim_loop_dropctl.py" >/dev/null; then
  rm -f "$DROP_FLAG"
  : > "$LOG_DIR/sim.log"        # so the readiness poll cannot match a stale run
  (cd "$REPO" && nohup .venv_sim/bin/python "$SR/run_sim_loop_dropctl.py" \
      > "$LOG_DIR/sim.log" 2>&1 &)
  # the sim is up once its 1 Hz pelvis trace starts; keep a few seconds of margin
  _wait_bar "sim startup" 30 "grep -q '\[trace\]' '$LOG_DIR/sim.log' 2>/dev/null" || {
    echo "sim failed to start, see $LOG_DIR/sim.log"; exit 1; }
  sleep 3
fi

# 2) WBC deploy, band still holding the robot up. Any deploy left from a previous
#    replay is stopped first - a second one would fight it for :5557.
pkill -f target/release/g1_deploy_onnx_ref 2>/dev/null || true
rm -f "$DROP_FLAG"; sleep 3
: > "$LOG_DIR/deploy.log"       # ditto: a stale "Init Done" would end the wait early
nohup "$SR/run_wbc_deploy.sh" > "$LOG_DIR/deploy.log" 2>&1 </dev/null &
_wait_bar "WBC init" 80 "grep -q 'Init Done' '$LOG_DIR/deploy.log' 2>/dev/null" || {
  echo "deploy failed to init, see $LOG_DIR/deploy.log"; exit 1; }

# 3) start control, lower the robot onto its feet, let it settle. start_control's
#    own output goes to a log so it does not cut through the progress bar.
(cd "$REPO" && .venv_teleop/bin/python -u "$SR/start_control.py" 14 \
    > "$LOG_DIR/start_control.log" 2>&1) &
PRELUDE=$!
sleep 3; touch "$DROP_FLAG"
_wait_bar "lowering robot onto the ground" 30 "! kill -0 $PRELUDE 2>/dev/null" || true
wait $PRELUDE 2>/dev/null || true

# 4) stream the episode, recording the viewer for exactly its duration
echo "[replay_in_mujoco] streaming: $EPISODE"
_start_recording
(cd "$REPO" && .venv_teleop/bin/python -u replay_sonic.py \
    --in "$EPISODE" --warmup 2.0 --no-stop 2>&1 | tee "$LOG_DIR/replay.log")
_stop_recording

# 5) tear down: WBC first (so it stops writing to a sim that is going away), then
#    the sim itself, which is what owns the MuJoCo viewer window. KEEP_SIM=1 keeps
#    both up if you want to poke at the final pose.
if [ "${KEEP_SIM:-0}" = "1" ]; then
  echo "[replay_in_mujoco] episode done; WBC still running and holding the pose."
  echo "                   logs: $LOG_DIR   stop WBC: pkill -f g1_deploy_onnx_ref"
else
  echo "[replay_in_mujoco] episode done; closing WBC and the MuJoCo viewer..."
  pkill -f target/release/g1_deploy_onnx_ref 2>/dev/null || true
  pkill -f "$SR/run_sim_loop_dropctl.py" 2>/dev/null || true
  for _ in $(seq 1 20); do
    pgrep -f "run_sim_loop_dropctl.py" >/dev/null || break
    sleep 0.5
  done
  pkill -9 -f "run_sim_loop_dropctl.py" 2>/dev/null || true
  rm -f "$DROP_FLAG"
  echo "                   logs: $LOG_DIR   (KEEP_SIM=1 leaves the sim running)"
fi
