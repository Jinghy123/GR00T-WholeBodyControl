#!/bin/bash
# Replay one episode of a LeRobot dataset in the MuJoCo sim.
#
#   usage: sim_replay/replay_lerobot.sh [dataset_root] [episode_index]
#
# The MuJoCo viewer is recorded for the episode to
# $WORK_DIR/<dataset>_ep<N>.mp4; NO_VIDEO=1 skips it, VIDEO=<path> overrides.
#
# The dataset's action layout (meta/modality.json) is hand(14)+neck(2)+token(64)
# at 30 Hz - identical to record_sonic.py's include_neck layout - so the episode
# only needs repacking into a replay pickle, no re-ordering or rescaling.
set -eu

REPO="$(cd "$(dirname "$(dirname "$(readlink -f "$0")")")" && pwd)"
SR="$REPO/sim_replay"
ROOT="${1:-$REPO/.data/g1_sonic_lerobot_0810_merged_val}"
EP="${2:-0}"
WORK_DIR="${WORK_DIR:-/tmp/g1_sim_replay}"
# Neither repo venv has pyarrow; point this at any python that does.
PYARROW_PY="${PYARROW_PY:-/home/songlin/Projects/cosmos/.venv/bin/python}"
mkdir -p "$WORK_DIR"

"$PYARROW_PY" "$SR/lerobot_extract.py" "$ROOT" "$EP" "$WORK_DIR/lerobot_ep$EP.npz"
"$REPO/.venv_teleop/bin/python" "$SR/lerobot_to_replay.py" \
    "$WORK_DIR/lerobot_ep$EP.npz" "$WORK_DIR/lerobot_ep$EP.pkl"

# name the viewer recording after the dataset, not after the throwaway pickle,
# so replays of different datasets do not overwrite each other's video
VIDEO="${VIDEO:-$WORK_DIR/$(basename "${ROOT%/}")_ep$EP.mp4}"
export VIDEO
exec "$SR/replay_in_mujoco.sh" "$WORK_DIR/lerobot_ep$EP.pkl"
