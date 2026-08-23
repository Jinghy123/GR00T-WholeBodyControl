#!/bin/bash
# Replay one episode of a LeRobot dataset in the MuJoCo sim.
#
#   usage: sim_replay/replay_lerobot.sh [dataset_root] [episode_index]
#                                       [--max-allowed-frames N]
#
# --max-allowed-frames truncates the episode to its first N frames: only those
# are pulled out of the parquet, so a long episode can be smoke-tested without
# extracting (or replaying) the whole thing. The work files and the video are
# suffixed _f<N> in that case, so a truncated run never overwrites the full one.
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

MAX_FRAMES=""
POS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --max-allowed-frames)
            [ $# -ge 2 ] || { echo "$0: --max-allowed-frames needs a value" >&2; exit 2; }
            MAX_FRAMES="$2"; shift 2 ;;
        --max-allowed-frames=*)
            MAX_FRAMES="${1#*=}"; shift ;;
        -h|--help)
            sed -n '2,17p' "$0"; exit 0 ;;
        --) shift; POS+=("$@"); break ;;
        -*) echo "$0: unknown option $1" >&2; exit 2 ;;
        *)  POS+=("$1"); shift ;;
    esac
done
set -- ${POS+"${POS[@]}"}

if [ -n "$MAX_FRAMES" ]; then
    case "$MAX_FRAMES" in
        ''|*[!0-9]*) echo "$0: --max-allowed-frames must be a positive integer" >&2; exit 2 ;;
    esac
    [ "$MAX_FRAMES" -gt 0 ] || { echo "$0: --max-allowed-frames must be > 0" >&2; exit 2; }
fi

ROOT="${1:-$REPO/.data/g1_sonic_lerobot_0810_merged_val}"
EP="${2:-0}"
WORK_DIR="${WORK_DIR:-$REPO/.data/g1_sim_replay}"
# Neither repo venv has pyarrow; point this at any python that does.
PYARROW_PY="${PYARROW_PY:-/home/songlin/Projects/cosmos/.venv/bin/python}"
mkdir -p "$WORK_DIR"

# Truncated runs get their own file names so they never clobber a full episode.
SUFFIX=""
EXTRACT_ARGS=()
if [ -n "$MAX_FRAMES" ]; then
    SUFFIX="_f$MAX_FRAMES"
    EXTRACT_ARGS=(--max-frames "$MAX_FRAMES")
fi
STEM="$WORK_DIR/lerobot_ep$EP$SUFFIX"

"$PYARROW_PY" "$SR/lerobot_extract.py" "$ROOT" "$EP" "$STEM.npz" \
    ${EXTRACT_ARGS+"${EXTRACT_ARGS[@]}"}
"$REPO/.venv_teleop/bin/python" "$SR/lerobot_to_replay.py" "$STEM.npz" "$STEM.pkl"

# name the viewer recording after the dataset, not after the throwaway pickle,
# so replays of different datasets do not overwrite each other's video
VIDEO="${VIDEO:-$WORK_DIR/$(basename "${ROOT%/}")_ep$EP$SUFFIX.mp4}"
export VIDEO
exec "$SR/replay_in_mujoco.sh" "$STEM.pkl"
