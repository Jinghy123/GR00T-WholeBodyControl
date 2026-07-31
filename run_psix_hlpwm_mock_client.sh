#!/usr/bin/env bash
# Off-robot launcher for the production HLP + WM + RTC-VLA client.
#
# This does NOT run a simplified mock orchestration client. It starts byte-
# compatible fake G1 camera/state endpoints, then runs the exact production
# psix_rtc_sonic_hlpwm_client.py in --dry-run mode. HLP, WM, RTC provenance,
# hard gates, watchdogs, manual commands, logging and action decoding are real;
# only the WBC publication sink is replaced by NullAdapter.

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SONIC_ENV="${SONIC_ENV:-sonic}"
MOCK_CAMERA_PORT="${MOCK_CAMERA_PORT:-5558}"
MOCK_STATE_PORT="${MOCK_STATE_PORT:-5557}"
MOCK_CONTROL_PORT="${MOCK_CONTROL_PORT:-5559}"
MOCK_REPLAY_FPS="${MOCK_REPLAY_FPS:-30}"
MOCK_REPLAY_MODE="${MOCK_REPLAY_MODE:-hold}"
MOCK_ACTION_TRACE_PERIOD="${MOCK_ACTION_TRACE_PERIOD:-1.0}"
MOCK_FORCE_LEASE="${MOCK_FORCE_LEASE:-1}"
MOCK_LOG_DIR="${MOCK_LOG_DIR:-$ROOT/.logs/mock_g1}"

# Exact raw source for cleanup_table_1_episode_11. Its nine goal landmarks are
# frames 212,320,486,611,794,925,1095,1211,1318 (duck,tiger,corn,bottle,retract).
MOCK_EPISODE_DIR="${MOCK_EPISODE_DIR:-/mnt/data/weiduo/heng/data/demonstration_2026-07-01_19-25-55/episode_19}"

SCENE_MODE="${MOCK_SCENE_MODE:-dynamic}"
START_PAUSED=0
USE_SYNTHETIC=0
CLIENT_ARGS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [mock options] [production-client options...]

Mock options:
  --dynamic-scene       HLP owns object order; TASK_KEY is empty (default)
  --fixed-scene         enforce cleanup_table_1_episode_11 order in the client
  --manual              HLP off; Enter advances episode prompts manually
  --mock-episode DIR    recorded episode containing data.json + color/
  --synthetic           moving synthetic camera + sinusoidal state
  --start-paused        hold recorded replay at frame 0
  --replay-fps FPS      recorded replay rate (default: $MOCK_REPLAY_FPS)
  --no-action-trace     disable full 78-D action samples in client JSONL
  -h, --help            show this help

Safety: --real is rejected. This launcher always uses production --dry-run and
never creates the WBC publisher.

Required before launch:
  VLA http://127.0.0.1:8014       (real RTC server)
  HLP http://127.0.0.1:8015       (unless --manual)
  WM  http://192.168.123.240:8016 (production G1-wired direct path)

For a deliberate all-local unit test only, set:
  WM_HOST=127.0.0.1 PSIX_ALLOW_NON_G1_WM=1

Replay control while running:
  curl -s http://127.0.0.1:$MOCK_CONTROL_PORT/state
  curl -s -X POST http://127.0.0.1:$MOCK_CONTROL_PORT/pause
  curl -s -X POST 'http://127.0.0.1:$MOCK_CONTROL_PORT/goto?frame=486'
  curl -s -X POST http://127.0.0.1:$MOCK_CONTROL_PORT/resume
EOF
}

while (($#)); do
    case "$1" in
        --dynamic-scene) SCENE_MODE="dynamic" ;;
        --fixed-scene) SCENE_MODE="fixed" ;;
        --manual) SCENE_MODE="manual" ;;
        --mock-episode)
            (($# >= 2)) || { echo "[mock-launcher] --mock-episode needs DIR" >&2; exit 2; }
            MOCK_EPISODE_DIR="$2"
            shift
            ;;
        --mock-episode=*) MOCK_EPISODE_DIR="${1#*=}" ;;
        --synthetic) USE_SYNTHETIC=1 ;;
        --start-paused) START_PAUSED=1 ;;
        --replay-fps)
            (($# >= 2)) || { echo "[mock-launcher] --replay-fps needs FPS" >&2; exit 2; }
            MOCK_REPLAY_FPS="$2"
            shift
            ;;
        --replay-fps=*) MOCK_REPLAY_FPS="${1#*=}" ;;
        --no-action-trace) MOCK_ACTION_TRACE_PERIOD=0 ;;
        --real)
            echo "[mock-launcher] REFUSED: --real is not allowed in the mock launcher" >&2
            exit 2
            ;;
        -h|--help) usage; exit 0 ;;
        --)
            shift
            CLIENT_ARGS+=("$@")
            break
            ;;
        *) CLIENT_ARGS+=("$1") ;;
    esac
    shift
done

case "$SCENE_MODE" in
    dynamic)
        export HLP_MODE=active
        export TASK_KEY=""
        ;;
    fixed)
        export HLP_MODE=active
        export TASK_KEY="${TASK_KEY:-cleanup_table_1_episode_11}"
        ;;
    manual)
        export HLP_MODE=off
        export TASK_KEY="${TASK_KEY:-cleanup_table_1_episode_11}"
        ;;
    *)
        echo "[mock-launcher] invalid MOCK_SCENE_MODE=$SCENE_MODE" >&2
        exit 2
        ;;
esac

for port in "$MOCK_CAMERA_PORT" "$MOCK_STATE_PORT" "$MOCK_CONTROL_PORT"; do
    if ss -H -ltn "sport = :$port" | grep -q .; then
        echo "[mock-launcher] local port $port is already occupied" >&2
        ss -H -ltnp "sport = :$port" >&2 || true
        exit 1
    fi
done

if ((USE_SYNTHETIC == 0)); then
    [[ -f "$MOCK_EPISODE_DIR/data.json" ]] || {
        echo "[mock-launcher] missing $MOCK_EPISODE_DIR/data.json" >&2
        exit 1
    }
    [[ -d "$MOCK_EPISODE_DIR/color" ]] || {
        echo "[mock-launcher] missing $MOCK_EPISODE_DIR/color" >&2
        exit 1
    }
fi

command -v conda >/dev/null 2>&1 || {
    echo "[mock-launcher] conda not found" >&2
    exit 1
}
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$SONIC_ENV"

mkdir -p "$MOCK_LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
MOCK_LOG="$MOCK_LOG_DIR/mock_g1_${STAMP}.log"

MOCK_ARGS=(
    --bind-host 127.0.0.1
    --camera-port "$MOCK_CAMERA_PORT"
    --state-port "$MOCK_STATE_PORT"
    --control-host 127.0.0.1
    --control-port "$MOCK_CONTROL_PORT"
    --replay-fps "$MOCK_REPLAY_FPS"
    --replay-mode "$MOCK_REPLAY_MODE"
)
if ((USE_SYNTHETIC == 0)); then
    MOCK_ARGS+=(--episode-dir "$MOCK_EPISODE_DIR")
fi
if ((START_PAUSED)); then
    MOCK_ARGS+=(--start-paused)
fi

python -u "$ROOT/mock_g1_obs.py" "${MOCK_ARGS[@]}" \
    </dev/null >"$MOCK_LOG" 2>&1 &
MOCK_PID=$!

cleanup() {
    if kill -0 "$MOCK_PID" 2>/dev/null; then
        kill "$MOCK_PID" 2>/dev/null || true
        wait "$MOCK_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

deadline=$((SECONDS + 10))
while ((SECONDS < deadline)); do
    if ! kill -0 "$MOCK_PID" 2>/dev/null; then
        echo "[mock-launcher] mock G1 process exited; log follows:" >&2
        sed -n '1,160p' "$MOCK_LOG" >&2
        exit 1
    fi
    if ss -H -ltn "sport = :$MOCK_CAMERA_PORT" | grep -q . \
            && ss -H -ltn "sport = :$MOCK_STATE_PORT" | grep -q . \
            && curl --noproxy '*' -fsS --max-time 1 \
                "http://127.0.0.1:$MOCK_CONTROL_PORT/state" >/dev/null; then
        break
    fi
    sleep 0.1
done

if ! curl --noproxy '*' -fsS --max-time 1 \
        "http://127.0.0.1:$MOCK_CONTROL_PORT/state" >/dev/null; then
    echo "[mock-launcher] fake G1 endpoints did not become ready; see $MOCK_LOG" >&2
    exit 1
fi

export CAMERA_ADDRESS="tcp://127.0.0.1:$MOCK_CAMERA_PORT"
export WM_HOST="${WM_HOST:-${WM_LOCAL_HOST:-192.168.123.240}}"
export WM_PORT="${WM_PORT:-${WM_LOCAL_PORT:-8016}}"

echo "[mock-launcher] fake G1 ready (camera=$MOCK_CAMERA_PORT state=$MOCK_STATE_PORT)"
echo "[mock-launcher] hardware log: $MOCK_LOG"
echo "[mock-launcher] replay control: http://127.0.0.1:$MOCK_CONTROL_PORT/state"
echo "[mock-launcher] scene mode: $SCENE_MODE; WBC publication: DISABLED"
if ((USE_SYNTHETIC == 0)); then
    echo "[mock-launcher] episode: $MOCK_EPISODE_DIR @ $MOCK_REPLAY_FPS fps"
    echo "[mock-launcher] episode11 landmarks: 0,212,320,486,611,794,925,1095,1211,1318"
fi

FORCE_ARGS=()
if [[ "$HLP_MODE" != "off" && "$MOCK_FORCE_LEASE" == "1" ]]; then
    FORCE_ARGS+=(--force-lease)
fi

"$ROOT/run_psix_hlpwm_client.sh" \
    --dry-run \
    --hlp-mode "$HLP_MODE" \
    "${FORCE_ARGS[@]}" \
    --zmq-host 127.0.0.1 \
    --zmq-sub-port "$MOCK_STATE_PORT" \
    --zmq-sub-topic g1_debug \
    --camera-address "$CAMERA_ADDRESS" \
    --action-trace-period "$MOCK_ACTION_TRACE_PERIOD" \
    "${CLIENT_ARGS[@]}"
