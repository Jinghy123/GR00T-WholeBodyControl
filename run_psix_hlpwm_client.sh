#!/usr/bin/env bash
# Launcher for the HLP + WM + RTC-VLA robot client.
#
# Responsibilities:
#   1. Fail closed unless the production G1-wired link is active, then verify
#      VLA (:8014), HLP (:8015, unless mode=off), and direct WM (.240:8016).
#   2. Activate the sonic conda environment. HLP active/shadow launches the
#      combined client; HLP off launches the WM/manual client (same entrypoint).
#
# It never starts SSH and never changes DNS, routes, addresses, or profiles.

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VLA_HOST="${VLA_HOST:-127.0.0.1}"
VLA_PORT="${VLA_PORT:-8014}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-}"
HLP_HOST="${HLP_HOST:-127.0.0.1}"
HLP_PORT="${HLP_PORT:-8015}"
HLP_MODE="${HLP_MODE:-off}"
GOAL_SOURCE="${GOAL_SOURCE:-wm}"

WM_HOST="${WM_HOST:-${WM_LOCAL_HOST:-192.168.123.240}}"
WM_PORT="${WM_PORT:-${WM_LOCAL_PORT:-8016}}"
WM_READY_TIMEOUT="${WM_READY_TIMEOUT:-5}"
PSIX_ALLOW_NON_G1_WM="${PSIX_ALLOW_NON_G1_WM:-0}"

SONIC_ENV="${SONIC_ENV:-sonic}"
PROFILE="${PROFILE:-$ROOT/profiles/cleanup_table_fine_v1.json}"
PROFILE_HASH="${PROFILE_HASH:-80b1aa7c2b71e314aa88dffa080b689957dbe8c6a8aedc4a4c66e81bc1b8cb78}"
# Whether the operator pinned these explicitly. Recorded BEFORE defaults are
# applied, because afterwards "unset" and "set to the default" look identical —
# and only an unset one may be auto-resolved from the task key below.
_DATA_ROOT_PINNED="${DATA_ROOT+1}"
_PROMPTS_JSON_PINNED="${PROMPTS_JSON+1}"
_EPISODE_DIR_PINNED="${EPISODE_DIR+1}"
# Auto-resolution applies only to a key the operator actually chose (TASK_KEY in
# the environment, or --task). The built-in default key exists in several
# prompts.json files and has always been disambiguated by the DATA_ROOT default
# below, so resolving it would turn a working bare invocation into an ambiguity
# error. --task sets this too, in the argument loop.
_TASK_KEY_CHOSEN="${TASK_KEY+1}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/real_pick_place_0709_psix_train_val_general_prompt}"
PROMPTS_JSON="${PROMPTS_JSON:-$DATA_ROOT/prompts.json}"
TASK_KEY="${TASK_KEY-real_pick_place_0709_psix_val_episode_0}"
EPISODE_DIR="${EPISODE_DIR:-$DATA_ROOT/${TASK_KEY:-real_pick_place_0709_psix_val_episode_0}}"
# Replaces prompts.json[TASK_KEY].task_description for this run. TASK_KEY still
# selects the entry, because the per-stage subtasks always come from the file —
# only the task text is overridden. Empty means "use the file's text".
INPUT_PROMPT="${INPUT_PROMPT:-}"
CAMERA_ADDRESS="${CAMERA_ADDRESS:-tcp://192.168.123.164:5558}"
SUBGOAL_LOG_DIR="${SUBGOAL_LOG_DIR:-/home/weiduoyuan/Desktop/psi/.logs/HLP_WM_logs}"
WM_DUMP_DIR="${WM_DUMP_DIR:-/home/weiduoyuan/Desktop/psi/.logs/bagel_gen_images/$(date +%Y%m%d-%H%M%S)}"
NECK_PUB_HOST="${NECK_PUB_HOST:-*}"
NECK_PUB_PORT="${NECK_PUB_PORT:-5570}"
NECK_STATE_ZMQ="${NECK_STATE_ZMQ:-tcp://192.168.123.164:5560}"
WM_GOAL_HARD_AGE="${WM_GOAL_HARD_AGE:-30}"
CLIENT_LOCK_FILE="${CLIENT_LOCK_FILE:-/tmp/psix_rtc_robot_client.lock}"

MODE="dry-run"
CHECK_ONLY=0
LIST_TASKS=0
EXTRA_ARGS=()

# Every prompts.json under data/, as "<task key>\t<file>" lines. This is the one
# place that knows the layout; --task, --list-tasks and the error messages all
# read from it, so they can never disagree about what exists.
task_index() {
    local f
    for f in "$ROOT"/data/*/prompts.json; do
        [[ -f "$f" ]] || continue
        python3 - "$f" <<'PY' 2>/dev/null || true
import json, sys
path = sys.argv[1]
try:
    d = json.load(open(path))
except Exception:
    sys.exit(0)
if isinstance(d, dict):
    for k in d:
        print(f"{k}\t{path}")
PY
    done
}

# Fill DATA_ROOT / PROMPTS_JSON / EPISODE_DIR from TASK_KEY alone. A pinned value
# always wins, so existing DATA_ROOT-based invocations keep working unchanged.
resolve_task_paths() {
    [[ -n "$TASK_KEY" ]] || return 0
    [[ -n "$_TASK_KEY_CHOSEN" ]] || return 0

    # A pinned DATA_ROOT/PROMPTS_JSON normally wins, but only while it can
    # actually serve the requested key. A leftover `export DATA_ROOT=...` from an
    # earlier task is the common case, and letting it beat an explicit --task
    # just produced a confusing client-side "task-key not found" much later.
    # So: honour the pin when it contains the key, otherwise resolve and say so.
    if [[ -n "$_DATA_ROOT_PINNED" || -n "$_PROMPTS_JSON_PINNED" ]]; then
        local pinned_key="${TASK_KEY##*:}"
        if python3 - "$PROMPTS_JSON" "$pinned_key" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if isinstance(d, dict) and sys.argv[2] in d else 1)
PY
        then
            TASK_KEY="$pinned_key"
            # Same staleness trap one level down: an EPISODE_DIR exported for the
            # previous key would still point at that key's directory. Keep it only
            # when it already matches the key being run.
            if [[ -n "$_EPISODE_DIR_PINNED" && "${EPISODE_DIR##*/}" != "$TASK_KEY" ]]; then
                EPISODE_DIR="$DATA_ROOT/$TASK_KEY"
            fi
            return 0
        fi
        echo "[launcher] NOTE: task '$TASK_KEY' is not in the pinned $PROMPTS_JSON;" >&2
        echo "[launcher] ignoring the exported DATA_ROOT/PROMPTS_JSON and resolving by key" >&2
        _EPISODE_DIR_PINNED=""
    fi

    # Roughly half the keys under data/ are duplicated on purpose — the 0709 set
    # ships a `general_prompt` and a `corrected_prompt` variant of the same
    # episodes. So a bare key is not always unique, and an optional
    # "<data subdir>:<key>" qualifier picks the variant without having to fall
    # back to exporting DATA_ROOT. Only the bare key is passed to the client.
    local want_dir="" key="$TASK_KEY"
    if [[ "$TASK_KEY" == *:* ]]; then
        want_dir="${TASK_KEY%%:*}"
        key="${TASK_KEY#*:}"
    fi

    local hits
    hits="$(task_index | awk -F'\t' -v k="$key" '$1 == k {print $2}')"
    if [[ -n "$want_dir" ]]; then
        hits="$(printf '%s\n' "$hits" | awk -v d="/data/$want_dir/prompts.json" \
                    'index($0, d) { print }')"
    fi
    hits="$(printf '%s\n' "$hits" | grep . || true)"
    local n
    n="$(printf '%s' "$hits" | grep -c . || true)"

    if [[ "$n" == "0" ]]; then
        if [[ -n "$want_dir" ]]; then
            echo "[launcher] ERROR: task key '$key' not found in data/$want_dir/prompts.json" >&2
        else
            echo "[launcher] ERROR: task key '$key' not found in $ROOT/data/*/prompts.json" >&2
        fi
        echo "[launcher] available keys ($(basename "$0") --list-tasks for detail):" >&2
        task_index | awk -F'\t' '{print "  " $1}' | sort -u >&2
        exit 2
    fi
    if [[ "$n" != "1" ]]; then
        # Print each candidate as a ready-to-paste qualified key, so the fix is a
        # copy rather than a lookup of what DATA_ROOT should have been.
        echo "[launcher] ERROR: task key '$key' is defined in more than one prompts.json." >&2
        echo "[launcher] re-run with one of these qualified keys:" >&2
        while read -r f; do
            [[ -n "$f" ]] || continue
            local d="${f%/prompts.json}"
            echo "  --task ${d##*/}:$key" >&2
        done <<<"$hits"
        exit 2
    fi

    PROMPTS_JSON="$hits"
    DATA_ROOT="$(dirname "$PROMPTS_JSON")"
    TASK_KEY="$key"
    [[ -n "$_EPISODE_DIR_PINNED" ]] || EPISODE_DIR="$DATA_ROOT/$TASK_KEY"
}

list_tasks() {
    local idx cur="" dupes
    idx="$(task_index | sort -t'	' -k2,2 -k1,1)"
    if [[ -z "$idx" ]]; then
        echo "No prompts.json found under $ROOT/data/" >&2
        return 1
    fi
    # Keys living in more than one prompts.json need the "<subdir>:<key>" form.
    dupes="$(printf '%s\n' "$idx" | cut -f1 | sort | uniq -d)"
    while IFS=$'\t' read -r key file; do
        if [[ "$file" != "$cur" ]]; then
            cur="$file"
            echo
            echo "${file#$ROOT/}"
        fi
        local shown="$key"
        if grep -qxF "$key" <<<"$dupes"; then
            local d="${file%/prompts.json}"
            shown="${d##*/}:$key"
        fi
        printf '  %-22s %s\n' "$shown" \
            "$(python3 - "$file" "$key" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print(d[sys.argv[2]].get("task_description", ""))
PY
)"
    done <<<"$idx"
    echo
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run|--real|--check-only] [client options...]

  --dry-run     Full goal/VLA flow, but publish no robot/neck commands (default).
  --real        Enable real robot command publication.
  --check-only  Verify G1 wired routing and required services, then exit.
  --goal-source wm|episode
                Use remote WM goals (default), or fixed GT images from
                EPISODE_DIR/color_subgoal without contacting a WM server.
  --task KEY    Select the prompt by task key alone. DATA_ROOT, PROMPTS_JSON and
                EPISODE_DIR are resolved by finding KEY in data/*/prompts.json,
                so none of them need to be exported. Setting DATA_ROOT or
                PROMPTS_JSON explicitly disables the lookup and wins.
                Keys that exist in several prompts.json (the 0709 set ships a
                general_prompt and a corrected_prompt variant) take the
                qualified form "<data subdir>:<key>"; --list-tasks prints them
                that way already, and an ambiguous key errors with the exact
                qualified alternatives to paste.
  --prompt TEXT Run TEXT as the task instruction instead of the task_description
                in prompts.json. TASK_KEY still selects the entry, because the
                per-stage subtasks always come from the file. Env: INPUT_PROMPT.
  --list-tasks  Print every task key found in data/*/prompts.json, then exit.

With --hlp-mode off, prompts come from PROMPTS_JSON/TASK_KEY. Enter advances to
the next episode prompt/goal. Manual text and ':resume' are available only for
WM goals; fixed episode GT mode has no matching image for arbitrary text.

Other client arguments are appended to the selected Python client. Service
endpoints, goal source and embodiment are launcher-level settings so preflight
and runtime cannot diverge. Examples:

  export EMBODIMENT_TAG=psix_g1_sonic_neck
  $(basename "$0") --list-tasks                           # what can I run?
  $(basename "$0") --dry-run --task pick_cloth_1
  $(basename "$0") --real    --task pour_water_2
  $(basename "$0") --real    --task water_flower_1 -- --wm-seconds 3.2
  $(basename "$0") --check-only
  $(basename "$0") --dry-run --goal-source episode        # fixed GT goals, no WM
  $(basename "$0") --embodiment-tag psix_g1_sonic_neck --dry-run

Useful environment overrides:
  WM_HOST=$WM_HOST
  WM_PORT=$WM_PORT
  HLP_MODE=$HLP_MODE (default: off; active, shadow, or off)
  GOAL_SOURCE=$GOAL_SOURCE (default: wm; episode requires HLP_MODE=off)
  TASK_KEY=$TASK_KEY (default: real_pick_place_0709_psix_val_episode_0)
  EPISODE_DIR=$EPISODE_DIR
  EMBODIMENT_TAG=${EMBODIMENT_TAG:-<required>} (checked against VLA /info)
  NECK_STATE_ZMQ=$NECK_STATE_ZMQ
  NECK_PUB_PORT=$NECK_PUB_PORT
  SUBGOAL_LOG_DIR=$SUBGOAL_LOG_DIR
  WM_DUMP_DIR=$WM_DUMP_DIR (rollout manifest/events; also WM pairs in WM mode)
  WM_GOAL_HARD_AGE=$WM_GOAL_HARD_AGE (0 disables; default: 30s)
  CLIENT_LOCK_FILE=$CLIENT_LOCK_FILE
EOF
}

while (($#)); do
    case "$1" in
        --dry-run) MODE="dry-run" ;;
        --real) MODE="real" ;;
        --check-only) CHECK_ONLY=1 ;;
        --hlp-mode)
            if (($# < 2)); then
                echo "[launcher] ERROR: --hlp-mode needs active, shadow, or off" >&2
                exit 2
            fi
            HLP_MODE="$2"
            shift
            ;;
        --hlp-mode=*) HLP_MODE="${1#*=}" ;;
        --goal-source)
            if (($# < 2)); then
                echo "[launcher] ERROR: --goal-source needs wm or episode" >&2
                exit 2
            fi
            GOAL_SOURCE="$2"
            shift
            ;;
        --goal-source=*) GOAL_SOURCE="${1#*=}" ;;
        --embodiment-tag)
            if (($# < 2)); then
                echo "[launcher] ERROR: --embodiment-tag needs a value" >&2
                exit 2
            fi
            EMBODIMENT_TAG="$2"
            shift
            ;;
        --embodiment-tag=*) EMBODIMENT_TAG="${1#*=}" ;;
        --task)
            if (($# < 2)); then
                echo "[launcher] ERROR: --task needs a task key" >&2
                exit 2
            fi
            TASK_KEY="$2"
            _TASK_KEY_CHOSEN=1
            shift
            ;;
        --task=*) TASK_KEY="${1#*=}"; _TASK_KEY_CHOSEN=1 ;;
        --prompt)
            if (($# < 2)); then
                echo "[launcher] ERROR: --prompt needs an instruction string" >&2
                exit 2
            fi
            INPUT_PROMPT="$2"
            shift
            ;;
        --prompt=*) INPUT_PROMPT="${1#*=}" ;;
        --list-tasks) LIST_TASKS=1 ;;
        -h|--help) usage; exit 0 ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *) EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

# Pure inventory listing: answer before any config is required, so it works
# without EMBODIMENT_TAG and without touching the network, services or the lock.
if [[ "$LIST_TASKS" == "1" ]]; then
    list_tasks
    exit 0
fi

case "$HLP_MODE" in
    active|shadow|off) ;;
    *) echo "[launcher] ERROR: invalid HLP_MODE=$HLP_MODE" >&2; exit 2 ;;
esac
case "$GOAL_SOURCE" in
    wm|episode) ;;
    *) echo "[launcher] ERROR: invalid GOAL_SOURCE=$GOAL_SOURCE" >&2; exit 2 ;;
esac
if [[ "$GOAL_SOURCE" == "episode" && "$HLP_MODE" != "off" ]]; then
    echo "[launcher] ERROR: episode GT goals require HLP_MODE=off" >&2
    exit 2
fi
[[ -n "$EMBODIMENT_TAG" ]] || {
    echo "[launcher] ERROR: set EMBODIMENT_TAG to the VLA checkpoint embodiment" >&2
    exit 2
}
if [[ "$HLP_MODE" == "off" ]]; then
    [[ -n "$TASK_KEY" ]] || {
        echo "[launcher] ERROR: HLP off requires a non-empty TASK_KEY" >&2
        exit 2
    }
    resolve_task_paths
    [[ -f "$PROMPTS_JSON" ]] || {
        echo "[launcher] ERROR: prompts file not found: $PROMPTS_JSON" >&2
        exit 2
    }
    # Only the GT path reads EPISODE_DIR (it loads EPISODE_DIR/color_subgoal).
    # Under --goal-source wm the goals come from the WM server and the directory
    # is never opened, so requiring it there only forced empty placeholder dirs.
    if [[ "$GOAL_SOURCE" == "episode" ]]; then
        [[ -d "$EPISODE_DIR" ]] || {
            echo "[launcher] ERROR: episode directory not found: $EPISODE_DIR" >&2
            exit 2
        }
    fi
fi

# Keep the endpoints and embodiment that were preflighted identical to those
# passed to Python. Overrides use environment variables so argparse's
# "last value wins" rule cannot bypass the launcher contract.
for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
        --wm-host|--wm-host=*|--wm-port|--wm-port=*)
            echo "[launcher] ERROR: set WM_HOST/WM_PORT in the environment; do not pass $arg" >&2
            exit 2
            ;;
        --host|--host=*|--port|--port=*)
            echo "[launcher] ERROR: set VLA_HOST/VLA_PORT in the environment; do not pass $arg" >&2
            exit 2
            ;;
        --embodiment-tag|--embodiment-tag=*)
            echo "[launcher] ERROR: set EMBODIMENT_TAG in the environment; do not pass $arg" >&2
            exit 2
            ;;
        --task-key|--task-key=*)
            # argparse takes the last value, so this would silently win over the
            # key the launcher resolved paths and ran its preflight against —
            # prompts.json/EPISODE_DIR would then belong to a different task.
            echo "[launcher] ERROR: use --task (or TASK_KEY) instead of $arg;" >&2
            echo "[launcher] passing --task-key through bypasses task resolution and preflight" >&2
            exit 2
            ;;
        --prompts-json|--prompts-json=*|--episode-dir|--episode-dir=*)
            echo "[launcher] ERROR: set PROMPTS_JSON/EPISODE_DIR in the environment; do not pass $arg" >&2
            exit 2
            ;;
        --instruction|--instruction=*)
            echo "[launcher] ERROR: use --prompt (or INPUT_PROMPT) instead of $arg" >&2
            exit 2
            ;;
    esac
done

for cmd in curl python3 flock; do
    command -v "$cmd" >/dev/null 2>&1 \
        || { echo "[launcher] ERROR: required command not found: $cmd" >&2; exit 1; }
done

http_ok() {
    curl --noproxy '*' -fsS --max-time 2 "$1" >/dev/null 2>&1
}

wm_ready() {
    curl --noproxy '*' -fsS --max-time 2 "$1" 2>/dev/null \
        | python3 -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("ready") is True else 1)' \
        >/dev/null 2>&1
}

check_vla_contract() {
    local url="http://${VLA_HOST}:${VLA_PORT}/info"
    curl --noproxy '*' -fsS --max-time 3 "$url" \
        | python3 -c '
import json, sys
expected, mode = sys.argv[1:3]
info = json.load(sys.stdin)
served = str(info.get("embodiment_tag", ""))
try:
    action_dim = int(info["wire"]["action_dim"])
    state_dim = int(info["wire"]["state_dim"])
except (KeyError, TypeError, ValueError):
    raise SystemExit(
        "[launcher] ERROR: VLA /info missing wire.state_dim/action_dim; "
        "deploy the matching serve_psix.py")
if served != expected:
    raise SystemExit(
        f"[launcher] ERROR: embodiment mismatch: expected {expected!r}, "
        f"VLA serves {served!r}")
if (state_dim, action_dim) not in ((43, 78), (45, 80)):
    raise SystemExit(
        f"[launcher] ERROR: unsupported VLA dims={state_dim}/{action_dim}; "
        "expected 43/78 or 45/80")
if mode != "off" and (state_dim, action_dim) != (43, 78):
    raise SystemExit(
        "[launcher] ERROR: active/shadow combined client currently supports only 43/78; "
        "use HLP_MODE=off for this embodiment")
print(f"[launcher] VLA contract: embodiment={served} dims={state_dim}/{action_dim}")
' "$EMBODIMENT_TAG" "$HLP_MODE"
}

wait_http() {
    local name="$1" url="$2" timeout_s="$3"
    local deadline=$((SECONDS + timeout_s))
    while ((SECONDS < deadline)); do
        if http_ok "$url"; then
            echo "[launcher] $name ready: $url"
            return 0
        fi
        sleep 1
    done
    echo "[launcher] ERROR: $name not ready after ${timeout_s}s: $url" >&2
    return 1
}

wait_wm_ready() {
    local url="$1" timeout_s="$2"
    local deadline=$((SECONDS + timeout_s))
    while ((SECONDS < deadline)); do
        if wm_ready "$url"; then
            echo "[launcher] WM ready=true: $url"
            return 0
        fi
        sleep 1
    done
    echo "[launcher] ERROR: WM did not report ready=true after ${timeout_s}s: $url" >&2
    return 1
}

VLA_HEALTH_URL="http://${VLA_HOST}:${VLA_PORT}/health"
HLP_HEALTH_URL="http://${HLP_HOST}:${HLP_PORT}/health"
WM_READY_URL="http://${WM_HOST}:${WM_PORT}/ready"

if [[ "$GOAL_SOURCE" == "episode" ]]; then
    "$ROOT/g1_teleop_network.sh" check
    echo "[launcher] GT goal source: skipping WM endpoint contract/health"
elif [[ "$WM_HOST" == "192.168.123.240" ]]; then
    "$ROOT/g1_teleop_network.sh" check
elif [[ "$MODE" == "dry-run" && "$PSIX_ALLOW_NON_G1_WM" == "1" ]]; then
    echo "[launcher] WARNING: test-only non-G1 WM endpoint: $WM_HOST:$WM_PORT" >&2
else
    echo "[launcher] ERROR: production WM must be 192.168.123.240 over g1 TELEOP" >&2
    echo "[launcher] PSIX_ALLOW_NON_G1_WM=1 is accepted only with --dry-run for local tests" >&2
    exit 1
fi

wait_http "VLA" "$VLA_HEALTH_URL" 5
check_vla_contract
if [[ "$HLP_MODE" == "off" ]]; then
    echo "[launcher] HLP disabled: skipping HLP health/lease entirely"
else
    wait_http "HLP" "$HLP_HEALTH_URL" 5
fi

if [[ "$GOAL_SOURCE" == "wm" ]]; then
    if ! wait_wm_ready "$WM_READY_URL" "$WM_READY_TIMEOUT"; then
        echo "[launcher] G1 route passed; start/bind the WM server at 192.168.123.240:8016" >&2
        exit 1
    fi
fi

if ((CHECK_ONLY)); then
    echo "[launcher] all services are ready; check-only complete"
    exit 0
fi

# One robot/VLA client at a time. This catches duplicate dry/real launchers
# before either can acquire the WebSocket session or bind a command port.
exec 9<>"$CLIENT_LOCK_FILE"
if ! flock -n 9; then
    IFS= read -r lock_owner <"$CLIENT_LOCK_FILE" || lock_owner="unknown owner"
    echo "[launcher] ERROR: another robot client owns $CLIENT_LOCK_FILE ($lock_owner)" >&2
    exit 1
fi
printf 'pid=%s mode=%s hlp=%s goal_source=%s started=%s\n' \
    "$$" "$MODE" "$HLP_MODE" "$GOAL_SOURCE" \
    "$(date --iso-8601=seconds)" >"$CLIENT_LOCK_FILE"

# A client started outside this launcher will not own the lock. Report the
# existing neck publisher explicitly instead of failing much later in Python.
if [[ "$MODE" == "real" && "$HLP_MODE" == "off" && "$EMBODIMENT_TAG" == *neck* ]] \
        && command -v ss >/dev/null 2>&1; then
    neck_owner="$(ss -H -ltnp "sport = :$NECK_PUB_PORT" 2>/dev/null || true)"
    if [[ -n "$neck_owner" ]]; then
        echo "[launcher] ERROR: neck publisher port $NECK_PUB_PORT is already listening:" >&2
        echo "$neck_owner" >&2
        exit 1
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[launcher] ERROR: conda not found" >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$SONIC_ENV"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Lets the client's run manifest record the psi (server) repo identity too.
export PSI_REPO_DIR="${PSI_REPO_DIR:-/home/weiduoyuan/Desktop/psi}"
mkdir -p "$SUBGOAL_LOG_DIR"
cd "$ROOT"

if [[ "$HLP_MODE" == "off" ]]; then
    CLIENT_PROGRAM="psix_rtc_sonic_wm_client.py"
    CLIENT_ARGS=(
        --host "$VLA_HOST"
        --port "$VLA_PORT"
        --embodiment-tag "$EMBODIMENT_TAG"
        --goal-source "$GOAL_SOURCE"
        --wm-host "$WM_HOST"
        --wm-port "$WM_PORT"
        --wm-goal-hard-age "$WM_GOAL_HARD_AGE"
        --wm-dump-dir "$WM_DUMP_DIR"
        --camera-address "$CAMERA_ADDRESS"
        --episode-dir "$EPISODE_DIR"
        --prompts-json "$PROMPTS_JSON"
        --task-key "$TASK_KEY"
        --neck-pub-host "$NECK_PUB_HOST"
        --neck-pub-port "$NECK_PUB_PORT"
        --neck-state-zmq "$NECK_STATE_ZMQ"
    )
    if [[ -n "$INPUT_PROMPT" ]]; then
        CLIENT_ARGS+=(--instruction "$INPUT_PROMPT")
        echo "[launcher] prompt override: $INPUT_PROMPT"
        echo "[launcher] (subtask stages still come from $TASK_KEY)"
    fi
    if [[ "$GOAL_SOURCE" == "episode" ]]; then
        echo "[launcher] scene mode: HLP OFF; fixed episode GT images; WM disabled"
    else
        echo "[launcher] scene mode: HLP OFF; episode prompts + manual takeover"
    fi
    echo "[launcher] prompts: $PROMPTS_JSON [$TASK_KEY]"
    echo "[launcher] episode dir: $EPISODE_DIR"
    echo "[launcher] rollout telemetry: $WM_DUMP_DIR"
    if [[ "$GOAL_SOURCE" == "episode" ]]; then
        echo "[launcher] controls: Enter=next GT goal | :restart | :mark LABEL"
    else
        echo "[launcher] controls: Enter=next | text/:ov TEXT=manual | :resume | :restart"
    fi
else
    CLIENT_PROGRAM="psix_rtc_sonic_hlpwm_client.py"
    CLIENT_ARGS=(
        --host "$VLA_HOST"
        --port "$VLA_PORT"
        --hlp-host "$HLP_HOST"
        --hlp-port "$HLP_PORT"
        --hlp-mode "$HLP_MODE"
        --wm-host "$WM_HOST"
        --wm-port "$WM_PORT"
        --profile "$PROFILE"
        --profile-hash "$PROFILE_HASH"
        --camera-address "$CAMERA_ADDRESS"
        --episode-dir "$EPISODE_DIR"
        --prompts-json "$PROMPTS_JSON"
        --task-key "$TASK_KEY"
        --allow-raw-override
        --subgoal-log-dir "$SUBGOAL_LOG_DIR"
    )
    if [[ -n "$TASK_KEY" ]]; then
        echo "[launcher] scene mode: FIXED replay order from task key $TASK_KEY"
        echo "[launcher] episode dir: $EPISODE_DIR"
    else
        echo "[launcher] scene mode: DYNAMIC live HLP order (no fixed episode scene)"
    fi
fi

if [[ "$MODE" == "dry-run" ]]; then
    CLIENT_ARGS+=(--dry-run)
    echo "[launcher] starting DRY-RUN client (no robot command publication)"
else
    echo "[launcher] starting REAL client"
fi

echo "[launcher] client/HLP logs: $SUBGOAL_LOG_DIR"
# Persist the client console; gate/safety events were unrecoverable when the
# terminal scrollback was the only copy. tee leaves stdin on the terminal, so
# the interactive Enter/:ov prompt commands keep working.
CLIENT_LOG="$SUBGOAL_LOG_DIR/$(date +%Y%m%d-%H%M%S)_client.log"
echo "[launcher] client console log: $CLIENT_LOG"
python -u "$CLIENT_PROGRAM" "${CLIENT_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$CLIENT_LOG"
