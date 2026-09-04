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
# VLA transport. ws = the /ws RTC stream (psix_rtc_sonic_wm_client.py);
# http = one POST /act per chunk, 30 Hz playback, hold-in-place between chunks
# (psix_http_sonic_wm_client.py). http needs the VLA served with rtc_mode=off
# and is only wired for HLP_MODE=off.
TRANSPORT="${TRANSPORT:-ws}"
# Default is the fixed GT goals under DATA_ROOT/<TASK_KEY>/color_subgoal, paired
# 1:1 with the prompts.json subtasks. The pnp wmgoal checkpoint trains on both
# goal kinds, and the GT arm is the reproducible one: no WM server in the loop,
# no generation jitter. Pass --goal-source wm for the WM future-frame arm.
GOAL_SOURCE="${GOAL_SOURCE:-episode}"

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
# the environment, or --task); the built-in default is left to pair with the
# DATA_ROOT default below instead. That kept a bare invocation working back when
# the default key existed in several prompts.json files and resolving it raised
# an ambiguity error; the current pnp_psix default is unique, so the two paths
# now agree, and pinning is simply one less lookup. --task sets this too, in the
# argument loop.
_TASK_KEY_CHOSEN="${TASK_KEY+1}"
# Same idea for WM_DUMP_DIR: recorded before its timestamped default is
# computed, so the METHOD_NAME auto-routing below (after TASK_KEY has its
# final, resolved value) can tell "operator pinned it" from "still default".
_WM_DUMP_DIR_CHOSEN="${WM_DUMP_DIR+1}"

# The pick_place goal set the current pnp wmgoal checkpoint is served against is
# TWO-level, unlike every older root under $ROOT/data:
#
#   <DATA_ROOT>/prompts.json                       4 task keys x 3 subtasks
#   <DATA_ROOT>/<TASK_KEY>/<EPISODE_KEY>/color_subgoal/frame_*.jpg
#
# TASK_KEY selects the prompt entry, EPISODE_KEY selects which recorded episode's
# goal frames to replay (~20 per task). An empty EPISODE_KEY means "the first
# episode", resolved numerically by resolve_episode_dir. Flat roots (goal frames
# directly under <DATA_ROOT>/<TASK_KEY>) still work -- resolve_episode_dir detects
# the layout instead of requiring a mode flag.
_DATA_ROOT_DEFAULT="/home/weiduoyuan/data/pick_place"
DATA_ROOT="${DATA_ROOT:-$_DATA_ROOT_DEFAULT}"
PROMPTS_JSON="${PROMPTS_JSON:-$DATA_ROOT/prompts.json}"
TASK_KEY="${TASK_KEY-pick_place_1}"
EPISODE_KEY="${EPISODE_KEY:-}"
# Left empty on purpose: resolve_episode_dir fills it once TASK_KEY/EPISODE_KEY
# are final. An exported EPISODE_DIR still wins (_EPISODE_DIR_PINNED).
EPISODE_DIR="${EPISODE_DIR:-}"
# Replaces prompts.json[TASK_KEY].task_description for this run. TASK_KEY still
# selects the entry, because the per-stage subtasks always come from the file —
# only the task text is overridden. Empty means "use the file's text".
INPUT_PROMPT="${INPUT_PROMPT:-}"
# Label for the VLA checkpoint/ablation arm under test (e.g. statehist_80k,
# goaldrop_80k, generalist_40k). Purely a label: embedded in run_manifest.json
# and the init-frame sidecar, and (when set) routes WM_DUMP_DIR under
# .logs/main_comparisons/<task_key>/<method_name>/ instead of the flat
# .logs/psix_rollouts/ default -- see the WM_DUMP_DIR auto-routing below.
METHOD_NAME="${METHOD_NAME:-}"
CAMERA_ADDRESS="${CAMERA_ADDRESS:-tcp://192.168.123.164:5558}"
SUBGOAL_LOG_DIR="${SUBGOAL_LOG_DIR:-/home/weiduoyuan/Desktop/psi/.logs/HLP_WM_logs}"
WM_DUMP_DIR="${WM_DUMP_DIR:-/home/weiduoyuan/Desktop/psi/.logs/psix_rollouts/$(date +%Y%m%d-%H%M%S)}"
NECK_PUB_HOST="${NECK_PUB_HOST:-*}"
NECK_PUB_PORT="${NECK_PUB_PORT:-5570}"
NECK_STATE_ZMQ="${NECK_STATE_ZMQ:-tcp://192.168.123.164:5560}"
WM_GOAL_HARD_AGE="${WM_GOAL_HARD_AGE:-30}"
CLIENT_LOCK_FILE="${CLIENT_LOCK_FILE:-/tmp/psix_rtc_robot_client.lock}"

MODE="dry-run"
CHECK_ONLY=0
LIST_TASKS=0
EXTRA_ARGS=()
# Client flags an operator actually types, accepted directly so nobody has to know
# where the "--" goes. Forwarded before EXTRA_ARGS, so an explicit `-- --wm-seconds X`
# still wins. Only flags psix_rtc_sonic_wm_client.py defines belong here -- the
# HLP client does not define them, which is why they are gated on HLP_MODE=off below.
WM_CLIENT_ARGS=()

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
    # The effective PROMPTS_JSON -- pinned or default -- gets first refusal. That
    # keeps a pin authoritative while it can serve the key, and it also stops the
    # default root from losing to a same-named key elsewhere under $ROOT/data
    # (e.g. pick_place_1 also exists in data/zedmini10_pick_place).
    if [[ -f "$PROMPTS_JSON" ]]; then
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
            if [[ -n "$_EPISODE_DIR_PINNED" && "$EPISODE_DIR" != "$DATA_ROOT/$TASK_KEY"* ]]; then
                _EPISODE_DIR_PINNED=""   # stale pin; let resolve_episode_dir recompute
            fi
            return 0
        fi
        if [[ -n "$_DATA_ROOT_PINNED" || -n "$_PROMPTS_JSON_PINNED" ]]; then
            echo "[launcher] NOTE: task '$TASK_KEY' is not in the pinned $PROMPTS_JSON;" >&2
            echo "[launcher] ignoring the exported DATA_ROOT/PROMPTS_JSON and resolving by key" >&2
        fi
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
}

# Episode directory names under a task, ordered numerically so episode_2 comes
# before episode_10 (plain sort would not).
list_episodes() {
    find "$1" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null \
        | sort -t_ -k2 -n
}

# Fill EPISODE_DIR from TASK_KEY + EPISODE_KEY, detecting the root layout rather
# than requiring a mode flag:
#   nested  <DATA_ROOT>/<TASK_KEY>/<EPISODE_KEY>/color_subgoal   (pick_place)
#   flat    <DATA_ROOT>/<TASK_KEY>/color_subgoal                 (older roots)
# Only the GT goal path reads EPISODE_DIR, so a --goal-source wm run is unaffected.
resolve_episode_dir() {
    [[ -z "$_EPISODE_DIR_PINNED" ]] || return 0
    local task_dir="$DATA_ROOT/$TASK_KEY"

    if [[ -d "$task_dir/color_subgoal" ]]; then
        if [[ -n "$EPISODE_KEY" ]]; then
            echo "[launcher] ERROR: --episode '$EPISODE_KEY' given, but $task_dir is a" >&2
            echo "[launcher] flat root (goal frames sit directly in color_subgoal/)" >&2
            exit 2
        fi
        EPISODE_DIR="$task_dir"
        return 0
    fi

    # Missing task dir: leave the path as-is so the caller's own -d check reports
    # it with the usual message instead of a second, competing error here.
    [[ -d "$task_dir" ]] || { EPISODE_DIR="$task_dir"; return 0; }

    if [[ -n "$EPISODE_KEY" ]]; then
        # Accept both "episode_7" and a bare "7".
        local cand="$task_dir/$EPISODE_KEY"
        [[ -d "$cand" ]] || cand="$task_dir/episode_$EPISODE_KEY"
        if [[ ! -d "$cand" ]]; then
            echo "[launcher] ERROR: episode '$EPISODE_KEY' not found under $task_dir" >&2
            echo "[launcher] available: $(list_episodes "$task_dir" | tr '\n' ' ')" >&2
            exit 2
        fi
        EPISODE_DIR="$cand"
    else
        local first
        first="$(list_episodes "$task_dir" | head -n 1)"
        if [[ -z "$first" ]]; then
            echo "[launcher] ERROR: no episode directories under $task_dir" >&2
            exit 2
        fi
        EPISODE_DIR="$task_dir/$first"
        echo "[launcher] no --episode given; using the first one: $first"
    fi
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
  --transport ws|http
                ws (default): the /ws RTC action stream. http: POST /act once
                per chunk, play it back at 30 Hz, hold in place (encoder token,
                last hand/neck) until the next chunk lands. http requires the
                VLA served with RTC_MODE=off and HLP_MODE=off. Env: TRANSPORT.
  --goal-source wm|episode
                Fixed GT images from EPISODE_DIR/color_subgoal, no WM server
                contacted (default), or remote WM goals.
  --episode KEY Which recorded episode's GT goal frames to replay, for two-level
                roots like the default (<DATA_ROOT>/<task>/<episode>/color_subgoal).
                Accepts "episode_7" or a bare "7". Omit for the first episode.
                Rejected on flat roots, where the task dir holds color_subgoal
                directly. Only used by --goal-source episode.
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
  --method-name NAME
                Label for the VLA checkpoint/ablation arm under test (e.g.
                statehist_80k, goaldrop_80k, generalist_40k). Purely a label,
                embedded in run_manifest.json and the init-frame sidecar. When
                set and WM_DUMP_DIR is not pinned, also routes rollout
                telemetry to .logs/main_comparisons/<task_key>/<method_name>/
                instead of the flat .logs/psix_rollouts/ default -- use
                this for quantitative cross-method comparison runs. Env:
                METHOD_NAME.
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
  $(basename "$0") --dry-run --goal-source wm             # WM future frames instead
  $(basename "$0") --embodiment-tag psix_g1_sonic_neck --dry-run

Useful environment overrides:
  WM_HOST=$WM_HOST
  WM_PORT=$WM_PORT
  HLP_MODE=$HLP_MODE (default: off; active, shadow, or off)
  TRANSPORT=$TRANSPORT (default: ws; ws or http)
  GOAL_SOURCE=$GOAL_SOURCE (default: episode; episode requires HLP_MODE=off)
  TASK_KEY=$TASK_KEY (default: pick_place_1, from $DATA_ROOT)
  EPISODE_KEY=${EPISODE_KEY:-<first episode>} (see --episode)
  DATA_ROOT=$DATA_ROOT
  EPISODE_DIR=${EPISODE_DIR:-<resolved from TASK_KEY/EPISODE_KEY>}
  EMBODIMENT_TAG=${EMBODIMENT_TAG:-<required>} (checked against VLA /info)
  NECK_STATE_ZMQ=$NECK_STATE_ZMQ
  NECK_PUB_PORT=$NECK_PUB_PORT
  SUBGOAL_LOG_DIR=$SUBGOAL_LOG_DIR
  METHOD_NAME=${METHOD_NAME:-<unset>} (label; auto-routes WM_DUMP_DIR under
                main_comparisons/<task_key>/<method_name>/ unless pinned)
  WM_DUMP_DIR=$WM_DUMP_DIR (rollout manifest/events; also WM pairs in WM mode)
  WM_GOAL_HARD_AGE=$WM_GOAL_HARD_AGE (0 disables; default: 30s)
  CLIENT_LOCK_FILE=$CLIENT_LOCK_FILE
EOF
}

# A flag on the command line invalidates every environment pin it derives from.
# Without this, a leftover `export DATA_ROOT=...` from the previous task silently
# beat an explicit --task, and a stale EPISODE_DIR survived an --episode switch
# within the same task -- which is the only reason every command in the notes was
# prefixed with `env -u DATA_ROOT -u PROMPTS_JSON -u EPISODE_DIR ...`.
_drop_root_pins() {
    if [[ -n "$_DATA_ROOT_PINNED$_PROMPTS_JSON_PINNED$_EPISODE_DIR_PINNED" ]]; then
        echo "[launcher] NOTE: --task given; ignoring exported DATA_ROOT/PROMPTS_JSON/EPISODE_DIR"
    fi
    _DATA_ROOT_PINNED=""
    _PROMPTS_JSON_PINNED=""
    _EPISODE_DIR_PINNED=""
    DATA_ROOT="${_DATA_ROOT_DEFAULT}"
    PROMPTS_JSON="${DATA_ROOT}/prompts.json"
    EPISODE_DIR=""
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
        --transport)
            if (($# < 2)); then
                echo "[launcher] ERROR: --transport needs ws or http" >&2
                exit 2
            fi
            TRANSPORT="$2"
            shift
            ;;
        --transport=*) TRANSPORT="${1#*=}" ;;
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
            _drop_root_pins
            shift
            ;;
        --task=*) TASK_KEY="${1#*=}"; _TASK_KEY_CHOSEN=1; _drop_root_pins ;;
        --episode)
            if (($# < 2)); then
                echo "[launcher] ERROR: --episode needs an episode key" >&2
                exit 2
            fi
            EPISODE_KEY="$2"
            _EPISODE_DIR_PINNED=""
            shift
            ;;
        --episode=*) EPISODE_KEY="${1#*=}"; _EPISODE_DIR_PINNED="" ;;
        --method-name)
            if (($# < 2)); then
                echo "[launcher] ERROR: --method-name needs a value" >&2
                exit 2
            fi
            METHOD_NAME="$2"
            shift
            ;;
        --method-name=*) METHOD_NAME="${1#*=}" ;;
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
        # Common psix_rtc_sonic_wm_client.py flags, accepted directly.
        --wm-seconds|--wm-period|--wm-mode)
            if (($# < 2)); then
                echo "[launcher] ERROR: $1 needs a value" >&2
                exit 2
            fi
            WM_CLIENT_ARGS+=("$1" "$2")
            shift
            ;;
        --wm-seconds=*|--wm-period=*|--wm-mode=*) WM_CLIENT_ARGS+=("$1") ;;
        --show-goal|--subtask-prompt|--no-subtask-prompt|--neck-reset|--no-neck-reset)
            WM_CLIENT_ARGS+=("$1") ;;
        --neck-reset-yaw|--neck-reset-pitch|--neck-reset-hold|--neck-reset-tol|--neck-reset-on-fail)
            if (($# < 2)); then
                echo "[launcher] ERROR: $1 needs a value" >&2
                exit 2
            fi
            WM_CLIENT_ARGS+=("$1" "$2")
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        # A bare word here is almost always a value that lost its flag ("--task X 3"
        # meaning "--episode 3"). Passing it through only surfaced as an argparse
        # error after preflight and after the client lock was taken.
        --*) EXTRA_ARGS+=("$1") ;;
        *)
            echo "[launcher] ERROR: stray argument '$1' (did you mean --episode $1?)" >&2
            exit 2
            ;;
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
    wm|episode|none) ;;
    *) echo "[launcher] ERROR: invalid GOAL_SOURCE=$GOAL_SOURCE" >&2; exit 2 ;;
esac
case "$TRANSPORT" in
    ws|http) ;;
    *) echo "[launcher] ERROR: invalid TRANSPORT=$TRANSPORT (ws or http)" >&2; exit 2 ;;
esac
if [[ "$TRANSPORT" == "http" && "$HLP_MODE" != "off" ]]; then
    echo "[launcher] ERROR: --transport http is only wired for HLP_MODE=off" >&2
    exit 2
fi
if ((${#WM_CLIENT_ARGS[@]})) && [[ "$HLP_MODE" != "off" ]]; then
    echo "[launcher] ERROR: ${WM_CLIENT_ARGS[0]} is only available with --hlp-mode off;" >&2
    echo "[launcher] psix_rtc_sonic_hlpwm_client.py does not define it" >&2
    exit 2
fi

if [[ ( "$GOAL_SOURCE" == "episode" || "$GOAL_SOURCE" == "none" ) && "$HLP_MODE" != "off" ]]; then
    echo "[launcher] ERROR: $GOAL_SOURCE goals require HLP_MODE=off" >&2
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
        resolve_episode_dir
        [[ -d "$EPISODE_DIR" ]] || {
            echo "[launcher] ERROR: episode directory not found: $EPISODE_DIR" >&2
            exit 2
        }
    fi
    # Quantitative comparison runs: route telemetry to
    # .logs/main_comparisons/<task_key>/<method_name>/<timestamp> instead of
    # the flat .logs/psix_rollouts/<timestamp> default, so rollouts group
    # by task and by the method/checkpoint under test. Only when the operator
    # set METHOD_NAME and did not pin WM_DUMP_DIR themselves -- an explicit
    # WM_DUMP_DIR always wins.
    if [[ -z "$_WM_DUMP_DIR_CHOSEN" && -n "$METHOD_NAME" ]]; then
        WM_DUMP_DIR="/home/weiduoyuan/Desktop/psi/.logs/main_comparisons/$TASK_KEY/$METHOD_NAME/$(date +%Y%m%d-%H%M%S)"
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
expected, mode, transport = sys.argv[1:4]
info = json.load(sys.stdin)
served = str(info.get("embodiment_tag", ""))
rtc_mode = info.get("rtc_mode")
if transport == "http" and rtc_mode != "off":
    raise SystemExit(
        f"[launcher] ERROR: --transport http needs the VLA served with rtc_mode=off, "
        f"but /info reports rtc_mode={rtc_mode!r}; restart the server with "
        "RTC_MODE=off bash scripts/deploy/serve_psix_rtc.sh")
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
print(f"[launcher] VLA contract: embodiment={served} dims={state_dim}/{action_dim} "
      f"rtc_mode={rtc_mode} transport={transport}")
' "$EMBODIMENT_TAG" "$HLP_MODE" "$TRANSPORT"
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

if [[ "$GOAL_SOURCE" == "episode" || "$GOAL_SOURCE" == "none" ]]; then
    # No goal comes from the WM machine (episode = local disk, none = no goal at
    # all), so its route leg is skipped too, not just the /ready probe.
    # skip its route leg too, not just the /ready probe. The robot leg is still
    # checked -- camera and neck come over the same wire.
    "$ROOT/g1_teleop_network.sh" check --no-wm
    echo "[launcher] goal-source=$GOAL_SOURCE: skipping WM route, endpoint contract and health"
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
printf 'pid=%s mode=%s hlp=%s transport=%s goal_source=%s started=%s\n' \
    "$$" "$MODE" "$HLP_MODE" "$TRANSPORT" "$GOAL_SOURCE" \
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
    if [[ "$TRANSPORT" == "http" ]]; then
        # Same CLI as the RTC client (it imports that parser), plus --http-timeout.
        CLIENT_PROGRAM="psix_http_sonic_wm_client.py"
        echo "[launcher] transport: HTTP /act chunk playback (open loop, hold between chunks)"
    else
        CLIENT_PROGRAM="psix_rtc_sonic_wm_client.py"
        echo "[launcher] transport: WebSocket /ws RTC stream"
    fi
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
    if [[ -n "$METHOD_NAME" ]]; then
        CLIENT_ARGS+=(--method-name "$METHOD_NAME")
        echo "[launcher] method under test: $METHOD_NAME"
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
python -u "$CLIENT_PROGRAM" "${CLIENT_ARGS[@]}" "${WM_CLIENT_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$CLIENT_LOG"
