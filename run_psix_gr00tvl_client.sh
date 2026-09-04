#!/bin/bash
# GR00T-VL client launcher.
#
# Thin wrapper over run_psix_hlpwm_client.sh that pins the three things this arm
# needs and that are easy to get wrong by hand. It does not fork the client:
# psix_rtc_sonic_wm_client.py already speaks this server's contract.
#
#   1. --frozen-action, explicitly.
#      The GR00T-VL server runs rtc_mode=off with action_chunk_size ==
#      action_exec_horizon == 30, so there is NO continuity mechanism across
#      chunks. When one runs out before the next is installed the server repeats
#      its last predicted row and flags rtc_repeat_last; psi's own gr00tvl commit
#      ("drop the chunk-exhaustion hold") deleted the server-side hold and left
#      that gap to the client on purpose. --frozen-action is what fills it: the
#      body token is re-encoded from the CURRENT measured pose via
#      model_encoder.onnx, so the WBC is told "stay here" instead of re-driving a
#      token from a plan that has already moved on. Hand and neck keep their last
#      commanded values. It is the client default, but this arm is the one where
#      it actually fires, so it is passed explicitly rather than relied upon.
#
#   2. A goal source that always yields a goal.
#      This checkpoint trained with subgoal_prob=1.0 -- every sample carried a
#      goal image (30% GT / 43% WM). --goal-source none is therefore out of
#      distribution here and is refused below.
#
#   3. PSI_REPO_DIR / log roots pointed at the psi-gr00tvl checkout, so the run
#      manifest records the branch that actually served the rollout.
#
# Everything else -- task keys, --wm-seconds, --show-goal, dry-run/real, the
# network contract -- is the hlpwm launcher's, unchanged. Any flag it does not
# recognise is forwarded to the client as usual.
#
# Usage:
#   ./run_psix_gr00tvl_client.sh --dry-run --task pick_place_1
#   ./run_psix_gr00tvl_client.sh --real    --task pick_place_1
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00TVL_REPO="${GR00TVL_REPO:-/home/weiduoyuan/Desktop/psi-gr00tvl}"

[[ -d "$GR00TVL_REPO" ]] || {
    echo "[gr00tvl] ERROR: GR00TVL_REPO not found: $GR00TVL_REPO" >&2
    exit 2
}

# Goal source: wm (default) or episode. "none" would run a prompt shape this
# checkpoint never trained (subgoal_prob=1.0), so refuse it rather than let the
# rollout look merely bad.
GOAL_SOURCE="${GOAL_SOURCE:-wm}"
if [[ "$GOAL_SOURCE" == "none" ]]; then
    echo "[gr00tvl] ERROR: this arm trained with subgoal_prob=1.0 -- every sample" >&2
    echo "[gr00tvl] carried a goal image, so --goal-source none is out of distribution." >&2
    echo "[gr00tvl] Use GOAL_SOURCE=wm (default) or GOAL_SOURCE=episode." >&2
    exit 2
fi
export GOAL_SOURCE

export HLP_MODE="${HLP_MODE:-off}"
export EMBODIMENT_TAG="${EMBODIMENT_TAG:-psix_g1_sonic_neck}"
export PSI_REPO_DIR="${PSI_REPO_DIR:-$GR00TVL_REPO}"
export SUBGOAL_LOG_DIR="${SUBGOAL_LOG_DIR:-$GR00TVL_REPO/.logs/HLP_WM_logs}"

# Same auto-routing rule the hlpwm launcher uses, just rooted in this checkout:
# METHOD_NAME set -> main_comparisons/<task>/<method>/, otherwise a flat run dir.
if [[ -z "${WM_DUMP_DIR+x}" && -z "${METHOD_NAME:-}" ]]; then
    export WM_DUMP_DIR="$GR00TVL_REPO/.logs/psix_rollouts/$(date +%Y%m%d-%H%M%S)"
fi

echo "[gr00tvl] repo=$GR00TVL_REPO"
echo "[gr00tvl] goal_source=$GOAL_SOURCE hlp=$HLP_MODE embodiment=$EMBODIMENT_TAG"
echo "[gr00tvl] --frozen-action pinned (rtc_mode=off: every chunk boundary can starve)"

# WM refresh cadence for this arm. The client ships 1.6/1.6; here the period is
# 1.0 while the horizon stays 1.6, i.e. the goal is refreshed before the previous
# one is fully consumed. Only applied when the caller did not say otherwise --
# the launcher takes the LAST value, so scanning the knob still works:
#   ./run_psix_gr00tvl_client.sh --real --task X --wm-period 1.6
WM_ARGS=()
case " $* " in *" --wm-period "*|*" --wm-period="*) ;; *) WM_ARGS+=(--wm-period "${WM_PERIOD:-1.0}") ;; esac
case " $* " in *" --wm-seconds "*|*" --wm-seconds="*) ;; *) WM_ARGS+=(--wm-seconds "${WM_SECONDS:-1.6}") ;; esac
[[ ${#WM_ARGS[@]} -gt 0 ]] && echo "[gr00tvl] wm defaults: ${WM_ARGS[*]}"

# --frozen-action goes first so an explicit --no-frozen-action from the caller
# still wins (argparse takes the last value), e.g. to measure the plain
# repeat-last baseline.
exec "$ROOT/run_psix_hlpwm_client.sh" --frozen-action "${WM_ARGS[@]}" "$@"
