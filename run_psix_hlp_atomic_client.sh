#!/usr/bin/env bash
# HLP picks the instructions and breaks the registered cleaning/drink tasks into
# atomic steps. Needs an HLP server whose /health reports "hierarchical": true.
# See HLP_INSTRUCTION_CLIENT.md.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PSIX_PYTHON:-$HOME/miniforge3/envs/sonic/bin/python}"

exec "$PYTHON" -u "$ROOT/psix_client.py" \
    --prompt serve_drink_clean_collect \
    --hlp-mode hlp \
    --hlp-atomic \
    --rtc-mode test_time \
    --encoder-version v1_1 \
    --wm-period 0.8 \
    --wm-seconds 3.2 \
    --hlp-period 1.0 \
    --show-goal \
    "$@"
