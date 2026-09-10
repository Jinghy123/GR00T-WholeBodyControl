#!/usr/bin/env bash
# Manual instructions: Enter walks the --next-prompt list, i types one.
# See HLP_INSTRUCTION_CLIENT.md.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PSIX_PYTHON:-$HOME/miniforge3/envs/sonic/bin/python}"

exec "$PYTHON" -u "$ROOT/psix_client.py" \
    --prompt "Gather the fruits in the basket, grab the drink, turn right, and place them on the black cart on the right." \
    --next-prompt "Pick the table cloth from the cabinet, put on the black cart, push cart near kitchen island." \
    --next-prompt "Pick up the table cloth, walk to the kitchen island, pick up the cup, clean the surface, put down the cup." \
    --next-prompt "Serve the basket and iced tea at the kitchen island beside the cup." \
    --next-prompt "Turn right to the bed, place the yellow shirt and blue shirt into the laundry basket." \
    --hlp-mode manual \
    --rtc-mode test_time \
    --encoder-version v1_1 \
    --wm-period 0.8 \
    --wm-seconds 2.4 \
    --hlp-period 1.0 \
    --show-goal \
    "$@"
