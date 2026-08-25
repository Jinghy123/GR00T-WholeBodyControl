#!/usr/bin/env python3
"""
g1_sonic_client_hack.py
──────────────────────────────────────────────────────────────────────────────
TEMPORARY HACK: 15 Hz deploy for a mis-recorded 15 Hz episode.

Same problem as g1_sonic_client_rtc_hack.py, but for the normal (non-RTC,
open-loop chunk-by-chunk) client `g1_sonic_client.py`.

One dataset was accidentally recorded at 15 Hz, but the normal deploy runs the
30 Hz control loop. A policy trained on that 15 Hz data emits a 24-action chunk
meant to be *executed* at 15 Hz (24 actions = 1.6 s of motion). Publishing it at
30 Hz plays it back in 0.8 s -> the robot moves at ~2x speed ("加一倍速").

Fix (temporary): drop the control-loop publish frequency from 30 Hz to 15 Hz.
The publish loop advances one action per tick and re-plans a fresh 24-action
chunk only after the previous one is fully executed; everything downstream of
`FREQ_POLICY` is measured in ticks/actions, not wall clock, so stretching the
tick period simply plays each chunk at the rate it was recorded. The 4 frames
sent per chunk (RELATIVE_OFFSETS across the just-executed 24-tick window) are
unchanged, and at 15 Hz their real-time spacing doubles to match the 15 Hz
training data.

The downstream WBC tracks the latest published target (see `publish_token` in
g1_sonic_client.py -- no internal 30 Hz assumption), so halving the publish rate
directly halves the motion speed. Nothing else changes.

Usage (identical to g1_sonic_client.py, just a different entrypoint):

    python g1_sonic_client_hack.py \
        --policy-host 127.0.0.1 --policy-port 5000 --include-neck \
        --prompt "..." --action-only

Optionally override the deploy rate (default 15 Hz) via the DEPLOY_HZ env var,
e.g. `DEPLOY_HZ=15 python g1_sonic_client_hack.py ...`.

DELETE THIS FILE once the bad episode is re-recorded at 30 Hz.
"""

import os

# Import the real client module and patch its control-loop frequency BEFORE any
# client is constructed / the publish loop starts. `FREQ_POLICY` is a module
# global defined in g1_sonic_client.py and read at runtime inside `_publish_loop`
# (`dt = 1.0 / FREQ_POLICY`), so overriding the name here is sufficient -- no need
# to edit g1_sonic_client.py itself.
import g1_sonic_client as _c

DEPLOY_HZ = float(os.environ.get("DEPLOY_HZ", "15"))

_orig_hz = _c.FREQ_POLICY
_c.FREQ_POLICY = DEPLOY_HZ

print(
    f"[HACK] 15 Hz deploy hack ACTIVE: control-loop frequency "
    f"{_orig_hz} Hz -> {DEPLOY_HZ} Hz (for the mis-recorded 15 Hz episode). "
    f"Set DEPLOY_HZ to change."
)


if __name__ == "__main__":
    _c.main()
