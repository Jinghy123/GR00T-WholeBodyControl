#!/usr/bin/env python3
"""
g1_sonic_client_rtc_hack.py
──────────────────────────────────────────────────────────────────────────────
TEMPORARY HACK: 15 Hz deploy for a mis-recorded 15 Hz episode.

One dataset was accidentally recorded at 15 Hz, but our normal RTC deploy
(`g1_sonic_client_rtc.py`) runs the 30 Hz control loop. A policy trained on that
15 Hz data emits a 24-action chunk that is meant to be *executed* at 15 Hz
(24 actions = 1.6 s of motion). Publishing it at 30 Hz plays it back in 0.8 s ->
the robot moves at ~2x speed ("加一倍速").

Fix (temporary): drop the control-loop publish frequency from 30 Hz to 15 Hz.
Because the ENTIRE RTC control/timing logic is expressed in units of "ticks =
published actions" (execution_horizon s=12, frozen prefix d, obs_buffer indices,
the mid/fresh frame spacing), simply stretching the tick period:
  * plays the 24-action chunk at the rate it was recorded (no more 2x speed), AND
  * spaces the 2 re-plan frames (mid @ s//2 ticks back, fresh @ now) at the 15 Hz
    interval the model saw in training -> image temporal spacing also matches.
The only wall-clock-dependent quantity (inference latency measured in ticks,
`d_real`) just becomes smaller; it is adaptive + clamped, and 15 Hz actually
gives the async re-plan MORE wall-clock budget per step, so it is strictly safer.

The downstream WBC tracks the latest published target (see `publish_token` in
g1_sonic_client.py -- no internal 30 Hz assumption), so halving the publish rate
directly halves the motion speed. Nothing else changes.

Usage (identical to g1_sonic_client_rtc.py, just a different entrypoint):

    python g1_sonic_client_rtc_hack.py \
        --policy-host 127.0.0.1 --policy-port 5000 --include-neck \
        --prompt "..." \
        --execution-horizon 12 --inference-delay 10 --guidance-weight 5.0 \
        --kv-scheme stride1

Optionally override the deploy rate (default 15 Hz) via the DEPLOY_HZ env var,
e.g. `DEPLOY_HZ=15 python g1_sonic_client_rtc_hack.py ...`.

DELETE THIS FILE once the bad episode is re-recorded at 30 Hz.
"""

import os

# Import the real RTC client module and patch its control-loop frequency BEFORE
# any client is constructed / the publish loop starts. `_publish_loop` reads
# `FREQ_POLICY` as a module-global (imported into g1_sonic_client_rtc's namespace
# via `from g1_sonic_client import (... FREQ_POLICY ...)`), so overriding the name
# here is sufficient -- no need to touch g1_sonic_client.py or the RTC class.
import g1_sonic_client_rtc as _rtc

DEPLOY_HZ = float(os.environ.get("DEPLOY_HZ", "15"))

_orig_hz = _rtc.FREQ_POLICY
_rtc.FREQ_POLICY = DEPLOY_HZ

print(
    f"[RTC-HACK] 15 Hz deploy hack ACTIVE: control-loop frequency "
    f"{_orig_hz} Hz -> {DEPLOY_HZ} Hz (for the mis-recorded 15 Hz episode). "
    f"Set DEPLOY_HZ to change."
)


if __name__ == "__main__":
    _rtc.main()
