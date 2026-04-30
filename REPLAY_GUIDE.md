# Replaying Recorded Episodes — `replay_token.py` / `replay_token_latency.py`

These two scripts re-publish the actions captured in a recorded
`data.json` so the G1 reproduces the demonstration. They run on the
**desktop** and stream over ZMQ to the G1 at `192.168.123.164`.

| Script | Use when… |
|---|---|
| [replay_token.py](replay_token.py) | You want a vanilla, frame-by-frame replay at the original recording rate (30 Hz). Token-only Protocol v4 stream — what the teleop pipeline emits in real time. |
| [replay_token_latency.py](replay_token_latency.py) | You want to **simulate inference latency**. Sends `--chunk-size` tokens at 30 Hz, then freezes for `--latency-frames` ticks (computing a "freeze token" from current robot state via the encoder) before the next chunk. Use this to test how the policy will behave with a real on-device inference budget. |

Both scripts can additionally drive the **2-DOF neck motors** by
re-publishing `actions['neck']` on the same ZMQ port that
`pico_manus_thread_server.py` (or `pose_publisher.py` fallback) would use.
Add `--enable-neck` to either invocation.

---

## What goes over the wire

```
┌──────────────────────────────────────┐               ┌───────────────────────────┐
│   Desktop                            │               │   G1  192.168.123.164     │
│                                      │               │                           │
│  replay_token*.py     ── PUB :5556 ─►│──────────────►│  WBC controller (sub)     │
│   (token + hands)        topic=pose  │               │  (g1_deploy_onnx_ref)     │
│                                      │               │                           │
│  replay_token*.py     ── PUB :5570 ─►│──────────────►│  realsense_server.py      │
│   (--enable-neck)        JSON [y,p]  │               │  --pose-zmq               │
│                                      │               │   (NeckMotor SUB)         │
│                                      │               │                           │
│  replay_token_latency  ◄── SUB :5557 ┤◄──────────────┤  WBC state PUB            │
│   (encoder freeze)       topic=g1_   │               │   (body_q_measured,       │
│                          debug       │               │    base_quat_measured)    │
└──────────────────────────────────────┘               └───────────────────────────┘
```

- **Tokens (`:5556`)** — Protocol v4 messages packed by
  `gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message`
  with `version=4`. Contains `token_state` (latent), optional
  `left_hand_joints` / `right_hand_joints` (7-D each).
- **Neck (`:5570`)** — JSON-encoded `[yaw_rad, pitch_rad]`. Identical to
  `pico_manus_thread_server.py`'s neck PUB and `pose_publisher.py`'s wire
  format; `realsense_server.py`'s `--pose-zmq` subscriber consumes it
  without any change. (Port chosen at 5570 to stay clear of the Wuji
  block 5559-5561: 5559 wuji_hand tracking, 5560 wuji_state, 5561
  wuji_replay command port.)
- **WBC state (`:5557`, latency variant only)** — msgpack from the WBC
  controller, used by `replay_token_latency.py` to compute a freeze
  token through the encoder during the WAITING phase.

---

## Prerequisites

### Required upstream processes (on the G1)
1. **WBC controller** — typically `g1_deploy_onnx_ref` with
   `--output-type zmq` (or `all`). Listens for token messages on
   `:5556` and publishes its own state on `:5557`.
2. **`realsense_server.py` with neck enabled** (only if using `--enable-neck`):
   ```bash
   python realsense_server.py \
       --zed-only \
       --zmq-bind tcp://0.0.0.0:5558 \
       --enable-neck-motor \
       --pose-zmq tcp://<DESKTOP_IP>:5570
   ```
   See [NECK_TELEOP_GUIDE.md](NECK_TELEOP_GUIDE.md) for full setup.
   Don't forget `sudo chmod 777 /dev/ttyUSB0`.

### Required on the desktop
1. The recorded episode folder containing `data.json` (and per-frame
   image folders, but the replay scripts only need `data.json`).
2. The `gear_sonic_deploy/policy/release/model_encoder.onnx` model
   (latency variant only — used to compute freeze tokens). Path is
   resolved relative to `_GROOT_ROOT` at the top of each script.
3. **Stop any running neck publisher** before using `--enable-neck` —
   in standard flow that's `pico_manus_thread_server.py`, but
   `pose_publisher.py` is also possible:
   ```bash
   pkill -f pico_manus_thread_server.py
   pkill -f pose_publisher.py
   ```
   All three bind port 5570; otherwise the replay's bind fails fast
   with a clear `NeckPublisher bind failed on tcp://*:5570` error.

### What's in a record
The replay scripts read from each frame:

| Path in `data.json` | Used by | What it is |
|---|---|---|
| `actions.token` | both | Latent vector (required for the EXEC phase). |
| `actions.hand_joints` | both | Optional 14-D (left 7 + right 7). |
| `actions.neck` | both, only with `--enable-neck` | 2-D `[yaw_rad, pitch_rad]` written by `g1_data_server.py --neck-zmq` (commanded angle, post-smoothing). |

`states.neck` (motor present-position) is **not** used by replay — the
canonical replay source is the command, not the measured state. If you
want measured-state replay, see [Caveats](#caveats) below.

---

## Quick reference

### Vanilla replay
```bash
python replay_token.py --episode-dir /home/xiawei/data/<task>_<session>/episode_0
```

### Vanilla replay + neck
```bash
pkill -f pico_manus_thread_server.py; pkill -f pose_publisher.py
python replay_token.py \
    --episode-dir /home/xiawei/data/<task>_<session>/episode_0 \
    --enable-neck
```

### Latency-simulating replay (16-token chunks, 533 ms latency)
```bash
python replay_token_latency.py \
    --episode-dir /home/xiawei/data/<task>_<session>/episode_0 \
    --chunk-size 16 \
    --latency-frames 16
```

### Latency-simulating replay + neck
```bash
pkill -f pico_manus_thread_server.py; pkill -f pose_publisher.py
python replay_token_latency.py \
    --episode-dir /home/xiawei/data/<task>_<session>/episode_0 \
    --chunk-size 24 \
    --latency-frames 30 \
    --enable-neck
```

### Dry-run (parse only, no ZMQ traffic, no encoder)
```bash
python replay_token.py --episode-dir <path> --dry-run
python replay_token_latency.py --episode-dir <path> --dry-run
```
Useful for sanity-checking that an episode's `data.json` parses and has
tokens/neck before you light up the robot.

---

## What you should see

### `replay_token.py`
```
[ReplayToken] Loaded N frames at 30 Hz from <path>/data.json
[ReplayToken] Neck replay enabled: K/N frames have neck data.
[NeckPublisher] PUB bound to tcp://*:5570
[ReplayToken] Waiting for ZMQ connections...
[TokenPublisher] Command: start=True stop=False planner=True
[ReplayToken] Starting token replay of N frames...
[ReplayToken] Frame    1/N  t=0.03s  token_dim=64 L=… R=…
…
[ReplayToken] Done. Sent N frames.
[ReplayToken] Holding last token. Press Ctrl+C to stop.
```
Token streaming runs at 30 Hz; the neck publisher emits one
`[yaw, pitch]` per token frame at the same rate (the G1 NeckMotor's
SUB uses `CONFLATE`, so the 30 Hz feed lands cleanly in its 50 Hz
control loop).

### `replay_token_latency.py`
```
[ReplayLatency] Loaded N frames at 30 Hz from <path>/data.json
[ReplayLatency] Neck replay enabled: K/M token frames have neck data.
[NeckPublisher] PUB bound to tcp://*:5570
[WBCState] Subscribed to localhost:5557 topic=g1_debug
[ReplayLatency] Waiting for ZMQ connections...
[TokenPublisher] Command: start=True stop=False planner=True
[ReplayLatency] M tokens → C chunks
[ReplayLatency] chunk_size=24  latency_frames=30 (1000 ms)
[ReplayLatency] chunk 1/C  frame    1/M  [EXEC  ]
…
[ReplayLatency] chunk 1/C  hold      1/30  [WAITING]
[ReplayLatency] Encoder freeze token computed from robot state.
…
```
During EXEC the recorded neck angles stream through. During WAITING
the **last neck angle of the chunk is held** — no encoder-equivalent
exists for the neck, so the head simply parks at end-of-exec until the
next chunk starts. That matches the policy's intent (head doesn't move
while we're "thinking").

---

## On the G1, you should see

If `realsense_server.py` is running with `--enable-neck-motor`:
```
[Neck] yaw  +12.6° (tick 2347)  pitch  -3.7° (tick 2982)
[Neck] yaw  +14.1° (tick 2406)  pitch  -3.5° (tick 2986)
…
```
The `tick` values must change as the recorded yaw/pitch change —
otherwise motor writes are off (see `NECK_TELEOP_GUIDE.md`
troubleshooting "tick stays at the zero tick").

---

## Caveats

- **Replay source is `actions['neck']` (post-smoothing command).** It is
  *not* `states['neck']` (the measured present-position). The two
  differ by motor lag, smoothing, and any saturation. To replay
  measured state instead, change `_extract_neck`'s key from
  `"neck"` (under `actions`) to `"neck"` under `states`. Easy
  one-line edit in both replay files; ask if you want it added as a
  flag.
- **Sign and smoothing convention must match record time.**
  `realsense_server.py` re-applies `NECK_YAW_SIGN`, `NECK_PITCH_SIGN`,
  and `NECK_SMOOTH_ALPHA` on the way to the motors. If any of these
  changed between record and replay, the trajectory will be wrong.
  Either freeze them before recording or save them in
  `episode_meta.json` so you can verify match at replay time.
- **Frames without `actions['neck']` hold the last value.** This keeps
  motors smooth across mid-episode dropouts but means a fully
  neck-less episode never moves the neck (the publisher reports
  "no frame has actions['neck']; skipping neck replay" and continues
  with token-only replay).
- **The desktop neck publisher and replay can't both run at once.**
  Whichever one is binding 5570 — `pico_manus_thread_server.py` (the
  default in standard flow) or `pose_publisher.py` (fallback) — has to
  stop before replay binds. The replay's bind error tells you which to
  pkill.
- **`WBC_HOST` in `replay_token_latency.py` is hardcoded to `localhost`**
  (line 28). When running off-robot, override with `--wbc-host
  192.168.123.164` so the encoder freeze can read real WBC state.
  Without a real state the freeze falls back to repeating the last
  recorded token, which is fine for a quick test but not for latency
  realism.
- **Hard-coded paths.** `_GROOT_ROOT` at the top of both scripts is
  `~/hsc/GR00T-WholeBodyControl`, which is *not* this repo's path on
  the desktop (`~/Desktop/GR00T-WholeBodyControl`). Edit those two
  lines once or `ln -s` the directory before running, otherwise the
  encoder model load and the import of `gear_sonic.utils.teleop.zmq…`
  will fail.

---

## Troubleshooting

**`NeckPublisher bind failed on tcp://*:5570`**
A desktop neck publisher is still bound — `pico_manus_thread_server.py`
in standard flow, `pose_publisher.py` if you ran the fallback. Kill
whichever:
```bash
pkill -f pico_manus_thread_server.py
pkill -f pose_publisher.py
```
then re-run. Same error if some other process holds 5570 — find it
with `sudo fuser -v 5570/tcp`.

**`--enable-neck set but no frame has actions['neck']`**
The recorded episode predates the neck-recording feature, or the
recorder didn't have `--neck-zmq` set. The replay continues without
neck; the head won't move during playback.

**Tokens stream but the G1 neck doesn't move**
Check in order:
1. `realsense_server.py` on the G1 logs `[Neck] ZMQ SUB neck-angle
   source: tcp://<desktop-ip>:5570` — if the address is `localhost`
   or wrong, the G1 isn't subscribed to your replay.
2. The G1 logs periodic `[Neck] yaw … pitch …` lines that change. If
   they're stuck at the zero tick, motor writes are disabled in
   `realsense_server.py`'s neck loop.
3. Bus permission: `sudo chmod 777 /dev/ttyUSB0` on the G1 between
   sessions.

**`[ReplayLatency] No robot state available — repeating last token.`**
The WBC state subscriber timed out (no `body_q_measured` arriving on
`:5557`). Either the WBC isn't publishing (start `g1_deploy_onnx_ref
--output-type zmq`) or the host/port are wrong (override with
`--wbc-host <robot-ip> --wbc-port 5557`).

**`FileNotFoundError: data.json not found in <dir>`**
Double-check `--episode-dir`. The path must point at the
`episode_<N>` folder, not at the per-task `<task>_<session>` parent.

**`ImportError: gear_sonic.utils.teleop.zmq.zmq_planner_sender`**
The hard-coded `_GROOT_ROOT` doesn't match where this repo lives on
your machine. Edit line 37 of `replay_token.py` and line 46 of
`replay_token_latency.py` to your actual path, or set
`PYTHONPATH=<path-to-repo>` before running.

**Replay is jittery / not matching the original speed**
Both scripts use `time.perf_counter()` + `time.sleep()` for cadence —
fine on a quiet machine, but heavy CPU load (browser, GPU jobs) can
slip frames. Close other apps or run with `nice -n -10`.
