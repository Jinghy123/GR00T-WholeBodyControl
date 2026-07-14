# Policy Server Communication Protocol

This describes how `psi_rtc_sonic_client.py` talks to the VLA policy server. The
client subscribes to G1 robot state and camera streaming, streams observations
to the policy server, and publishes the returned actions to the G1 Sonic
controller.

## Transport

- **WebSocket**, full-duplex, at `ws://<host>:<port>/ws` (default
  `localhost:8014`).
- A single persistent connection runs two concurrent flows:
  - a **send thread** that streams observations at the robot's tick rate,
  - the WebSocket `on_message` callback that receives actions asynchronously.
- No request/response pairing or sequence id — it is a free-running stream in
  both directions (real-time control, RTC).
- No explicit handshake/auth; opening the connection (`_on_open`) just unblocks
  the send loop.

## Message Encoding

- Every message is a **JSON object** (`json.dumps` / `json.loads`).
- NumPy arrays are embedded with a custom codec
  (`numpy_serialize` / `numpy_deserialize`):

  ```json
  {"__numpy__": "<base64 of raw bytes>", "dtype": "<descr>", "shape": [...]}
  ```

  `convert_numpy_in_dict` walks the dict/list tree to encode on send and decode
  on receive.

## Client → Server (observation)

One JSON payload per tick:

```json
{
  "image":   {"observation.images.egocentric": "<uint8 HxWx3 RGB ndarray>"},
  "state":   {"states": "<float32 ndarray>"},
  "gt_action": null,
  "dataset_name": null,
  "instruction": "<task string>",
  "history": null,
  "condition": null,
  "timestamp": null
}
```

- **Image**: single egocentric RGB frame, BGR→RGB converted, `uint8`.
- **State vector** (`joint_positions` ordering `[hand_L7 | hand_R7 | arm14 | leg15]`):
  - default: **43-D** = hand(14) + arm(14) + leg(15)
  - `--include-neck`: **45-D** = 43-D + neck(2) appended.

## Server → Client (action)

```json
{"action": "<ndarray>", "version": "<int>"}
```

- **Action vector** layout `[hand_joints(14) | body_token(64) | (neck(2))]`:
  - default: **78-D**
  - `--include-neck`: **80-D**, neck as the trailing 2 dims.
- The client post-processes before publishing to the controller:
  - the 64-D body token is **FSQ-quantized** (`fsq_quantize`, step `1/16`,
    range `±0.625`),
  - reordered to the controller's expected
    `[token(64) | left_hand(7) | right_hand(7)]`,
  - published over ZMQ (Protocol v4) to the G1 Sonic WBC; neck (if present) is
    published separately as `[yaw, pitch]` JSON.

## Notes

- The protocol is **stateless and latest-only**: each tick sends the most recent
  state + frame (`history: null`), and actions are consumed as they arrive.
- The `version` field is informational (logged, used to tag the publish).

---

# DreamZero Policy Server Protocol

This describes how `g1_junjie_client.py` (the DreamZero real-G1 client) talks to
the DreamZero policy server (`serve_dreamzero_g1_real.py`). It speaks SIMPLE's
HTTP `/act` protocol and returns **action chunks** rather than the RTC stream
above.

## Transport

- **HTTP** (SIMPLE-style), default `http://<host>:<port>` (default
  `localhost:22085`), via `requests`.
- Request/response per call (synchronous), not a persistent stream.
- Endpoints:
  - `GET  /health` — readiness probe (polled at startup until 200).
  - `GET  /config` — returns model framing (`action_horizon`,
    `video_frames_per_chunk`, `video_stride`); used to override client defaults.
  - `POST /act` — main inference call (observation → action chunk).
  - `POST /flush` — reset server-side history/state (called on client reset/stop).
- The client runs two background threads: an **inference worker** (issues `/act`
  calls) and a **30 Hz publish loop** (plays the returned chunk to the WBC).

## Message Encoding

- JSON body, same custom numpy codec as the RTC client
  (`_np_serialize` / `_np_deserialize`):

  ```json
  {"__numpy__": "<base64 raw bytes>", "dtype": "<descr>", "shape": [...]}
  ```

## Client → Server (`POST /act`)

```json
{
  "image":   {"rgb_head": "<uint8 (T,H,W,3) RGB video ndarray>"},
  "instruction": "<task prompt>",
  "history": {"session_id": "<str>", "episode_index": -1,
              "step_index": "<int>", "reset": true},
  "state":   {"states": "<float32 (1,43) ndarray>"},
  "condition": {},
  "gt_action": [],
  "dataset_name": "dreamzero_g1_real",
  "timestamp": "<str>"
}
```

- **Image**: a short **video clip**, not a single frame —
  `video_frames_per_chunk` frames sampled at `video_stride`, **end-aligned**
  (most recent observation last).
- **State vector** (43-D) = qpos(29) + left_hand_q(7) + right_hand_q(7).
- **History**: carries `session_id` / `step_index`; `reset: true` is sent only
  on the first call after a reset, so the server maintains per-session state.

## Server → Client (`/act` response)

```json
{"action": "<(action_horizon, 78) ndarray>", "err": 0.0}
```

- Returns a full **action chunk** of length `action_horizon` (default 48 ≈ 1.6 s
  at 30 Hz), each row 78-D `[hand_joints(14) | body_token(64)]`.
- `err` may be a float or an error string (logged).
- Client post-processing: the 64-D body token half is **FSQ-quantized**
  (`fsq_quantize`, step `1/16`, range `±0.625`), the chunk reordered to
  `[token(64) | left_hand(7) | right_hand(7)]`, and rows published at 30 Hz over
  ZMQ (Protocol v4) to the WBC.

## Chunk playback & freezing

- The publish loop steps through the chunk one row per 30 Hz tick while the next
  `/act` runs in the background.
- When a chunk is **exhausted before the next arrives**, the client re-anchors:
  it reads current robot pose, runs the local ONNX encoder
  (`model_encoder.onnx`) to recompute the body token for the current pose, and
  holds hand joints at their last commanded value (a "frozen" action) until the
  new chunk lands. This keeps the body token consistent with the robot's actual
  pose during the gap.

## Notes

- Unlike the RTC client, this protocol is **chunked and session-stateful**:
  observations are video clips, responses are multi-step action horizons, and
  the server tracks history across calls via `session_id` / `step_index`.

---

# DreamZero Non-RTC (openpi) Policy Server Protocol

This describes how `dz_client.py` (the non-RTC DreamZero G1-neck client) talks
to the DreamZero policy server (`baselines/dreamzero/socket_test_g1_neck.py`). It
uses an **openpi `WebsocketPolicyServer`** with a msgpack-numpy wire format, and
like `g1_junjie_client.py` it returns **action chunks** played open-loop at 30 Hz
with encoder token-freeze between chunks. The neck (2 dims) is first-class here.

## Transport

- **WebSocket** carrying **msgpack-numpy** frames (openpi protocol), default
  server port `5000` (client default `48014` via SSH tunnel → `nebula102:5000`).
- Driven by `openpi_client.WebsocketClientPolicy` (`policy.infer(obs)`), not raw
  JSON.
- **Handshake**: on connect the server sends one msgpack **metadata** frame
  (consumed by `WebsocketClientPolicy.__init__`; readable via
  `get_server_metadata()`).
- Per request: client sends one **flat** msgpack obs dict; server replies with a
  msgpack action array. Request/response, synchronous.
- Two background threads: an **inference worker** (`infer` calls) and a **30 Hz
  publish loop** (chunk playback + encoder freeze).

## Message Encoding

- **msgpack-numpy** (handled inside openpi). The client passes/receives plain
  numpy arrays — no manual base64/JSON codec as in the HTTP/RTC variants.

## Client → Server (obs dict, FLAT top-level keys)

```python
{
  "observation/head":        head,        # uint8 RGB, (9, H, W, 3) video window
  "observation/hand_joints": hand,        # float32 (14,)  = L_hand(7) + R_hand(7)
  "observation/qpos":        body_q,      # float32 (29,)  raw, leg/base(15) + arm(14)
  "observation/neck":        neck,        # float32 (2,)   [yaw, pitch]
  "prompt":                  instruction, # str
  "session_id":              session_id,  # str (per-run uuid)
  "endpoint":                "infer",     # "infer" to run, "reset" to reset
}
```

- **Image** is a **video window**: 9 frames at stride-4, offsets
  `[-32,-28,…,-4,0]` pulled from a continuous 30 Hz ego ring buffer
  (left-padded), pre-resized to `672x384` before sending.
- **State is sent raw and unsplit** — the server reorders `qpos` + `hand_joints`
  + `neck` into the model state internally (unlike the RTC/HTTP clients, which
  pre-assemble the 43/45-D state vector).
- **`endpoint`** must be set explicitly — openpi does not auto-add it; the
  server reads `obs["endpoint"]` first to dispatch run vs. reset.
- **`session_id`** is a per-run uuid; the server resets its video frame buffer
  when it changes.
- **Instruction** is composed as `Task: <task>. Subtask: <subtask>`, with the
  subtask advanced by the operator pressing Enter (`InstructionManager`).

## Server → Client (action chunk)

- Returns a chunk `(T, 80)`, each row `[hand_joints(14) | neck(2) | token(64)]`
  (note: **neck before token**, unlike the 78-D `[hand | token]` HTTP/RTC layout).
- Client post-processing: FSQ-quantize the 64-D token, keep the publish layout
  `hand(14) + neck(2) + token(64)`, then per 30 Hz tick:
  - publish `token + hands` over ZMQ (Protocol v4) to the WBC,
  - publish `neck [yaw, pitch]` separately to the G1 NeckMotor.

## Chunk playback & freezing

- Same scheme as `g1_junjie_client.py`: step one row per tick while the next
  `infer` runs in the background.
- On chunk exhaustion, re-anchor: read current pose, run the local ONNX encoder
  (`model_encoder.onnx`) to recompute the body token, keep last hand+neck, and
  hold that "frozen" action until the next chunk arrives. The 30 Hz ego ring
  buffer keeps filling during the freeze so the next video window stays
  continuous.

## Notes

- Three transports for the same downstream WBC: **WebSocket+JSON RTC**
  (`psi_rtc_sonic_client.py`, per-tick streaming), **HTTP /act**
  (`g1_junjie_client.py`, chunked), and **openpi msgpack-WebSocket**
  (`dz_client.py`, chunked). The latter two share chunk playback + encoder
  freeze; only transport, obs layout, and action ordering differ.

## `dz_client.py` end-to-end walkthrough

A step-by-step trace of the chunked client (also the template for
`cosmos_http_client.py`).

### Camera frames

- Camera is a ZMQ **REQ/REP** server on the robot
  (`tcp://192.168.123.164:5558`). `RSCamera` (default) or `ZedNeckCamera`
  (`--include-neck`) send `b"get_frame"` and `cv2.imdecode` slot 0 (ego RGB) of
  the multipart reply.
- Frames are **not** pulled per request. The 30 Hz publish loop continuously
  calls `_capture_frame()`: grab → BGR→RGB → resize to `672x384` → append to a
  `deque(maxlen=64)` ring buffer.
- Per request, `_video_window()` selects **9 frames at stride-4** (offsets
  `[-32,-28,…,-4,0]`, left-padded with the oldest) → `(9, H, W, 3)` video block.

### G1 state subscription

- `RobotStateSubscriber` — ZMQ **SUB** on `localhost:5557`, topic `g1_debug`
  (published by the WBC). A background thread strips the topic prefix,
  `msgpack.unpackb`s the payload, and stores the latest dict under a lock.
- Keys used: `body_q_measured` (29 = leg/base15 + arm14),
  `left_hand_q_measured` / `right_hand_q_measured` (7 each),
  `base_quat_measured` (4, wxyz).
- Neck state (`--include-neck`) comes from a separate SUB, `NeckStateReader`,
  reading JSON `[yaw, pitch]` from `tcp://192.168.123.164:5560`.

### Sending the request

- Policy link is openpi's `WebsocketClientPolicy` (msgpack-numpy).
- `_build_observation_payload()` assembles the flat obs dict (head video,
  hand_joints(14), raw qpos(29), neck(2), prompt, session_id, `endpoint="infer"`).
- `_get_policy_chunk()` calls `policy.infer(payload)` on the **inference worker
  thread**, validates the `(N, 80)` reply `[hand(14)|neck(2)|token(64)]`,
  FSQ-quantizes the token, and stores the chunk in `_pending_chunk`. The worker
  is gated by `_sequence_done_event` (set by the publish loop when it needs the
  next chunk).

### Chunk playback + freeze (the 30 Hz publish loop)

- **Executing** (`idx < len(chunk)`): publish `chunk[idx]`, save `last_action`,
  `idx += 1`.
- **Exhausted** (`idx == len(chunk)`): build a **frozen action** so the robot
  holds pose — read current state, convert qpos Mujoco→IsaacLab, tile
  qpos/`base_quat` to `(10, …)` with zero velocity, run the **ONNX sonic
  encoder** to get a 64-D token for the *current* pose, and assemble
  `[last hand(14) | last neck(2) | encoder token(64)]`. Signal
  `_sequence_done_event` to request the next chunk.
- Keep publishing the frozen action each tick until `_pending_chunk` is ready,
  then swap in the new chunk and reset `idx=0`.
- Each tick publishes `token + hands` over ZMQ Protocol-v4 and `neck` to the
  NeckMotor, then `_capture_frame()` keeps the ego buffer continuous (even
  during freeze).

The freeze works by **re-encoding the robot's live proprioception through the
sonic tokenizer** every gap-tick, so the WBC always receives a valid body token
consistent with where the robot actually is — holding it steady instead of
drifting while waiting on the next chunk.
