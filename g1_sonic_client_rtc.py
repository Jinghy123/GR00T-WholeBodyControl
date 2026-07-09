#!/usr/bin/env python3
"""
g1_sonic_client_rtc.py
Real-robot Real-Time Chunking (RTC) client for the DreamZero g1_sonic RTC server
(DreamZero-private/scripts/serve/g1_sonic_server_rtc.py).

Usage:
    python g1_sonic_client_rtc.py --policy-host localhost --policy-port 5000 \
        --zmq-host "*" --zmq-port 5556 --prompt "your task prompt" \
        [--include-neck] [--execution-horizon 12] [--inference-delay 10] \
        [--guidance-weight 5.0]

──────────────────────────────────────────────────────────────────────────────
Why the control loop lives on the CLIENT (server needs NO changes)
──────────────────────────────────────────────────────────────────────────────
Psi0's RTC keeps the 30 Hz control loop, the `A_prev` shift and the delay
buffer on the *server*. Our server is different: every `client.infer(obs)` call
runs exactly ONE (blocking) inference, and the RTC guidance shift
(`new[0:H-s] tracks prev[s:H]`) is done INTERNALLY on the server, which keeps its
own previous prediction (`_rtc_prev_action_norm`). So the client sends NO prev
actions -- only obs + RTC params. That means the entire control/timing logic
(30 Hz streaming, when to re-plan, which image/state to feed, the delay buffer)
belongs on the client, and the server file is reused unchanged.

The only coupling between client and server is the CADENCE: the server shifts its
previous prediction by exactly `execution_horizon = s`, so the client MUST fire
one inference every `s` executed actions, feeding the observation that is `s`
world-actions ahead of the previous one. This client guarantees that.

──────────────────────────────────────────────────────────────────────────────
Timing / image indexing (the crux -- read this)
──────────────────────────────────────────────────────────────────────────────
30 Hz control tick, `t` = index of the next action to publish from the current
chunk `A_cur` (A_cur[0] == world action W):

  each tick:
    (0) if a fresh chunk arrived from the worker: A_cur = new; t -= s
        (the s actions we "burned" on the old chunk's overlap during inference)
    (1) publish A_cur[t]; t += 1
    (2) capture (image, state) NOW -> buffer.
        A command published at tick k only physically takes effect ~1 tick
        later, so the frame captured right after publishing A_cur[t] reflects
        the PREVIOUS action A_cur[t-2] (world W+t-2). i.e. buffer[-1] == the
        observation *after* the last-but-one published action.
    (3) if t == s+1  -> we have just published A_cur[s] (world W+s) and the
        freshest capture buffer[-1] is the image observed AFTER executing
        A_cur[s-1] (== image at world W+s == "image_12" for s=12). Fire ONE
        background inference with that image/state. The server will re-plan a
        chunk that starts at world W+s (shift-by-s guidance).
    (4) sleep to keep 30 Hz.

So the re-plan input is `image_{W+s}` / `state_{W+s}` -- the observation that
truly follows the s-th executed action -- which is the "+1 tick" correction over
Psi0 (Psi0 fires right after step() using the pre-action image). We send 2 frames
per re-plan (mid-window + freshest) which the stride-1 server duplicates to 4 and
VAE-encodes into 1 latent, exactly like the open-loop test client.

Delay buffer (adaptive frozen-prefix `d`):
  After each swap, the post-swap `t` equals `d_real` = how many actions were
  executed off the old chunk while the inference ran (i.e. real latency in ticks).
  We keep a small deque of recent d_real and send `inference_delay = max(deque)`
  (clamped to [1, min(s, H-s)]). The server uses it as the RTC frozen-prefix so
  the in-flight actions are the ones held identical across the chunk switch.

Server must be launched with RTC_STRIDE1=1 (this client fires one commit per
re-plan; stride-1 commits every step so rtc_commit is ignored server-side).
"""

import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np

# Import all hardware / protocol helpers from the proven non-RTC client so the
# neck vs no-neck I/O, ZMQ wire formats, FSQ quantization and the websocket
# policy client stay byte-for-byte identical and never drift. Importing this
# module also sets up sys.path (gear_sonic, eval_utils, encoder_client).
from g1_sonic_client import (
    RSCamera,
    ZedNeckCamera,
    WBCStateReader,
    TokenPublisher,
    NeckStateReader,
    NeckPublisher,
    fsq_quantize,
    _mujoco29_to_isaaclab29,
    ENCODER_MODEL,
    POLICY_IMAGE_RESOLUTION,
    CAMERA_KEY,
    FREQ_POLICY,
    ACTION_HORIZON,
    HAND_DIM,
    NECK_DIM,
    TOKEN_DIM,
    ACTION_DIM_DEFAULT,
    ACTION_DIM_NECK,
    REALSENSE_HOST,
    REALSENSE_PORT,
    WBC_HOST,
    WBC_PORT,
    WBC_TOPIC,
    DEFAULT_ZMQ_HOST,
    DEFAULT_ZMQ_PORT,
    DEFAULT_ZMQ_TOPIC,
    DEFAULT_NECK_PUB_HOST,
    DEFAULT_NECK_PUB_PORT,
    DEFAULT_NECK_STATE_ZMQ,
    TASK_PROMPT,
)
from g1_sonic_client import WebsocketClientPolicy  # only set when policy client is available

try:
    from encoder_client import EncoderClient
    _ENCODER_AVAILABLE = True
except Exception:  # pragma: no cover - encoder optional (only used for freeze)
    _ENCODER_AVAILABLE = False

DEFAULT_POLICY_PORT = 5000


class RTCTokenPolicyClient:
    """Client-side RTC controller: 30 Hz publish loop + background re-plan worker."""

    def __init__(
        self,
        policy_host,
        policy_port,
        prompt,
        zmq_host,
        zmq_port,
        zmq_topic,
        camera_host,
        camera_port,
        wbc_host,
        wbc_port,
        wbc_topic,
        neck_pub_host,
        neck_pub_port,
        neck_state_zmq,
        include_neck=False,
        execution_horizon=12,
        inference_delay=10,
        guidance_weight=5.0,
        mask_schedule="exponential",
        stabilize_sec=2.0,
        kv_scheme="stride1",
        tick_log_every=1,
    ):
        self._include_neck = include_neck
        self._prompt = prompt
        self._action_dim = ACTION_DIM_NECK if include_neck else ACTION_DIM_DEFAULT
        self._tick_log_every = max(int(tick_log_every), 1)  # per-tick heartbeat throttle

        # KV scheme: "stride1" (server ignores rtc_commit, commit every step) or
        # "optionc" (stride-2: server commits a new block only every 2nd re-plan, so
        # rtc_commit must alternate True/False). The control timing is identical for
        # both -- only the rtc_commit flag differs.
        self._optionc = (kv_scheme == "optionc")
        self._infer_step = 0  # counts EVERY infer (init=0, then each re-plan); drives Option-C commit

        # ---- RTC knobs ----
        self._H = ACTION_HORIZON            # chunk length (24)
        self._s = int(execution_horizon)    # s_min: executed actions per re-plan (12)
        self._d_init = int(inference_delay)  # initial frozen-prefix estimate (10)
        self._gw = float(guidance_weight)
        self._mask_schedule = mask_schedule
        self._stabilize_sec = float(stabilize_sec)
        assert 0 < self._s < self._H, f"need 0<s<H, got s={self._s}, H={self._H}"
        # d must satisfy the server constraint 0 < d <= s <= H - d.
        self._d_max = min(self._s, self._H - self._s)
        self._d = int(np.clip(self._d_init, 1, self._d_max))
        # Adaptive delay buffers (Psi0-style: d = max(recent real latencies in ticks)).
        # We keep TWO estimators because the server has two latency regimes:
        #   * normal re-plans (fast, append one latent to the KV cache), and
        #   * periodic KV-wrap re-plans (~every local_attn_size steps: the server
        #     re-encodes the CLIP anchor + rebuilds the KV cache -> a latency spike).
        # Mixing the spike into the normal deque would inflate d for every fast step
        # (needless extra frozen prefix). Instead the server tells us (via
        # rtc_next_is_wrap) when the NEXT re-plan will wrap; we then send the larger
        # d_wrap for that one step and route its measured latency to _Qw only, so the
        # normal estimate _Q stays tight and the spike never causes a "frozen too few"
        # discontinuity.
        self._Q = deque([self._d], maxlen=12)                  # normal-step latencies
        self._Qw = deque([self._d_max], maxlen=10)             # wrap-step latencies (start conservative)
        self._d_normal = self._d                              # d to send on normal steps
        self._d_wrap = self._d_max                            # d to send on wrap steps
        # Whether the NEXT re-plan we fire is predicted (by the server) to be a KV-wrap.
        self._next_replan_is_wrap = False

        # ---- hardware / protocol I/O (identical to the non-RTC client) ----
        if include_neck:
            self._camera = ZedNeckCamera(host=camera_host, port=camera_port)
        else:
            self._camera = RSCamera(host=camera_host, port=camera_port)
        self._state_reader = WBCStateReader(host=wbc_host, port=wbc_port, topic=wbc_topic)
        self._token_publisher = TokenPublisher(
            host=zmq_host, port=zmq_port, topic=zmq_topic, include_neck=include_neck
        )
        if include_neck:
            self._neck_publisher = NeckPublisher(host=neck_pub_host, port=neck_pub_port)
            self._neck_state_reader = NeckStateReader(neck_state_zmq)
        else:
            self._neck_publisher = None
            self._neck_state_reader = None

        self._encoder = None
        if _ENCODER_AVAILABLE:
            try:
                self._encoder = EncoderClient(ENCODER_MODEL, mode=0)
            except Exception as e:
                print(f"[RTC] encoder unavailable ({e}); freeze will repeat last action")

        # ---- policy websocket client ----
        import uuid
        self._client = WebsocketClientPolicy(host=policy_host, port=policy_port)
        metadata = self._client.get_server_metadata()
        print(f"[RTC] Server metadata: {metadata}")
        self._session_id = str(uuid.uuid4())
        print(f"[RTC] Session ID: {self._session_id}")

        # ---- shared control state ----
        self._running = threading.Event()
        self._A_cur = None            # current (H, action_dim) chunk (publish loop only)
        self._t = 0                   # next index into A_cur (publish loop only)
        self._last_action = None      # last real (non-freeze) action published
        self._frozen_action = None    # cached encoder-freeze action while WAITING

        # obs samples captured one-per-tick: {img, qpos, hand, neck}
        self._obs_buffer = deque(maxlen=100)

        # publish loop -> worker: the queued re-plan request
        self._infer_req = None
        self._infer_req_lock = threading.Lock()
        self._infer_event = threading.Event()
        self._in_flight = False       # a re-plan is queued/running (publish loop only)

        # worker -> publish loop: the finished chunk
        self._pending = None
        self._pending_lock = threading.Lock()

    # ------------------------------------------------------------------ obs I/O

    def _capture_sample(self):
        """Capture one (image, state[, neck]) sample. Returns dict or None."""
        frame = self._camera.get_frame()
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
        # ZED (neck) outputs 672x376 -> resize to the server's expected 672x384.
        # RealSense (no-neck) frames are sent raw; the server transform resizes.
        if self._include_neck:
            rgb = cv2.resize(rgb, POLICY_IMAGE_RESOLUTION)

        state = self._state_reader.get_state()
        if state is None:
            return None
        qpos = state["qpos"].astype(np.float32)                                   # (29,)
        hand = np.concatenate([state["left_hand_q"], state["right_hand_q"]]).astype(np.float32)  # (14,)

        neck = None
        if self._include_neck:
            nl = self._neck_state_reader.get_latest()
            neck = (
                np.asarray(nl, dtype=np.float32).reshape(NECK_DIM)
                if nl is not None
                else np.zeros(NECK_DIM, dtype=np.float32)
            )
        return {"img": rgb, "qpos": qpos, "hand": hand, "neck": neck}

    def _take_commit_flag(self):
        """rtc_commit for the current infer, then advance the step counter.
        stride-1: always True (server ignores it). Option C: alternate True/False
        (True on even infer steps -> init=True, replan1=False, replan2=True, ...)."""
        commit = True if not self._optionc else (self._infer_step % 2 == 0)
        self._infer_step += 1
        return commit

    def _build_obs(self, images, qpos, hand, neck, commit):
        """Build the RTC obs payload for one infer() call.
        `images` is (H,W,3) for the init call, (2,H,W,3) for re-plan calls."""
        obs = {
            CAMERA_KEY: images,
            "observation/qpos": qpos,
            "observation/hand_joints": hand,
            "prompt": self._prompt,
            "session_id": self._session_id,
            # action-only inference (RTC only supports this path)
            "action_only_inference": True,
            "action_attend_to_noisy_video": False,
            # ---- RTC control params (consumed by the server, not model inputs) ----
            "rtc": True,
            "rtc_commit": bool(commit),  # stride-1: ignored; Option C: alternates
            "inference_delay": int(self._d),
            "execution_horizon": int(self._s),
            "guidance_weight": float(self._gw),
            "mask_schedule": self._mask_schedule,
            # Ask the server for the richer response (adds rtc_next_is_wrap so we can
            # pre-size d for the periodic KV-wrap latency spike). Legacy servers ignore
            # this key and still return a bare ndarray, which _infer handles.
            "rtc_return_meta": True,
        }
        if self._include_neck:
            obs["observation/neck"] = (
                neck if neck is not None else np.zeros(NECK_DIM, dtype=np.float32)
            )
        return obs

    def _quantize_chunk(self, chunk):
        """FSQ-quantize the body token of every action in the chunk (like the
        non-RTC client's get_action). Layout: hand(14)[+neck(2)]+token(64)."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if self._include_neck:
            hand = chunk[:, :HAND_DIM]
            neck = chunk[:, HAND_DIM:HAND_DIM + NECK_DIM]
            token = fsq_quantize(chunk[:, HAND_DIM + NECK_DIM:HAND_DIM + NECK_DIM + TOKEN_DIM])
            return np.concatenate([hand, neck, token], axis=-1)
        hand = chunk[:, :HAND_DIM]
        token = fsq_quantize(chunk[:, HAND_DIM:HAND_DIM + TOKEN_DIM])
        return np.concatenate([hand, token], axis=-1)

    def _infer(self, images, qpos, hand, neck):
        commit = self._take_commit_flag()
        obs = self._build_obs(images, qpos, hand, neck, commit)
        resp = self._client.infer(obs)
        # New server returns {"actions", "rtc_next_is_wrap"}; legacy server returns a
        # bare (H, action_dim) ndarray. Support both so the client never hard-depends
        # on the wrap-prediction protocol.
        if isinstance(resp, dict):
            raw = resp["actions"]
            next_is_wrap = bool(resp.get("rtc_next_is_wrap", False))
        else:
            raw = resp
            next_is_wrap = False
        assert raw.shape[0] == self._H, f"expected {self._H} actions, got {raw.shape}"
        return self._quantize_chunk(raw), next_is_wrap

    # ------------------------------------------------------------------ freeze

    def _freeze_action(self):
        """Action to hold while WAITING for a late chunk (should be rare in RTC).
        Re-encode the body token from the current pose so the robot holds still;
        keep hand/neck from the last real action. Cached until the next swap."""
        if self._frozen_action is not None:
            return self._frozen_action
        last = self._last_action
        state = self._state_reader.get_state()
        if self._encoder is not None and state is not None and last is not None:
            qpos = _mujoco29_to_isaaclab29(state["qpos"])
            base_quat = state["base_quat"]
            joint_pos = np.tile(qpos, (10, 1)).astype(np.float32)
            joint_vel = np.zeros((10, 29), dtype=np.float32)
            body_quat = np.tile(base_quat, (10, 1)).astype(np.float32)
            enc_token = self._encoder.encode(joint_pos, joint_vel, body_quat)  # (64,)
            if self._include_neck:
                self._frozen_action = np.concatenate(
                    [last[:HAND_DIM], last[HAND_DIM:HAND_DIM + NECK_DIM], enc_token]
                )
            else:
                self._frozen_action = np.concatenate([last[:HAND_DIM], enc_token])
        elif last is not None:
            self._frozen_action = last.copy()
        else:
            self._frozen_action = np.zeros(self._action_dim, dtype=np.float32)
        return self._frozen_action

    # ------------------------------------------------------------------ publish

    def _publish_action(self, action):
        self._token_publisher.publish_token(action)
        if self._include_neck and self._neck_publisher is not None:
            neck = action[HAND_DIM:HAND_DIM + NECK_DIM]
            self._neck_publisher.publish(neck[0], neck[1])

    # ------------------------------------------------------------------ startup

    def _wait_for_state(self, timeout_sec=30.0):
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            if self._state_reader.get_state() is not None:
                return True
            print("[RTC] waiting for robot state...")
            time.sleep(0.5)
        return False

    def _init_controller(self):
        """Capture the first observation and run the init inference (1 frame)."""
        sample = None
        while sample is None and self._running.is_set():
            sample = self._capture_sample()
            if sample is None:
                time.sleep(0.05)
        if sample is None:
            return False
        self._obs_buffer.append(sample)
        # First call: single (H,W,3) frame -> server treats it as the RTC init
        # (current_start_frame == 0, encodes the single frame as the anchor).
        print("[RTC] init inference (1 frame)...")
        t0 = time.time()
        chunk, next_is_wrap = self._infer(
            sample["img"], sample["qpos"], sample["hand"], sample["neck"]
        )
        # Seed the wrap prediction for the FIRST re-plan (normally False right after init).
        self._next_replan_is_wrap = next_is_wrap
        print(f"[RTC] init done in {(time.time()-t0)*1000:.1f}ms, chunk={chunk.shape}, "
              f"next_is_wrap={next_is_wrap}")
        self._A_cur = chunk
        self._t = 0
        self._last_action = chunk[0].copy()
        return True

    def start(self):
        print("[RTC] Starting...")
        self._running.set()
        if not self._wait_for_state():
            print("[RTC] no robot state; aborting")
            return False

        # Let the robot settle in its holding pose before we look at it.
        if self._stabilize_sec > 0:
            print(f"[RTC] stabilizing for {self._stabilize_sec:.1f}s...")
            time.sleep(self._stabilize_sec)

        # Enter planner/token-streaming mode, then plan the first chunk.
        self._token_publisher.send_command(start=True, stop=False, planner=True)
        time.sleep(0.2)  # let the WBC subscribe before streaming
        if not self._init_controller():
            print("[RTC] init inference failed; aborting")
            return False

        self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._infer_thread.start()
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()
        print("[RTC] Started.")
        return True

    # ------------------------------------------------------------------ worker

    def _inference_worker(self):
        """Runs the (blocking) re-plan inference off the 30 Hz loop."""
        while self._running.is_set():
            if not self._infer_event.wait(timeout=0.5):
                continue
            self._infer_event.clear()
            with self._infer_req_lock:
                req = self._infer_req
                self._infer_req = None
            if req is None:
                continue
            try:
                t0 = time.time()
                chunk, next_is_wrap = self._infer(
                    req["images"], req["qpos"], req["hand"], req["neck"]
                )
                dt = (time.time() - t0) * 1000
                with self._pending_lock:
                    self._pending = {
                        "chunk": chunk,
                        # was THIS re-plan a wrap? (routes its latency to the right deque)
                        "fire_is_wrap": bool(req.get("fire_is_wrap", False)),
                        # will the NEXT re-plan be a wrap? (server prediction)
                        "next_is_wrap": next_is_wrap,
                    }
                print(f"[RTC] re-plan inference {dt:.1f}ms  (d={self._d} "
                      f"{'WRAP' if req.get('fire_is_wrap') else 'norm'}"
                      f" -> next_wrap={next_is_wrap})")
            except Exception as e:
                print(f"[RTC] re-plan inference failed: {e}")

    def _fire_inference(self):
        """Queue a re-plan using the freshest obs (image_{W+s}/state_{W+s}) and a
        mid-window frame (~s//2 ticks earlier) for the stride-1 2-frame VAE."""
        n = len(self._obs_buffer)
        fresh = self._obs_buffer[-1]
        mid_off = self._s // 2
        used_mid = -1 - mid_off if n > mid_off else 0
        mid = self._obs_buffer[used_mid]
        if n <= mid_off:
            print(f"[RTC][WARN] obs_buffer too short (n={n}<= s//2={mid_off}); "
                  f"using buffer[0] as mid frame (only at very start)")
        images = np.stack([mid["img"], fresh["img"]], axis=0)  # (2, H, W, 3), earliest first
        # If the server predicted this upcoming re-plan is a KV-wrap (spike), send the
        # larger, wrap-specific frozen prefix so the extra latency is fully covered and
        # we don't fall into a "frozen too few" discontinuity. Tag the request so the
        # swap handler routes the measured latency to the right estimator.
        fire_is_wrap = self._next_replan_is_wrap
        self._d = self._d_wrap if fire_is_wrap else self._d_normal
        req = {
            "images": images,
            "qpos": fresh["qpos"],
            "hand": fresh["hand"],
            "neck": fresh["neck"],
            "fire_is_wrap": fire_is_wrap,
        }
        with self._infer_req_lock:
            self._infer_req = req
        self._infer_event.set()
        return used_mid  # for logging

    # ------------------------------------------------------------------ 30 Hz loop

    def _publish_loop(self):
        dt = 1.0 / FREQ_POLICY
        next_tick = time.perf_counter()
        gtick = 0                 # global tick counter == world action index
        was_freezing = False
        print(f"[RTC] control loop @ {FREQ_POLICY}Hz | H={self._H} s={self._s} "
              f"d_init={self._d} gw={self._gw} scheme={'optionc' if self._optionc else 'stride1'}")
        while self._running.is_set():
            # (0) swap in a freshly-planned chunk, if ready.
            pending = None
            with self._pending_lock:
                if self._pending is not None:
                    pending = self._pending
                    self._pending = None
            if pending is not None:
                new_chunk = pending["chunk"]
                fire_is_wrap = pending["fire_is_wrap"]     # was the just-finished re-plan a wrap?
                next_is_wrap = pending["next_is_wrap"]     # will the NEXT re-plan be a wrap?
                # Bookkeeping: A_new[0] aligns with the OLD chunk's world index W+s
                # (server shift-by-s). We already executed s worth of overlap while
                # inference ran, so drop those s indices from t.
                t_old = self._t
                self._A_cur = new_chunk
                self._t = self._t - self._s
                d_real = self._t if self._t > 0 else 0  # ticks burned during inference
                # Route this latency to the matching estimator so the (slower) wrap
                # spike never inflates the fast-path d, and vice-versa.
                if fire_is_wrap:
                    self._Qw.append(max(d_real, 1))
                else:
                    self._Q.append(max(d_real, 1))
                self._d_normal = int(np.clip(max(self._Q), 1, self._d_max))
                self._d_wrap = int(np.clip(max(self._Qw), 1, self._d_max))
                # Prediction (from the server) for the re-plan we fire next, plus the d
                # we will send for it. Keeping self._d in sync makes the heartbeat/trigger
                # logs show the value that will actually be sent.
                self._next_replan_is_wrap = next_is_wrap
                self._d = self._d_wrap if next_is_wrap else self._d_normal
                self._frozen_action = None
                self._in_flight = False
                outran = self._t < 0
                if outran:
                    self._t = 0  # inference outran the chunk; resume at head
                print(f"[RTC][swap] gtick={gtick} new chunk in: t {t_old}->{self._t} "
                      f"d_real={d_real} ({'WRAP' if fire_is_wrap else 'norm'}) "
                      f"d={self._d} (dn={self._d_normal} dw={self._d_wrap}) "
                      f"next_wrap={next_is_wrap} Q={list(self._Q)} Qw={list(self._Qw)} "
                      f"chunk={tuple(new_chunk.shape)}")
                if outran:
                    print(f"[RTC][WARN] gtick={gtick} inference OUTRAN the chunk "
                          f"(t_old={t_old} > H+? ); clamped t->0. Inference too slow for s={self._s}.")

            # (1) pick + publish this tick's action.
            if 0 <= self._t < len(self._A_cur):
                action = self._A_cur[self._t]
                self._last_action = action.copy()
                exec_idx = self._t
                freezing = False
            else:
                action = self._freeze_action()  # WAITING: hold pose
                exec_idx = -1
                freezing = True
            self._publish_action(action)
            self._t += 1

            # freeze enter/exit is abnormal for RTC -> always announce the transition.
            if freezing and not was_freezing:
                print(f"[RTC][WARN] gtick={gtick} chunk EXHAUSTED -> FREEZE (holding pose). "
                      f"t={self._t - 1} len(A_cur)={len(self._A_cur)} in_flight={self._in_flight} "
                      f"d={self._d} s={self._s}. Re-plan is lagging > {self._H - self._s} ticks.")
            elif was_freezing and not freezing:
                print(f"[RTC] gtick={gtick} FREEZE ended, resumed at A[{exec_idx}].")
            was_freezing = freezing

            # (2) capture obs (reflects the previous action; buffer[-1] == post
            #     A_cur[t-2] == "image after the last-but-one published action").
            sample = self._capture_sample()
            if sample is not None:
                self._obs_buffer.append(sample)
            else:
                print(f"[RTC][WARN] gtick={gtick} capture failed (camera/state None); obs skipped")

            # (3) re-plan trigger: once we have published A_cur[s] (t == s+1) the
            #     freshest capture is image_{W+s} (after the s-th executed action).
            fired = False
            if (
                not self._in_flight
                and sample is not None
                and self._t >= self._s + 1
                and self._A_cur is not None
            ):
                used_mid = self._fire_inference()
                self._in_flight = True
                fired = True
                print(f"[RTC][trigger] gtick={gtick} infer_step={self._infer_step} "
                      f"t={self._t} s={self._s} d={self._d} "
                      f"input=image_after_A[{self._s - 1}] (buffer[-1]); mid=buffer[{used_mid}] "
                      f"-> re-plan starts @ world {gtick}")

            # per-tick heartbeat (throttled). Shows d/t/s live.
            if (gtick % self._tick_log_every == 0) and not freezing and not fired:
                print(f"[RTC][tick] gtick={gtick} exec A[{exec_idx}] "
                      f"t={self._t} s={self._s} d={self._d} in_flight={self._in_flight}")

            # (4) keep a strict 30 Hz cadence (camera+publish are inside the budget).
            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"[RTC][WARN] gtick={gtick} missed 30Hz tick by {-sleep_time * 1000:.1f}ms")
                next_tick = time.perf_counter()  # resync
            gtick += 1

    # ------------------------------------------------------------------ shutdown

    def stop(self):
        print("[RTC] Stopping...")
        self._running.clear()
        try:
            self._token_publisher.send_command(start=False, stop=True, planner=False)
        except Exception:
            pass
        # Join the worker/publish threads first so nothing else is using the
        # websocket when we call reset() below (WebsocketClientPolicy is not
        # thread-safe; the worker may be mid-infer()).
        self._infer_event.set()  # wake the worker so it can observe _running=False
        for th in ("_publish_thread", "_infer_thread"):
            t = getattr(self, th, None)
            if t is not None:
                t.join(timeout=2.0)
        try:
            self._client.reset({})  # flush server session / save videos
        except Exception as e:
            print(f"[RTC] reset failed: {e}")
        try:
            self._camera.close()
        except Exception:
            pass
        try:
            self._state_reader.close()
        except Exception:
            pass
        try:
            self._token_publisher.stop()
        except Exception:
            pass
        if self._neck_publisher is not None:
            self._neck_publisher.stop()
        if self._neck_state_reader is not None:
            self._neck_state_reader.close()
        print("[RTC] Stopped.")


def main():
    parser = argparse.ArgumentParser(description="G1 Sonic real-robot RTC client")
    parser.add_argument("--policy-host", type=str, default="localhost")
    parser.add_argument("--policy-port", type=int, default=DEFAULT_POLICY_PORT)
    parser.add_argument("--prompt", type=str, default=TASK_PROMPT)

    parser.add_argument("--zmq-host", type=str, default=DEFAULT_ZMQ_HOST)
    parser.add_argument("--zmq-port", type=int, default=DEFAULT_ZMQ_PORT)
    parser.add_argument("--zmq-topic", type=str, default=DEFAULT_ZMQ_TOPIC)

    parser.add_argument("--camera-host", type=str, default=REALSENSE_HOST)
    parser.add_argument("--camera-port", type=int, default=REALSENSE_PORT)

    parser.add_argument("--wbc-host", type=str, default=WBC_HOST)
    parser.add_argument("--wbc-port", type=int, default=WBC_PORT)
    parser.add_argument("--wbc-topic", type=str, default=WBC_TOPIC)

    parser.add_argument("--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST)
    parser.add_argument("--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT)
    parser.add_argument("--neck-state-zmq", type=str, default=DEFAULT_NECK_STATE_ZMQ)
    parser.add_argument("--include-neck", action="store_true",
                        help="g1_sonic_neck variant: action 80-dim (hand14+neck2+token64), "
                             "state +observation/neck(2), ZED neck camera + neck pub/sub. "
                             "Server must be launched with the same --include-neck flag.")

    # ---- RTC knobs (must be consistent with how the server/model were trained) ----
    parser.add_argument("--execution-horizon", type=int, default=12,
                        help="s / s_min: executed actions per re-plan (chunk overlap = H - s).")
    parser.add_argument("--inference-delay", type=int, default=10,
                        help="d_init: initial RTC frozen-prefix (ticks). Adapts online to the "
                             "measured re-plan latency via a max-deque.")
    parser.add_argument("--guidance-weight", type=float, default=5.0)
    parser.add_argument("--mask-schedule", type=str, default="exponential",
                        choices=["exponential", "linear", "hard"])
    parser.add_argument("--kv-scheme", type=str, default="stride1",
                        choices=["stride1", "optionc"],
                        help="Must match the server: stride1 (RTC_STRIDE1=1, default) or optionc "
                             "(stride-2 commit gating; client alternates rtc_commit).")
    parser.add_argument("--stabilize-sec", type=float, default=2.0,
                        help="Seconds to wait for the robot to settle before the init inference.")
    parser.add_argument("--tick-log-every", type=int, default=1,
                        help="Print the per-tick heartbeat (t/s/d) every N ticks. 1=every tick "
                             "(30 lines/s); events (trigger/swap/freeze/miss) always print.")
    args = parser.parse_args()

    print(f"[RTC] include_neck={args.include_neck}, "
          f"action_dim={ACTION_DIM_NECK if args.include_neck else ACTION_DIM_DEFAULT}")
    print(f"[RTC] H={ACTION_HORIZON} s={args.execution_horizon} d_init={args.inference_delay} "
          f"gw={args.guidance_weight} mask={args.mask_schedule}")

    client = RTCTokenPolicyClient(
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        prompt=args.prompt,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        zmq_topic=args.zmq_topic,
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        wbc_host=args.wbc_host,
        wbc_port=args.wbc_port,
        wbc_topic=args.wbc_topic,
        neck_pub_host=args.neck_pub_host,
        neck_pub_port=args.neck_pub_port,
        neck_state_zmq=args.neck_state_zmq,
        include_neck=args.include_neck,
        execution_horizon=args.execution_horizon,
        inference_delay=args.inference_delay,
        guidance_weight=args.guidance_weight,
        mask_schedule=args.mask_schedule,
        stabilize_sec=args.stabilize_sec,
        kv_scheme=args.kv_scheme,
        tick_log_every=args.tick_log_every,
    )

    try:
        if not client.start():
            print("[RTC] Failed to start client")
            return
        print("[RTC] Running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[RTC] Caught Ctrl+C, stopping...")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
