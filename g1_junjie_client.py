#!/usr/bin/env python3
"""
g1_dreamzero_client.py

Real-G1 client for the DreamZero HTTP policy server (`serve_dreamzero_g1_real.py`).

Mirrors the structure of `g1_sonic_client.py`:
  - RSCamera (ZMQ REQ/REP) for RealSense head camera frames
  - WBCStateReader (ZMQ SUB) for live robot proprio + base quaternion
  - TokenPublisher (ZMQ PUB) for streaming actions to the WBC executor
  - Background inference thread + 30Hz publish thread

Replaces `PolicyClientManager` (websocket → encoder/onnx) with an
`HttpActionClient` that speaks SIMPLE's /act protocol against the
DreamZero server. Action layout is identical (78-D = hand_joints[14] +
token[64]); we apply FSQ quantization to the token half before publishing.

Usage:
    python g1_junjie_client.py \\
        --policy-host localhost --policy-port 22085 \\
        --zmq-host "*" --zmq-port 5556 \\
        --prompt "pick up the red box and pour water into the orange cup"
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from base64 import b64decode, b64encode
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import imageio
import msgpack
import numpy as np
import requests
import zmq
from numpy.lib.format import descr_to_dtype, dtype_to_descr

_GROOT_ROOT = os.path.expanduser("~/hongyi/GR00T-WholeBodyControl")
if os.path.isdir(_GROOT_ROOT):
    sys.path.insert(0, _GROOT_ROOT)

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)
from ours.common.encoder_client import EncoderClient

# WBC publishes qpos in Mujoco order; the ONNX encoder expects IsaacLab order.
# (Same constant as g1_sonic_client.py.)
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
     5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)


def _mujoco29_to_isaaclab29(qpos: np.ndarray) -> np.ndarray:
    return np.asarray(qpos, dtype=np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB_DOF].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# RealSense camera
REALSENSE_HOST = "192.168.123.164"
REALSENSE_PORT = 5558

# WBC state subscriber
WBC_HOST = "localhost"
WBC_PORT = 5557
WBC_TOPIC = "g1_debug"

# ZMQ token publisher (to onnx low-level controller)
DEFAULT_ZMQ_HOST = "*"
DEFAULT_ZMQ_PORT = 5556
DEFAULT_ZMQ_TOPIC = "pose"

# DreamZero HTTP policy server (matches serve_dreamzero_g1_real.py defaults)
DEFAULT_POLICY_HOST = "localhost"
DEFAULT_POLICY_PORT = 22085

# ONNX encoder used to re-anchor the body token to the robot's current pose
# while waiting for the next chunk (mirrors g1_sonic_client.py freeze logic).
ENCODER_MODEL = os.path.join(_GROOT_ROOT, "gear_sonic_deploy/policy/release/model_encoder.onnx")

# Real G1 control rate. Training data is 30 fps, so we publish at 30 Hz
# and the action chunk (action_horizon=48) covers 48/30 ≈ 1.6 s.
FREQ_POLICY = 30
CAMERA_KEY = "rgb_head"

# FSQ configuration (matches g1_sonic_client.py — must mirror training-time
# tokenizer, otherwise the WBC decoder produces nonsense).
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625

# Defaults for DreamZero G1-real, overridden by /config response.
DEFAULT_ACTION_HORIZON = 48
DEFAULT_VIDEO_FRAMES_PER_CHUNK = 8
DEFAULT_VIDEO_STRIDE = 6  # 48 / 8
ACTION_DIM = 78  # 14 (hand) + 64 (body token)
STATE_DIM = 43   # 29 (qpos) + 7 (left hand) + 7 (right hand)

# Generous buffer so we never drop the oldest frame the model needs.
IMAGE_BUFFER_SIZE = 256

TASK_PROMPT = "pick up the red box and pour water into the orange cup"


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    return np.clip(quantized, fsq_min, fsq_max)


# ──────────────────────────────────────────────────────────────────────────────
# SIMPLE numpy-in-JSON wire format (mirrors simple/baselines/client.py)
# ──────────────────────────────────────────────────────────────────────────────


def _np_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _np_deserialize(dct):
    if "__numpy__" in dct:
        arr = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        shape = dct["shape"]
        return arr.reshape(shape) if shape else arr[0]
    return dct


def _convert_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {k: _convert_in_dict(v, func) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_in_dict(v, func) for v in data]
    if isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    return data


class HttpActionClient:
    """Same wire protocol as simple/baselines/client.py:HttpActionClient."""

    def __init__(self, host: str, port: int, request_timeout: float = 60.0):
        self.host = host
        self.port = port
        self.request_timeout = request_timeout

    @property
    def timestamp(self) -> str:
        return str(datetime.now()).replace(" ", "_").replace(":", "-")

    def query_action(
        self,
        image_dict: Dict[str, Any],
        instruction: str,
        state_dict: Dict[str, Any],
        condition_dict: Dict[str, Any],
        history: Optional[Dict[str, Any]] = None,
        dataset: str = "dreamzero_g1_real",
    ):
        if history is None:
            history = {}
        msg = {
            "image": image_dict,
            "instruction": instruction,
            "history": history,
            "state": state_dict,
            "condition": condition_dict,
            "gt_action": [],
            "dataset_name": dataset,
            "timestamp": self.timestamp,
        }
        payload = _convert_in_dict(msg, _np_serialize)
        url = f"http://{self.host}:{self.port}/act"
        resp = requests.post(url, json=payload, timeout=self.request_timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"[/act] {resp.status_code}: {resp.text[:300]}")
        body = _convert_in_dict(resp.json(), _np_deserialize)
        err = body.get("err", 0.0)
        if isinstance(err, str):
            print(f"[HttpActionClient] server reported error: {err}")
        return body["action"], err

    def query_config(self) -> dict:
        try:
            r = requests.get(f"http://{self.host}:{self.port}/config", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"[HttpActionClient] /config failed: {e}")
        return {}

    def health(self, timeout_s: float = 3.0) -> bool:
        try:
            r = requests.get(f"http://{self.host}:{self.port}/health", timeout=timeout_s)
            return r.status_code == 200
        except Exception:
            return False

    def flush(self, timeout_s: float = 30.0) -> None:
        try:
            requests.post(f"http://{self.host}:{self.port}/flush", timeout=timeout_s)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers (RealSense, WBC state, ZMQ token publisher)
#   — copied from g1_sonic_client.py with no logic changes
# ──────────────────────────────────────────────────────────────────────────────


class RSCamera:
    def __init__(self, host=REALSENSE_HOST, port=REALSENSE_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"[RSCamera] Connected to {host}:{port}")

    def get_frame(self):
        self.socket.send(b"get_frame")
        rgb_bytes, _, _ = self.socket.recv_multipart()
        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        return cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)

    def close(self):
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()


class WBCStateReader:
    def __init__(self, host=WBC_HOST, port=WBC_PORT, topic=WBC_TOPIC):
        self._topic_bytes = topic.encode()
        self._topic_len = len(self._topic_bytes)
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, self._topic_bytes)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.connect(f"tcp://{host}:{port}")
        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[WBCState] Subscribed to {host}:{port} topic={topic}")

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                raw = self._sock.recv()
                payload = raw[self._topic_len:]
                data = msgpack.unpackb(payload, raw=False)
                with self._lock:
                    self._latest = {
                        "qpos": np.array(data["body_q_measured"], dtype=np.float32),
                        "left_hand_q": np.array(data.get("left_hand_q_measured", [0]*7), dtype=np.float32),
                        "right_hand_q": np.array(data.get("right_hand_q_measured", [0]*7), dtype=np.float32),
                        "base_quat": np.array(data["base_quat_measured"], dtype=np.float32),
                    }
            except zmq.Again:
                pass
            except Exception as e:
                print(f"[WBCState] Recv error: {e}")

    def get_state(self):
        with self._lock:
            if self._latest is None:
                return None
            return {k: v.copy() for k, v in self._latest.items()}

    def close(self):
        self._stop.set()
        if self._sock:
            self._sock.close()
        self._ctx.term()


class TokenPublisher:
    """Same Protocol-v4 publisher used by g1_sonic_client.py.

    Action layout (78,) = [left_hand_q(7), right_hand_q(7), token(64)]; the
    first 14 are routed as joint commands, the last 64 as the body token.
    """

    def __init__(self, host="*", port=5556, topic="pose"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")
        self._topic = topic
        self._frame_index = 0

    def send_command(self, start=False, stop=False, planner=False):
        msg = build_command_message(start=start, stop=stop, planner=planner)
        self._socket.send(msg)
        print(f"[TokenPublisher] Command: start={start} stop={stop} planner={planner}")

    def publish_token(self, action, body_quat_w=None):
        action = action.reshape(1, -1)
        # pose_data = {
        #     "token_state": action[:, 14:],         # (1, 64)
        #     "left_hand_joints": action[:, :7],     # (1, 7)
        #     "right_hand_joints": action[:, 7:14],  # (1, 7)
        # }
        pose_data = {
            "token_state": action[:, :-14],         # (1, 64)
            "left_hand_joints": action[:, -14:-7],     # (1, 7)
            "right_hand_joints": action[:, -7:],  # (1, 7)
        }
        if body_quat_w is not None:
            pose_data["body_quat_w"] = np.asarray(body_quat_w, dtype=np.float32).reshape(1, 4)
        msg = pack_pose_message(pose_data, topic=self._topic, version=4)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()


# ──────────────────────────────────────────────────────────────────────────────
# DreamZero policy client
# ──────────────────────────────────────────────────────────────────────────────


class DreamZeroPolicyManager:
    """Wraps HttpActionClient with the SIMPLE-style framing the DreamZero
    server expects (video.egocentric + state.joint_position + history)."""

    def __init__(self, host: str, port: int, prompt: str):
        self._client = HttpActionClient(host, port)
        self._prompt = prompt
        self._session_id = f"dreamzero-real-{os.getpid()}-{int(time.time())}"
        self._reset_history = True
        self._step_idx = 0

        self.action_horizon = DEFAULT_ACTION_HORIZON
        self.video_frames_per_chunk = DEFAULT_VIDEO_FRAMES_PER_CHUNK
        self.video_stride = DEFAULT_VIDEO_STRIDE

    def wait_and_configure(self, timeout_s: float = 900.0, poll_s: float = 5.0) -> None:
        t0 = time.time()
        attempt = 0
        while time.time() - t0 < timeout_s:
            attempt += 1
            if self._client.health():
                print(f"[Policy] /health ok after {time.time()-t0:.1f}s")
                break
            if attempt == 1 or attempt % 6 == 0:
                print(f"[Policy] waiting for server at "
                      f"{self._client.host}:{self._client.port} "
                      f"({time.time()-t0:.0f}s elapsed)")
            time.sleep(poll_s)
        cfg = self._client.query_config()
        if cfg:
            print(f"[Policy] server config: {cfg}")
            self.action_horizon = int(cfg.get("action_horizon", self.action_horizon))
            self.video_frames_per_chunk = int(cfg.get("video_frames_per_chunk", self.video_frames_per_chunk))
            self.video_stride = int(cfg.get("video_stride", self.video_stride))

    def reset(self):
        self._client.flush()
        self._reset_history = True
        self._step_idx = 0
        self._session_id = f"dreamzero-real-{os.getpid()}-{int(time.time())}"

    def get_action(self, video: np.ndarray, state43: np.ndarray) -> np.ndarray:
        """Send one /act request and return (action_horizon, 78) chunk.

        Args:
            video:   (T, H, W, 3) uint8 — typically video_frames_per_chunk
                     frames sampled at video_stride, oldest-first.
            state43: (43,) or (1, 43) float — qpos(29) + L_hand(7) + R_hand(7)
        """
        if state43.ndim == 1:
            state43 = state43[None]
        history = {
            "session_id": self._session_id,
            "episode_index": -1,
            "step_index": int(self._step_idx),
        }
        if self._reset_history:
            history["reset"] = True
            self._reset_history = False

        image_dict = {CAMERA_KEY: video.astype(np.uint8, copy=False)}
        state_dict = {"states": state43.astype(np.float32, copy=False)}
        action, _ = self._client.query_action(
            image_dict, self._prompt, state_dict, {}, history=history,
            dataset="dreamzero_g1_real",
        )
        # Apply FSQ quantization to the body-token half of the action.
        hand = action[:, -14:]
        token = fsq_quantize(action[:, :-14])
        return np.concatenate([token, hand], axis=-1).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main client
# ──────────────────────────────────────────────────────────────────────────────


class DreamZeroG1Client:
    def __init__(self, policy_host, policy_port, prompt,
                 zmq_host, zmq_port, zmq_topic,
                 camera_host, camera_port,
                 wbc_host, wbc_port, wbc_topic,
                 video_output_dir: Optional[str] = None,
                 run_id: str = "dreamzero_g1_real"):
        self._camera = RSCamera(host=camera_host, port=camera_port)
        self._state_reader = WBCStateReader(host=wbc_host, port=wbc_port, topic=wbc_topic)
        self._token_publisher = TokenPublisher(host=zmq_host, port=zmq_port, topic=zmq_topic)
        self._policy = DreamZeroPolicyManager(host=policy_host, port=policy_port, prompt=prompt)
        self._encoder = EncoderClient(ENCODER_MODEL, mode=0)

        self._running = threading.Event()
        self._sequence_done_event = threading.Event()
        self._chunk_lock = threading.Lock()
        self._pending_chunk: Optional[np.ndarray] = None

        self.image_buffer = deque(maxlen=IMAGE_BUFFER_SIZE)
        self.image_buffer_lock = threading.Lock()

        # Rollout video — accumulates the exact frames sent to DreamZero on
        # every /act call, in temporal order. After the run, dumped as one
        # MP4 at the policy's effective video fps (30 / video_stride = 5 fps
        # for action_horizon=48). Mirrors the predicted-video MP4 the server
        # writes, so client/server videos stay frame-aligned.
        self._video_output_dir = video_output_dir
        self._run_id = run_id
        self._rollout_frames: list[np.ndarray] = []
        self._rollout_lock = threading.Lock()
        if video_output_dir:
            os.makedirs(video_output_dir, exist_ok=True)
            print(f"[DreamZeroG1Client] Rollout video → {video_output_dir}")

    def start(self) -> bool:
        print("[DreamZeroG1Client] Connecting to policy server...")
        self._policy.wait_and_configure()
        print(f"[DreamZeroG1Client] action_horizon={self._policy.action_horizon}, "
              f"video_frames_per_chunk={self._policy.video_frames_per_chunk}, "
              f"video_stride={self._policy.video_stride}")

        self._running.set()
        threading.Thread(target=self._inference_worker, daemon=True).start()
        threading.Thread(target=self._publish_loop, daemon=True).start()
        return True

    def _grab_frame(self) -> None:
        frame_bgr = self._camera.get_frame()
        if frame_bgr is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self.image_buffer_lock:
            self.image_buffer.append(frame_rgb)

    def _select_video_frames(self) -> np.ndarray:
        """Pick `video_frames_per_chunk` frames at stride `video_stride`,
        aligned to the END so the most recent observation is always last
        — same convention as `simple/baselines/dreamzero.py`."""
        with self.image_buffer_lock:
            buf = list(self.image_buffer)
        if len(buf) == 0:
            raise RuntimeError("[Inference] image buffer empty")
        if len(buf) == 1:
            return np.stack(buf, axis=0)
        n_want = self._policy.video_frames_per_chunk
        stride = self._policy.video_stride
        indices = [len(buf) - 1 - i * stride for i in range(n_want)]
        indices = [max(0, idx) for idx in reversed(indices)]
        return np.stack([buf[i] for i in indices], axis=0)

    def _build_state43(self, state: dict) -> np.ndarray:
        """qpos(29) + left_hand_q(7) + right_hand_q(7) → 43-D."""
        return np.concatenate([
            state["qpos"][:29],
            state["left_hand_q"][:7],
            state["right_hand_q"][:7],
        ], axis=0).astype(np.float32)

    def _inference_worker(self):
        # Wait for the first state message
        while self._running.is_set():
            if self._state_reader.get_state() is not None:
                break
            print("[Inference] waiting for first WBC state...")
            time.sleep(1.0)

        step = 0
        while self._running.is_set():
            self._sequence_done_event.wait()
            try:
                if step == 0:
                    print("=== Initial: grab single frame ===")
                    self._grab_frame()

                video = self._select_video_frames()
                state = self._state_reader.get_state()
                if state is None:
                    raise RuntimeError("WBC state went missing")
                state43 = self._build_state43(state)

                # Record the exact frames the policy is conditioning on so
                # the rollout MP4 matches DreamZero's input timeline:
                #   step 0: real_img_0
                #   step 1: real_img_1 ... real_img_8   (8 stride-6 samples)
                #   step 2: real_img_9 ... real_img_16
                #   ...
                if self._video_output_dir is not None:
                    with self._rollout_lock:
                        for f in video:
                            self._rollout_frames.append(np.ascontiguousarray(f))

                t0 = time.time()
                chunk = self._policy.get_action(video, state43)
                dt = time.time() - t0
                print(f"[Inference] chunk shape={chunk.shape} "
                      f"range=[{chunk.min():.3f},{chunk.max():.3f}] dt={dt:.2f}s")

                with self._chunk_lock:
                    self._pending_chunk = chunk
                # Drop frames we already conditioned on; keep the most recent
                # one so the next chunk's fallback (single-frame) still works.
                with self.image_buffer_lock:
                    if len(self.image_buffer) > 1:
                        last = self.image_buffer[-1]
                        self.image_buffer.clear()
                        self.image_buffer.append(last)
                step += 1
                self._sequence_done_event.clear()
            except Exception as e:
                print(f"[Inference] error: {e}")
                self._running.clear()
                return

    def _publish_loop(self):
        dt = 1.0 / FREQ_POLICY
        self._token_publisher.send_command(start=True, stop=False, planner=True)
        print("[PublishLoop] Requesting first inference...")
        self._sequence_done_event.set()
        while self._sequence_done_event.is_set() and self._running.is_set():
            time.sleep(0.05)
        if not self._running.is_set():
            return

        with self._chunk_lock:
            chunk = self._pending_chunk
        idx = 0
        last_action = None
        frozen_action = None
        print(f"[PublishLoop] First chunk: shape={chunk.shape}. Starting execution.")

        while self._running.is_set():
            t_start = time.perf_counter()

            if idx < len(chunk):
                action = chunk[idx]
                last_action = action.copy()
                idx += 1
                using_last_action = False
            else:
                # ── WAITING for next chunk ─────────────────────────────────
                # First tick after chunk exhausted: read robot state → run
                # ONNX encoder → freeze body token to current pose. Hand
                # joints stay at the last commanded value.
                if idx == len(chunk):
                    state = self._state_reader.get_state()
                    if state is not None and last_action is not None:
                        qpos      = _mujoco29_to_isaaclab29(state["qpos"])
                        base_quat = state["base_quat"]
                        joint_pos = np.tile(qpos,      (10, 1)).astype(np.float32)
                        joint_vel = np.zeros((10, 29),               dtype=np.float32)
                        body_quat = np.tile(base_quat, (10, 1)).astype(np.float32)
                        enc_token = self._encoder.encode(joint_pos, joint_vel, body_quat)
                        frozen_action = np.concatenate([enc_token, last_action[-14:]])
                        print(f"[PublishLoop] Chunk done ({len(chunk)} actions), "
                              f"encoder freeze token computed.")
                    elif last_action is not None:
                        frozen_action = last_action.copy()
                        print(f"[PublishLoop] Chunk done, no robot state — "
                              f"repeating last action.")
                    self._sequence_done_event.set()
                    idx += 1

                if not self._sequence_done_event.is_set():  # inference done
                    with self._chunk_lock:
                        chunk = self._pending_chunk
                    frozen_action = None
                    idx = 0
                    print(f"[PublishLoop] New chunk received: shape={chunk.shape}. "
                          f"Resuming execution.")
                    action = chunk[idx]
                    last_action = action.copy()
                    idx += 1
                    using_last_action = False
                else:
                    action = frozen_action
                    using_last_action = True

            self._token_publisher.publish_token(action)

            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Keep gathering camera frames for the next inference window.
            if not using_last_action:
                self._grab_frame()

    def _save_rollout_video(self) -> None:
        """Dump accumulated `/act` conditioning frames as one MP4.

        Effective fps = control_freq / video_stride = 30 / 6 = 5 fps for
        action_horizon=48. This matches the server's predicted-video MP4
        so the two can be played side-by-side.
        """
        if not self._video_output_dir:
            return
        with self._rollout_lock:
            frames = list(self._rollout_frames)
            self._rollout_frames = []
        if not frames:
            return
        try:
            from PIL import Image, ImageDraw
            annotated = []
            for gi, frame in enumerate(frames):
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                text = f"r{gi}"
                bbox = draw.textbbox((4, 4), text)
                draw.rectangle(bbox, fill=(0, 0, 0))
                draw.text((4, 4), text, fill=(255, 255, 255))
                annotated.append(np.array(img))

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fps = max(1, int(round(FREQ_POLICY / max(1, self._policy.video_stride))))
            path = os.path.join(
                self._video_output_dir,
                f"rollout-{self._run_id}-{ts}_n{len(annotated)}.mp4",
            )
            imageio.mimsave(path, annotated, fps=fps, codec="libx264")
            print(f"[DreamZeroG1Client] saved rollout {len(annotated)} frames @ {fps}fps → {path}")
        except Exception as e:
            print(f"[DreamZeroG1Client] failed to save rollout video: {e}")

    def stop(self):
        print("[DreamZeroG1Client] Stopping...")
        self._running.clear()
        self._token_publisher.send_command(start=False, stop=True, planner=False)
        time.sleep(0.5)
        self._save_rollout_video()
        self._camera.close()
        self._state_reader.close()
        self._token_publisher.stop()
        self._policy.reset()
        print("[DreamZeroG1Client] Stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DreamZero G1 real-robot client")
    parser.add_argument("--policy-host", type=str, default=DEFAULT_POLICY_HOST)
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
    parser.add_argument("--run-id", type=str, default="dreamzero_g1_real",
                        help="Run name used in the rollout MP4 filename.")
    parser.add_argument("--video-output-dir", type=str, default=None,
                        help="Where to dump the rollout MP4. Disabled if unset; "
                             "set to e.g. data/rollouts/<run-id>_real to enable.")
    args = parser.parse_args()

    client = DreamZeroG1Client(
        policy_host=args.policy_host, policy_port=args.policy_port, prompt=args.prompt,
        zmq_host=args.zmq_host, zmq_port=args.zmq_port, zmq_topic=args.zmq_topic,
        camera_host=args.camera_host, camera_port=args.camera_port,
        wbc_host=args.wbc_host, wbc_port=args.wbc_port, wbc_topic=args.wbc_topic,
        video_output_dir=args.video_output_dir,
        run_id=args.run_id,
    )
    try:
        if not client.start():
            print("Failed to start client")
            return
        print("[Main] Running. Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Caught Ctrl+C, stopping...")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
