#!/usr/bin/env python3
"""
g1_sonic_fastwam_client.py

FastWAM-specific variant of g1_sonic_client.py.

This keeps the original G1 SONIC client untouched while adding:
  - GR00T_ROOT override support
  - --dry-run
  - --max-chunks

Usage:
    GR00T_ROOT=$PWD python g1_sonic_fastwam_client.py \\
        --policy-host localhost --policy-port 5000 \\
        --prompt "Your task prompt here" \\
        --dry-run --max-chunks 2
"""

import argparse
import json
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import zmq
import msgpack

_GROOT_ROOT = os.environ.get("GR00T_ROOT")
if _GROOT_ROOT:
    _GROOT_ROOT = os.path.abspath(os.path.expanduser(_GROOT_ROOT))
else:
    _GROOT_ROOT = str(Path(__file__).resolve().parent)
if _GROOT_ROOT not in sys.path:
    sys.path.insert(0, _GROOT_ROOT)

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)
from ours.common.encoder_client import EncoderClient

# Joint order conversion: WBC publishes in Mujoco order, encoder expects IsaacLab order
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)

def _mujoco29_to_isaaclab29(qpos: np.ndarray) -> np.ndarray:
    return np.asarray(qpos, dtype=np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB_DOF].copy()

# Policy client imports
try:
    import eval_utils.policy_server as policy_server
    from eval_utils.policy_client import WebsocketClientPolicy
    POLICY_CLIENT_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Policy client not available. Make sure eval_utils is in the path.")
    POLICY_CLIENT_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Encoder model path
ENCODER_MODEL = os.path.join(_GROOT_ROOT, "gear_sonic_deploy/policy/release/model_encoder.onnx")

# RealSense camera configuration
REALSENSE_HOST = "192.168.123.164"
REALSENSE_PORT = 5558

# WBC state subscriber configuration
WBC_HOST = "localhost"
WBC_PORT = 5557
WBC_TOPIC = "g1_debug"

# ZMQ publisher configuration (to onnx policy)
DEFAULT_ZMQ_HOST = "*"
DEFAULT_ZMQ_PORT = 5556
DEFAULT_ZMQ_TOPIC = "pose"

# Policy server configuration
DEFAULT_POLICY_HOST = "localhost"
DEFAULT_POLICY_PORT = 5000

# Control frequencies
FREQ_POLICY = 30  # Hz - frequency to query policy server

VIDEO_FREQ = 30
CAMERA_KEY = "observation/head"


# FSQ configuration
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16
FSQ_LEVELS = 21 

# Action configuration
RELATIVE_OFFSETS = [-23 - 1, -16 - 1, -8 - 1, 0 - 1]
ACTION_HORIZON = 24

    # g1_sonic action layout: hand_joints(14) + token(64) = 78
ACTION_DIM = 78

# Image buffer
IMAGE_BUFFER_SIZE = 100

# TASK_PROMPT = "pick up the bottle and place it on the left"
TASK_PROMPT = "pick up the red box and pour water into the orange cup"


@dataclass
class FirstChunkSanityReport:
    ok: bool
    reason: str
    hand_delta_l2: float | None
    hand_delta_max: float | None
    token_clip_fraction: float
    token_range: tuple[float, float]


@dataclass
class HandStateHealthReport:
    ok: bool
    reason: str
    saturation_fraction: float | None
    startup_sentinel_l2: float | None
    hand_range: tuple[float, float] | None


# WBC fallback/default value observed in bad live traces before a real hand
# command/state has been supplied. It is published under *_measured fields in
# the current WBC debug stream, but it is far outside the FastWAM 0422 initial
# state distribution.
STARTUP_HAND_SENTINEL = np.array(
    [0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75, 0.0, 0.0, -1.75, 1.57, 1.75, 1.57, 1.75],
    dtype=np.float32,
)

# RealWorld/0422 dataset-derived first-frame hand fallback. Across the 141
# available 0422 episodes, the first states.hand_joints median/min/max are all
# this zero vector, so this is the least surprising policy-conditioning value
# when WBC is still publishing its default hand fallback.
DATASET_INITIAL_HAND_FALLBACK = np.zeros(14, dtype=np.float32)


def parse_hand_state_fallback(mode="dataset_initial", values=None):
    mode = str(mode or "none").strip().lower()
    if mode in ("none", "off", "disabled"):
        return None
    if mode in ("dataset_initial", "0422_initial"):
        return DATASET_INITIAL_HAND_FALLBACK.copy()
    if mode == "zeros":
        return np.zeros(14, dtype=np.float32)
    if mode == "custom":
        if values is None or str(values).strip() == "":
            raise ValueError("--hand-state-fallback-values is required when fallback mode is custom")
        parsed = np.fromstring(str(values), dtype=np.float32, sep=",")
        if parsed.shape != (14,):
            raise ValueError(
                f"--hand-state-fallback-values must contain 14 comma-separated floats, got {parsed.shape}"
            )
        if not np.isfinite(parsed).all():
            raise ValueError("--hand-state-fallback-values contains non-finite values")
        return parsed.astype(np.float32, copy=True)
    raise ValueError(f"Unknown invalid hand state fallback mode: {mode}")


def state_with_hand_fallback(state, fallback):
    if state is None or fallback is None:
        return state
    hand = np.asarray(fallback, dtype=np.float32).reshape(-1)
    if hand.shape != (14,):
        raise ValueError(f"hand fallback must have shape (14,), got {hand.shape}")
    patched = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            patched[key] = value.copy()
        else:
            patched[key] = value
    patched["left_hand_q"] = hand[:7].copy()
    patched["right_hand_q"] = hand[7:].copy()
    return patched


class RtcPrefetchState:
    """Small pure state machine for one-prefetch-per-active-chunk RTC scheduling."""

    def __init__(self, prefetch_step=15, timeout_sec=1.0):
        self.prefetch_step = max(1, int(prefetch_step))
        self.timeout_sec = max(0.0, float(timeout_sec))
        self.active_chunk_id = 0
        self.prefetch_started_for_chunk = False
        self.inflight = False
        self.inflight_chunk_id = None
        self.ready_chunk_id = None
        self.start_time = 0.0

    def adopt_chunk(self):
        self.active_chunk_id += 1
        self.prefetch_started_for_chunk = False
        return self.active_chunk_id

    def should_start(self, executed_steps, chunk_len):
        threshold = min(self.prefetch_step, int(chunk_len))
        return (
            int(executed_steps) >= threshold
            and not self.prefetch_started_for_chunk
            and not self.inflight
            and self.ready_chunk_id is None
        )

    def mark_started(self, now):
        self.prefetch_started_for_chunk = True
        self.inflight = True
        self.inflight_chunk_id = self.active_chunk_id
        self.start_time = float(now)
        return self.inflight_chunk_id

    def mark_ready(self, chunk_id):
        if not self.inflight or chunk_id != self.inflight_chunk_id:
            return False
        self.inflight = False
        self.ready_chunk_id = chunk_id
        return True

    def consume_ready(self):
        if self.ready_chunk_id != self.active_chunk_id:
            return False
        self.ready_chunk_id = None
        return True

    def clear_ready(self):
        self.ready_chunk_id = None

    def timeout_expired(self, now):
        if not self.inflight or self.timeout_sec <= 0:
            return False
        if float(now) - self.start_time <= self.timeout_sec:
            return False
        self.inflight = False
        self.inflight_chunk_id = None
        return True

def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)

    quantized = np.round(clipped / fsq_step) * fsq_step

    quantized = np.clip(quantized, fsq_min, fsq_max)

    return quantized


def token_clip_fraction(token):
    token = np.asarray(token, dtype=np.float32)
    if token.size == 0:
        return 0.0
    return float(np.mean((token <= FSQ_MIN) | (token >= FSQ_MAX)))


def first_chunk_sanity_report(
    chunk,
    state,
    hand_l2_threshold=2.0,
    hand_max_threshold=1.0,
    token_clip_threshold=0.95,
):
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] <= 0 or chunk.shape[1] != ACTION_DIM:
        return FirstChunkSanityReport(False, f"invalid_chunk_shape={chunk.shape}", None, None, 1.0, (0.0, 0.0))
    token = chunk[0, 14:]
    clip_fraction = token_clip_fraction(token)
    token_range = (float(np.min(token)), float(np.max(token)))
    if state is None:
        return FirstChunkSanityReport(False, "missing_wbc_state", None, None, clip_fraction, token_range)
    current_hand = np.concatenate([state["left_hand_q"], state["right_hand_q"]], axis=-1).astype(np.float32)
    hand_delta = chunk[0, :14] - current_hand
    hand_l2 = float(np.linalg.norm(hand_delta))
    hand_max = float(np.max(np.abs(hand_delta)))
    reasons = []
    if hand_l2 > float(hand_l2_threshold):
        reasons.append(f"hand_l2={hand_l2:.4f}>{float(hand_l2_threshold):.4f}")
    if hand_max > float(hand_max_threshold):
        reasons.append(f"hand_max={hand_max:.4f}>{float(hand_max_threshold):.4f}")
    if clip_fraction > float(token_clip_threshold):
        reasons.append(f"token_clip_fraction={clip_fraction:.4f}>{float(token_clip_threshold):.4f}")
    return FirstChunkSanityReport(
        ok=not reasons,
        reason="ok" if not reasons else ";".join(reasons),
        hand_delta_l2=hand_l2,
        hand_delta_max=hand_max,
        token_clip_fraction=clip_fraction,
        token_range=token_range,
    )


def hand_state_health_report(
    state,
    saturation_abs_threshold=1.55,
    saturation_fraction_threshold=0.6,
    startup_sentinel_l2_threshold=0.25,
):
    if state is None:
        return HandStateHealthReport(False, "missing_wbc_state", None, None, None)
    try:
        hand = np.concatenate([state["left_hand_q"], state["right_hand_q"]], axis=-1).astype(np.float32)
    except Exception as exc:
        return HandStateHealthReport(False, f"invalid_hand_state:{exc}", None, None, None)
    if hand.shape != (14,):
        return HandStateHealthReport(False, f"invalid_hand_shape={hand.shape}", None, None, None)
    if not np.isfinite(hand).all():
        return HandStateHealthReport(False, "nonfinite_hand_state", None, None, None)

    saturation_fraction = float(np.mean(np.abs(hand) >= float(saturation_abs_threshold)))
    startup_sentinel_l2 = float(np.linalg.norm(hand - STARTUP_HAND_SENTINEL))
    hand_range = (float(np.min(hand)), float(np.max(hand)))
    reasons = []
    if startup_sentinel_l2 <= float(startup_sentinel_l2_threshold):
        reasons.append(f"startup_sentinel_l2={startup_sentinel_l2:.4f}")
    if saturation_fraction > float(saturation_fraction_threshold):
        reasons.append(
            f"hand_saturation_fraction={saturation_fraction:.4f}>{float(saturation_fraction_threshold):.4f}"
        )
    return HandStateHealthReport(
        ok=not reasons,
        reason="ok" if not reasons else ";".join(reasons),
        saturation_fraction=saturation_fraction,
        startup_sentinel_l2=startup_sentinel_l2,
        hand_range=hand_range,
    )



class RSCamera:
    """RealSense camera client - matches client_AR.py implementation."""

    def __init__(self, host=REALSENSE_HOST, port=REALSENSE_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"[RSCamera] Connected to {host}:{port}")

    def get_frame(self):
        """Get RGB frame from RealSense server."""
        self.socket.send(b"get_frame")
        rgb_bytes, _, _ = self.socket.recv_multipart()
        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        return rgb_image

    def close(self):
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()


# ──────────────────────────────────────────────────────────────────────────────
# WBC State Subscriber
# ──────────────────────────────────────────────────────────────────────────────


class WBCStateReader:
    """
    Background-thread subscriber to the deploy's ZMQ state publisher.
    Provides access to robot state (qpos, quat, hand_joints, etc.)
    """

    def __init__(self, host=WBC_HOST, port=WBC_PORT, topic=WBC_TOPIC):
        self._topic_bytes = topic.encode()
        self._topic_len = len(self._topic_bytes)

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, self._topic_bytes)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt(zmq.RCVHWM, 1)  # always get latest
        self._sock.connect(f"tcp://{host}:{port}")

        self._lock = threading.Lock()
        self._latest = None
        self._ref_quat = None
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

    def reset_ref(self):
        """Reset the quaternion reference frame."""
        with self._lock:
            self._ref_quat = None

    def get_state(self):
        """
        Returns state dict or None if no data yet.
        State includes: qpos(29), left_hand_q(7), right_hand_q(7), base_quat(4, wxyz)
        """
        with self._lock:
            if self._latest is None:
                return None

            qpos = self._latest["qpos"].copy()
            left_hand_q = self._latest["left_hand_q"].copy()
            right_hand_q = self._latest["right_hand_q"].copy()
            base_quat = self._latest["base_quat"].copy()

            return {
                "qpos": qpos,
                "left_hand_q": left_hand_q,
                "right_hand_q": right_hand_q,
                "base_quat": base_quat,
            }

    def get_base_quat(self):
        """Returns current robot base quaternion (w,x,y,z) or None if no data yet."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest["base_quat"].copy()

    def close(self):
        self._stop.set()
        if self._sock:
            self._sock.close()
        self._ctx.term()



class TokenPublisher:
    """ZMQ publisher for token-only streaming (Protocol v4)."""

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
        """
        Publish action-only message (Protocol v4).

        Args:
            action: np.ndarray of shape (14+64,) - latent vector from encoder
            body_quat_w: optional np.ndarray of shape (4,) or (1,4), (w,x,y,z).
                         If provided, included in message so WBC holds current heading.
        """
        action = action.reshape(1, -1)
        pose_data = {
            "token_state": action[:, 14:],       # (1, 64)
            "left_hand_joints": action[:, :7],    # (1, 7)
            "right_hand_joints": action[:, 7:14], # (1, 7)
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
# Policy Client Manager
# ──────────────────────────────────────────────────────────────────────────────

class PolicyClientManager:
    """Manages communication with autonomous policy server."""

    def __init__(self, host, port, prompt, action_only=False):
        if not POLICY_CLIENT_AVAILABLE:
            raise RuntimeError("Policy client not available!")

        self._host = host
        self._port = port
        self._prompt = prompt
        self._action_only = action_only
        self._client = None
        self._session_id = None
        self._infer_lock = threading.Lock()

    def connect(self):
        """Connect to policy server and initialize session."""
        print(f"[PolicyClient] Connecting to {self._host}:{self._port}...")
        self._client = WebsocketClientPolicy(host=self._host, port=self._port)

        metadata = self._client.get_server_metadata()
        print(f"[PolicyClient] Server metadata: {metadata}")

        try:
            server_config = policy_server.PolicyServerConfig(**metadata)
            print(f"[PolicyClient] Server config: {server_config}")
        except Exception as e:
            print(f"[PolicyClient] Error parsing metadata: {e}")
            raise

        # Generate unique session ID
        import uuid
        self._session_id = str(uuid.uuid4())
        print(f"[PolicyClient] Session ID: {self._session_id}")
        if self._action_only:
            print(f"[PolicyClient] Action-only mode: ON")
        print(f"[PolicyClient] Connected successfully!")

    def get_action(self, images, state, rtc_metadata=None, recording_frames=None):
        """
        Send observation to policy server and get action.

        Args:
            images: RGB images (T, H, W, 3) or (H, W, 3)
            state: dict with robot state

        Returns:
            action dict with 'token' and 'hand_states'
        """
        if self._client is None:
            raise RuntimeError("Not connected to policy server")

        # Build observation dict (similar to client_AR.py)
        obs = {
            CAMERA_KEY: images,
            "observation/hand_joints": np.concatenate([state["left_hand_q"], state["right_hand_q"]], axis=-1),
            "observation/qpos": state["qpos"],
            "prompt": self._prompt,
            "session_id": self._session_id,
        }
        if self._action_only:
            obs["action_only_inference"] = True
            obs["action_attend_to_noisy_video"] = False
        if rtc_metadata:
            obs.update(rtc_metadata)
        if recording_frames is not None:
            obs["recording/head"] = recording_frames

        # Get action from policy server
        try:
            start = time.time()
            with self._infer_lock:
                action_from_policy = self._client.infer(obs)
            end = time.time()
            print(f"[PolicyClient] Inference time: {end - start:.4f} seconds")

            hand_joints = action_from_policy[:, :14] # (N, 14)
            token_ori = action_from_policy[:, 14:] # (N, 64)

            
            # 量化token到FSQ级别
            token_qtz = fsq_quantize(token_ori) # (N, 64)
            print(f"[PolicyClient] Token quantized: shape={token_ori.shape}, "
                    f"original_range=[{token_ori.min():.4f},{token_ori.max():.4f}], "
                    f"quantized_range=[{token_qtz.min():.4f},{token_qtz.max():.4f}]")

            action = np.concatenate([hand_joints, token_qtz], axis=-1) # (N, 14+64)

            return action
        except Exception as e:
            print(f"[PolicyClient] Error getting action: {e}")
            return None

    def reset(self, reset_info=None):
        """Send reset signal to policy server."""
        if self._client:
            try:
                self._client.reset({} if reset_info is None else dict(reset_info))
                print("[PolicyClient] Reset signal sent successfully.")
            except Exception as e:
                print(f"[PolicyClient] Failed to send reset: {e}")

    def close(self):
        """Clean up resources."""
        if self._client:
            self.reset()
            self._client = None


# ──────────────────────────────────────────────────────────────────────────────
# Main Client
# ──────────────────────────────────────────────────────────────────────────────

class TokenPolicyClient:
    """Main client that orchestrates camera, state, policy, and ZMQ publishing."""

    def __init__(self, policy_host, policy_port, prompt,
                 zmq_host, zmq_port, zmq_topic,
                 camera_host, camera_port,
                 wbc_host, wbc_port, wbc_topic,
                 action_only=False,
                 dry_run=False,
                 max_chunks=0,
                 enable_rtc_prefetch=False,
                 rtc_prefetch_step=15,
                 rtc_prefetch_timeout_sec=1.0,
                 first_chunk_sanity="warn",
                 first_chunk_hand_l2_threshold=2.0,
                 first_chunk_hand_max_threshold=1.0,
                 first_chunk_token_clip_threshold=0.95,
                 start_after_first_chunk=True,
                 require_valid_hand_state=True,
                 valid_hand_state_timeout_sec=5.0,
                 hand_state_saturation_fraction_threshold=0.6,
                 invalid_hand_state_fallback="dataset_initial",
                 hand_state_fallback_values=None,
                 send_recording_frames=True,
                 max_recording_frames=120):
        self._dry_run = bool(dry_run)
        self._max_chunks = max(0, int(max_chunks))
        self._enable_rtc_prefetch = bool(enable_rtc_prefetch)
        self._rtc_prefetch_step = max(1, int(rtc_prefetch_step))
        self._rtc_prefetch_timeout_sec = max(0.0, float(rtc_prefetch_timeout_sec))
        self._rtc_state = RtcPrefetchState(
            prefetch_step=self._rtc_prefetch_step,
            timeout_sec=self._rtc_prefetch_timeout_sec,
        )
        self._first_chunk_sanity = str(first_chunk_sanity)
        self._first_chunk_hand_l2_threshold = float(first_chunk_hand_l2_threshold)
        self._first_chunk_hand_max_threshold = float(first_chunk_hand_max_threshold)
        self._first_chunk_token_clip_threshold = float(first_chunk_token_clip_threshold)
        self._start_after_first_chunk = bool(start_after_first_chunk)
        self._require_valid_hand_state = bool(require_valid_hand_state)
        self._valid_hand_state_timeout_sec = max(0.0, float(valid_hand_state_timeout_sec))
        self._hand_state_saturation_fraction_threshold = float(hand_state_saturation_fraction_threshold)
        self._invalid_hand_state_fallback_mode = str(invalid_hand_state_fallback)
        self._invalid_hand_state_fallback = parse_hand_state_fallback(
            self._invalid_hand_state_fallback_mode,
            hand_state_fallback_values,
        )
        self._invalid_hand_state_fallback_count = 0
        self._last_policy_state_for_sanity = None
        self._send_recording_frames = bool(send_recording_frames)
        self._max_recording_frames = max(1, int(max_recording_frames))

        # Initialize components
        self._camera = RSCamera(host=camera_host, port=camera_port)
        self._state_reader = WBCStateReader(host=wbc_host, port=wbc_port, topic=wbc_topic)
        self._token_publisher = None if self._dry_run else TokenPublisher(host=zmq_host, port=zmq_port, topic=zmq_topic)
        self._policy_client = PolicyClientManager(host=policy_host, port=policy_port, prompt=prompt, action_only=action_only)
        self._encoder = None if self._dry_run else EncoderClient(ENCODER_MODEL, mode=0)

        # Threading components
        self._running = threading.Event()
        self._sequence_done_event = threading.Event()

        self._pending_chunk: np.ndarray | None = None  # latest chunk from inference worker
        self._pending_chunk_seq = 0
        self._pending_chunk_parent_id: int | None = None
        self._pending_chunk_source = "regular"
        self._regular_request_inflight = False
        self._regular_request_parent_id: int | None = None
        self._regular_request_label = "regular"
        self._chunk_lock = threading.Lock()
        self._prefetch_lock = threading.Lock()
        self._prefetch_inflight = False
        self._prefetch_chunk: np.ndarray | None = None
        self._prefetch_elapsed_ticks: int | None = None
        self._prefetch_chunk_id: int | None = None
        self._prefetch_start_time = 0.0
        self._prefetch_start_step: int | None = None
        self._prefetch_freeze_ticks = 0
        self._last_rtc_elapsed_ticks: int | None = None

        self.image_buffer = deque(maxlen=IMAGE_BUFFER_SIZE)
        self._recording_frames = deque(maxlen=self._max_recording_frames)
        self.image_buffer_lock = threading.Lock()

    def start(self):
        """Start the client."""
        print("[TokenPolicyClient] Starting...")

        # Connect to policy server
        try:
            self._policy_client.connect()
        except Exception as e:
            print(f"[TokenPolicyClient] Failed to connect to policy server: {e}")
            return False

        self._running.set()

        # Background Autonomous Policy inference thread
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()

        # Main 30Hz publish/execute thread, or a no-publish dry-run scheduler.
        loop_target = self._dry_run_loop if self._dry_run else self._publish_loop
        self._publish_thread = threading.Thread(target=loop_target, daemon=True)
        self._publish_thread.start()

        print("[TokenPolicyClient] Started successfully!")
        return True

    def _get_image(self,):
        frame = self._camera.get_frame()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self.image_buffer_lock:
            self.image_buffer.append(frame_rgb)
            if self._send_recording_frames:
                self._recording_frames.append(frame_rgb.copy())

    def _request_next_chunk(self, label="regular"):
        with self._chunk_lock:
            if self._regular_request_inflight:
                return False
            self._regular_request_inflight = True
            self._regular_request_parent_id = self._rtc_state.active_chunk_id
            self._regular_request_label = str(label)
            self._sequence_done_event.set()
        print(
            f"[Inference] Requested {label} policy chunk "
            f"parent_chunk_id={self._regular_request_parent_id}."
        )
        return True

    def _latest_image_from_buffer(self):
        with self.image_buffer_lock:
            if len(self.image_buffer) > 0:
                return self.image_buffer[-1].copy()
        self._get_image()
        with self.image_buffer_lock:
            return self.image_buffer[-1].copy()

    def _get_policy_state(self):
        deadline = None
        if self._valid_hand_state_timeout_sec > 0:
            deadline = time.perf_counter() + self._valid_hand_state_timeout_sec
        last_log = 0.0
        while self._running.is_set():
            state = self._state_reader.get_state()
            if not self._require_valid_hand_state:
                self._last_policy_state_for_sanity = state
                return state
            report = hand_state_health_report(
                state,
                saturation_fraction_threshold=self._hand_state_saturation_fraction_threshold,
            )
            if report.ok:
                self._last_policy_state_for_sanity = state
                return state
            if state is not None and self._invalid_hand_state_fallback is not None:
                patched = state_with_hand_fallback(state, self._invalid_hand_state_fallback)
                patched_report = hand_state_health_report(
                    patched,
                    saturation_fraction_threshold=self._hand_state_saturation_fraction_threshold,
                )
                if patched_report.ok:
                    self._invalid_hand_state_fallback_count += 1
                    self._last_policy_state_for_sanity = patched
                    print(
                        "[WBCState] Replacing invalid WBC hand state for policy request "
                        f"with {self._invalid_hand_state_fallback_mode} fallback. "
                        f"original_reason={report.reason} original_range={report.hand_range} "
                        f"fallback_count={self._invalid_hand_state_fallback_count}"
                    )
                    return patched
                print(
                    "[WBCState] Configured hand fallback is still invalid; refusing it. "
                    f"fallback_mode={self._invalid_hand_state_fallback_mode} "
                    f"fallback_reason={patched_report.reason}"
                )
            now = time.perf_counter()
            if deadline is not None and now >= deadline:
                print(
                    "[WBCState] Invalid hand state timeout; refusing policy request. "
                    f"reason={report.reason} saturation_fraction={report.saturation_fraction} "
                    f"startup_sentinel_l2={report.startup_sentinel_l2} hand_range={report.hand_range}"
                )
                return None
            if now - last_log >= 1.0:
                print(
                    "[WBCState] Waiting for valid measured hand state before policy request. "
                    f"reason={report.reason} saturation_fraction={report.saturation_fraction} "
                    f"startup_sentinel_l2={report.startup_sentinel_l2} hand_range={report.hand_range}"
                )
                last_log = now
            time.sleep(0.05)
        return None

    def _pop_recording_frames(self):
        if not self._send_recording_frames:
            return None
        with self.image_buffer_lock:
            if not self._recording_frames:
                return None
            frames = np.stack([frame.copy() for frame in self._recording_frames], axis=0)
            self._recording_frames.clear()
        return frames

    def _get_policy_chunk(self, frame_indices, rtc_metadata=None, latest_only=False):
        """
        Capture current observation and query policy server.
        Returns np.ndarray of shape (N, action_dim), or None on failure.
        """
        if latest_only:
            selected = self._latest_image_from_buffer()
        else:
            with self.image_buffer_lock:
                selected = [self.image_buffer[i].copy() for i in frame_indices]  # (T, H, W, 3)
            selected = np.stack(selected, axis=0) # (T, H, W, 3) or (1, H, W, 3)
            if len(frame_indices) == 1:
                selected = selected[0]  # (H, W, 3)

        state = self._get_policy_state()
        if state is None:
            print("[Inference] No valid WBC state available for policy request.")
            return None
        self._last_policy_state_for_sanity = state

        recording_frames = self._pop_recording_frames()
        if recording_frames is not None:
            print(f"[Recording] Sending {len(recording_frames)} continuous frames with policy request.")
        action = self._policy_client.get_action(
            selected,
            state,
            rtc_metadata=rtc_metadata,
            recording_frames=recording_frames,
        )
        return action

    def _adopt_chunk(self, chunk, adopt_idx=0, source="regular", chunk_seq=None):
        active_id = self._rtc_state.adopt_chunk()
        idx = int(np.clip(adopt_idx, 0, max(0, len(chunk) - 1)))
        if chunk_seq is not None:
            with self._chunk_lock:
                self._pending_chunk_seq = max(self._pending_chunk_seq, int(chunk_seq))
        print(
            f"[Chunk] Adopted chunk_id={active_id} source={source} "
            f"adopt_idx={idx} shape={getattr(chunk, 'shape', None)}"
        )
        if idx >= self._rtc_prefetch_step:
            self._maybe_start_rtc_prefetch(idx, len(chunk))
        return chunk, idx

    def _maybe_start_rtc_prefetch(self, executed_steps, chunk_len):
        if not self._enable_rtc_prefetch:
            return False
        if not self._rtc_state.should_start(executed_steps, chunk_len):
            return False
        exec_steps = int(np.clip(executed_steps, 1, chunk_len))
        return self._start_rtc_prefetch(exec_steps)

    def _expire_rtc_prefetch_if_needed(self):
        with self._prefetch_lock:
            expired = self._rtc_state.timeout_expired(time.perf_counter())
            if not expired:
                return False
            chunk_id = self._prefetch_chunk_id
            self._prefetch_inflight = False
            self._prefetch_chunk = None
            self._prefetch_elapsed_ticks = None
            self._prefetch_chunk_id = None
            self._prefetch_start_step = None
            self._prefetch_freeze_ticks = 0
        print(f"[RTC] Prefetch timeout chunk_id={chunk_id}; allowing fallback.")
        return True

    def _start_rtc_prefetch(self, exec_steps):
        if not self._enable_rtc_prefetch:
            return False
        exec_steps = int(exec_steps)
        if exec_steps <= 0:
            return False

        with self._prefetch_lock:
            if self._prefetch_inflight or self._prefetch_chunk is not None:
                return False
            chunk_id = self._rtc_state.mark_started(time.perf_counter())
            self._prefetch_inflight = True
            self._prefetch_chunk = None
            self._prefetch_elapsed_ticks = None
            self._prefetch_chunk_id = chunk_id
            self._prefetch_start_time = self._rtc_state.start_time
            self._prefetch_start_step = exec_steps
            self._prefetch_freeze_ticks = 0

        rtc_metadata = {
            "rtc_prefetch": True,
            "rtc_exec_steps": exec_steps,
            "rtc_request_step": exec_steps,
        }
        if self._last_rtc_elapsed_ticks is not None:
            rtc_metadata["rtc_client_elapsed_ticks"] = int(self._last_rtc_elapsed_ticks)

        print(
            f"[RTC] Prefetch start chunk_id={chunk_id} step={exec_steps} "
            f"prev_elapsed_ticks={rtc_metadata.get('rtc_client_elapsed_ticks')}"
        )

        def _worker():
            try:
                t0 = time.perf_counter()
                chunk = self._get_policy_chunk([-1], rtc_metadata=rtc_metadata, latest_only=True)
                elapsed_s = time.perf_counter() - t0
                elapsed_ticks = max(0, int(np.ceil(elapsed_s * FREQ_POLICY)))
                if chunk is None:
                    raise RuntimeError("RTC prefetch returned no chunk")
                if not isinstance(chunk, np.ndarray) or chunk.ndim != 2 or chunk.shape[0] != ACTION_HORIZON:
                    raise RuntimeError(f"RTC prefetch returned invalid chunk shape: {getattr(chunk, 'shape', None)}")
                with self._prefetch_lock:
                    if not self._rtc_state.mark_ready(chunk_id):
                        print(f"[RTC] Dropping stale prefetch result chunk_id={chunk_id}")
                        return
                    self._prefetch_chunk = chunk
                    self._prefetch_elapsed_ticks = elapsed_ticks
                    self._prefetch_inflight = False
                print(
                    f"[RTC] Prefetch ready chunk_id={chunk_id} elapsed_ticks={elapsed_ticks} "
                    f"shape={chunk.shape}"
                )
            except Exception as e:
                with self._prefetch_lock:
                    self._rtc_state.inflight = False
                    self._rtc_state.inflight_chunk_id = None
                    self._prefetch_inflight = False
                    self._prefetch_chunk = None
                    self._prefetch_elapsed_ticks = None
                    self._prefetch_chunk_id = None
                print(f"[RTC] Prefetch failed: {e}")

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return True

    def _has_rtc_prefetch_inflight(self):
        self._expire_rtc_prefetch_if_needed()
        with self._prefetch_lock:
            return self._prefetch_inflight

    def _consume_rtc_prefetch(self, current_idx, current_chunk_len):
        self._expire_rtc_prefetch_if_needed()
        with self._prefetch_lock:
            if self._prefetch_chunk is None:
                return None
            if not self._rtc_state.consume_ready():
                stale_id = self._prefetch_chunk_id
                self._prefetch_chunk = None
                self._prefetch_elapsed_ticks = None
                self._prefetch_chunk_id = None
                self._prefetch_start_step = None
                self._prefetch_freeze_ticks = 0
                print(f"[RTC] Dropping stale ready prefetch chunk_id={stale_id}")
                return None
            chunk = self._prefetch_chunk
            elapsed_ticks = 0 if self._prefetch_elapsed_ticks is None else int(self._prefetch_elapsed_ticks)
            freeze_ticks = int(self._prefetch_freeze_ticks)
            chunk_id = self._prefetch_chunk_id
            self._prefetch_chunk = None
            self._prefetch_elapsed_ticks = None
            self._prefetch_chunk_id = None
            self._prefetch_start_step = None
            self._prefetch_freeze_ticks = 0

        handoff_idx = int(np.clip(elapsed_ticks, 0, max(0, len(chunk) - 1)))
        ready_before_boundary = int(current_idx) < int(current_chunk_len)
        self._last_rtc_elapsed_ticks = elapsed_ticks
        print(
            f"[RTC] Handoff chunk_id={chunk_id} elapsed_ticks={elapsed_ticks} handoff_idx={handoff_idx} "
            f"ready_before_boundary={ready_before_boundary} freeze_ticks={freeze_ticks}"
        )
        return chunk, handoff_idx

    def _increment_rtc_freeze_ticks(self):
        self._expire_rtc_prefetch_if_needed()
        with self._prefetch_lock:
            if self._prefetch_inflight:
                self._prefetch_freeze_ticks += 1
                return self._prefetch_freeze_ticks
            return self._prefetch_freeze_ticks

    def _run_first_chunk_sanity(self, chunk):
        mode = self._first_chunk_sanity
        if mode == "off" or self._dry_run:
            return True
        state = self._last_policy_state_for_sanity
        if state is None:
            state = self._state_reader.get_state()
        report = first_chunk_sanity_report(
            chunk,
            state,
            hand_l2_threshold=self._first_chunk_hand_l2_threshold,
            hand_max_threshold=self._first_chunk_hand_max_threshold,
            token_clip_threshold=self._first_chunk_token_clip_threshold,
        )
        print(
            "[FirstChunkSanity] "
            f"ok={report.ok} reason={report.reason} "
            f"hand_l2={report.hand_delta_l2} hand_max={report.hand_delta_max} "
            f"token_clip_fraction={report.token_clip_fraction:.4f} token_range={report.token_range}"
        )
        if mode == "block" and not report.ok:
            print("[FirstChunkSanity] Blocking first policy chunk before WBC start/publish.")
            return False
        return True

    def _log_action(self, actions: np.ndarray, dt: float) -> None:

        """Pretty-print action shape, range, and timing."""
        assert isinstance(actions, np.ndarray), f"Expected numpy array, got {type(actions)}"
        assert actions.ndim == 2, f"Expected 2D array, got shape {actions.shape}"
        assert actions.shape[0] == ACTION_HORIZON, f"Expected {ACTION_HORIZON} actions, got {actions.shape[0]}"

        print(
            f"  Action shape: {actions.shape}, "
            f"range: [{actions.min():.4f}, {actions.max():.4f}], "
            f"time: {dt:.2f}s"
        )

    def _inference_worker(self):
        """
        Background thread: waits for an inference request, runs policy, posts result.
        Retries automatically on failure.
        """

        # wait for state ready
        while True:
            time.sleep(1)
            state = self._state_reader.get_state()
            if state is not None:
                break
            print("[VLA] robot state is empty, waiting for robot state to be updated...")

        
        step = 0 # step counter
        while self._running.is_set():
            self._sequence_done_event.wait()
            try:
                with self._chunk_lock:
                    request_parent_id = self._regular_request_parent_id
                    request_label = self._regular_request_label
                # Step 0: initial single frame
                if step == 0:
                    print(("=== Initial: frame [0] ==="))
                    frame_indices = [-1]
                    self._get_image() # append first image into image buffer

                else:
                    frame_indices = RELATIVE_OFFSETS # get previous 4 frames relative to current step
                    with self.image_buffer_lock:
                        len_image_buffer = len(self.image_buffer)
                    assert len_image_buffer >= ACTION_HORIZON, f"Expected at least {ACTION_HORIZON} frames in image buffer, got {len_image_buffer}"
                
                t0 = time.time()
                chunk = self._get_policy_chunk(frame_indices)
                dt = time.time() - t0
                if chunk is not None:
                    self._log_action(chunk, dt)
                    with self._chunk_lock:
                        self._pending_chunk = chunk
                        self._pending_chunk_seq += 1
                        self._pending_chunk_parent_id = request_parent_id
                        self._pending_chunk_source = request_label
                        self._regular_request_inflight = False
                    with self.image_buffer_lock:
                        self.image_buffer.clear()
                    step += 1
                    self._sequence_done_event.clear()
                else:
                    raise RuntimeError("[Inference] Failed to get chunk.")
            except RuntimeError as e:
                print(f"[Inference] {e}")
                with self._chunk_lock:
                    self._regular_request_inflight = False
                self._sequence_done_event.clear()
                self._running.clear()
                return

    def _dry_run_loop(self):
        """
        No-publish 30 Hz scheduler for bench testing.

        It requests policy chunks, captures 24 camera frames between chunks,
        and exits after --max-chunks when provided. It never sends WBC start,
        stop, or token messages.
        """
        dt = 1.0 / FREQ_POLICY
        chunks_seen = 0

        print("[DryRun] Requesting first policy inference; WBC publish is disabled.")
        self._request_next_chunk("initial dry-run")
        while self._sequence_done_event.is_set() and self._running.is_set():
            time.sleep(0.05)

        carried_chunk = None
        carried_idx = 0
        carried_source = "dry-run"
        while self._running.is_set():
            if carried_chunk is None:
                with self._chunk_lock:
                    chunk = None if self._pending_chunk is None else self._pending_chunk.copy()
                    chunk_seq = self._pending_chunk_seq
                if chunk is None:
                    print("[DryRun] No chunk available; stopping.")
                    self._running.clear()
                    return
                start_idx = 0
                source = "dry-run"
            else:
                chunk = carried_chunk
                start_idx = carried_idx
                source = carried_source
                with self._chunk_lock:
                    chunk_seq = self._pending_chunk_seq
                carried_chunk = None
                carried_idx = 0
                carried_source = "dry-run"
            chunk, start_idx = self._adopt_chunk(chunk, adopt_idx=start_idx, source=source)

            chunks_seen += 1
            print(
                f"[DryRun] Chunk {chunks_seen}: shape={chunk.shape}, "
                f"range=[{chunk.min():.4f},{chunk.max():.4f}]"
            )
            if self._max_chunks and chunks_seen >= self._max_chunks:
                print(f"[DryRun] Reached max_chunks={self._max_chunks}; stopping.")
                self._running.clear()
                return

            for step in range(start_idx, ACTION_HORIZON):
                if not self._running.is_set():
                    return
                t_start = time.perf_counter()
                self._get_image()
                exec_steps = step + 1
                self._maybe_start_rtc_prefetch(exec_steps, ACTION_HORIZON)
                prefetched = self._consume_rtc_prefetch(exec_steps, ACTION_HORIZON)
                if prefetched is not None:
                    carried_chunk, carried_idx = prefetched
                    carried_source = "rtc_prefetch"
                    break
                elapsed = time.perf_counter() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if carried_chunk is not None:
                continue

            if self._enable_rtc_prefetch:
                prefetched = self._consume_rtc_prefetch(ACTION_HORIZON, ACTION_HORIZON)
                if prefetched is not None:
                    carried_chunk, carried_idx = prefetched
                    carried_source = "rtc_prefetch"
                    continue

            if self._enable_rtc_prefetch and self._has_rtc_prefetch_inflight():
                while self._running.is_set():
                    prefetched = self._consume_rtc_prefetch(ACTION_HORIZON, ACTION_HORIZON)
                    if prefetched is not None:
                        carried_chunk, carried_idx = prefetched
                        carried_source = "rtc_prefetch"
                        break
                    freeze_ticks = self._increment_rtc_freeze_ticks()
                    print(f"[RTC] Dry-run waiting for prefetch freeze_ticks={freeze_ticks}")
                    time.sleep(dt)
                if carried_chunk is not None:
                    continue

            self._request_next_chunk("dry-run fallback")
            while self._running.is_set():
                with self._chunk_lock:
                    ready = self._pending_chunk_seq > chunk_seq
                if ready:
                    break
                time.sleep(0.05)

    def _publish_loop(self):
        """
        Main 30 Hz control loop.

        State machine:
          - EXECUTING: iterate through action chunk, one token per tick (1/30 s)
          - WAITING:   chunk exhausted; repeat last token until new chunk arrives
        """
        dt = 1.0 / FREQ_POLICY  # 1/30 s
        assert self._token_publisher is not None
        assert self._encoder is not None

        # FastWAM can take a moment to produce the first chunk. By default we
        # wait until the first chunk passes sanity checks before telling WBC to
        # start, so WBC does not move on stale/default policy input.
        if not self._start_after_first_chunk:
            self._token_publisher.send_command(start=True, stop=False, planner=True)
        print("[PublishLoop] Requesting first policy inference...")
        self._request_next_chunk("initial publish")
        while self._sequence_done_event.is_set() and self._running.is_set():
            time.sleep(0.05)
        if not self._running.is_set():
            return
        with self._chunk_lock:
            chunk = self._pending_chunk
            chunk_seq = self._pending_chunk_seq
        if chunk is None:
            print("[PublishLoop] No initial chunk available; stopping.")
            self._running.clear()
            return
        if not self._run_first_chunk_sanity(chunk):
            self._running.clear()
            self._token_publisher.send_command(start=False, stop=True, planner=False)
            return
        if self._start_after_first_chunk:
            self._token_publisher.send_command(start=True, stop=False, planner=True)

        chunk, idx = self._adopt_chunk(chunk, adopt_idx=0, source="initial", chunk_seq=chunk_seq)
        using_last_action = False
        frozen_action = None   # encoder-derived freeze token, set once when chunk exhausts
        print(f"[PublishLoop] First chunk: shape={chunk.shape}. Starting execution.")

        while self._running.is_set():
            t_start = time.perf_counter()
            prefetch_step_after_capture = None

            if self._enable_rtc_prefetch:
                prefetched = self._consume_rtc_prefetch(idx, len(chunk))
                if prefetched is not None:
                    chunk, idx = self._adopt_chunk(
                        prefetched[0],
                        adopt_idx=prefetched[1],
                        source="rtc_prefetch",
                    )
                    frozen_action = None

            if idx < len(chunk):
                # ── EXECUTING ──────────────────────────────────────────────
                action = chunk[idx]
                last_action = action.copy()
                idx += 1
                prefetch_step_after_capture = idx
                using_last_action = False
            else:
                # ── WAITING for next chunk ─────────────────────────────────

                # First tick after chunk exhausted: read robot state → run encoder → freeze token
                if idx == len(chunk):
                    state = self._state_reader.get_state()
                    if state is not None:
                        qpos      = _mujoco29_to_isaaclab29(state["qpos"])  # (29,) reordered to IsaacLab
                        base_quat = state["base_quat"]                      # (4,) wxyz

                        joint_pos = np.tile(qpos,      (10, 1)).astype(np.float32)  # (10, 29)
                        joint_vel = np.zeros((10, 29), dtype=np.float32)
                        body_quat = np.tile(base_quat, (10, 1)).astype(np.float32)  # (10, 4)

                        enc_token = self._encoder.encode(joint_pos, joint_vel, body_quat)  # (64,)
                        # keep hand joints from last action, replace body token
                        frozen_action = np.concatenate([last_action[:14], enc_token])
                        print(f"[PublishLoop] Chunk done ({len(chunk)} tokens), "
                              f"encoder freeze token computed.")
                    else:
                        frozen_action = last_action.copy()
                        print(f"[PublishLoop] Chunk done, no robot state — repeating last action.")
                    if not self._enable_rtc_prefetch:
                        self._request_next_chunk("boundary")
                    elif not self._has_rtc_prefetch_inflight():
                        self._request_next_chunk("rtc fallback")
                    idx += 1

                if self._enable_rtc_prefetch:
                    prefetched = self._consume_rtc_prefetch(idx, len(chunk))
                    if prefetched is not None:
                        chunk, idx = self._adopt_chunk(
                            prefetched[0],
                            adopt_idx=prefetched[1],
                            source="rtc_prefetch",
                        )
                        frozen_action = None
                        print(f"[PublishLoop] RTC chunk received: shape={chunk.shape}. Resuming execution.")
                    elif not self._has_rtc_prefetch_inflight():
                        self._request_next_chunk("rtc fallback")

                with self._chunk_lock:
                    regular_ready = self._pending_chunk_seq > chunk_seq
                    pending_chunk = self._pending_chunk
                    pending_seq = self._pending_chunk_seq
                    pending_parent_id = self._pending_chunk_parent_id
                    pending_source = self._pending_chunk_source

                if regular_ready and pending_parent_id != self._rtc_state.active_chunk_id:
                    chunk_seq = pending_seq
                    print(
                        f"[PublishLoop] Ignoring stale {pending_source} chunk "
                        f"parent_id={pending_parent_id} active_id={self._rtc_state.active_chunk_id}."
                    )
                    action = frozen_action
                    using_last_action = True
                elif regular_ready:
                    chunk = pending_chunk
                    chunk_seq = pending_seq
                    frozen_action = None
                    chunk, idx = self._adopt_chunk(
                        chunk,
                        adopt_idx=0,
                        source=pending_source,
                        chunk_seq=chunk_seq,
                    )
                    print(f"[PublishLoop] New chunk received: shape={chunk.shape}. "
                          f"Resuming execution.")

                    # don't wait, directly execute the first action
                    action = chunk[idx]
                    last_action = action.copy()
                    idx += 1
                    prefetch_step_after_capture = idx
                    using_last_action = False
                elif self._enable_rtc_prefetch and idx < len(chunk):
                    action = chunk[idx]
                    last_action = action.copy()
                    idx += 1
                    prefetch_step_after_capture = idx
                    using_last_action = False
                else:
                    action = frozen_action
                    using_last_action = True
                    if self._enable_rtc_prefetch and self._has_rtc_prefetch_inflight():
                        freeze_ticks = self._increment_rtc_freeze_ticks()
                        print(f"[RTC] Waiting for prefetch freeze_ticks={freeze_ticks}")

            self._token_publisher.publish_token(action)
            
            # Maintain 30 Hz with relative delay
            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if not using_last_action:
                self._get_image()
                if prefetch_step_after_capture is not None:
                    self._maybe_start_rtc_prefetch(prefetch_step_after_capture, len(chunk))

    def stop(self):
        """Stop the client."""
        print("[TokenPolicyClient] Stopping...")
        self._running.clear()

        # Send stop command
        if self._token_publisher is not None:
            self._token_publisher.send_command(start=False, stop=True, planner=False)

        # Wait for threads to finish
        time.sleep(0.5)

        # Clean up
        self._camera.close()
        self._state_reader.close()
        if self._token_publisher is not None:
            self._token_publisher.stop()
        reset_info = {}
        recording_frames = self._pop_recording_frames()
        if recording_frames is not None:
            reset_info["recording/head"] = recording_frames
            print(f"[Recording] Sending {len(recording_frames)} final frames with reset.")
            self._policy_client.reset(reset_info)
        self._policy_client.close()

        print("[TokenPolicyClient] Stopped!")

    def is_running(self):
        return self._running.is_set()


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Client that queries autonomous policy server and sends tokens to onnx policy"
    )
    parser.add_argument("--policy-host", type=str, default=DEFAULT_POLICY_HOST,
                       help="Policy server host (default: localhost)")
    parser.add_argument("--policy-port", type=int, default=DEFAULT_POLICY_PORT,
                       help="Policy server port (default: 5000)")
    parser.add_argument("--prompt", type=str, default=TASK_PROMPT,
                       help="Task prompt for policy server")

    parser.add_argument("--zmq-host", type=str, default=DEFAULT_ZMQ_HOST,
                       help="ZMQ publisher bind host (default: *)")
    parser.add_argument("--zmq-port", type=int, default=DEFAULT_ZMQ_PORT,
                       help="ZMQ publisher port (default: 5556)")
    parser.add_argument("--zmq-topic", type=str, default=DEFAULT_ZMQ_TOPIC,
                       help="ZMQ topic (default: pose)")

    parser.add_argument("--camera-host", type=str, default=REALSENSE_HOST,
                       help="RSCamera server host (default: 192.168.123.164)")
    parser.add_argument("--camera-port", type=int, default=REALSENSE_PORT,
                       help="RSCamera server port (default: 5558)")

    parser.add_argument("--wbc-host", type=str, default=WBC_HOST,
                       help="WBC state publisher host (default: localhost)")
    parser.add_argument("--wbc-port", type=int, default=WBC_PORT,
                       help="WBC state publisher port (default: 5557)")
    parser.add_argument("--wbc-topic", type=str, default=WBC_TOPIC,
                       help="WBC state topic (default: g1_debug)")
    parser.add_argument("--action-only", action="store_true",
                       help="Enable action-only inference (skip video denoising for faster speed)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Query policy chunks and camera/state at 30Hz without sending WBC start/stop/token messages")
    parser.add_argument("--max-chunks", type=int, default=0,
                       help="Dry-run: stop after this many policy chunks; 0 means run until Ctrl+C")
    parser.add_argument("--enable-rtc-prefetch", action="store_true",
                       help="Start the next FastWAM inference mid-chunk with RTC metadata")
    parser.add_argument("--rtc-prefetch-step", type=int, default=15,
                       help="Action step at which to start RTC prefetch")
    parser.add_argument("--rtc-prefetch-timeout-sec", type=float, default=1.0,
                       help="Drop an inflight RTC prefetch after this many seconds and allow fallback")
    parser.add_argument("--first-chunk-sanity", choices=("off", "warn", "block"), default="block",
                       help="Check first policy action against current WBC state before publishing")
    parser.add_argument("--first-chunk-hand-l2-threshold", type=float, default=2.0,
                       help="First-chunk sanity hand L2 threshold")
    parser.add_argument("--first-chunk-hand-max-threshold", type=float, default=1.0,
                       help="First-chunk sanity max per-joint hand threshold")
    parser.add_argument("--first-chunk-token-clip-threshold", type=float, default=0.95,
                       help="First-chunk sanity token clip-fraction threshold")
    parser.add_argument("--start-after-first-chunk", action=argparse.BooleanOptionalAction, default=True,
                       help="Send WBC start only after the first policy chunk passes sanity checks")
    parser.add_argument("--require-valid-hand-state", action=argparse.BooleanOptionalAction, default=True,
                       help="Validate WBC hand state before policy requests")
    parser.add_argument("--valid-hand-state-timeout-sec", type=float, default=5.0,
                       help="Seconds to wait for a valid hand state before failing the policy request when fallback is disabled; 0 waits indefinitely")
    parser.add_argument("--hand-state-saturation-fraction-threshold", type=float, default=0.6,
                       help="Reject hand states with more than this fraction of joints near +/- hand limits")
    parser.add_argument("--invalid-hand-state-fallback",
                       choices=("none", "dataset_initial", "zeros", "custom"),
                       default="dataset_initial",
                       help="Replace invalid WBC fallback/default hand state before policy requests; dataset_initial is the 0422 first-frame median hand_joints")
    parser.add_argument("--hand-state-fallback-values", default=None,
                       help="14 comma-separated floats used when --invalid-hand-state-fallback=custom")
    parser.add_argument("--send-recording-frames", action=argparse.BooleanOptionalAction, default=True,
                       help="Send continuous record-only camera frames to FastWAM server for complete input_video.mp4")
    parser.add_argument("--max-recording-frames", type=int, default=120,
                       help="Maximum continuous record-only frames to batch into one policy/reset request")

    args = parser.parse_args()

    if args.action_only:
        print("[Main] Action-only mode enabled: video denoising will be skipped")
    if args.dry_run:
        print("[Main] Dry-run enabled: no WBC start/stop/token messages will be sent")
    if args.enable_rtc_prefetch:
        print(
            f"[Main] RTC prefetch enabled: step={args.rtc_prefetch_step} "
            f"timeout={args.rtc_prefetch_timeout_sec}s"
        )
    if not args.dry_run:
        print(
            f"[Main] First chunk sanity={args.first_chunk_sanity}; "
            f"start_after_first_chunk={args.start_after_first_chunk}; "
            f"require_valid_hand_state={args.require_valid_hand_state}; "
            f"invalid_hand_state_fallback={args.invalid_hand_state_fallback}; "
            f"send_recording_frames={args.send_recording_frames}"
        )

    # Create and start client
    client = TokenPolicyClient(
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
        action_only=args.action_only,
        dry_run=args.dry_run,
        max_chunks=args.max_chunks,
        enable_rtc_prefetch=args.enable_rtc_prefetch,
        rtc_prefetch_step=args.rtc_prefetch_step,
        rtc_prefetch_timeout_sec=args.rtc_prefetch_timeout_sec,
        first_chunk_sanity=args.first_chunk_sanity,
        first_chunk_hand_l2_threshold=args.first_chunk_hand_l2_threshold,
        first_chunk_hand_max_threshold=args.first_chunk_hand_max_threshold,
        first_chunk_token_clip_threshold=args.first_chunk_token_clip_threshold,
        start_after_first_chunk=args.start_after_first_chunk,
        require_valid_hand_state=args.require_valid_hand_state,
        valid_hand_state_timeout_sec=args.valid_hand_state_timeout_sec,
        hand_state_saturation_fraction_threshold=args.hand_state_saturation_fraction_threshold,
        invalid_hand_state_fallback=args.invalid_hand_state_fallback,
        hand_state_fallback_values=args.hand_state_fallback_values,
        send_recording_frames=args.send_recording_frames,
        max_recording_frames=args.max_recording_frames,
    )

    try:
        if not client.start():
            print("Failed to start client")
            return

        print("[Main] Running. Press Ctrl+C to stop.")
        while client.is_running():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Main] Caught Ctrl+C, stopping...")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
