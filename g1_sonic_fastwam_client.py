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
from encoder_client import EncoderClient

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

def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)

    quantized = np.round(clipped / fsq_step) * fsq_step

    quantized = np.clip(quantized, fsq_min, fsq_max)

    return quantized



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

    def get_action(self, images, state, rtc_metadata=None):
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

    def reset(self):
        """Send reset signal to policy server."""
        if self._client:
            try:
                self._client.reset({})
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
                 rtc_prefetch_step=15):
        self._dry_run = bool(dry_run)
        self._max_chunks = max(0, int(max_chunks))
        self._enable_rtc_prefetch = bool(enable_rtc_prefetch)
        self._rtc_prefetch_step = max(1, int(rtc_prefetch_step))

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
        self._regular_request_inflight = False
        self._chunk_lock = threading.Lock()
        self._prefetch_lock = threading.Lock()
        self._prefetch_inflight = False
        self._prefetch_chunk: np.ndarray | None = None
        self._prefetch_elapsed_ticks: int | None = None
        self._prefetch_start_time = 0.0
        self._prefetch_start_step: int | None = None
        self._prefetch_freeze_ticks = 0
        self._last_rtc_elapsed_ticks: int | None = None

        self.image_buffer = deque(maxlen=IMAGE_BUFFER_SIZE)
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

    def _request_next_chunk(self, label="regular"):
        with self._chunk_lock:
            if self._regular_request_inflight:
                return False
            self._regular_request_inflight = True
            self._sequence_done_event.set()
        print(f"[Inference] Requested {label} policy chunk.")
        return True

    def _latest_image_from_buffer(self):
        with self.image_buffer_lock:
            if len(self.image_buffer) > 0:
                return self.image_buffer[-1].copy()
        self._get_image()
        with self.image_buffer_lock:
            return self.image_buffer[-1].copy()

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

        state = self._state_reader.get_state()
        assert state is not None

        action = self._policy_client.get_action(selected, state, rtc_metadata=rtc_metadata)
        return action

    def _start_rtc_prefetch(self, exec_steps):
        if not self._enable_rtc_prefetch:
            return False
        exec_steps = int(exec_steps)
        if exec_steps <= 0:
            return False

        with self._prefetch_lock:
            if self._prefetch_inflight or self._prefetch_chunk is not None:
                return False
            self._prefetch_inflight = True
            self._prefetch_chunk = None
            self._prefetch_elapsed_ticks = None
            self._prefetch_start_time = time.perf_counter()
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
            f"[RTC] Prefetch start step={exec_steps} "
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
                    self._prefetch_chunk = chunk
                    self._prefetch_elapsed_ticks = elapsed_ticks
                    self._prefetch_inflight = False
                print(
                    f"[RTC] Prefetch ready elapsed_ticks={elapsed_ticks} "
                    f"shape={chunk.shape}"
                )
            except Exception as e:
                with self._prefetch_lock:
                    self._prefetch_inflight = False
                    self._prefetch_chunk = None
                    self._prefetch_elapsed_ticks = None
                print(f"[RTC] Prefetch failed: {e}")

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return True

    def _has_rtc_prefetch_inflight(self):
        with self._prefetch_lock:
            return self._prefetch_inflight

    def _consume_rtc_prefetch(self, current_idx, current_chunk_len):
        with self._prefetch_lock:
            if self._prefetch_chunk is None:
                return None
            chunk = self._prefetch_chunk
            elapsed_ticks = 0 if self._prefetch_elapsed_ticks is None else int(self._prefetch_elapsed_ticks)
            freeze_ticks = int(self._prefetch_freeze_ticks)
            self._prefetch_chunk = None
            self._prefetch_elapsed_ticks = None
            self._prefetch_start_step = None
            self._prefetch_freeze_ticks = 0

        handoff_idx = int(np.clip(elapsed_ticks, 0, max(0, len(chunk) - 1)))
        ready_before_boundary = int(current_idx) < int(current_chunk_len)
        self._last_rtc_elapsed_ticks = elapsed_ticks
        print(
            f"[RTC] Handoff elapsed_ticks={elapsed_ticks} handoff_idx={handoff_idx} "
            f"ready_before_boundary={ready_before_boundary} freeze_ticks={freeze_ticks}"
        )
        return chunk, handoff_idx

    def _increment_rtc_freeze_ticks(self):
        with self._prefetch_lock:
            if self._prefetch_inflight:
                self._prefetch_freeze_ticks += 1
                return self._prefetch_freeze_ticks
            return self._prefetch_freeze_ticks

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

        while self._running.is_set():
            with self._chunk_lock:
                chunk = None if self._pending_chunk is None else self._pending_chunk.copy()
                chunk_seq = self._pending_chunk_seq
            if chunk is None:
                print("[DryRun] No chunk available; stopping.")
                self._running.clear()
                return

            chunks_seen += 1
            print(
                f"[DryRun] Chunk {chunks_seen}: shape={chunk.shape}, "
                f"range=[{chunk.min():.4f},{chunk.max():.4f}]"
            )
            if self._max_chunks and chunks_seen >= self._max_chunks:
                print(f"[DryRun] Reached max_chunks={self._max_chunks}; stopping.")
                self._running.clear()
                return

            next_chunk = None
            for step in range(ACTION_HORIZON):
                if not self._running.is_set():
                    return
                t_start = time.perf_counter()
                self._get_image()
                exec_steps = step + 1
                if self._enable_rtc_prefetch and exec_steps == min(self._rtc_prefetch_step, ACTION_HORIZON):
                    self._start_rtc_prefetch(exec_steps)
                prefetched = self._consume_rtc_prefetch(exec_steps, ACTION_HORIZON)
                if prefetched is not None:
                    next_chunk, _ = prefetched
                    break
                elapsed = time.perf_counter() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if next_chunk is not None:
                with self._chunk_lock:
                    self._pending_chunk = next_chunk
                    self._pending_chunk_seq += 1
                continue

            if self._enable_rtc_prefetch:
                prefetched = self._consume_rtc_prefetch(ACTION_HORIZON, ACTION_HORIZON)
                if prefetched is not None:
                    next_chunk, _ = prefetched
                    with self._chunk_lock:
                        self._pending_chunk = next_chunk
                        self._pending_chunk_seq += 1
                    continue

            if self._enable_rtc_prefetch and self._has_rtc_prefetch_inflight():
                while self._running.is_set():
                    prefetched = self._consume_rtc_prefetch(ACTION_HORIZON, ACTION_HORIZON)
                    if prefetched is not None:
                        next_chunk, _ = prefetched
                        with self._chunk_lock:
                            self._pending_chunk = next_chunk
                            self._pending_chunk_seq += 1
                        break
                    freeze_ticks = self._increment_rtc_freeze_ticks()
                    print(f"[RTC] Dry-run waiting for prefetch freeze_ticks={freeze_ticks}")
                    time.sleep(dt)
                if next_chunk is not None:
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

        # Send start command first, then request first chunk, then publish
        # immediately once it arrives (matches psi_sonic_client pattern).
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


        idx = 0
        using_last_action = False
        frozen_action = None   # encoder-derived freeze token, set once when chunk exhausts
        print(f"[PublishLoop] First chunk: shape={chunk.shape}. Starting execution.")

        while self._running.is_set():
            t_start = time.perf_counter()
            prefetch_step_after_capture = None

            if self._enable_rtc_prefetch:
                prefetched = self._consume_rtc_prefetch(idx, len(chunk))
                if prefetched is not None:
                    chunk, idx = prefetched
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
                        chunk, idx = prefetched
                        frozen_action = None
                        print(f"[PublishLoop] RTC chunk received: shape={chunk.shape}. Resuming execution.")
                    elif not self._has_rtc_prefetch_inflight():
                        self._request_next_chunk("rtc fallback")

                with self._chunk_lock:
                    regular_ready = self._pending_chunk_seq > chunk_seq

                if regular_ready:
                    with self._chunk_lock:
                        chunk = self._pending_chunk
                        chunk_seq = self._pending_chunk_seq
                    frozen_action = None
                    idx = 0
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
                if (
                    self._enable_rtc_prefetch
                    and prefetch_step_after_capture == min(self._rtc_prefetch_step, len(chunk))
                ):
                    self._start_rtc_prefetch(prefetch_step_after_capture)

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

    args = parser.parse_args()

    if args.action_only:
        print("[Main] Action-only mode enabled: video denoising will be skipped")
    if args.dry_run:
        print("[Main] Dry-run enabled: no WBC start/stop/token messages will be sent")
    if args.enable_rtc_prefetch:
        print(f"[Main] RTC prefetch enabled: step={args.rtc_prefetch_step}")

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
