#!/usr/bin/env python3
"""
g1_sonic_client.py
Usage:
    python g1_sonic_client.py --policy-host localhost --policy-port 8014 \\
                           --zmq-host "*" --zmq-port 5556 \\
                           --prompt "Your task prompt here"
"""

import argparse
import json
import os
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np
import zmq
import msgpack

_GROOT_ROOT = os.path.expanduser("~/hsc/GR00T-WholeBodyControl")
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

TASK_PROMPT = "pick up the bottle and place it on the left"

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

    def get_action(self, images, state):
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

        # Get action from policy server
        try:
            start = time.time()
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
                 action_only=False):
        # Initialize components
        self._camera = RSCamera(host=camera_host, port=camera_port)
        self._state_reader = WBCStateReader(host=wbc_host, port=wbc_port, topic=wbc_topic)
        self._token_publisher = TokenPublisher(host=zmq_host, port=zmq_port, topic=zmq_topic)
        self._policy_client = PolicyClientManager(host=policy_host, port=policy_port, prompt=prompt, action_only=action_only)
        self._encoder = EncoderClient(ENCODER_MODEL, mode=0)

        # Threading components
        self._running = threading.Event()
        self._sequence_done_event = threading.Event()

        self._pending_chunk: np.ndarray | None = None  # latest chunk from inference worker
        self._chunk_lock = threading.Lock()

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

        # Main 30Hz publish/execute thread
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()

        print("[TokenPolicyClient] Started successfully!")
        return True

    def _get_image(self,):
        frame = self._camera.get_frame()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self.image_buffer_lock:
            self.image_buffer.append(frame_rgb)


    def _get_policy_chunk(self, frame_indices):
        """
        Capture current observation and query policy server.
        Returns np.ndarray of shape (N, action_dim), or None on failure.
        """
        
        with self.image_buffer_lock:
            selected = [self.image_buffer[i].copy() for i in frame_indices]  # (T, H, W, 3)
        selected = np.stack(selected, axis=0) # (T, H, W, 3) or (1, H, W, 3)
        if len(frame_indices) == 1:
            selected = selected[0]  # (H, W, 3)

        state = self._state_reader.get_state()
        assert state is not None

        action = self._policy_client.get_action(selected, state)
        return action

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
                    assert len_image_buffer == ACTION_HORIZON, f"Expected {ACTION_HORIZON} frames in image buffer, got {len_image_buffer}"
                
                t0 = time.time()
                chunk = self._get_policy_chunk(frame_indices)
                dt = time.time() - t0
                self._log_action(chunk, dt)
                if chunk is not None:
                    with self._chunk_lock:
                        self._pending_chunk = chunk
                    self.image_buffer.clear()
                    step += 1
                    self._sequence_done_event.clear()
                else:
                    raise RuntimeError("[Inference] Failed to get chunk.")
            except RuntimeError as e:
                print(f"[Inference] {e}")
                self._running.clear()
                return

    def _publish_loop(self):
        """
        Main 30 Hz control loop.

        State machine:
          - EXECUTING: iterate through action chunk, one token per tick (1/30 s)
          - WAITING:   chunk exhausted; repeat last token until new chunk arrives
        """
        dt = 1.0 / FREQ_POLICY  # 1/30 s

        # Send start command first, then request first chunk, then publish
        # immediately once it arrives (matches psi_sonic_client pattern).
        self._token_publisher.send_command(start=True, stop=False, planner=True)
        print("[PublishLoop] Requesting first policy inference...")
        self._sequence_done_event.set()
        while self._sequence_done_event.is_set() and self._running.is_set():
            time.sleep(0.05)
        if not self._running.is_set():
            return
        with self._chunk_lock:
            chunk = self._pending_chunk


        idx = 0
        using_last_action = False
        frozen_action = None   # encoder-derived freeze token, set once when chunk exhausts
        print(f"[PublishLoop] First chunk: shape={chunk.shape}. Starting execution.")

        while self._running.is_set():
            t_start = time.perf_counter()

            if idx < len(chunk):
                # ── EXECUTING ──────────────────────────────────────────────
                action = chunk[idx]
                last_action = action.copy()
                idx += 1
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
                    self._sequence_done_event.set()
                    idx += 1

                if not self._sequence_done_event.is_set():  # means inference done
                    with self._chunk_lock:
                        chunk = self._pending_chunk
                    frozen_action = None
                    idx = 0
                    print(f"[PublishLoop] New chunk received: shape={chunk.shape}. "
                          f"Resuming execution.")

                    # don't wait, directly execute the first action
                    action = chunk[idx]
                    last_action = action.copy()
                    idx += 1
                    using_last_action = False
                else:
                    action = frozen_action
                    using_last_action = True

            self._token_publisher.publish_token(action)
            
            # Maintain 30 Hz with relative delay
            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if not using_last_action:
                self._get_image()

    def stop(self):
        """Stop the client."""
        print("[TokenPolicyClient] Stopping...")
        self._running.clear()

        # Send stop command
        self._token_publisher.send_command(start=False, stop=True, planner=False)

        # Wait for threads to finish
        time.sleep(0.5)

        # Clean up
        self._camera.close()
        self._state_reader.close()
        self._token_publisher.stop()
        self._policy_client.close()

        print("[TokenPolicyClient] Stopped!")


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

    args = parser.parse_args()

    if args.action_only:
        print("[Main] Action-only mode enabled: video denoising will be skipped")

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
    )

    try:
        if not client.start():
            print("Failed to start client")
            return

        print("[Main] Running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Main] Caught Ctrl+C, stopping...")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
