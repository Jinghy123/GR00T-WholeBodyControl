#!/usr/bin/env python3
"""
client_token.py
Usage:
    python client_token.py --policy-host localhost --policy-port 8014 \\
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
DEFAULT_POLICY_PORT = 8014

# Control frequencies
FREQ_POLICY = 30  # Hz - frequency to query policy server

VIDEO_FREQ = 30
CAMERA_KEY = "observation/egocentric"


# FSQ configuration
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16
FSQ_LEVELS = 21 



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
        State includes: qpos(29), left_hand_q(7), right_hand_q(7)
        """
        with self._lock:
            if self._latest is None:
                return None

            qpos = self._latest["qpos"].copy()
            left_hand_q = self._latest["left_hand_q"].copy()
            right_hand_q = self._latest["right_hand_q"].copy()

            return {
                "qpos": qpos,
                "left_hand_q": left_hand_q,
                "right_hand_q": right_hand_q,
            }

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

    def publish_token(self, token, left_hand=None, right_hand=None):
        """
        Publish token-only message (Protocol v4).

        Args:
            token: np.ndarray of shape (N,) - latent vector from encoder
            left_hand: np.ndarray of shape (7,) - left hand 7-DOF joint positions
            right_hand: np.ndarray of shape (7,) - right hand 7-DOF joint positions
        """
        pose_data = {
            "token_state": token.astype(np.float32).reshape(1, -1),
        }

        if left_hand is not None:
            left_hand = np.array(left_hand, dtype=np.float32)
            pose_data["left_hand_joints"] = left_hand.reshape(1, -1)

        if right_hand is not None:
            right_hand = np.array(right_hand, dtype=np.float32)
            pose_data["right_hand_joints"] = right_hand.reshape(1, -1)

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

    def __init__(self, host, port, prompt):
        if not POLICY_CLIENT_AVAILABLE:
            raise RuntimeError("Policy client not available!")

        self._host = host
        self._port = port
        self._prompt = prompt
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
        print(f"[PolicyClient] Connected successfully!")

    def get_action(self, image, state):
        """
        Send observation to policy server and get action.

        Args:
            image: RGB image (H, W, 3)
            state: dict with robot state

        Returns:
            action dict with 'token' and 'hand_states'
        """
        if self._client is None:
            raise RuntimeError("Not connected to policy server")

        # Build observation dict (similar to client_AR.py)
        obs = {
            CAMERA_KEY: image,
            "observation/left_hand": state["left_hand_q"],
            "observation/right_hand": state["right_hand_q"],
            "observation/left_arm": state["qpos"][15:22],
            "observation/right_arm": state["qpos"][22:29],
            "prompt": self._prompt,
            "session_id": self._session_id,
        }

        # Get action from policy server
        try:
            action = self._client.infer(obs)

            # 应用FSQ量化到token
            if 'token' in action and action['token'] is not None:
                token = np.array(action['token'], dtype=np.float32)
                # 量化token到FSQ级别
                action['token'] = fsq_quantize(token)
                print(f"[PolicyClient] Token quantized: shape={token.shape}, "
                      f"original_range=[{token.min():.4f},{token.max():.4f}], "
                      f"quantized_range=[{action['token'].min():.4f},{action['token'].max():.4f}]")

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
                 wbc_host, wbc_port, wbc_topic):
        # Initialize components
        self._camera = RSCamera(host=camera_host, port=camera_port)
        self._state_reader = WBCStateReader(host=wbc_host, port=wbc_port, topic=wbc_topic)
        self._token_publisher = TokenPublisher(host=zmq_host, port=zmq_port, topic=zmq_topic)
        self._policy_client = PolicyClientManager(host=policy_host, port=policy_port, prompt=prompt)

        # Threading components
        self._running = threading.Event()
        self._infer_request = threading.Event()   # signal inference worker to run
        self._chunk_lock = threading.Lock()
        self._pending_chunk: np.ndarray | None = None  # latest chunk from inference worker
        self._chunk_ready = threading.Event()     # new chunk is available

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

    def _get_policy_chunk(self):
        """
        Capture current observation and query policy server.
        Returns np.ndarray of shape (N, action_dim), or None on failure.
        """
        frame = self._camera.get_frame()
        if frame is None:
            print("[Inference] No camera frame, skipping.")
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        state = self._state_reader.get_state()
        if state is None:
            print("[Inference] No robot state, skipping.")
            return None

        action = self._policy_client.get_action(frame_rgb, state)
        if action is None:
            return None

        token = action.get('token', None)
        if token is None:
            return None

        token = np.array(token, dtype=np.float32)
        # Ensure shape is (N, action_dim)
        if token.ndim == 1:
            token = token.reshape(1, -1)
        return token

    def _inference_worker(self):
        """
        Background thread: waits for an inference request, runs policy, posts result.
        Retries automatically on failure.
        """
        while self._running.is_set():
            if not self._infer_request.wait(timeout=0.1):
                continue
            self._infer_request.clear()

            try:
                chunk = self._get_policy_chunk()
                if chunk is not None:
                    with self._chunk_lock:
                        self._pending_chunk = chunk
                    self._chunk_ready.set()
                    print(f"[Inference] New chunk ready: shape={chunk.shape}")
                else:
                    print("[Inference] Failed to get chunk, will retry.")
                    self._infer_request.set()   # retry immediately
            except Exception as e:
                print(f"[Inference] Error: {e}")
                self._infer_request.set()       # retry on exception

    def _publish_loop(self):
        """
        Main 30 Hz control loop.

        State machine:
          - EXECUTING: iterate through action chunk, one token per tick (1/30 s)
          - WAITING:   chunk exhausted; repeat last token until new chunk arrives
        """
        dt = 1.0 / FREQ_POLICY  # 1/30 s

        # Warm-up: send start command
        time.sleep(1.0)
        self._token_publisher.send_command(start=True, stop=False, planner=True)
        time.sleep(0.2)

        # Request and block until first chunk arrives
        print("[PublishLoop] Requesting first policy inference...")
        self._infer_request.set()
        self._chunk_ready.wait()
        self._chunk_ready.clear()
        with self._chunk_lock:
            chunk = self._pending_chunk
            self._pending_chunk = None
        assert chunk is not None

        idx = 0
        last_token = chunk[-1].copy()
        infer_triggered = False
        print(f"[PublishLoop] First chunk: shape={chunk.shape}. Starting execution.")

        while self._running.is_set():
            t0 = time.time()

            if idx < len(chunk):
                # ── EXECUTING ──────────────────────────────────────────────
                token = chunk[idx]
                last_token = token.copy()
                idx += 1
                infer_triggered = False
            else:
                # ── WAITING for next chunk ─────────────────────────────────
                token = last_token

                # First tick after chunk exhausted (1/30s after last token was sent):
                # now capture observation and trigger policy inference
                if not infer_triggered:
                    print(f"[PublishLoop] Chunk done ({len(chunk)} tokens), "
                          f"triggering policy inference...")
                    self._infer_request.set()
                    infer_triggered = True

                if self._chunk_ready.is_set():
                    self._chunk_ready.clear()
                    with self._chunk_lock:
                        chunk = self._pending_chunk
                        self._pending_chunk = None
                    idx = 0
                    infer_triggered = False
                    print(f"[PublishLoop] New chunk received: shape={chunk.shape}. "
                          f"Resuming execution.")

            # Get fresh hand states for this tick
            state = self._state_reader.get_state()
            left_hand = state["left_hand_q"] if state else None
            right_hand = state["right_hand_q"] if state else None

            self._token_publisher.publish_token(token, left_hand, right_hand)

            # Maintain 30 Hz
            elapsed = time.time() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

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
                       help="Policy server port (default: 8014)")
    parser.add_argument("--prompt", type=str, required=True,
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

    args = parser.parse_args()

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
