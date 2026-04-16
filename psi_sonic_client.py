import os
import sys
import time
import threading
import json
import signal

import cv2
import numpy as np
import zmq
import msgpack
import requests
import json_numpy

json_numpy.patch()

# Add project root to path for imports
_GROOT_ROOT = os.path.expanduser("~/hsc/GR00T-WholeBodyControl")
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message,
    build_command_message,
)
from encoder_client import EncoderClient

# ---------------- Configuration ----------------
TASK_INSTRUCTION = "pick bottle and turn and pour into cup"

# FSQ configuration (must match g1_sonic_client / encoder)
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16

# Control frequency
FREQ_POLICY = 30  # Hz

# Action layout: token(64) + hand_joints(14) = 78
ACTION_DIM = 78

# Encoder model path (for frozen token between chunks)
ENCODER_MODEL = os.path.join(_GROOT_ROOT, "gear_sonic_deploy/policy/release/model_encoder.onnx")

# Joint order conversion: WBC publishes in Mujoco order, encoder expects IsaacLab order
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)


def _mujoco29_to_isaaclab29(qpos: np.ndarray) -> np.ndarray:
    return np.asarray(qpos, dtype=np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB_DOF].copy()


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    quantized = np.clip(quantized, fsq_min, fsq_max)
    return quantized


# ---------------- Serialization utilities ----------------
# json_numpy.patch() (above) monkey-patches json to handle np.ndarray transparently.


# ---------------- RSCamera ----------------
class RSCamera:
    def __init__(self, address="tcp://192.168.123.164:5558"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)

    def get_frame(self):
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


# ---------------- RobotStateSubscriber ----------------
class RobotStateSubscriber:
    """Subscribe to robot state published by g1_deploy_onnx_ref on ZMQ PUB port."""

    def __init__(self, host="localhost", port=5557, topic="g1_debug"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout (for fast shutdown)
        self._socket.setsockopt(zmq.RCVHWM, 1)

        self._topic = topic
        self._lock = threading.Lock()
        self._latest_state = None
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                msg = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break

            topic_bytes = self._topic.encode("utf-8")
            if msg.startswith(topic_bytes):
                payload = msg[len(topic_bytes):]
            else:
                payload = msg

            try:
                state = msgpack.unpackb(payload, raw=False)
                with self._lock:
                    self._latest_state = state
            except Exception as e:
                print(f"[StateSubscriber] Unpack error: {e}")

    def get_state(self):
        with self._lock:
            return self._latest_state

    def stop(self):
        self._running = False
        self._thread.join(timeout=0.5)
        self._socket.close(linger=0)
        self._context.term()


# ---------------- TokenPublisher ----------------
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

    def publish_token(self, action):
        """
        Publish action token message (Protocol v4).

        Args:
            action: np.ndarray of shape (78,) — token(64) + hand_joints(14)
        """
        action = action.astype(np.float32).reshape(1, -1)
        pose_data = {
            "token_state": action[:, :64],          # (1, 64)
            "left_hand_joints": action[:, 64:71],   # (1, 7)
            "right_hand_joints": action[:, 71:78],  # (1, 7)
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=4)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close(linger=0)
        self._context.term()


# ---------------- Main Client ----------------
class PsiSonicClient:
    """
    Chunk-based client: HTTP policy server (POST /act) returns a chunk of
    actions per inference call, publish loop iterates at 30 Hz, encoder
    freezes token between chunks. Matches g1_sonic_client.py logic.
    """

    def __init__(self, server_url, state_subscriber, camera, token_publisher, instruction,
                 http_timeout=30.0):
        self._server_url = server_url
        self._state_sub = state_subscriber
        self._camera = camera
        self._token_publisher = token_publisher
        self._instruction = instruction
        self._http_timeout = http_timeout

        # Encoder for frozen token
        self._encoder = EncoderClient(ENCODER_MODEL, mode=0)

        # Threading / synchronization (mirrors g1_sonic_client)
        self._running = threading.Event()
        self._sequence_done_event = threading.Event()
        self._pending_chunk = None  # latest chunk from inference worker
        self._chunk_lock = threading.Lock()

        # HTTP session (reuses TCP connection)
        self._session = requests.Session()

        self._inference_thread = None
        self._publish_thread = None

    # ---------- Policy request ----------
    def _build_observation_payload(self):
        """Capture current state + frame, build payload dict (json_numpy-aware)."""
        state = self._state_sub.get_state()
        assert state is not None, "Robot state not available"

        body_q = np.array(state["body_q_measured"], dtype=np.float32)
        left_hand_states = np.array(state["left_hand_q"], dtype=np.float32)
        right_hand_states = np.array(state["right_hand_q"], dtype=np.float32)
        states = np.concatenate((body_q, left_hand_states, right_hand_states), axis=0)

        frame = self._camera.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.uint8)

        payload = {
            "image": {"observation.images.egocentric": frame},
            "state": {"states": states.reshape(1, -1)},
            "gt_action": None,
            "dataset_name": None,
            "instruction": self._instruction,
            "history": None,
            "condition": None,
            "timestamp": None,
        }
        return payload

    def _get_policy_chunk(self):
        """
        POST current observation to HTTP policy server and return a chunk of
        actions, shape (N, 78), with FSQ quantization applied to token.
        Returns None on failure.
        """
        payload = self._build_observation_payload()

        try:
            resp = self._session.post(self._server_url, json=payload,
                                      timeout=self._http_timeout)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Inference] HTTP request error: {e}")
            return None

        try:
            data = resp.json()
            action = data["action"] if isinstance(data, dict) else data
            chunk = np.asarray(action, dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
        except Exception as e:
            print(f"[Inference] Response parse error: {e}")
            return None

        if chunk.shape[-1] != ACTION_DIM:
            print(f"[Inference] Unexpected action dim: {chunk.shape}, expected (*, {ACTION_DIM})")
            return None

        # Apply FSQ quantization on token part, leave hand joints unchanged
        token_ori = chunk[:, :64]
        hand_joints = chunk[:, 64:78]
        token_qtz = fsq_quantize(token_ori)
        chunk_out = np.concatenate([token_qtz, hand_joints], axis=-1).astype(np.float32)

        print(f"[Inference] Chunk received: shape={chunk_out.shape}, "
              f"token range=[{token_ori.min():.4f},{token_ori.max():.4f}] → "
              f"[{token_qtz.min():.4f},{token_qtz.max():.4f}]")
        return chunk_out

    # ---------- Threads ----------
    def _inference_worker(self):
        """Wait for request, run policy, post result. Matches g1_sonic_client."""
        # Wait for first robot state
        while self._running.is_set():
            state = self._state_sub.get_state()
            if state is not None:
                break
            print("[Inference] waiting for robot state...")
            time.sleep(1.0)

        while self._running.is_set():
            self._sequence_done_event.wait()
            try:
                t0 = time.time()
                chunk = self._get_policy_chunk()
                dt = time.time() - t0
                if chunk is None:
                    raise RuntimeError("Failed to get chunk")
                print(f"[Inference] Policy returned chunk shape={chunk.shape} in {dt:.2f}s")
                with self._chunk_lock:
                    self._pending_chunk = chunk
                self._sequence_done_event.clear()
            except RuntimeError as e:
                print(f"[Inference] {e}")
                self._running.clear()
                return

    def _publish_loop(self):
        """
        Main 30 Hz control loop.

        State machine:
          - EXECUTING: iterate through action chunk, one token per tick (1/30 s)
          - WAITING:   chunk exhausted; send encoder-derived frozen token until
                       next chunk arrives
        """
        dt = 1.0 / FREQ_POLICY

        # Send start command and request first chunk in parallel
        # (same pattern as psi_rtc_sonic_client: start first, then wait for action)
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
        frozen_action = None
        last_action = None
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
                    state = self._state_sub.get_state()
                    if state is not None:
                        qpos = _mujoco29_to_isaaclab29(state["body_q_measured"])  # (29,)
                        base_quat = np.array(state["base_quat_measured"], dtype=np.float32)  # (4,) wxyz

                        joint_pos = np.tile(qpos, (10, 1)).astype(np.float32)       # (10, 29)
                        joint_vel = np.zeros((10, 29), dtype=np.float32)
                        body_quat = np.tile(base_quat, (10, 1)).astype(np.float32)  # (10, 4)

                        enc_token = self._encoder.encode(joint_pos, joint_vel, body_quat)  # (64,)
                        # action layout: token(64) + hand_joints(14)
                        # keep hand joints from last action, replace body token
                        frozen_action = np.concatenate([enc_token, last_action[64:78]])
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

                    # directly execute the first action
                    action = chunk[idx]
                    last_action = action.copy()
                    idx += 1
                    using_last_action = False
                else:
                    action = frozen_action
                    using_last_action = True

            self._token_publisher.publish_token(action)

            # Maintain 30 Hz
            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ---------- Lifecycle ----------
    def start(self):
        print("[PsiSonicClient] Starting...")
        self._running.set()

        # Start inference worker
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()

        # Start publish loop
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()

        print("[PsiSonicClient] Started successfully!")
        return True

    def stop(self):
        print("[PsiSonicClient] Stopping...")
        self._running.clear()

        # Send stop command to WBC
        try:
            self._token_publisher.send_command(start=False, stop=True, planner=True)
        except Exception as e:
            print(f"[PsiSonicClient] Error sending stop command: {e}")

        # Close HTTP session
        try:
            self._session.close()
        except Exception:
            pass

        # Clean up
        try:
            self._camera.close()
        except Exception:
            pass
        self._state_sub.stop()
        self._token_publisher.stop()

        print("[PsiSonicClient] Stopped.")


# ---------------- Main ----------------
def main(server_url, zmq_host, zmq_pub_port, zmq_sub_port, zmq_topic, zmq_sub_topic,
         camera_address, instruction):
    print("[MAIN] Initializing components...")

    # 1. Initialize token publisher (ZMQ PUB, Protocol v4)
    token_publisher = TokenPublisher(host="*", port=zmq_pub_port, topic=zmq_topic)
    print(f"[MAIN] TokenPublisher bound on port {zmq_pub_port}, topic='{zmq_topic}'")

    # 2. Initialize robot state subscriber (ZMQ SUB)
    state_sub = RobotStateSubscriber(host=zmq_host, port=zmq_sub_port, topic=zmq_sub_topic)
    print(f"[MAIN] State subscriber connected to {zmq_host}:{zmq_sub_port}, topic='{zmq_sub_topic}'")

    # 3. Initialize camera
    camera = RSCamera(address=camera_address)
    print(f"[MAIN] Camera connected to {camera_address}")

    # 4. Wait briefly for ZMQ PUB socket to establish connections
    time.sleep(1.0)

    # 5. Wait for first robot state
    print("[MAIN] Waiting for robot state...")
    for i in range(30):
        state = state_sub.get_state()
        if state is not None:
            print(f"[MAIN] Got robot state with keys: {list(state.keys())}")
            break
        time.sleep(0.5)
    else:
        print("[MAIN] WARNING: No robot state received after 15s, proceeding anyway...")

    # 6. Create and start client
    client = PsiSonicClient(
        server_url=server_url,
        state_subscriber=state_sub,
        camera=camera,
        token_publisher=token_publisher,
        instruction=instruction,
    )

    if not client.start():
        print("[MAIN] Client failed to start")
        client.stop()
        return

    print("[MAIN] Running. Ctrl+C to stop.")

    def signal_handler(sig, frame):
        print("\n[MAIN] Caught signal, shutting down...")
        client._running.clear()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while client._running.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[MAIN] Caught Ctrl+C, exiting...")

    client.stop()
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk-based VLA Policy Client (token streaming via Protocol v4)"
    )
    parser.add_argument("--host", type=str, default="localhost",
                        help="VLA policy server host")
    parser.add_argument("--port", type=int, default=22085,
                        help="VLA policy server port")
    parser.add_argument("--zmq-host", type=str, default="localhost",
                        help="ZMQ host for robot state subscriber")
    parser.add_argument("--zmq-pub-port", type=int, default=5556,
                        help="ZMQ PUB port for sending pose to WBC")
    parser.add_argument("--zmq-sub-port", type=int, default=5557,
                        help="ZMQ SUB port for receiving robot state")
    parser.add_argument("--zmq-topic", type=str, default="pose",
                        help="ZMQ topic for pose messages")
    parser.add_argument("--zmq-sub-topic", type=str, default="g1_debug",
                        help="ZMQ topic for robot state subscription")
    parser.add_argument("--camera-address", type=str, default="tcp://192.168.123.164:5558",
                        help="Camera ZMQ address")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Task instruction for VLA policy")

    args = parser.parse_args()

    instruction = args.instruction if args.instruction else TASK_INSTRUCTION
    server_url = f"http://{args.host}:{args.port}/act"

    main(
        server_url=server_url,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_sub_port=args.zmq_sub_port,
        zmq_topic=args.zmq_topic,
        zmq_sub_topic=args.zmq_sub_topic,
        camera_address=args.camera_address,
        instruction=instruction,
    )
