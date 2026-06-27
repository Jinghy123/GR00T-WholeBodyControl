"""Non-RTC WebSocket client for the DreamZero (DZ) G1-neck policy server.

Talks to  baselines/dreamzero/socket_test_g1_neck.py  (the NON-rtc server), which
serves over an openpi `WebsocketPolicyServer` (msgpack_numpy protocol, default
port 5000). Each request returns one action chunk; the client plays the chunk
open-loop at 30 Hz and, between chunks, freezes the body token via the encoder
(identical playback to psix_sonic_client.py — only the transport + obs layout differ).

Protocol (openpi WebsocketPolicyServer):
  * On connect the server sends one msgpack metadata frame (handled by
    WebsocketClientPolicy.__init__).
  * Each request: client sends a msgpack-packed FLAT obs dict, server replies with
    a msgpack-packed action array.
  * The obs dict MUST carry an "endpoint" field ("infer" to run, "reset" to reset).

Obs dict (FLAT, top-level keys — same as g1_sonic_client.py --include-neck):
  {
      "observation/head":        frame,   # uint8, RGB, (H, W, 3) or (T, H, W, 3)
      "observation/hand_joints": hand,    # float32, (14,)
      "observation/qpos":        qpos,    # float32, (29,) layout leg/base15 + arm14
      "observation/neck":        neck,    # float32, (2,)
      "prompt":                  instruction,  # str
      "session_id":              session_id,   # str
  }

Server reply: action chunk, shape (T, 80), layout [hand_joints(14) | neck(2) | token(64)].
We FSQ-quantize the token and publish in internal layout hand(14) + neck(2) + token(64),
matching g1_sonic_client.py --include-neck.
"""

import os
import sys
import time
import threading
import json
import signal
import uuid
from collections import deque

import cv2
import numpy as np
import zmq
import msgpack

from openpi_client.websocket_client_policy import WebsocketClientPolicy

# Add project root to path for imports
_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message,
    build_command_message,
)
from ours.common.encoder_client import EncoderClient

# ---------------- Configuration ----------------
TASK_INSTRUCTION = "pick up the green grapes and place it into the green bowl"

# FSQ configuration (must match g1_sonic_client / encoder)
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16

# Control frequency
FREQ_POLICY = 30  # Hz

# Internal action layout (after reorder in _get_policy_chunk), matching
# g1_sonic_client.py --include-neck:
#   hand_joints(14) + neck(2) + token(64) = 80
# Server reply layout is [hand(14) | neck(2) | token(64)] (always 80-D).
HAND_DIM = 14
NECK_DIM = 2
TOKEN_DIM = 64
ACTION_DIM = 80

# Video: send 9 frames at stride-4 (training video delta [0,4,...,32], span 32)
# selected from a continuous 30 Hz ego ring buffer. The server uses them directly.
VIDEO_OFFSETS = (-32, -28, -24, -20, -16, -12, -8, -4, 0)
VIDEO_BUFFER_SIZE = 64
# Pre-resize frames to the policy resolution before sending (server is 384x672);
# cv2.resize takes (W, H). Keeps the 9-frame payload small.
POLICY_IMAGE_WH = (672, 384)

# Encoder model path (for frozen token between chunks)
ENCODER_MODEL = "gear_sonic_deploy/policy/release/model_encoder.onnx"

# Neck publisher configuration (to G1 NeckMotor, matches pose_publisher.py wire format)
DEFAULT_NECK_PUB_HOST = "*"
DEFAULT_NECK_PUB_PORT = 5570

# Neck state subscriber (ZMQ SUB <- realsense_server.py on the robot, port 5560)
DEFAULT_NECK_STATE_ZMQ = "tcp://192.168.123.164:5560"

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


# ---------------- InstructionManager ----------------
class InstructionManager:
    """Holds the per-stage subtask prompts and advances through them on Enter.

    DreamZero has no goal image, so this only tracks the text subtask; the assembled
    instruction switches as the operator presses Enter.
    """

    def __init__(self, subtasks=None):
        subtasks = [str(s) for s in (subtasks or [])]
        self._subtasks = subtasks if subtasks else [""]
        self._idx = 0
        self._lock = threading.Lock()

        print(f"[Instruction] Loaded {len(self._subtasks)} subtask stage(s):")
        for i, s in enumerate(self._subtasks):
            print(f"  [{i}] subtask: {s!r}")
        print("[Instruction] Current index: 0  (press Enter to advance)")

    def get_subtask(self):
        with self._lock:
            return self._subtasks[self._idx]

    def get_stage(self):
        with self._lock:
            return self._idx, self._subtasks[self._idx]

    def advance(self):
        with self._lock:
            if self._idx < len(self._subtasks) - 1:
                self._idx += 1
                print(f"[Instruction] -> index {self._idx}: subtask {self._subtasks[self._idx]!r}")
            else:
                print(f"[Instruction] Already at last subtask (index {self._idx})")


# ---------------- RSCamera ----------------
class RSCamera:
    def __init__(self, address="tcp://192.168.123.164:5558"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)

    def get_frame(self):
        self.socket.send(b"get_frame")
        parts = self.socket.recv_multipart()  # 3-part RealSense or 4-part ZED; slot 0 is ego RGB
        rgb_array = np.frombuffer(parts[0], np.uint8)
        rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        return rgb_image

    def close(self):
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()


# ---------------- ZedNeckCamera ----------------
class ZedNeckCamera:
    """Neck-mounted ZED camera (--include-neck). Server reply is 4-part
    multipart [ego_rgb, ego_stereo, left_wrist, right_wrist]; only slot 0 used."""

    def __init__(self, address="tcp://192.168.123.164:5558"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)

    def get_frame(self):
        self.socket.send(b"get_frame")
        parts = self.socket.recv_multipart()
        while len(parts) < 4:
            parts.append(b"")
        ego_rgb_jpeg = parts[0]
        if not ego_rgb_jpeg:
            return None
        arr = np.frombuffer(ego_rgb_jpeg, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

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
        """Publish action token message (Protocol v4).

        Args:
            action: np.ndarray of shape (80,) — hand(14) + neck(2) + token(64),
                matching g1_sonic_client.py --include-neck. The neck slot is
                forwarded separately by NeckPublisher; only token + hands go here.
        """
        action = action.astype(np.float32).reshape(1, -1)
        token_start = HAND_DIM + NECK_DIM  # 16
        pose_data = {
            "token_state": action[:, token_start:token_start + TOKEN_DIM],  # (1, 64)
            "left_hand_joints": action[:, :7],    # (1, 7)
            "right_hand_joints": action[:, 7:14],  # (1, 7)
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=4)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close(linger=0)
        self._context.term()


# ---------------- NeckStateReader / NeckPublisher ----------------
class NeckStateReader:
    """SUB to realsense_server.py's neck present-position stream (JSON [yaw, pitch])."""

    def __init__(self, addr):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(addr)
        self._latest = None

    def get_latest(self):
        try:
            raw = self._sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return self._latest
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._latest
        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
            self._latest = [float(msg[0]), float(msg[1])]
        return self._latest

    def stop(self):
        self._sock.close(linger=0)


class NeckPublisher:
    """PUB of [yaw, pitch] JSON for the G1 NeckMotor (matches pose_publisher.py wire format)."""

    def __init__(self, host=DEFAULT_NECK_PUB_HOST, port=DEFAULT_NECK_PUB_PORT):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{port}")

    def publish(self, yaw, pitch):
        msg = json.dumps([float(yaw), float(pitch)]).encode("utf-8")
        self._sock.send(msg)

    def stop(self):
        self._sock.close(linger=0)
        self._ctx.term()


# ---------------- Main Client ----------------
class DZSonicClient:
    """
    Chunk-based non-RTC DreamZero client. Each inference returns a chunk of actions;
    the publish loop iterates it at 30 Hz, and the encoder freezes the body token
    between chunks. Identical playback to psix_sonic_client.py — only the transport
    (openpi msgpack WebSocket) and the flat obs layout differ.
    """

    def __init__(self, policy_client, state_subscriber, camera, token_publisher,
                 instruction_manager, task_instruction,
                 include_neck=False, neck_publisher=None, neck_state_reader=None):
        self._policy = policy_client
        self._state_sub = state_subscriber
        self._camera = camera
        self._token_publisher = token_publisher
        self._instruction_manager = instruction_manager
        self._task = task_instruction
        self._dbg_last_idx = -1
        self._include_neck = include_neck
        self._neck_publisher = neck_publisher
        self._neck_state_reader = neck_state_reader

        # One session id per run; server resets its video frame buffer when it changes.
        self._session_id = uuid.uuid4().hex
        print(f"[DZSonicClient] session_id={self._session_id}")

        # Encoder for frozen token
        self._encoder = EncoderClient(ENCODER_MODEL, mode=0)

        self._running = threading.Event()
        self._sequence_done_event = threading.Event()
        self._pending_chunk = None
        self._chunk_lock = threading.Lock()

        self._inference_thread = None
        self._publish_thread = None

        # Continuous 30 Hz ego-frame ring buffer; inference pulls 9 stride-4 frames
        # out of it. Filled by the publish loop (one capture per control tick).
        self._frame_buffer = deque(maxlen=VIDEO_BUFFER_SIZE)
        self._frame_lock = threading.Lock()

    # ---------- Policy request ----------
    def _capture_frame(self):
        """Grab one ego frame, convert to RGB + resize, append to the ring buffer."""
        frame = self._camera.get_frame()
        if frame is None:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, POLICY_IMAGE_WH).astype(np.uint8)
        with self._frame_lock:
            self._frame_buffer.append(frame)

    def _video_window(self):
        """Select the 9 stride-4 frames [-32,-28,...,0] from the ring buffer
        (left-padded with the oldest). Returns (9, H, W, 3) or None if empty."""
        with self._frame_lock:
            frames = list(self._frame_buffer)
        if not frames:
            return None
        latest = len(frames) - 1
        idxs = [max(0, latest + off) for off in VIDEO_OFFSETS]
        return np.stack([frames[i] for i in idxs], axis=0)

    def _build_observation_payload(self):
        """Capture current state + ego frame, build the flat DreamZero obs dict.

        FLAT top-level keys, same as g1_sonic_client.py --include-neck:
        observation/head, observation/hand_joints (14), observation/qpos (29, raw
        body_q = leg/base15 + arm14), observation/neck (2), prompt, session_id.
        The server reorders qpos+hand+neck into the model state internally.
        """
        state = self._state_sub.get_state()
        assert state is not None, "Robot state not available"

        body_q = np.array(state["body_q_measured"], dtype=np.float32)        # (29,) = leg/base15 + arm14
        left_hand_states = np.array(state["left_hand_q_measured"], dtype=np.float32)   # (7,)
        right_hand_states = np.array(state["right_hand_q_measured"], dtype=np.float32)  # (7,)
        hand_joints = np.concatenate((left_hand_states, right_hand_states), axis=0)    # (14,)

        neck_latest = (
            self._neck_state_reader.get_latest()
            if self._neck_state_reader is not None
            else None
        )
        neck_state = (
            np.asarray(neck_latest, dtype=np.float32).reshape(NECK_DIM)
            if neck_latest is not None
            else np.zeros(NECK_DIM, dtype=np.float32)
        )

        # 9 stride-4 ego frames from the 30 Hz ring buffer (filled by the publish
        # loop). The server uses them directly as the video block.
        head = self._video_window()
        if head is None:
            raise RuntimeError(
                "No camera frames in buffer yet; the publish loop should prime it"
            )

        # Instruction: verbatim subtask if set, else the task string as-is.
        base = str(self._task).strip()
        subtask = str(self._instruction_manager.get_subtask()).strip()
        instruction = f"Task: {base.lower()}. Subtask: {subtask}" if subtask else base

        # --- DEBUG: dump latest ego frame whenever the instruction stage changes ---
        stage_idx, stage_subtask = self._instruction_manager.get_stage()
        if stage_idx != self._dbg_last_idx:
            self._dbg_last_idx = stage_idx
            cv2.imwrite("/tmp/sent_ego.jpg", cv2.cvtColor(head[-1], cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] stage idx={stage_idx} | subtask={stage_subtask!r} "
                  f"| instruction={instruction!r} | head={head.shape} | dumped /tmp/sent_ego.jpg")

        payload = {
            "observation/head": head,                                   # (9, H, W, 3) uint8
            "observation/hand_joints": hand_joints.astype(np.float32),  # (14,)
            "observation/qpos": body_q.astype(np.float32),              # (29,) leg/base15 + arm14
            "observation/neck": neck_state.astype(np.float32),          # (2,)
            "prompt": instruction,
            "session_id": self._session_id,
            # openpi_client.WebsocketClientPolicy does NOT auto-add the endpoint
            # field; the server's handshake reads obs["endpoint"] first.
            "endpoint": "infer",
        }
        return payload

    def _get_policy_chunk(self):
        """POST current observation to the DZ server and return a chunk of actions
        in internal publish layout hand(14) + neck(2) + token(64), with FSQ
        quantization applied to the token. Returns None on failure.

        Server returns each action as [hand_joints(14) | neck(2) | token(64)].
        """
        payload = self._build_observation_payload()

        try:
            result = self._policy.infer(payload)
        except Exception as e:
            print(f"[Inference] WS request error: {e}")
            return None

        try:
            chunk = np.asarray(result, dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
        except Exception as e:
            print(f"[Inference] Response parse error: {e}")
            return None

        if chunk.shape[-1] != ACTION_DIM:
            print(f"[Inference] Unexpected action dim: {chunk.shape}, expected (*, {ACTION_DIM})")
            return None

        hand_joints = chunk[:, :HAND_DIM]                                       # (N, 14)
        neck = chunk[:, HAND_DIM:HAND_DIM + NECK_DIM]                           # (N, 2)
        token_ori = chunk[:, HAND_DIM + NECK_DIM:HAND_DIM + NECK_DIM + TOKEN_DIM]  # (N, 64)
        token_qtz = fsq_quantize(token_ori)
        chunk_out = np.concatenate([hand_joints, neck, token_qtz], axis=-1).astype(np.float32)

        print(f"[Inference] Chunk received: shape={chunk_out.shape}, "
              f"token range=[{token_ori.min():.4f},{token_ori.max():.4f}] -> "
              f"[{token_qtz.min():.4f},{token_qtz.max():.4f}]")
        return chunk_out

    # ---------- Threads ----------
    def _inference_worker(self):
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
        """Main 30 Hz control loop (open-loop chunk playback + encoder token freeze
        between chunks). Internal action layout is hand(14) + neck(2) + token(64)."""
        dt = 1.0 / FREQ_POLICY

        self._token_publisher.send_command(start=True, stop=False, planner=True)
        self._capture_frame()  # prime the ring buffer before the first inference
        print("[PublishLoop] Requesting first policy inference...")
        self._sequence_done_event.set()
        while self._sequence_done_event.is_set() and self._running.is_set():
            time.sleep(0.05)
        if not self._running.is_set():
            return
        with self._chunk_lock:
            chunk = self._pending_chunk

        idx = 0
        frozen_action = None
        last_action = None
        print(f"[PublishLoop] First chunk: shape={chunk.shape}. Starting execution.")

        while self._running.is_set():
            t_start = time.perf_counter()

            if idx < len(chunk):
                # ── EXECUTING ──
                action = chunk[idx]
                last_action = action.copy()
                idx += 1
            else:
                # ── WAITING for next chunk ──
                if idx == len(chunk):
                    state = self._state_sub.get_state()
                    if state is not None:
                        qpos = _mujoco29_to_isaaclab29(state["body_q_measured"])  # (29,)
                        base_quat = np.array(state["base_quat_measured"], dtype=np.float32)  # (4,) wxyz

                        joint_pos = np.tile(qpos, (10, 1)).astype(np.float32)
                        joint_vel = np.zeros((10, 29), dtype=np.float32)
                        body_quat = np.tile(base_quat, (10, 1)).astype(np.float32)

                        enc_token = self._encoder.encode(joint_pos, joint_vel, body_quat)  # (64,)
                        # Layout hand(14) + neck(2) + token(64): keep hand+neck from the
                        # last action, replace the body token with the encoder freeze.
                        frozen_action = np.concatenate([
                            last_action[:HAND_DIM],                     # hand14
                            last_action[HAND_DIM:HAND_DIM + NECK_DIM],  # neck2
                            enc_token,                                  # token64
                        ])
                        print(f"[PublishLoop] Chunk done ({len(chunk)} tokens), "
                              f"encoder freeze token computed.")
                    else:
                        frozen_action = last_action.copy()
                        print("[PublishLoop] Chunk done, no robot state — repeating last action.")
                    self._sequence_done_event.set()
                    idx += 1

                if not self._sequence_done_event.is_set():  # inference done
                    with self._chunk_lock:
                        chunk = self._pending_chunk
                    frozen_action = None
                    idx = 0
                    print(f"[PublishLoop] New chunk received: shape={chunk.shape}. Resuming.")
                    action = chunk[idx]
                    last_action = action.copy()
                    idx += 1
                else:
                    action = frozen_action

            self._token_publisher.publish_token(action)
            if self._neck_publisher is not None:
                neck = action[HAND_DIM:HAND_DIM + NECK_DIM]
                self._neck_publisher.publish(neck[0], neck[1])

            # Keep the 30 Hz ego buffer filled (incl. during freeze) so the next
            # inference pulls a continuous stride-4 window from it.
            self._capture_frame()

            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ---------- Lifecycle ----------
    def start(self):
        print("[DZSonicClient] Starting...")
        self._running.set()
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()
        print("[DZSonicClient] Started successfully!")
        return True

    def stop(self):
        print("[DZSonicClient] Stopping...")
        self._running.clear()
        try:
            self._token_publisher.send_command(start=False, stop=True, planner=True)
        except Exception as e:
            print(f"[DZSonicClient] Error sending stop command: {e}")
        try:
            self._camera.close()
        except Exception:
            pass
        self._state_sub.stop()
        self._token_publisher.stop()
        if self._neck_publisher is not None:
            self._neck_publisher.stop()
        if self._neck_state_reader is not None:
            self._neck_state_reader.stop()
        print("[DZSonicClient] Stopped.")


# ---------------- Main ----------------
def main(host, port, zmq_host, zmq_pub_port, zmq_sub_port, zmq_topic, zmq_sub_topic,
         camera_address, task_instruction, subtasks,
         include_neck=False, neck_pub_host=DEFAULT_NECK_PUB_HOST, neck_pub_port=DEFAULT_NECK_PUB_PORT,
         neck_state_zmq=DEFAULT_NECK_STATE_ZMQ):
    print("[MAIN] Initializing components...")

    token_publisher = TokenPublisher(host="*", port=zmq_pub_port, topic=zmq_topic)
    print(f"[MAIN] TokenPublisher bound on port {zmq_pub_port}, topic='{zmq_topic}'")

    state_sub = RobotStateSubscriber(host=zmq_host, port=zmq_sub_port, topic=zmq_sub_topic)
    print(f"[MAIN] State subscriber connected to {zmq_host}:{zmq_sub_port}, topic='{zmq_sub_topic}'")

    camera = ZedNeckCamera(address=camera_address) if include_neck else RSCamera(address=camera_address)
    print(f"[MAIN] Camera connected to {camera_address} (include_neck={include_neck})")

    neck_publisher = None
    neck_state_reader = None
    if include_neck:
        neck_publisher = NeckPublisher(host=neck_pub_host, port=neck_pub_port)
        neck_state_reader = NeckStateReader(neck_state_zmq)
        print(f"[MAIN] Neck publisher bound on {neck_pub_host}:{neck_pub_port}, "
              f"state reader connected to {neck_state_zmq}")

    instruction_manager = InstructionManager(subtasks=subtasks)
    print(f"[MAIN] Task instruction: {task_instruction!r}")

    # Connect to the DZ policy server (openpi WebsocketPolicyServer; handshake on connect)
    print(f"[MAIN] Connecting to DZ policy server at ws://{host}:{port} ...")
    policy_client = WebsocketClientPolicy(host=host, port=port)
    print(f"[MAIN] Connected. Server metadata: {policy_client.get_server_metadata()}")

    time.sleep(1.0)

    token_publisher.send_command(start=True, stop=False, planner=True)

    print("[MAIN] Waiting for robot state...")
    for i in range(30):
        state = state_sub.get_state()
        if state is not None:
            print(f"[MAIN] Got robot state with keys: {list(state.keys())}")
            body_q = np.array(state.get("body_q_measured", []))
            print(f"[MAIN] body_q_measured shape: {body_q.shape}")
            break
        time.sleep(0.5)
    else:
        print("[MAIN] WARNING: No robot state received after 15s, proceeding anyway...")

    client = DZSonicClient(
        policy_client=policy_client,
        state_subscriber=state_sub,
        camera=camera,
        token_publisher=token_publisher,
        instruction_manager=instruction_manager,
        task_instruction=task_instruction,
        include_neck=include_neck,
        neck_publisher=neck_publisher,
        neck_state_reader=neck_state_reader,
    )

    if not client.start():
        print("[MAIN] Client failed to start")
        client.stop()
        return

    def stdin_listener():
        print("[MAIN] Press Enter to advance subtask.")
        while client._running.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break
            instruction_manager.advance()

    t_stdin = threading.Thread(target=stdin_listener, daemon=True)
    t_stdin.start()

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

    parser = argparse.ArgumentParser(description="Non-RTC DreamZero WebSocket Policy Client")
    parser.add_argument("--host", type=str, default="localhost",
                        help="DreamZero policy server host")
    parser.add_argument("--port", type=int, default=48014,
                        help="Local port reaching the DZ non-RTC server (SSH tunnel "
                             "48014 -> nebula102:5000)")
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
    parser.add_argument("--prompts-json", type=str,
                        default="/home/xiawei/data/multi-task/prompts.json",
                        help="JSON mapping task-key -> {task_description, subtasks[]}")
    parser.add_argument("--task-key", type=str, default="pick_place_1",
                        help="Key into prompts.json; selects task_description and per-stage subtasks.")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override task instruction (sent verbatim; skips prompts.json)")
    parser.add_argument("--include-neck", action="store_true",
                        help="Neck variant: send real 45-dim state (+neck2) and publish the "
                             "action's neck(2) + use the neck-mounted ZED camera. Default (off) "
                             "sends 43-dim state (server pads to 45) and ignores the action neck.")
    parser.add_argument("--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST,
                        help=f"Neck PUB bind host (default: {DEFAULT_NECK_PUB_HOST})")
    parser.add_argument("--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT,
                        help=f"Neck PUB port (default: {DEFAULT_NECK_PUB_PORT})")
    parser.add_argument("--neck-state-zmq", type=str, default=DEFAULT_NECK_STATE_ZMQ,
                        help=f"Neck-state SUB address (default: {DEFAULT_NECK_STATE_ZMQ})")

    args = parser.parse_args()

    task_instruction = args.instruction
    subtasks = []
    if task_instruction is None and args.task_key:
        with open(args.prompts_json) as f:
            prompts = json.load(f)
        if args.task_key not in prompts:
            raise SystemExit(
                f"[MAIN] task-key '{args.task_key}' not found in {args.prompts_json}; "
                f"available: {list(prompts)}"
            )
        entry = prompts[args.task_key]
        if task_instruction is None:
            task_instruction = entry.get("task_description")
        subtasks = entry.get("subtasks", [])
    if task_instruction is None:
        task_instruction = TASK_INSTRUCTION

    main(
        host=args.host,
        port=args.port,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_sub_port=args.zmq_sub_port,
        zmq_topic=args.zmq_topic,
        zmq_sub_topic=args.zmq_sub_topic,
        camera_address=args.camera_address,
        task_instruction=task_instruction,
        subtasks=subtasks,
        include_neck=args.include_neck,
        neck_pub_host=args.neck_pub_host,
        neck_pub_port=args.neck_pub_port,
        neck_state_zmq=args.neck_state_zmq,
    )
