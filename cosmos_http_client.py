"""Non-RTC HTTP client for the Cosmos G1 action-policy server.

The Cosmos policy server is the ``action_policy_server`` HTTP service (the same
one driven open-loop by ``examples/eval_g1_openloop.py``), reachable locally
after port-forwarding:

    curl -X GET http://localhost:8000/info

It exposes two endpoints:
  * ``GET  /info``    -> model metadata (run_name, checkpoint, sampling params).
  * ``POST /predict`` -> one action chunk (+ predicted video) for a single ego
                         observation frame.

This client plays each chunk open-loop at 30 Hz and, between chunks, freezes the
body token via the encoder (identical playback to dz_client.py — only the policy
transport differs). It subscribes to the real robot state and streams actions to
the WBC exactly like ``psi_rtc_sonic_client.py`` / ``psix_sonic_client.py``.

Request payload (mirrors eval_g1_openloop.query_server):
  {
      "image":       "<base64 PNG, RGB single ego frame>",
      "prompt":      instruction,          # str
      "domain_name": "g1_sonic_neck" | "g1_sonic_neckless",
      "image_size":  256,                  # server resize target
      "view_point":  "ego_view",
      "state":       [float, ...],         # raw proprio, 43-D (or 45-D w/ neck)
  }

The state layout MUST match training (use_state=True): the g1 policies prepend it
as conditioning row 0. Raw proprio is [hand(L7,R7) | arm(14) | leg(15)] (43-D),
plus [neck(2)] appended when --include-neck (45-D) — same as psix_sonic_client.py.

Server reply: {"action": [T, D], "video": [...]}. Only "action" is used here.
The g1 action layout is [body_token(64) | hand(14) | neck(2)] (80-D; a neckless
model returns 78-D and we zero-pad the neck). We FSQ-quantize the body token and
republish in internal layout hand(14) + neck(2) + token(64), matching the
TokenPublisher / encoder-freeze code below.
"""

import os
import sys
import io
import time
import threading
import json
import base64
import signal
from collections import deque

import cv2
import numpy as np
import zmq
import msgpack
import requests
from PIL import Image

# Add project root to path for imports
_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message,
    build_command_message,
)
from encoder_client import EncoderClient


# ---------------- Configuration ----------------
TASK_INSTRUCTION = "Clean up the table"

# FSQ configuration (must match g1_sonic_client / encoder)
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16

# Control frequency
FREQ_POLICY = 30  # Hz

# Server action layout: [body_token(64) | hand(14) | neck(2)] (80-D). A neckless
# model returns 78-D (no neck). We republish in internal layout
# hand(14) + neck(2) + token(64) = 80, matching TokenPublisher.publish_token and
# the encoder-freeze path below.
HAND_DIM = 14
NECK_DIM = 2
TOKEN_DIM = 64
ACTION_DIM_NECK = 80      # body_token(64) + hand(14) + neck(2)
ACTION_DIM_NECKLESS = 78  # body_token(64) + hand(14)

# Ego frame resize target sent to the server (server resizes to this; matches
# eval_g1_openloop default for g1).
DEFAULT_IMAGE_SIZE = 256

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


def _frame_to_png_b64(frame_rgb: np.ndarray) -> str:
    """[H,W,3] uint8 RGB -> base64 PNG (matches eval_g1_openloop.frame_to_png_b64)."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame_rgb)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------- CosmosPolicyClient (HTTP) ----------------
class CosmosPolicyClient:
    """HTTP client for the Cosmos action-policy server.

    Mirrors eval_g1_openloop's query_server/query_info: each predict() POSTs a
    single ego frame + proprio state to /predict and returns the raw action chunk.
    """

    def __init__(self, server, domain_name, image_size=DEFAULT_IMAGE_SIZE, timeout=1200):
        self._server = server.rstrip("/")
        self._domain_name = domain_name
        self._image_size = image_size
        self._timeout = timeout

    def info(self):
        r = requests.get(self._server + "/info", timeout=30)
        r.raise_for_status()
        return r.json()

    def predict(self, frame_rgb, prompt, state):
        """POST /predict; return the raw predicted action chunk as (T, D) float32.

        frame_rgb: (H, W, 3) uint8 RGB ego frame.
        prompt:    instruction string.
        state:     raw proprio vector (43-D, or 45-D with neck).
        """
        payload = {
            "image": _frame_to_png_b64(frame_rgb),
            "prompt": prompt,
            "domain_name": self._domain_name,
            "image_size": self._image_size,
            "view_point": "ego_view",
            "state": np.asarray(state, dtype=np.float32).tolist(),
        }
        resp = requests.post(self._server + "/predict", json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        return np.asarray(data["action"], dtype=np.float32)  # (T, D)


# ---------------- InstructionManager ----------------
class InstructionManager:
    """Holds the per-stage subtask prompts and advances through them on Enter.

    Cosmos has no goal image, so this only tracks the text subtask; the assembled
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
                internal layout. The neck slot is forwarded separately by
                NeckPublisher; only token + hands go here.
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
class CosmosSonicClient:
    """
    Chunk-based non-RTC Cosmos client. Each /predict returns a chunk of actions;
    the publish loop iterates it at 30 Hz, and the encoder freezes the body token
    between chunks. Identical playback to dz_client.py — the Cosmos policy server
    is the HTTP action-policy server exercised by eval_g1_openloop.py.
    """

    def __init__(self, policy_client, state_subscriber, camera, token_publisher,
                 instruction_manager, task_instruction,
                 include_neck=False, neck_publisher=None, neck_state_reader=None,
                 camera_address=None, record_dir=None, record_fps=15):
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

        # Frame recording: a dedicated recorder thread grabs frames via its own
        # camera socket and only writes them while the robot is executing a real
        # chunk action (self._is_moving set) — freeze frames are skipped.
        self._camera_address = camera_address
        self._record_dir = record_dir
        self._record_fps = record_fps
        self._is_moving = threading.Event()
        self._recorder_thread = None

        # Encoder for frozen token
        self._encoder = EncoderClient(ENCODER_MODEL, mode=0)

        self._running = threading.Event()
        self._sequence_done_event = threading.Event()
        self._pending_chunk = None
        self._chunk_lock = threading.Lock()

        self._inference_thread = None
        self._publish_thread = None

    # ---------- Observation ----------
    def _build_state(self):
        """Assemble the raw proprio conditioning state, same as psix_sonic_client.py.

        Model expects state.joint_positions = [hand(L7,R7) | arm(14) | leg(15)] (43-D),
        plus [neck(2)] appended when --include-neck (45-D). body_q_measured is (29,)
        = [leg/base(15) | arm(14)] in Mujoco order.
        """
        state = self._state_sub.get_state()
        assert state is not None, "Robot state not available"

        body_q = np.array(state["body_q_measured"], dtype=np.float32)                  # (29,)
        left_hand_states = np.array(state["left_hand_q_measured"], dtype=np.float32)    # (7,)
        right_hand_states = np.array(state["right_hand_q_measured"], dtype=np.float32)  # (7,)

        leg_states = body_q[:15]    # base/leg
        arm_states = body_q[15:29]  # arm
        states = np.concatenate(
            (left_hand_states, right_hand_states, arm_states, leg_states), axis=0
        )  # (43,)

        if self._include_neck:
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
            states = np.concatenate((states, neck_state), axis=0)  # (45,)

        return states.astype(np.float32)

    def _build_instruction(self):
        """Assemble "Task: <task>. Subtask: <subtask>" (subtask dropped when empty)."""
        base = str(self._task).strip().lower()
        subtask = str(self._instruction_manager.get_subtask()).strip()
        return f"Task: {base}. Subtask: {subtask}" if subtask else f"Task: {base}"

    def _get_policy_chunk(self):
        """Capture one ego frame + state, POST /predict, and return a chunk of actions
        in internal publish layout hand(14) + neck(2) + token(64), with FSQ quantization
        applied to the body token. Returns None on failure.

        Server returns each action as [body_token(64) | hand(14) | neck(2)] (80-D), or
        [body_token(64) | hand(14)] (78-D) for a neckless model.
        """
        frame = self._camera.get_frame()
        if frame is None:
            print("[Inference] No camera frame available")
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

        state = self._build_state()
        instruction = self._build_instruction()

        # --- DEBUG: dump latest ego frame whenever the instruction stage changes ---
        stage_idx, stage_subtask = self._instruction_manager.get_stage()
        if stage_idx != self._dbg_last_idx:
            self._dbg_last_idx = stage_idx
            cv2.imwrite("/tmp/sent_ego.jpg", frame)
            print(f"[DEBUG] stage idx={stage_idx} | subtask={stage_subtask!r} "
                  f"| instruction={instruction!r} | frame={frame_rgb.shape} "
                  f"| state={state.shape} | dumped /tmp/sent_ego.jpg")

        try:
            chunk = self._policy.predict(frame_rgb, instruction, state)
        except Exception as e:
            print(f"[Inference] /predict request error: {e}")
            return None

        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.shape[-1] not in (ACTION_DIM_NECK, ACTION_DIM_NECKLESS):
            print(f"[Inference] Unexpected action dim: {chunk.shape}, "
                  f"expected (*, {ACTION_DIM_NECKLESS}) or (*, {ACTION_DIM_NECK})")
            return None

        # Server layout [body_token(64) | hand(14) | neck(2)] -> internal
        # hand(14) + neck(2) + token(64). Neckless (78-D) has no neck: zero-pad it.
        token_ori = chunk[:, :TOKEN_DIM]                                # (N, 64)
        hand_joints = chunk[:, TOKEN_DIM:TOKEN_DIM + HAND_DIM]          # (N, 14)
        if chunk.shape[-1] >= TOKEN_DIM + HAND_DIM + NECK_DIM:
            neck = chunk[:, TOKEN_DIM + HAND_DIM:TOKEN_DIM + HAND_DIM + NECK_DIM]  # (N, 2)
        else:
            neck = np.zeros((chunk.shape[0], NECK_DIM), dtype=np.float32)
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
                self._is_moving.set()  # real chunk action -> record frames
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
                    self._is_moving.set()  # resumed real chunk -> record frames
                else:
                    action = frozen_action
                    self._is_moving.clear()  # encoder freeze -> skip frames

            self._token_publisher.publish_token(action)
            if self._neck_publisher is not None:
                neck = action[HAND_DIM:HAND_DIM + NECK_DIM]
                self._neck_publisher.publish(neck[0], neck[1])

            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _recorder_loop(self):
        """Grab ego frames on a dedicated camera socket and save only while the
        robot is executing a real chunk action (self._is_moving). Freeze frames
        (encoder body-token hold between chunks) are skipped.

        Uses its own camera connection: the main self._camera is a zmq.REQ socket
        driven by the inference thread and must not be shared across threads.
        """
        try:
            os.makedirs(self._record_dir, exist_ok=True)
            cam = (ZedNeckCamera if self._include_neck else RSCamera)(self._camera_address)
        except Exception as e:
            print(f"[Recorder] Failed to init ({e}); recording disabled.")
            return

        print(f"[Recorder] Saving moving frames to {self._record_dir} @ {self._record_fps} fps")
        period = 1.0 / max(1, self._record_fps)
        n = 0
        try:
            while self._running.is_set():
                t0 = time.perf_counter()
                if self._is_moving.is_set():
                    try:
                        frame = cam.get_frame()  # BGR
                    except Exception as e:
                        print(f"[Recorder] get_frame error: {e}")
                        frame = None
                    if frame is not None:
                        ts = time.time()
                        path = os.path.join(self._record_dir, f"{ts:.3f}_{n:06d}.jpg")
                        cv2.imwrite(path, frame)
                        n += 1
                sleep_time = period - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            try:
                cam.close()
            except Exception:
                pass
            print(f"[Recorder] Stopped. Saved {n} moving frame(s).")

    # ---------- Lifecycle ----------
    def start(self):
        print("[CosmosSonicClient] Starting...")
        self._running.set()
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()
        if self._record_dir:
            self._recorder_thread = threading.Thread(target=self._recorder_loop, daemon=True)
            self._recorder_thread.start()
        print("[CosmosSonicClient] Started successfully!")
        return True

    def stop(self):
        print("[CosmosSonicClient] Stopping...")
        self._running.clear()
        try:
            self._token_publisher.send_command(start=False, stop=True, planner=True)
        except Exception as e:
            print(f"[CosmosSonicClient] Error sending stop command: {e}")
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
        print("[CosmosSonicClient] Stopped.")


# ---------------- Main ----------------
def main(server, domain_name, image_size, zmq_host, zmq_pub_port, zmq_sub_port,
         zmq_topic, zmq_sub_topic, camera_address, task_instruction, subtasks,
         include_neck=False, neck_pub_host=DEFAULT_NECK_PUB_HOST, neck_pub_port=DEFAULT_NECK_PUB_PORT,
         neck_state_zmq=DEFAULT_NECK_STATE_ZMQ, record_dir=None, record_fps=15):
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

    # Connect to the Cosmos policy server (HTTP action-policy server).
    print(f"[MAIN] Cosmos policy server: {server} (domain={domain_name}, image_size={image_size})")
    policy_client = CosmosPolicyClient(server=server, domain_name=domain_name, image_size=image_size)
    try:
        info = policy_client.info()
        print(f"[MAIN] Server /info: run_name={info.get('run_name')} "
              f"checkpoint={info.get('checkpoint')}")
    except Exception as e:
        print(f"[MAIN] WARNING: could not GET {server}/info ({e}); proceeding anyway...")

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

    client = CosmosSonicClient(
        policy_client=policy_client,
        state_subscriber=state_sub,
        camera=camera,
        token_publisher=token_publisher,
        instruction_manager=instruction_manager,
        task_instruction=task_instruction,
        include_neck=include_neck,
        neck_publisher=neck_publisher,
        neck_state_reader=neck_state_reader,
        camera_address=camera_address,
        record_dir=record_dir,
        record_fps=record_fps,
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

    parser = argparse.ArgumentParser(description="Non-RTC Cosmos HTTP Policy Client")
    parser.add_argument("--server", type=str, default=None,
                        help="Cosmos policy server base URL (e.g. http://localhost:8000). "
                             "Overrides --host/--port when given.")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Cosmos policy server host (used when --server is unset)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Local port reaching the Cosmos HTTP server (port-forwarded 8000)")
    parser.add_argument("--domain-name", type=str, default=None,
                        choices=["g1_sonic_neck", "g1_sonic_neckless"],
                        help="Embodiment tag sent to the server. Default: derived from "
                             "--include-neck (g1_sonic_neck if set, else g1_sonic_neckless).")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
                        help="Server ego-frame resize target (matches training; 256 for g1)")
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
                        help=f"Task instruction sent verbatim (default: {TASK_INSTRUCTION!r})")
    parser.add_argument("--include-neck", action="store_true",
                        help="Neck variant: send 45-dim state (+neck2) and publish the "
                             "action's neck(2) + use the neck-mounted ZED camera. Default (off) "
                             "sends 43-dim state and ignores the action neck.")
    parser.add_argument("--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST,
                        help=f"Neck PUB bind host (default: {DEFAULT_NECK_PUB_HOST})")
    parser.add_argument("--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT,
                        help=f"Neck PUB port (default: {DEFAULT_NECK_PUB_PORT})")
    parser.add_argument("--neck-state-zmq", type=str, default=DEFAULT_NECK_STATE_ZMQ,
                        help=f"Neck-state SUB address (default: {DEFAULT_NECK_STATE_ZMQ})")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="If set, save ego frames here (JPG) ONLY while the robot is "
                             "executing chunk actions; encoder-freeze frames are skipped.")
    parser.add_argument("--record-fps", type=int, default=15,
                        help="Recorder frame rate when --record-dir is set (default: 15)")

    args = parser.parse_args()

    server = args.server if args.server else f"http://{args.host}:{args.port}"
    domain_name = args.domain_name or ("g1_sonic_neck" if args.include_neck else "g1_sonic_neckless")

    task_instruction = args.instruction if args.instruction else TASK_INSTRUCTION
    subtasks = []

    main(
        server=server,
        domain_name=domain_name,
        image_size=args.image_size,
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
        record_dir=args.record_dir,
        record_fps=args.record_fps,
    )
