import os
import sys
import time
import threading
import json
import signal
import struct

import cv2
import numpy as np
import zmq
import msgpack
from websocket import WebSocketApp

# Add project root to path for imports
_GROOT_ROOT = os.path.expanduser("~/hsc/GR00T-WholeBodyControl")
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message,
    build_command_message,
)

# ---------------- Configuration ----------------
# TASK_INSTRUCTION = "grasp the pink chip can and place it into the orange plate"
TASK_INSTRUCTION = "pick up the green grapes and place it into the green bowl"

# FSQ configuration (must match g1_sonic_client / encoder)
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16

# cap obs send rate; server control loop runs at 30Hz, sending faster just floods it
OBS_SEND_INTERVAL = 1.0 / 30.0

# Action/state layout (must match g1_sonic_client conventions):
#   default:        states(43) = hand(14) + arm(14) + leg(15)
#                    action(78) = hand_joints(14) + token(64)
#   --include-neck:  states(45) = states(43) + neck(2) [appended at the end]
#                    action(80) = hand_joints(14) + token(64) + neck(2) [neck is the last 2 dims]
HAND_DIM = 14
NECK_DIM = 2
TOKEN_DIM = 64
ACTION_DIM_DEFAULT = 78
ACTION_DIM_NECK = 80

# Neck publisher configuration (to G1 NeckMotor, matches pose_publisher.py wire format)
DEFAULT_NECK_PUB_HOST = "*"
DEFAULT_NECK_PUB_PORT = 5570

# Neck state subscriber (ZMQ SUB <- realsense_server.py on the robot, port 5560)
# JSON `[yaw_rad, pitch_rad]` of the Dynamixel present-position read each tick.
DEFAULT_NECK_STATE_ZMQ = "tcp://192.168.123.164:5560"


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    quantized = np.clip(quantized, fsq_min, fsq_max)
    return quantized

# Encode the current robot pose into a 64-D sonic body token (same encoder the non-RTC
# client uses to "freeze" the body token), to seed the server's first-chunk RTC prev-action.
from ours.common.encoder_client import EncoderClient
ENCODER_MODEL = "gear_sonic_deploy/policy/release/model_encoder.onnx"
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)
def _mujoco29_to_isaaclab29(qpos):
    return np.asarray(qpos, dtype=np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB_DOF].copy()

# ---------------- Serialization utilities ----------------
from base64 import b64encode, b64decode
from numpy.lib.format import dtype_to_descr, descr_to_dtype


def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(dct["shape"]) if dct["shape"] else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    else:
        return data


# ---------------- SubgoalManager ----------------
class SubgoalManager:
    """
    Holds the current subgoal image and advances through a fixed sequence on Enter.

    Sequence: every image in color_subgoal/ in order, then color[-1] as the final one.
    Further Enter presses after the last one are no-ops.
    """

    _IMG_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self, episode_dir, subtasks=None):
        self.episode_dir = episode_dir
        subgoal_dir = os.path.join(episode_dir, "color_subgoal")
        color_dir = os.path.join(episode_dir, "color")

        subgoal_files = self._list_images(subgoal_dir)
        color_files = self._list_images(color_dir)

        self.paths = list(subgoal_files)
        if color_files:
            self.paths.append(color_files[-1])

        if not self.paths:
            raise ValueError(f"[Subgoal] No images found under {episode_dir}")

        self._images = [self._load(p) for p in self.paths]
        self._idx = 0
        self._lock = threading.Lock()

        # Per-stage subtask prompts, parallel to self.paths. Advanced together with the
        # subgoal image on each Enter (the server expects the client to switch BOTH the
        # goal image and the subtask over time). Missing entries -> "" so the server
        # falls back to task-only conditioning for that stage.
        subtasks = list(subtasks or [])
        self._subtasks = [
            (subtasks[i] if i < len(subtasks) else "") for i in range(len(self.paths))
        ]
        if subtasks and len(subtasks) != len(self.paths):
            print(f"[Subgoal] WARNING: {len(subtasks)} subtasks but {len(self.paths)} "
                  f"subgoal stages; padded/truncated to match.")

        print(f"[Subgoal] Loaded {len(self._images)} subgoal images from {episode_dir}:")
        for i, p in enumerate(self.paths):
            print(f"  [{i}] {p}  | subtask: {self._subtasks[i]!r}")
        print(f"[Subgoal] Current index: 0  (press Enter to advance)")

    @classmethod
    def _list_images(cls, d):
        if not os.path.isdir(d):
            return []
        return sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith(cls._IMG_EXTS)
        )

    @staticmethod
    def _load(path):
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"[Subgoal] Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.uint8)

    def get(self):
        with self._lock:
            return self._images[self._idx]

    def get_subtask(self):
        with self._lock:
            return self._subtasks[self._idx]

    def get_stage(self):
        """(idx, subgoal_path, subtask) for the current stage — for debugging."""
        with self._lock:
            return self._idx, self.paths[self._idx], self._subtasks[self._idx]

    def advance(self):
        with self._lock:
            if self._idx < len(self._images) - 1:
                self._idx += 1
                print(f"[Subgoal] -> index {self._idx}: {self.paths[self._idx]}  "
                      f"| subtask: {self._subtasks[self._idx]!r}")
            else:
                print(f"[Subgoal] Already at last subgoal (index {self._idx})")


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

            # Strip topic prefix
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
        """Return the latest robot state dict, or None if not yet received."""
        with self._lock:
            return self._latest_state

    def stop(self):
        self._running = False
        self._thread.join(timeout=0.5)
        self._socket.close(linger=0)
        self._context.term()


# ---------------- TokenPublisher ----------------
class TokenPublisher:
    """ZMQ publisher for token-only streaming (Protocol v4), same as g1_sonic_client."""

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
            action: np.ndarray of shape (78,) — hand_joints(14) + token(64)
        """
        action = action.astype(np.float32).reshape(1, -1)
        pose_data = {
            "token_state": action[:, :64],       # (1, 64)
            "left_hand_joints": action[:, 64:71],    # (1, 7)
            "right_hand_joints": action[:, 71:78], # (1, 7)
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=4)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close()
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


# ---------------- Global state ----------------
running = threading.Event()
running.set()


# ---------------- RTCWebSocketClient ----------------
class RTCWebSocketClient:
    def __init__(self, server_url, state_subscriber, camera, token_publisher, subgoal_manager,
                 task_instruction, include_neck=False, neck_publisher=None, neck_state_reader=None):
        self.server_url = server_url
        self._running = True
        self._connected = threading.Event()
        self._ws = None
        self._send_lock = threading.Lock()
        self.start_time = time.time()

        self._state_sub = state_subscriber
        self._camera = camera
        self._token_publisher = token_publisher
        self._subgoal_manager = subgoal_manager
        self._task = task_instruction
        self._dbg_last_idx = -1  # debug: track stage changes for image/idx dumps
        self._include_neck = include_neck
        self._neck_publisher = neck_publisher
        self._neck_state_reader = neck_state_reader
        # Encoder for the first-frame current-pose token (seeds the server first-chunk RTC).
        try:
            self._encoder = EncoderClient(ENCODER_MODEL, mode=0)
        except Exception as e:
            print(f"[init-prev] encoder load failed ({e}); first chunk falls back to unconditioned")
            self._encoder = None
        self._sent_init_prev = False

    def execute_action(self, action):
        """
        Map the server action -> robot command and publish via Protocol v4.

        Server action layout is [hand_joints(14) | body_token(64)] (78-D default),
        or [hand_joints(14) | body_token(64) | neck(2)] (80-D --include-neck).
        publish_token expects [token(64) | left_hand(7) | right_hand(7)].
        """
        if action.ndim > 1:
            action = action[0]

        hand_joints = action[:HAND_DIM]
        token_ori = action[HAND_DIM:HAND_DIM + TOKEN_DIM]
        token_qtz = fsq_quantize(token_ori)

        action_out = np.concatenate([token_qtz, hand_joints])  # [token(64), LH(7), RH(7)]
        self._token_publisher.publish_token(action_out)

        if self._include_neck and self._neck_publisher is not None:
            neck = action[HAND_DIM + TOKEN_DIM:HAND_DIM + TOKEN_DIM + NECK_DIM]
            self._neck_publisher.publish(neck[0], neck[1])

    def _on_open(self, ws):
        print("[client] Connected!")
        self._connected.set()

    def _on_message(self, ws, message):
        interval = time.time() - self.start_time
        self.start_time = time.time()
        print(f"[client] recv_action interval: {interval:.3f}s")

        try:
            data = json.loads(message)
            action_data = data.get("action")
            version = data.get("version", -1)

            if action_data is not None:
                action = convert_numpy_in_dict(action_data, numpy_deserialize)
                if isinstance(action, np.ndarray):
                    self.execute_action(action)
                    print(f"[client] Received action, version={version}, shape={action.shape}")

        except Exception as e:
            print(f"[client] Message processing error: {e}")

    def _on_error(self, ws, error):
        print(f"[client] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[client] Connection closed: {close_status_code} - {close_msg}")
        self._running = False
        running.clear()

    def _send_thread(self):
        print("[client] Send thread started, waiting for connection...")
        self._connected.wait()
        print("[client] Connected, starting observation loop")

        prev_tick = time.perf_counter()

        while self._running and running.is_set():
            start = time.time()
            try:
                # Get robot state
                state = self._state_sub.get_state()
                if state is None:
                    print("[client] No robot state yet, waiting...")
                    time.sleep(0.1)
                    continue

                body_q    = np.array(state["body_q_measured"],   dtype=np.float32)  # (29,) = [leg/base(15) | arm(14)]
                left_hand_states = np.array(state["left_hand_q"], dtype=np.float32)   # (7,)
                right_hand_states = np.array(state["right_hand_q"], dtype=np.float32) # (7,)

                # Model expects state.joint_positions = [hand(L7,R7) | arm(14) | leg(15)].
                leg_states = body_q[:15]    # base/leg
                arm_states = body_q[15:29]  # arm
                states = np.concatenate(
                    (left_hand_states, right_hand_states, arm_states, leg_states), axis=0
                )  # (43,)

                if self._include_neck:
                    neck_latest = self._neck_state_reader.get_latest()
                    neck_state = (
                        np.asarray(neck_latest, dtype=np.float32)
                        if neck_latest is not None
                        else np.zeros(NECK_DIM, dtype=np.float32)
                    )
                    states = np.concatenate((states, neck_state), axis=0)  # (45,)

                # Get camera frame
                frame = self._camera.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.uint8)

                # Get current subgoal frame
                subgoal_frame = self._subgoal_manager.get()

                # --- DEBUG: confirm what is actually sent to the model ---
                # Dumps ego + goal to /tmp every time the stage (idx) changes, and logs
                # idx / subgoal filename / subtask so you can verify nothing auto-advanced
                # and the goal image is the one you expect.
                stage_idx, stage_path, stage_subtask = self._subgoal_manager.get_stage()
                if stage_idx != self._dbg_last_idx:
                    self._dbg_last_idx = stage_idx
                    cv2.imwrite("/tmp/sent_ego.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    cv2.imwrite("/tmp/sent_goal.jpg", cv2.cvtColor(subgoal_frame, cv2.COLOR_RGB2BGR))
                    print(f"[DEBUG] stage idx={stage_idx} | goal={os.path.basename(stage_path)} "
                          f"| subtask={stage_subtask!r} | dumped /tmp/sent_ego.jpg /tmp/sent_goal.jpg")

                # Build observation payload. Image keys MUST match the server's repack:
                #   ego image  -> repack.image_keys[0]  == "video.egocentric"
                #   goal image -> repack.subgoal_key[0] == "subgoal.egocentric"
                img_obs = {
                    "video.egocentric": frame,
                    "subgoal.egocentric": subgoal_frame,
                }
                state_obs = {"states": states}

                # First frame only: encode current pose -> 64-D sonic token, assemble a raw
                # pseudo prev-action [LH7 | RH7 | token64] (| neck2 if --include-neck) for the
                # server's first-chunk RTC.
                if not self._sent_init_prev and self._encoder is not None:
                    qpos = _mujoco29_to_isaaclab29(state["body_q_measured"])           # (29,)
                    base_quat = np.asarray(state.get("base_quat_measured", [1, 0, 0, 0]),
                                           dtype=np.float32).reshape(4)
                    jp = np.tile(qpos, (10, 1)).astype(np.float32)                     # (10,29)
                    jv = np.zeros((10, 29), dtype=np.float32)
                    bq = np.tile(base_quat, (10, 1)).astype(np.float32)                # (10,4)
                    enc_token = np.asarray(self._encoder.encode(jp, jv, bq),
                                           dtype=np.float32).reshape(64)               # (64,)
                    init_prev_action = np.concatenate(
                        [left_hand_states, right_hand_states, enc_token]).astype(np.float32)  # (78,)
                    if self._include_neck:
                        init_prev_action = np.concatenate(
                            [init_prev_action, states[-NECK_DIM:]]).astype(np.float32)  # (80,)
                    state_obs["init_prev_action"] = init_prev_action
                    self._sent_init_prev = True
                    print(f"[init-prev] first-frame pseudo prev-action sent; "
                          f"token range=[{enc_token.min():.3f},{enc_token.max():.3f}]")

                # Assemble the instruction string here (server feeds it to the VLM verbatim).
                # Must match the training format: "Task: <task>. Subtask: <subtask>"
                # (task lowercased, subtask dropped when empty).

                task = str(self._task).strip().lower()
                subtask = str(self._subgoal_manager.get_subtask()).strip()
                instruction = f"Task: {task}. Subtask: {subtask}" if subtask else f"Task: {task}"

                payload = {
                    "image": img_obs,
                    "state": state_obs,
                    "gt_action": None,
                    "dataset_name": None,
                    "instruction": instruction,
                    "history": None,
                    "condition": None,
                    "timestamp": None,
                }
                payload = convert_numpy_in_dict(payload, numpy_serialize)
                message = json.dumps(payload)

                # Send (thread-safe)
                with self._send_lock:
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send(message)
                    else:
                        print("[client] WebSocket not connected, skipping send")
                        break

            except Exception as e:
                print(f"[client] Send error: {e}")
                break

            sleep_time = max(0, OBS_SEND_INTERVAL - (time.time() - start))
            time.sleep(sleep_time)

            now = time.perf_counter()
            interval = now - prev_tick
            prev_tick = now
            print(f"[client] send interval: {interval:.3f}s")

        print("[client] Send thread stopped")

    def run(self):
        print(f"[client] Connecting to {self.server_url}")

        self._ws = WebSocketApp(
            self.server_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        send_thread = threading.Thread(target=self._send_thread, daemon=True)
        send_thread.start()

        self._ws.run_forever()

        self._running = False
        send_thread.join(timeout=0.5)
        print("[client] Client stopped")

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()


# ---------------- Main ----------------
def main(server_url, zmq_host, zmq_pub_port, zmq_sub_port, zmq_topic, zmq_sub_topic,
         camera_address, episode_dir, task_instruction, subtasks,
         include_neck=False, neck_pub_host=DEFAULT_NECK_PUB_HOST, neck_pub_port=DEFAULT_NECK_PUB_PORT,
         neck_state_zmq=DEFAULT_NECK_STATE_ZMQ):
    print("[MAIN] Initializing components...")

    # 1. Initialize token publisher (ZMQ PUB, Protocol v4)
    token_publisher = TokenPublisher(host="*", port=zmq_pub_port, topic=zmq_topic)
    print(f"[MAIN] TokenPublisher bound on port {zmq_pub_port}, topic='{zmq_topic}'")

    # 2. Initialize robot state subscriber (ZMQ SUB)
    state_sub = RobotStateSubscriber(host=zmq_host, port=zmq_sub_port, topic=zmq_sub_topic)
    print(f"[MAIN] State subscriber connected to {zmq_host}:{zmq_sub_port}, topic='{zmq_sub_topic}'")

    # 3. Initialize camera (neck-mounted ZED when --include-neck, else RealSense)
    camera = ZedNeckCamera(address=camera_address) if include_neck else RSCamera(address=camera_address)
    print(f"[MAIN] Camera connected to {camera_address} (include_neck={include_neck})")

    # 3c. Initialize neck publisher/state-reader when --include-neck
    neck_publisher = None
    neck_state_reader = None
    if include_neck:
        neck_publisher = NeckPublisher(host=neck_pub_host, port=neck_pub_port)
        neck_state_reader = NeckStateReader(neck_state_zmq)
        print(f"[MAIN] Neck publisher bound on {neck_pub_host}:{neck_pub_port}, "
              f"state reader connected to {neck_state_zmq}")

    # 3b. Initialize subgoal manager (subgoal images + per-stage subtask prompts)
    subgoal_manager = SubgoalManager(episode_dir=episode_dir, subtasks=subtasks)
    print(f"[MAIN] Task instruction: {task_instruction!r}")

    # 4. Wait briefly for ZMQ PUB socket to establish connections
    time.sleep(1.0)

    # 5. Send start command (planner mode for token streaming)
    token_publisher.send_command(start=True, stop=False, planner=True)

    # 6. Wait for first robot state
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

    # 7. Start WebSocket client
    client = RTCWebSocketClient(
        server_url=server_url,
        state_subscriber=state_sub,
        camera=camera,
        token_publisher=token_publisher,
        subgoal_manager=subgoal_manager,
        task_instruction=task_instruction,
        include_neck=include_neck,
        neck_publisher=neck_publisher,
        neck_state_reader=neck_state_reader,
    )

    # 7b. Start stdin listener: each Enter advances the subgoal index
    def stdin_listener():
        print("[MAIN] Press Enter to advance subgoal image.")
        while running.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:  # EOF
                break
            subgoal_manager.advance()

    t_stdin = threading.Thread(target=stdin_listener, daemon=True)
    t_stdin.start()

    def websocket_thread():
        client.run()
        print("[WS] WebSocket thread stopped")

    t_ws = threading.Thread(target=websocket_thread, daemon=True)
    t_ws.start()

    print("[MAIN] Running. Ctrl+C to stop.")

    # 8. Wait for shutdown
    def signal_handler(sig, frame):
        print("\n[MAIN] Caught signal, shutting down...")
        running.clear()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[MAIN] Caught Ctrl+C, exiting...")
        running.clear()

    # 9. Shutdown
    print("[MAIN] Shutting down...")
    client.stop()

    # Send stop command
    try:
        token_publisher.send_command(start=False, stop=True, planner=True)
    except Exception as e:
        print(f"[MAIN] Error sending stop command: {e}")

    state_sub.stop()
    token_publisher.stop()
    if neck_publisher is not None:
        neck_publisher.stop()
    if neck_state_reader is not None:
        neck_state_reader.stop()
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLA Policy Inference with WBC Stabilization")
    parser.add_argument("--host", type=str, default="localhost",
                        help="VLA policy server host")
    parser.add_argument("--port", type=int, default=8014,
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
    parser.add_argument("--episode-dir", type=str,
                        default="/mnt/data/weiduo/heng/GR00T-WholeBodyControl/data/real_pick_place/pick_place_eggplant_plastic_box/episode_1",
                        help="Episode folder containing color/ and color_subgoal/ for subgoal images")
    parser.add_argument("--prompts-json", type=str,
                        default="/mnt/data/weiduo/heng/GR00T-WholeBodyControl/data/real_pick_place/prompts.json",
                        help="JSON mapping task-key -> {task_description, subtasks[]}")
    parser.add_argument("--task-key", type=str, default="pick_place_eggplant_plastic_box",
                        help="Key into prompts.json (e.g. pick_place_1); selects the "
                             "task_description and the per-stage subtask prompts.")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override task instruction (else taken from prompts.json[task-key])")
    parser.add_argument("--include-neck", action="store_true",
                        help="Neck variant: states 45-dim (+neck2 appended), action 80-dim "
                             "(hand14 + token64 + neck2 appended), init_prev_action 80-dim "
                             "(+ first-frame neck2 appended). Also swaps RealSense for the "
                             "neck-mounted ZED camera. Default (off) keeps the legacy 43/78-dim path.")
    parser.add_argument("--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST,
                        help=f"Neck PUB bind host (default: {DEFAULT_NECK_PUB_HOST})")
    parser.add_argument("--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT,
                        help=f"Neck PUB port (default: {DEFAULT_NECK_PUB_PORT})")
    parser.add_argument("--neck-state-zmq", type=str, default=DEFAULT_NECK_STATE_ZMQ,
                        help=f"Neck-state SUB address (default: {DEFAULT_NECK_STATE_ZMQ})")

    args = parser.parse_args()

    # Resolve task instruction + per-stage subtasks from prompts.json (--task-key).
    task_instruction = args.instruction
    subtasks = []
    if args.task_key:
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

    server_url = f"ws://{args.host}:{args.port}/ws"
    main(
        server_url=server_url,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_sub_port=args.zmq_sub_port,
        zmq_topic=args.zmq_topic,
        zmq_sub_topic=args.zmq_sub_topic,
        camera_address=args.camera_address,
        episode_dir=args.episode_dir,
        task_instruction=task_instruction,
        subtasks=subtasks,
        include_neck=args.include_neck,
        neck_pub_host=args.neck_pub_host,
        neck_pub_port=args.neck_pub_port,
        neck_state_zmq=args.neck_state_zmq,
    )
