"""RTC client for the cosmos3 action-policy server (action_policy_server_rt.py).

Structure mirrors psi_rtc_sonic_client.py (same robot I/O: ZMQ state subscriber,
camera REQ/REP stream, Protocol-v4 token publisher, neck PUB/SUB); only the policy
request/response part is adapted to the cosmos3 RTC server's wire contract
(cosmos_framework/rtc/helpers.py RequestMessage / ResponseMessage, reference client
cosmos_framework/rtc/openloop_ws_client.py):

  request  = {image: {"observation.images.egocentric": (H,W,3) uint8},
              instruction: str (plain task text; server assembles the full VLM prompt),
              history: {"reset": True} on the FIRST frame of a connection, {} after,
              state: {"observation.state": (45,) float32
                       = [left_hand(7) | right_hand(7) | arm(14) | leg(15) | neck(2)]},
              condition: {}, gt_action: [],
              dataset_name: DOMAIN (e.g. "g1_sonic_neck_zedmini") -> selects action head,
              timestamp: str(time.time())}
  response = {action: (1,80) float32 DENORMALIZED [body_token(64) | hand(14) | neck(2)],
              err, traj_image, version}

The 80-D cosmos3 action layout equals the psi convention ([token64 | hand14 | neck2]),
so execute_action (FSQ-quantize token -> Protocol v4, neck -> NeckPublisher) is
unchanged from psi_rtc_sonic_client.py.

Server (workstation, fastest path):
    PSIX_QUANTIZE=fp8 PSIX_ENCODE_COND_ONLY=1 PSIX_PROFILE=1 \
    python cosmos_framework/rtc/action_policy_server_rt.py \
        --run-dir <run_dir> --sampler euler --num-steps 3

Client (this file, robot-side host):
    python cosmos3_rtc_sonic_client.py --host <server-ip> --port 8000 \
        --instruction "Pick up the eggplant and place it in the basket."

Add --record-lerobot to save the run as a LeRobot v2.0 eval episode (same recorder
and dataset layout as psi_rtc_sonic_client.py; see lerobot_eval_recorder.py).
"""

import os
import re
import sys
import time
import threading
import json
import signal

import cv2
import numpy as np
import zmq
import msgpack
from websocket import WebSocketApp

from lerobot_eval_recorder import (
    DEFAULT_SRC_DATASET,
    EvalDatasetRecorder,
    dataset_action,
    dataset_state,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message,
    build_command_message,
)

# ---------------- Configuration ----------------

# TASK_INSTRUCTION = "pick up the gray hippo toy and place it into the orange bowl"
# TASK_INSTRUCTION = "pick up the banana and place it into the wooden box"
# TASK_INSTRUCTION = "Pick up the eggplant and place it in the basket."
# TASK_INSTRUCTION = "pick up the green grapes and place it into the green bowl"

# TASK_INSTRUCTION = "hold the dustpan and sweep the white paper scraps into it with the brush"
# TASK_INSTRUCTION = "hold the dustpan and sweep the yellow and green bottle caps into it with the brush"
# TASK_INSTRUCTION = "hold the dustpan and sweep the black plastic pieces into it with the brush"

# TASK_INSTRUCTION = "grasp the backrest of the chair and push it straight under the table"
# TASK_INSTRUCTION = "grasp the backrest of the chair, turn left, and push it under the table"
# TASK_INSTRUCTION = "grasp the backrest of the chair, turn right, and push it under the table"

# TASK_INSTRUCTION = "pick up the foil bag and turn left and throw it into the trash can"
# TASK_INSTRUCTION = "pick up the red snack box and turn left and throw it into the trash can"
# TASK_INSTRUCTION = "pick up the snack bag and turn left and throw it into the trash can"
# TASK_INSTRUCTION = "pick up the paper ball and turn left and throw it into the trash can"
# TASK_INSTRUCTION = "pick up the snack bag and turn right and throw it into the trash can"
# TASK_INSTRUCTION = "pick up the red snack box and turn right and throw it into the trash can"
# TASK_INSTRUCTION = "pick up the foil bag and turn right and throw it into the trash can"
# TASK_INSTRUCTION = "pick up the paper ball and turn right and throw it into the trash can"

# TASK_INSTRUCTION = "kneel down, hook the beige shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_INSTRUCTION = "kneel down, hook the purple shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_INSTRUCTION = "kneel down, hook the white shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_INSTRUCTION = "kneel down, hook the blue shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"

# TASK_INSTRUCTION = "gather up the yellow shirt and turn right and put it into the laundry basket"
# TASK_INSTRUCTION = "gather up the gray shirt and turn right and put it into the laundry basket"
# TASK_INSTRUCTION = "gather up the yellow shirt with right hand and turn right and put it into the laundry basket"
# TASK_INSTRUCTION = "gather up the black trousers and turn right and put it into the laundry basket"
# TASK_INSTRUCTION = "gather up the yellow shirt and gray shirt and turn right and put them into the laundry basket"
# TASK_INSTRUCTION = "gather up the gray shirt with right hand and turn right and put it into the laundry basket"
# TASK_INSTRUCTION = "gather up the gray shirt and turn right and put it into the laundry basket"
# TASK_INSTRUCTION = "gather up the white shirt and turn right and put it into the laundry basket"
# TASK_INSTRUCTION = "gather up the black trousers and yellow shirt and turn right and put them into the laundry basket"
# TASK_INSTRUCTION = "gather up the yellow shirt and turn right and put it into the laundry basket"

# TASK_INSTRUCTION = "kneel down and scoop up the white pillow near the bed and put it at the head of the bed"
# TASK_INSTRUCTION = "kneel down and scoop up the beige pillow near the bed and put it at the head of the bed"
# TASK_INSTRUCTION = "kneel down and scoop up the beige pillow near the kitchen island and turn left and put it at the head of the bed"
# TASK_INSTRUCTION = "kneel down and scoop up the white pillow near the kitchen island and turn left and put it at the head of the bed"
# TASK_INSTRUCTION = "kneel down and scoop up the white pillow near the shoe rack and turn around and put it at the head of the bed"
# TASK_INSTRUCTION = "kneel down and scoop up the beige pillow near the shoe rack and turn around and put it at the head of the bed"
# TASK_INSTRUCTION = "kneel down and scoop up the beige pillow near the shoe rack and turn around and put it at the head of the bed"
# TASK_INSTRUCTION = "kneel down and scoop up the beige pillow near the laundry basket and turn around and put it at the head of the bed"
# TASK_INSTRUCTION = "kneel down and scoop up the white pillow near the laundry basket and turn around and put it at the head of the bed"

# TASK_INSTRUCTION = "open the top-right door of the cabinet, grab the purple flower, turn left, and place it at the top-right corner of the table"
# TASK_INSTRUCTION = "open the top-right door of the cabinet, grab the orange flower, turn left, and place it at the top-right corner of the table"
# TASK_INSTRUCTION = "open the top-right door of the cabinet, grab the white flower, turn left, and place it at the top-right corner of the table"
# TASK_INSTRUCTION = "open the top-right door of the cabinet, grab the pink flower, turn left, and place it at the top-right corner of the table"
# TASK_INSTRUCTION = "open the top-right door of the cabinet, grab the pink flower, turn left, and place it at the top-right corner of the table"
# TASK_INSTRUCTION = "open the top-left door of the cabinet, grab the white flower, close the cabinet door, turn left, and place it at the top-right corner of the table"
# TASK_INSTRUCTION = "open the top-left door of the cabinet, grab the pink flower, close the cabinet door, turn left, and place it at the top-right corner of the table"
# TASK_INSTRUCTION = "open the top-left door of the cabinet, grab the purple flower, close the cabinet door, turn left, and place it at the top-right corner of the table"
# TASK_INSTRUCTION = "open the top-left door of the cabinet, grab the orange flower, close the cabinet door, turn left, and place it at the top-right corner of the table"
# TASK_INSTRUCTION = "open the top-left door of the cabinet, grab the orange flower, close the cabinet door, turn left, and place it at the top-right corner of the table"


# TASK_INSTRUCTION = "grasp the yellow bottle, open the top drawer of the kitchen island, place the bottle inside, and close the drawer"
# TASK_INSTRUCTION = "grasp the green drink bottle, open the top drawer of the kitchen island, place the bottle inside, and close the drawer"
TASK_INSTRUCTION = "grasp the silver can, open the top drawer of the kitchen island, place the bottle inside, and close the drawer"

# cosmos3 embodiment domain -> action head + (de)normalization stats on the server.
#   g1_sonic_neck_zedmini / g1_sonic_neck_realsense: 80-D, neck active, 45-D state
#   g1_sonic_neckless: 80-D (neck dims inactive), no neck I/O
DEFAULT_DOMAIN = "g1_sonic_neck_zedmini"

# FSQ configuration (must match g1_sonic_client / encoder)
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16

# Action/state layout (identical to psi_rtc_sonic_client conventions):
#   action(80) = body_token(64) + hand_joints(14) [+ neck(2) when neck domain]
#   states(45) = hand(L7,R7) + arm(14) + leg(15) + neck(2)   (43 without neck)
HAND_DIM = 14
NECK_DIM = 2
TOKEN_DIM = 64

# cosmos3 request keys (openloop_ws_client / eval_g1_openloop conventions)
EGO_KEY = "observation.images.egocentric"
STATE_KEY = "observation.state"

# Neck publisher configuration (to G1 NeckMotor, matches pose_publisher.py wire format)
DEFAULT_NECK_PUB_HOST = "*"
DEFAULT_NECK_PUB_PORT = 5570

# Neck state subscriber (ZMQ SUB <- realsense_server.py on the robot, port 5560)
# JSON `[yaw_rad, pitch_rad]` of the Dynamixel present-position read each tick.
DEFAULT_NECK_STATE_ZMQ = "tcp://192.168.123.164:5560"

PAD_IMAGE_MULTIPLE = 16
_pad_logged = False

def pad_to_train_height(frame):
    """Pad the bottom with black rows so the height is a multiple of 16."""
    global _pad_logged
    if frame is None or not PAD_IMAGE_MULTIPLE:
        return frame
    h, w = frame.shape[:2]
    pad = (-h) % PAD_IMAGE_MULTIPLE
    if pad == 0:
        return frame
    if not _pad_logged:
        print(f"[client] padding camera frames {w}x{h} -> {w}x{h + pad} "
              f"({pad} black rows at the bottom, as in the training videos)")
        _pad_logged = True
    return cv2.copyMakeBorder(frame, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    quantized = np.clip(quantized, fsq_min, fsq_max)
    return quantized

def _task_slug(instruction, max_len=64):
    """Filesystem-safe folder name derived from the task instruction."""
    slug = re.sub(r"[^a-z0-9]+", "-", (instruction or "task").lower()).strip("-")
    return (slug[:max_len].rstrip("-") or "task")


# ---------------- Serialization utilities ----------------
# Same "__numpy__" base64 wire format as cosmos_framework/rtc/helpers.py.
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
    """Neck-mounted ZED camera (neck domains). Server reply is 4-part
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
            action: np.ndarray of shape (78,) — token(64) + hand_joints(14)
        """
        action = action.astype(np.float32).reshape(1, -1)
        pose_data = {
            "token_state": action[:, :64],           # (1, 64)
            "left_hand_joints": action[:, 64:71],    # (1, 7)
            "right_hand_joints": action[:, 71:78],   # (1, 7)
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
    def __init__(self, server_url, state_subscriber, camera, token_publisher,
                 domain_name=DEFAULT_DOMAIN, target_hz=30.0,
                 include_neck=False, neck_publisher=None, neck_state_reader=None,
                 recorder=None):
        self.server_url = server_url
        self._running = True
        self._connected = threading.Event()
        self._ws = None
        self._send_lock = threading.Lock()
        self.start_time = time.time()

        self._state_sub = state_subscriber
        self._camera = camera
        self._token_publisher = token_publisher
        self._domain_name = domain_name
        self._target_hz = target_hz
        self._include_neck = include_neck
        self._neck_publisher = neck_publisher
        self._neck_state_reader = neck_state_reader
        # First frame of a connection carries history={"reset": True}: the cosmos3
        # server (re)seeds its image/state history windows and starts the control loop.
        self._sent_first = False
        # Newest observation the send thread built, consumed by execute_action so each
        # recorded row is (what the policy saw, what it did) rather than two independently
        # sampled streams. Held under a lock: written by the send thread, read by the ws one.
        self._recorder = recorder
        self._obs_lock = threading.Lock()
        self._latest_obs = None

    def execute_action(self, action):
        """
        Map the server action -> robot command and publish via Protocol v4.

        cosmos3 action layout is [body_token(64) | hand_joints(14)] (+ [neck(2)] on
        neck domains) — same as psi. publish_token expects [token(64) | hand(14)].
        The action arrives DENORMALIZED (the server applies the domain's minmax stats).
        """
        if action.ndim > 1:
            action = action[0]

        token_ori = action[:TOKEN_DIM]
        hand_joints = action[TOKEN_DIM:TOKEN_DIM + HAND_DIM]
        token_qtz = fsq_quantize(token_ori)

        action_out = np.concatenate([token_qtz, hand_joints])
        self._token_publisher.publish_token(action_out)

        neck = (action[TOKEN_DIM + HAND_DIM:TOKEN_DIM + HAND_DIM + NECK_DIM]
                if self._include_neck else None)
        if neck is not None and self._neck_publisher is not None:
            self._neck_publisher.publish(neck[0], neck[1])

        # Record after publishing, so nothing sits between the action and the wire.
        if self._recorder is not None:
            with self._obs_lock:
                obs = self._latest_obs
            if obs is not None:
                frame_bgr, state = obs
                # Dataset layout, not the server's: hand + neck + token, with the
                # QUANTIZED token, i.e. exactly what the robot executed.
                self._recorder.add_frame(
                    frame_bgr, state, dataset_action(hand_joints, token_qtz, neck))

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
                    self.execute_action(np.asarray(action, dtype=np.float32))
                    print(f"[client] Received action, version={version}, shape={action.shape}")

        except Exception as e:
            print(f"[client] Message processing error: {e}")

    def _on_error(self, ws, error):
        print(f"[client] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[client] Connection closed: {close_status_code} - {close_msg}")
        self._running = False
        running.clear()

    def _build_payload(self, frame_rgb, states):
        """One cosmos3 RequestMessage-shaped dict, serialized to JSON.

        Field-for-field mirror of openloop_ws_client.build_request (the reference
        cosmos3 client); numpy arrays use the shared "__numpy__" base64 format.
        """
        reset = not self._sent_first
        payload = {
            "image": {EGO_KEY: frame_rgb},                 # {cam: (H,W,3) uint8}
            "instruction": TASK_INSTRUCTION,               # plain task text; server builds the VLM prompt
            "history": {"reset": True} if reset else {},
            "state": {STATE_KEY: states},                  # single raw vector; server normalizes
            "condition": {},
            "gt_action": [],
            "dataset_name": self._domain_name,             # -> predict_policy domain_name
            "timestamp": str(time.time()),
        }
        payload = convert_numpy_in_dict(payload, numpy_serialize)
        return json.dumps(payload)

    def _send_thread(self):
        print("[client] Send thread started, waiting for connection...")
        self._connected.wait()
        print("[client] Connected, starting observation loop")

        interval_target = 1.0 / self._target_hz
        prev_tick = time.perf_counter()

        while self._running and running.is_set():
            tick_start = time.perf_counter()
            try:
                # Get robot state (latest only, no history)
                state = self._state_sub.get_state()
                if state is None:
                    print("[client] No robot state yet, waiting...")
                    time.sleep(0.1)
                    continue

                body_q = np.array(state["body_q_measured"], dtype=np.float32)      # (29,) = [leg/base(15) | arm(14)]
                left_hand_states = np.array(state["left_hand_q"], dtype=np.float32)   # (7,)
                right_hand_states = np.array(state["right_hand_q"], dtype=np.float32) # (7,)

                # cosmos3 observation.state = [hand(L7,R7) | arm(14) | leg(15)] (+ neck(2)).
                leg_states = body_q[:15]    # base/leg
                arm_states = body_q[15:29]  # arm
                states = np.concatenate(
                    # (left_hand_states, right_hand_states, arm_states, leg_states), axis=0
                    (leg_states, arm_states, left_hand_states, right_hand_states), axis=0
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
                frame = pad_to_train_height(frame)      # what the policy was trained on
                
                if frame is None:
                    print("[client] No camera frame, retrying...")
                    time.sleep(0.05)
                    continue

                if self._recorder is not None:
                    # Keep the BGR frame (pre-cvtColor: that is what cv2.imwrite wants) and
                    # rebuild the state in the DATASET's order from the same raw fields,
                    # instead of un-permuting the model-order vector above.
                    # It is the PADDED frame, so the recording is pixel-for-pixel what the
                    # policy saw; the recorder derives info.json's shape from it.
                    with self._obs_lock:
                        self._latest_obs = (
                            frame.copy(),
                            dataset_state(body_q, left_hand_states, right_hand_states,
                                          neck_state if self._include_neck else None),
                        )

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.ascontiguousarray(frame.astype(np.uint8))

                message = self._build_payload(frame, states)

                # Send (thread-safe)
                with self._send_lock:
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send(message)
                        self._sent_first = True
                    else:
                        print("[client] WebSocket not connected, skipping send")
                        break

            except Exception as e:
                print(f"[client] Send error: {e}")
                break

            now = time.perf_counter()
            interval = now - prev_tick
            prev_tick = now
            print(f"[client] send interval: {interval:.3f}s")

            # Pace the obs stream at target_hz (the server's control loop runs on
            # wall-clock ticks; a stalled/flooded obs stream degrades RTC alignment).
            sleep = interval_target - (time.perf_counter() - tick_start)
            if sleep > 0:
                time.sleep(sleep)

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

        # ping_interval=0 disables keepalive pings: the server's first-obs model
        # warmup blocks its event loop for ~25 s and unanswered pings would kill
        # the connection (same reason openloop_ws_client passes ping_interval=None).
        self._ws.run_forever(ping_interval=0)

        self._running = False
        send_thread.join(timeout=0.5)
        print("[client] Client stopped")

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()


# ---------------- Main ----------------
def main(server_url, zmq_host, zmq_pub_port, zmq_sub_port, zmq_topic, zmq_sub_topic,
         camera_address, domain_name=DEFAULT_DOMAIN, target_hz=30.0, include_neck=False,
         neck_pub_host=DEFAULT_NECK_PUB_HOST, neck_pub_port=DEFAULT_NECK_PUB_PORT,
         neck_state_zmq=DEFAULT_NECK_STATE_ZMQ, record_lerobot=False,
         lerobot_root=None, lerobot_src=DEFAULT_SRC_DATASET):
    print("[MAIN] Initializing components...")

    # 1. Initialize token publisher (ZMQ PUB, Protocol v4)
    token_publisher = TokenPublisher(host="*", port=zmq_pub_port, topic=zmq_topic)
    print(f"[MAIN] TokenPublisher bound on port {zmq_pub_port}, topic='{zmq_topic}'")

    # 2. Initialize robot state subscriber (ZMQ SUB)
    state_sub = RobotStateSubscriber(host=zmq_host, port=zmq_sub_port, topic=zmq_sub_topic)
    print(f"[MAIN] State subscriber connected to {zmq_host}:{zmq_sub_port}, topic='{zmq_sub_topic}'")

    # 3. Initialize camera (neck-mounted ZED on neck domains, else RealSense)
    camera = ZedNeckCamera(address=camera_address) if include_neck else RSCamera(address=camera_address)
    print(f"[MAIN] Camera connected to {camera_address} (include_neck={include_neck})")

    # 3b. Initialize neck publisher/state-reader on neck domains
    neck_publisher = None
    neck_state_reader = None
    if include_neck:
        neck_publisher = NeckPublisher(host=neck_pub_host, port=neck_pub_port)
        neck_state_reader = NeckStateReader(neck_state_zmq)
        print(f"[MAIN] Neck publisher bound on {neck_pub_host}:{neck_pub_port}, "
              f"state reader connected to {neck_state_zmq}")

    # 3c. Eval dataset recorder (one LeRobot episode per client run).
    recorder = None
    if record_lerobot:
        root = lerobot_root or os.path.join(
            ".rollout", "cosmos3", f"{_task_slug(TASK_INSTRUCTION)}_{time.strftime('%y%m%d-%H%M%S')}")
        recorder = EvalDatasetRecorder(root, task=TASK_INSTRUCTION, src_dataset=lerobot_src)

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
        domain_name=domain_name,
        target_hz=target_hz,
        include_neck=include_neck,
        neck_publisher=neck_publisher,
        neck_state_reader=neck_state_reader,
        recorder=recorder,
    )

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

    # Encode + write the episode. Never let a recording failure mask the shutdown path.
    if recorder is not None:
        try:
            recorder.close()
        except Exception as e:
            print(f"[MAIN] Error saving eval dataset: {e}")

    state_sub.stop()
    token_publisher.stop()
    if neck_publisher is not None:
        neck_publisher.stop()
    if neck_state_reader is not None:
        neck_state_reader.stop()
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="cosmos3 RTC policy inference with WBC stabilization")
    parser.add_argument("--host", type=str, default="localhost",
                        help="cosmos3 RTC policy server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="cosmos3 RTC policy server port (ServerConfig default)")
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
                        help="Task instruction (plain text; server assembles the VLM prompt)")
    parser.add_argument("--domain-name", type=str, default=DEFAULT_DOMAIN,
                        choices=["g1_sonic_neck_zedmini", "g1_sonic_neck_realsense", "g1_sonic_neckless"],
                        help="cosmos3 embodiment domain (selects the server's action head + stats). "
                             "Neck domains use the ZED camera + neck I/O and a 45-D state; "
                             "g1_sonic_neckless keeps the RealSense + 43-D state.")
    parser.add_argument("--target-hz", type=float, default=30.0,
                        help="observation send frequency (should match the server's ctrl_hz)")
    parser.add_argument("--include-neck", action="store_true",
                        help="Neck variant (as in psi_rtc_sonic_client): ZED camera, neck pub/sub, "
                             "45-D state, neck(2) action dims active. Must be consistent with "
                             "--domain-name: required for g1_sonic_neck_*, forbidden for "
                             "g1_sonic_neckless (checked by assert).")
    parser.add_argument("--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST,
                        help=f"Neck PUB bind host (default: {DEFAULT_NECK_PUB_HOST})")
    parser.add_argument("--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT,
                        help=f"Neck PUB port (default: {DEFAULT_NECK_PUB_PORT})")
    parser.add_argument("--neck-state-zmq", type=str, default=DEFAULT_NECK_STATE_ZMQ,
                        help=f"Neck-state SUB address (default: {DEFAULT_NECK_STATE_ZMQ})")
    parser.add_argument("--record-lerobot", action="store_true",
                        help="Record this evaluation as a LeRobot v2.0 dataset laid out like "
                             "g1_sonic_lerobot_0810_merged_val, with meta/stats*.json copied "
                             "from --lerobot-src so normalization matches training.")
    parser.add_argument("--lerobot-root", type=str, default=None,
                        help="Dataset root (default: .eval_datasets/<task>_<timestamp>). Point "
                             "several runs at the same root to collect them as episodes.")
    parser.add_argument("--lerobot-src", type=str, default=DEFAULT_SRC_DATASET,
                        help=f"Dataset to copy meta/stats*.json from (default: {DEFAULT_SRC_DATASET})")

    args = parser.parse_args()

    if args.instruction:
        TASK_INSTRUCTION = args.instruction

    # Neck I/O (ZED camera, neck pub/sub, 45-D state) must match the server-side
    # embodiment domain: a neck-domain head expects the neck state dim and emits live
    # neck action dims; a neckless head does neither. A mismatch would silently feed
    # the model a mis-shaped/mis-ordered state, so fail fast here.
    domain_has_neck = "neckless" not in args.domain_name
    assert args.include_neck == domain_has_neck, (
        f"--include-neck={args.include_neck} is inconsistent with --domain-name="
        f"{args.domain_name!r} (neck domain: {domain_has_neck}). "
        + ("Pass --include-neck for g1_sonic_neck_* domains."
           if domain_has_neck else "Drop --include-neck for g1_sonic_neckless.")
    )

    server_url = f"ws://{args.host}:{args.port}/ws"
    main(
        server_url=server_url,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_sub_port=args.zmq_sub_port,
        zmq_topic=args.zmq_topic,
        zmq_sub_topic=args.zmq_sub_topic,
        camera_address=args.camera_address,
        domain_name=args.domain_name,
        target_hz=args.target_hz,
        include_neck=args.include_neck,
        neck_pub_host=args.neck_pub_host,
        neck_pub_port=args.neck_pub_port,
        neck_state_zmq=args.neck_state_zmq,
        record_lerobot=args.record_lerobot,
        lerobot_root=args.lerobot_root,
        lerobot_src=args.lerobot_src,
    )
