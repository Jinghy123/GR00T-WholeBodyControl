"""RTC client that drives Sonic from a served psi0 checkpoint.

Uploading recorded rollouts
---------------------------
Episodes accumulate under `.rollout/psi0/<task-slug>_<stamp>/`. To publish them
to the shared Hub dataset (arguments are positional -- `repo_id local_path
path_in_repo`, not a URL):

    hf upload Psi-X-share/data .rollout rollout --repo-type=dataset

This is incremental. Before sending anything the client hashes every local file
and asks the Hub which it already has (LFS blobs are deduped by sha256, small
files by git blob hash); matches are skipped, so re-running after adding new
episodes only uploads the new ones. Caveats:

  - Every file is still walked and hashed each run, so start-up cost scales with
    the whole tree, not with the delta.
  - The whole upload lands in one commit at the end. A run killed midway commits
    nothing, but the LFS blobs that did land are deduped on retry.
  - Files deleted locally are not removed on the Hub unless `--delete "*"`.

Once this tree grows past a few tens of GB, switch to the resumable variant,
which records per-file upload state in `.cache/huggingface/` inside the folder
and splits the work over many small commits:

    hf upload-large-folder Psi-X-share/data .rollout --repo-type=dataset

It always targets the repo root (no `path_in_repo`), so lay the local folder out
to match the paths you want on the Hub.
"""

import os
import re
import time
import threading
import json
import signal
import struct
import glob
import urllib.request

import cv2
import numpy as np
import zmq
import msgpack
from websocket import WebSocketApp

# Add project root to path for imports
# _GROOT_ROOT = os.path.expanduser("~/hsc/GR00T-WholeBodyControl")
# sys.path.insert(0, _GROOT_ROOT)
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
# TASK_INSTRUCTION = "Walk towards the table, pick up the grapes, and place them in the bowl."
# TASK_INSTRUCTION = "Walk towards the table, pick up the duck, and place it in the box."

# TASK_INSTRUCTION = "Pick up the eggplant and place it in the basket."
# TASK_INSTRUCTION = "Walk towards the table, pick up the eggplant, and place it in the basket."

# TASK_INSTRUCTION = "Pick up the crumpled paper ball and place it in the tray."
# TASK_INSTRUCTION = "Walk towards the table, pick up the crumpled paper ball, and place it in the tray."

# TASK_INSTRUCTION = "pick up the eggplant and place it into the transparent box"
# TASK_INSTRUCTION = "pick up the gray hippo toy and place it into the orange bowl"
# TASK_INSTRUCTION = "pick up the banana and place it into the wooden box"
# TASK_INSTRUCTION = "pick up the green grapes and place it into the green bowl"
# TASK_INSTRUCTION = "gather up the yellow shirt and turn right and put it into the laundry basket"

# TASK_INSTRUCTION = "grasp the backrest of the chair and push it straight under the table"
TASK_INSTRUCTION = "grasp the backrest of the chair, turn left, and push it under the table"
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

# TASK_INSTRUCTION = "pick up the foil bag and turn left and throw it into the trash can"
# TASK_INSTRUCTION = "pick up the snack bag and turn left and throw it into the trash can"

# TASK_INSTRUCTION = "kneel down, hook the beige shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_INSTRUCTION = "kneel down, hook the purple shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_INSTRUCTION = "kneel down, hook the white shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"
# TASK_INSTRUCTION = "kneel down, hook the blue shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"



# Must match the served checkpoint's data.transform.repack.image_keys
IMAGE_KEY = "observation.images.head"

# The training videos were h264-encoded, which rounds the frame height up to a
# multiple of 16 and fills the extra rows with black: the 376-row head camera was
# stored as 384 rows with an 8-row black strip along the bottom. The policy's
# vision encoder therefore always saw that strip, so the live frames get the same
# padding here - otherwise every frame is subtly off-distribution (real content
# shifted/rescaled relative to training). Set PAD_IMAGE_MULTIPLE = 0 to disable.
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

# FSQ configuration (must match g1_sonic_client / encoder)
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16

# Action/state layout. NOTE the psi0-sonic action puts the BODY TOKEN FIRST -- this is
# the SonicRepackTransform order (action.body_token, action[:14], action.neck), and it
# differs from the psix client's hand-first layout:
#   default:        states(43) = hand(14) + arm(14) + leg(15)
#                    action(78) = token(64) + hand_joints(14)
#   --include-neck:  states(45) = states(43) + neck(2) [appended at the end]
#                    action(80) = token(64) + hand_joints(14) + neck(2) [neck is the last 2 dims]
HAND_DIM = 14
NECK_DIM = 2
TOKEN_DIM = 64
ACTION_DIM_DEFAULT = 78
ACTION_DIM_NECK = 80

# Seed the server's FIRST action chunk with a pseudo prev-action encoding the robot's
# CURRENT pose, so chunk #0 is spliced onto where the robot actually is instead of being
# predicted open-loop (the mechanism psix_rtc_sonic_client ships). Opt-in on BOTH ends:
# the server ignores the field unless it too runs with PSI_RTC_INIT_PREV=1.
INIT_PREV_ENABLED = os.environ.get("PSI_RTC_INIT_PREV", "0").strip().lower() in (
    "1", "true", "yes", "on")

# Encoder that turns the current pose into the 64-D sonic body token (same one the
# non-RTC client uses to "freeze" the token between chunks).
from encoder_client import EncoderClient
ENCODER_MODEL = "gear_sonic_deploy/policy/release/model_encoder.onnx"
# WBC publishes joints in Mujoco order; the encoder expects IsaacLab order.
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)


def _mujoco29_to_isaaclab29(qpos):
    return np.asarray(qpos, dtype=np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB_DOF].copy()

# Neck publisher configuration (to G1 NeckMotor, matches pose_publisher.py wire format)
DEFAULT_NECK_PUB_HOST = "*"
DEFAULT_NECK_PUB_PORT = 5570

# Neck state subscriber (ZMQ SUB <- realsense_server.py on the robot, port 5560)
# JSON `[yaw_rad, pitch_rad]` of the Dynamixel present-position read each tick.
DEFAULT_NECK_STATE_ZMQ = "tcp://192.168.123.164:5560"

_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    quantized = np.clip(quantized, fsq_min, fsq_max)
    return quantized

# ---------------- Task slug ----------------
def _task_slug(instruction, max_len=128):
    """Filesystem-safe folder name derived from the task instruction."""
    slug = re.sub(r"[^a-z0-9]+", "-", (instruction or "task").lower()).strip("-")
    return (slug[:max_len].rstrip("-") or "task")


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


# ---------------- Task key / policy identity ----------------
def _fs_slug(value, fallback, limit=60):
    """Filesystem-safe slug for putting free text into a path or file name."""
    text = str(value) if value else fallback
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
    return slug[:limit].strip("_") or fallback


def _iter_prompt_entries(repo_root=_GROOT_ROOT):
    """Yield (task_key, task_description) from every data/*/prompts.json."""
    for path in sorted(glob.glob(os.path.join(repo_root, "data", "*", "prompts.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                prompts = json.load(f)
        except Exception:
            continue
        if not isinstance(prompts, dict):
            continue
        for key, entry in prompts.items():
            if isinstance(entry, dict):
                yield key, str(entry.get("task_description", ""))


def _lookup_instruction(task_key, repo_root=_GROOT_ROOT):
    """Forward lookup: the task_description for a task key, or None."""
    if not task_key:
        return None
    for key, desc in _iter_prompt_entries(repo_root):
        if key == task_key and desc:
            return desc
    return None


def _fetch_json(url, timeout=3.0):
    """Tolerant JSON GET for manifest identity; failures become error records.

    Proxies are disabled explicitly (the VLA server is on loopback / the wired
    G1 net, and an inherited http_proxy would otherwise swallow the request).
    """
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"url": url, "error": str(exc)}


def _info_url(server_url):
    """HTTP /info endpoint mirroring the /ws URL the client connects to."""
    return (server_url.replace("wss://", "https://")
                      .replace("ws://", "http://")
                      .replace("/ws", "/info"))


def _policy_tag(vla_info, base="psi0"):
    """`base` suffixed with the served checkpoint's timestamp (the run-dir suffix
    the server reports in /info), so a recording names the policy that produced it.
    Falls back to the bare base when /info is unreachable or has no timestamp."""
    ts = vla_info.get("timestamp") if isinstance(vla_info, dict) else None
    ts = str(ts).strip() if ts else ""
    return f"{base}_{_fs_slug(ts, base)}" if ts else base


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
    def __init__(self, server_url, state_subscriber, camera, token_publisher,
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
        self._include_neck = include_neck
        self._neck_publisher = neck_publisher
        self._neck_state_reader = neck_state_reader
        # Newest observation the send thread built, consumed by execute_action so each
        # recorded row is (what the policy saw, what it did) rather than two independently
        # sampled streams. Held under a lock: written by the send thread, read by the ws one.
        self._recorder = recorder
        self._obs_lock = threading.Lock()
        self._latest_obs = None
        # First-frame pseudo prev-action (see INIT_PREV_ENABLED). Encoder is loaded only
        # when the feature is on, so the default path keeps its old startup cost.
        self._sent_init_prev = False
        self._encoder = None
        if INIT_PREV_ENABLED:
            try:
                self._encoder = EncoderClient(ENCODER_MODEL, mode=0)
                print("[init-prev] enabled; first chunk will be RTC-conditioned on the current pose")
            except Exception as e:
                print(f"[init-prev] encoder load failed ({e}); first chunk falls back to unconditioned")
                self._encoder = None

    def execute_action(self, action):
        """
        Map the server action -> robot command and publish via Protocol v4.

        Server action layout is [hand_joints(14) | body_token(64)] (78-D default),
        or [hand_joints(14) | body_token(64) | neck(2)] (80-D --include-neck).
        publish_token expects [token(64) | left_hand(7) | right_hand(7)].
        """
        if action.ndim > 1:
            action = action[0]

        hand_joints = action[TOKEN_DIM:HAND_DIM + TOKEN_DIM]
        token_ori = action[:TOKEN_DIM]
        token_qtz = fsq_quantize(token_ori)

        action_out = np.concatenate([token_qtz, hand_joints])
        self._token_publisher.publish_token(action_out)

        neck = (action[HAND_DIM + TOKEN_DIM:HAND_DIM + TOKEN_DIM + NECK_DIM]
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
        # A reconnect gets a freshly warmed controller on the server, so the pose seed
        # has to be sent again -- the server drops it with the old connection's state.
        self._sent_init_prev = False
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
            try:
                # Get robot state (latest only, no history)
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
                frame = frame.astype(np.uint8)

                # Build observation payload
                img_obs = {IMAGE_KEY: frame}
                state_obs = {"states": states}

                # First frame only: encode the current pose -> 64-D sonic token and ship a
                # RAW pseudo prev-action in the server's action layout
                # [token(64) | hand(14) (| neck(2))]. The server normalizes it and warm-starts
                # the first chunk through the RTC path conditioned on it.
                if INIT_PREV_ENABLED and not self._sent_init_prev and self._encoder is not None:
                    qpos = _mujoco29_to_isaaclab29(state["body_q_measured"])           # (29,)
                    base_quat = np.asarray(state.get("base_quat_measured", [1, 0, 0, 0]),
                                           dtype=np.float32).reshape(4)
                    jp = np.tile(qpos, (10, 1)).astype(np.float32)                     # (10,29)
                    jv = np.zeros((10, 29), dtype=np.float32)
                    bq = np.tile(base_quat, (10, 1)).astype(np.float32)                # (10,4)
                    enc_token = np.asarray(self._encoder.encode(jp, jv, bq),
                                           dtype=np.float32).reshape(TOKEN_DIM)        # (64,)
                    init_prev_action = np.concatenate(
                        [enc_token, left_hand_states, right_hand_states]).astype(np.float32)  # (78,)
                    if self._include_neck:
                        init_prev_action = np.concatenate(
                            [init_prev_action, states[-NECK_DIM:]]).astype(np.float32)  # (80,)
                    state_obs["init_prev_action"] = init_prev_action
                    self._sent_init_prev = True
                    print(f"[init-prev] first-frame pseudo prev-action sent; "
                          f"token range=[{enc_token.min():.3f},{enc_token.max():.3f}]")

                payload = {
                    "image": img_obs,
                    "state": state_obs,
                    "gt_action": None,
                    "dataset_name": None,
                    "instruction": TASK_INSTRUCTION,
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
         camera_address, include_neck=False, neck_pub_host=DEFAULT_NECK_PUB_HOST,
         neck_pub_port=DEFAULT_NECK_PUB_PORT, neck_state_zmq=DEFAULT_NECK_STATE_ZMQ,
         record_lerobot=False, lerobot_root=None,
         lerobot_src=DEFAULT_SRC_DATASET):
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

    # 3b. Initialize neck publisher/state-reader when --include-neck
    neck_publisher = None
    neck_state_reader = None
    if include_neck:
        neck_publisher = NeckPublisher(host=neck_pub_host, port=neck_pub_port)
        neck_state_reader = NeckStateReader(neck_state_zmq)
        print(f"[MAIN] Neck publisher bound on {neck_pub_host}:{neck_pub_port}, "
              f"state reader connected to {neck_state_zmq}")

    # 3b2. Policy identity: one /info GET up front, before anything names a
    # directory. Its `timestamp` tags the recording dir.
    vla_info = _fetch_json(_info_url(server_url))
    policy_tag = _policy_tag(vla_info)
    if policy_tag == "psi0":
        print(f"[MAIN] WARNING: no policy timestamp from /info "
              f"({vla_info.get('error', 'field missing') if isinstance(vla_info, dict) else vla_info})")
    print(f"[MAIN] Serving policy tag: {policy_tag}")

    # 3c. Eval dataset recorder (one LeRobot episode per client run).
    recorder = None
    if record_lerobot:
        root = lerobot_root or os.path.join(
            ".rollout", policy_tag, f"{_task_slug(TASK_INSTRUCTION)}_{time.strftime('%y%m%d-%H%M%S')}")
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
    parser.add_argument("--instruction", type=str, default=None,
                        help="Task instruction for VLA policy")
    parser.add_argument("--include-neck", action="store_true",
                        help="Neck variant: states 45-dim (+neck2 appended), action 80-dim "
                             "(token64 + hand14 + neck2 appended). Also swaps RealSense for "
                             "the neck-mounted ZED camera. Default (off) keeps the legacy "
                             "43/78-dim path.")
    parser.add_argument("--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST,
                        help=f"Neck PUB bind host (default: {DEFAULT_NECK_PUB_HOST})")
    parser.add_argument("--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT,
                        help=f"Neck PUB port (default: {DEFAULT_NECK_PUB_PORT})")
    parser.add_argument("--neck-state-zmq", type=str, default=DEFAULT_NECK_STATE_ZMQ,
                        help=f"Neck-state SUB address (default: {DEFAULT_NECK_STATE_ZMQ})")
    parser.add_argument("--task-key", type=str, default=None,
                        help="prompts.json task key (e.g. pick_place_1). When given "
                             "without --instruction, the instruction is read from "
                             "prompts.json.")
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

    # --task-key without --instruction: read the instruction from prompts.json.
    if args.task_key and not args.instruction:
        looked_up = _lookup_instruction(args.task_key)
        if looked_up:
            TASK_INSTRUCTION = looked_up
            print(f"[MAIN] --task-key {args.task_key}: instruction from prompts.json: "
                  f"{TASK_INSTRUCTION!r}")
        else:
            print(f"[MAIN] WARNING: task key {args.task_key!r} not found in "
                  f"data/*/prompts.json; using the built-in default instruction")

    print(f"[MAIN] Task instruction: {TASK_INSTRUCTION!r}")

    server_url = f"ws://{args.host}:{args.port}/ws"
    main(
        server_url=server_url,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_sub_port=args.zmq_sub_port,
        zmq_topic=args.zmq_topic,
        zmq_sub_topic=args.zmq_sub_topic,
        camera_address=args.camera_address,
        include_neck=args.include_neck,
        neck_pub_host=args.neck_pub_host,
        neck_pub_port=args.neck_pub_port,
        neck_state_zmq=args.neck_state_zmq,
        record_lerobot=args.record_lerobot,
        lerobot_root=args.lerobot_root,
        lerobot_src=args.lerobot_src,
    )
