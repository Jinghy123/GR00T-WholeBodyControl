import os
import re
import sys
import time
import threading
import json
import signal
import struct
import glob
import queue
import hashlib
import subprocess
import urllib.request
from datetime import datetime

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

TASK_INSTRUCTION = "kneel down, hook the beige shoes on the first tier of the shoe rack, turn around, kneel down again, and place them at the foot of the bed"

# TASK_INSTRUCTION = "pick up the foil bag and turn left and throw it into the trash can"
# TASK_INSTRUCTION = "pick up the snack bag and turn left and throw it into the trash can"

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

# Rollout telemetry for cross-method comparison. Runs land in
#     <root>/<task_key>/<method_name>/<timestamp>/
# which is the same layout psix_rtc_sonic_wm_client writes, so a psi0 rollout
# sits directly beside the psix rollouts of the same task and both can be read
# by one set of tools. psi0 has no WM and no prompts.json task key of its own,
# so the key is recovered from the instruction text (see _resolve_task_key).
DEFAULT_COMPARISON_ROOT = os.environ.get(
    "PSI_COMPARISON_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".logs", "main_comparisons"),
)
DEFAULT_METHOD_NAME = "psi0"
DEFAULT_OBS_SAVE_EVERY_S = 1.0
_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    quantized = np.clip(quantized, fsq_min, fsq_max)
    return quantized

# ---------------- Task slug ----------------
def _task_slug(instruction, max_len=64):
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


# ---------------- Rollout logging ----------------
def _now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


def _now_stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


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


def _norm_text(s):
    return " ".join(str(s or "").lower().split())


def _resolve_task_key(instruction, repo_root=_GROOT_ROOT):
    """Recover a prompts.json task key from the instruction text, or None.

    The psix launcher groups rollouts by task key, but psi0 is driven by a
    free-text --instruction. Matching that text back to its key is what makes a
    psi0 run land in the SAME task folder as the psix runs it is compared with;
    without it the two methods would never line up under main_comparisons.
    """
    target = _norm_text(instruction)
    if not target:
        return None
    for key, desc in _iter_prompt_entries(repo_root):
        if desc and _norm_text(desc) == target:
            return key
    return None


def _lookup_instruction(task_key, repo_root=_GROOT_ROOT):
    """Forward lookup: the task_description for a task key, or None."""
    if not task_key:
        return None
    for key, desc in _iter_prompt_entries(repo_root):
        if key == task_key and desc:
            return desc
    return None


def _git_identity(repo_dir):
    """Best-effort commit + dirty record, so a rollout traces back to its code."""
    try:
        def _git(*args):
            return subprocess.check_output(
                ["git", "-C", repo_dir, *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        diff = _git("diff")
        return {
            "repo_dir": repo_dir,
            "sha": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(diff),
            "dirty_diff_sha256": (
                hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None
            ),
        }
    except Exception as exc:
        return {"repo_dir": repo_dir, "error": str(exc)}


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


def _grab_init_frame(camera_address, timeout_ms=3000):
    """Fetch one ego frame on a short-lived socket, with a hard timeout.

    Deliberately NOT the client's own camera object. That socket belongs to the
    control loop, and this client's camera classes set no timeouts: a blocking
    recv there would hang startup outright, and a REQ socket abandoned mid-request
    cannot legally send again, so it would poison the control loop too. A separate
    throwaway socket means a missing or wedged camera server costs nothing worse
    than a skipped init frame. Returns BGR, or None; never raises, never blocks
    past the timeout.
    """
    ctx = zmq.Context()
    sock = None
    try:
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(camera_address)
        sock.send(b"get_frame")
        parts = sock.recv_multipart()
        if not parts or not parts[0]:
            return None
        return cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        print(f"[log] WARNING: init-frame grab from {camera_address} failed "
              f"({type(exc).__name__}: {exc}) — is the camera server up? "
              f"Continuing without an init frame.")
        return None
    finally:
        if sock is not None:
            sock.close(linger=0)
        ctx.term()


class RolloutLogger:
    """Per-rollout telemetry, written entirely off the control path.

    Layout (mirrors psix_rtc_sonic_wm_client so both methods are diffable):

        <run_dir>/
          init_ego_<task_key>_<method>_<stamp>.jpg  frame at rollout start
          init_frame.json                           its sidecar
          run_manifest.json                         full launch config + VLA /info + git
          obs/obs_<seq>_<stamp>.jpg                 periodic ego frames
          obs.jsonl                                 one row per saved frame (states + file)
          actions.jsonl                             one row per action the server returned
          events.jsonl                              run_start / ws_* / run_stop

    The websocket and send threads only ever enqueue; a single writer thread
    performs every file write. The queue is bounded and DROPS on overflow rather
    than blocking -- telemetry is expendable, the 30 Hz control loop is not.
    """

    _STOP = object()

    def __init__(self, run_dir, task_key, method_name,
                 obs_save_every=DEFAULT_OBS_SAVE_EVERY_S, queue_size=4096):
        self.run_dir = os.path.abspath(os.path.expanduser(run_dir))
        self.obs_dir = os.path.join(self.run_dir, "obs")
        self.task_key = task_key
        self.method_name = method_name
        self._obs_every = float(obs_save_every)
        self._q = queue.Queue(maxsize=queue_size)
        self._thread = None
        self._dropped = 0
        self._obs_seq = 0
        self._action_count = 0
        self._last_obs_at = -float("inf")
        os.makedirs(self.obs_dir, exist_ok=True)

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(
            target=self._writer, name="rollout-logger", daemon=True)
        self._thread.start()
        self.log_event("run_start", run_dir=self.run_dir,
                       task_key=self.task_key, method_name=self.method_name)
        return self

    def stop(self, timeout=5.0):
        if self._thread is None:
            return
        self.log_event("run_stop", actions_recorded=self._action_count,
                       obs_recorded=self._obs_seq, dropped_records=self._dropped)
        try:
            self._q.put(self._STOP, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        if self._dropped:
            print(f"[log] WARNING: dropped {self._dropped} telemetry records (queue full)")
        print(f"[log] Rollout telemetry: {self.run_dir}")

    # -- producers (control / websocket threads; must never block) -----------
    def _put(self, item):
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._dropped += 1

    def log_event(self, kind, **fields):
        self._put(("event", dict(kind=kind, t_wall=_now_iso(),
                                 t_mono=time.monotonic(), **fields)))

    def record_action(self, action, version=None):
        """Record the action row that is actually executed on the robot."""
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim > 1:              # server may return (H, Da); execute_action uses row 0
            arr = arr[0]
        self._action_count += 1
        self._put(("action", {
            "t_wall": _now_iso(),
            "t_mono": time.monotonic(),
            "seq": self._action_count,
            "version": version,
            "action": [round(float(x), 5) for x in arr.reshape(-1)],
        }))

    def maybe_record_obs(self, frame_rgb, states):
        """Rate-limited ego snapshot. Copies, so callers may reuse their buffers."""
        now = time.monotonic()
        if now - self._last_obs_at < self._obs_every:
            return
        self._last_obs_at = now
        self._obs_seq += 1
        name = f"obs_{self._obs_seq:06d}_{_now_stamp()}.jpg"
        self._put(("obs", {
            "t_wall": _now_iso(),
            "t_mono": now,
            "seq": self._obs_seq,
            "image_file": os.path.join("obs", name),
            "states": [round(float(x), 6) for x in np.asarray(states).reshape(-1)],
            "_frame": np.ascontiguousarray(frame_rgb).copy(),
        }))

    # -- synchronous one-shots (startup only; safe to block) -----------------
    def save_init_frame(self, frame_bgr, instruction, camera_address, include_neck):
        """Persist the very first camera frame of this rollout.

        Grabbed before the send loop starts, so it is the true scene state at
        rollout start rather than whatever frame happened to be current later.
        This is the artifact used to confirm two methods began from a comparable
        physical scene. Best-effort: a failure here must never block startup.
        """
        try:
            if frame_bgr is None:
                raise RuntimeError("camera returned no frame")
            stamp = _now_stamp()
            name = (f"init_ego_{_fs_slug(self.task_key, 'task')}_"
                    f"{_fs_slug(self.method_name, 'nolabel')}_{stamp}.jpg")
            path = os.path.join(self.run_dir, name)
            if not cv2.imwrite(path, frame_bgr):
                raise RuntimeError(f"cv2.imwrite failed for {path}")
            meta = {
                "schema_version": "psi0-init-frame/1",
                "saved_at": _now_iso(),
                "task": instruction,
                "task_key": self.task_key,
                "method_name": self.method_name,
                "image_file": name,
                "camera_address": camera_address,
                "include_neck": bool(include_neck),
            }
            with open(os.path.join(self.run_dir, "init_frame.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"[log] Saved init frame: {path}")
            return path
        except Exception as exc:
            print(f"[log] WARNING: failed to save init frame ({exc})")
            return None

    def write_manifest(self, config, vla_info):
        """Everything needed to regroup and audit this run after the fact."""
        try:
            manifest = {
                "schema_version": "psi0-run-manifest/1",
                "written_at": _now_iso(),
                "argv": list(sys.argv),
                "config": config,
                "groot_repo": _git_identity(_GROOT_ROOT),
                "psi_repo": (
                    _git_identity(os.environ["PSI_REPO_DIR"])
                    if os.environ.get("PSI_REPO_DIR")
                    else {"error": "PSI_REPO_DIR not set"}
                ),
                "vla_info": vla_info,
                "fsq": {"min": FSQ_MIN, "max": FSQ_MAX, "step": FSQ_STEP},
            }
            path = os.path.join(self.run_dir, "run_manifest.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
            print(f"[log] Run manifest: {path}")
            return path
        except Exception as exc:
            print(f"[log] WARNING: failed to write run manifest ({exc})")
            return None

    # -- writer thread -------------------------------------------------------
    def _writer(self):
        paths = {
            "event": os.path.join(self.run_dir, "events.jsonl"),
            "action": os.path.join(self.run_dir, "actions.jsonl"),
            "obs": os.path.join(self.run_dir, "obs.jsonl"),
        }
        handles = {}
        try:
            for kind, path in paths.items():
                handles[kind] = open(path, "a", encoding="utf-8")
            while True:
                item = self._q.get()
                if item is self._STOP:
                    break
                try:
                    kind, record = item
                    if kind == "obs":
                        frame = record.pop("_frame")
                        out = os.path.join(self.run_dir, record["image_file"])
                        cv2.imwrite(out, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    fh = handles[kind]
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
                except Exception as exc:
                    print(f"[log] WARNING: telemetry write failed ({exc})")
        finally:
            for fh in handles.values():
                try:
                    fh.close()
                except Exception:
                    pass


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
                 logger=None,
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
        self._logger = logger
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
        if self._logger:
            self._logger.log_event("ws_open", server_url=self.server_url)

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
                    if self._logger:
                        self._logger.record_action(action, version=version)
                    print(f"[client] Received action, version={version}, shape={action.shape}")

        except Exception as e:
            print(f"[client] Message processing error: {e}")

    def _on_error(self, ws, error):
        print(f"[client] WebSocket error: {error}")
        if self._logger:
            self._logger.log_event("ws_error", error=str(error))

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[client] Connection closed: {close_status_code} - {close_msg}")
        if self._logger:
            self._logger.log_event("ws_close", code=close_status_code, msg=close_msg)
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

                if self._logger:
                    self._logger.maybe_record_obs(frame, states)

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
         task_key=None, method_name=DEFAULT_METHOD_NAME, dump_dir=None,
         obs_save_every=DEFAULT_OBS_SAVE_EVERY_S, enable_logging=True,
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

    # 3c. Eval dataset recorder (one LeRobot episode per client run).
    recorder = None
    if record_lerobot:
        root = lerobot_root or os.path.join(
            ".rollout", "psi0", f"{_task_slug(TASK_INSTRUCTION)}_{time.strftime('%y%m%d-%H%M%S')}")
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

    # 6b. Rollout telemetry. The init frame is grabbed HERE, before the send loop
    # starts: the camera is a REQ/REP socket with no other request outstanding
    # yet, which is the only safe window for a one-off synchronous get_frame().
    logger = None
    if enable_logging:
        run_dir = dump_dir or os.path.join(
            DEFAULT_COMPARISON_ROOT,
            _fs_slug(task_key, "unknown_task"),
            _fs_slug(method_name, DEFAULT_METHOD_NAME),
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
        logger = RolloutLogger(run_dir, task_key, method_name,
                               obs_save_every=obs_save_every).start()
        logger.save_init_frame(_grab_init_frame(camera_address), TASK_INSTRUCTION,
                               camera_address, include_neck)
        logger.write_manifest(
            config={
                "server_url": server_url,
                "task_key": task_key,
                "method_name": method_name,
                "task_instruction": TASK_INSTRUCTION,
                "camera_address": camera_address,
                "include_neck": bool(include_neck),
                "neck_pub_host": neck_pub_host,
                "neck_pub_port": neck_pub_port,
                "neck_state_zmq": neck_state_zmq,
                "zmq_host": zmq_host,
                "zmq_pub_port": zmq_pub_port,
                "zmq_sub_port": zmq_sub_port,
                "zmq_topic": zmq_topic,
                "zmq_sub_topic": zmq_sub_topic,
                "obs_save_every_s": obs_save_every,
                "action_dim": ACTION_DIM_NECK if include_neck else ACTION_DIM_DEFAULT,
                "state_dim": 45 if include_neck else 43,
            },
            vla_info=_fetch_json(
                server_url.replace("ws://", "http://").replace("/ws", "/info")),
        )

    # 7. Start WebSocket client
    client = RTCWebSocketClient(
        server_url=server_url,
        state_subscriber=state_sub,
        camera=camera,
        token_publisher=token_publisher,
        include_neck=include_neck,
        neck_publisher=neck_publisher,
        neck_state_reader=neck_state_reader,
        logger=logger,
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
    if logger is not None:
        logger.stop()
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
    parser.add_argument("--method-name", type=str, default=DEFAULT_METHOD_NAME,
                        help="Label for the checkpoint/method under test; names the "
                             f"telemetry folder (default: {DEFAULT_METHOD_NAME})")
    parser.add_argument("--task-key", type=str, default=None,
                        help="prompts.json task key (e.g. pick_place_1). Groups this "
                             "rollout with the psix runs of the same task. If omitted "
                             "it is recovered from --instruction; if given without "
                             "--instruction, the instruction is read from prompts.json.")
    parser.add_argument("--dump-dir", type=str, default=None,
                        help="Explicit telemetry directory; overrides the automatic "
                             f"{DEFAULT_COMPARISON_ROOT}/<task_key>/<method_name>/<timestamp>")
    parser.add_argument("--obs-save-every", type=float, default=DEFAULT_OBS_SAVE_EVERY_S,
                        help=f"Seconds between saved ego frames (default: {DEFAULT_OBS_SAVE_EVERY_S})")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable rollout telemetry entirely")
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

    # Task key and instruction are two views of the same choice: resolve whichever
    # the operator left out, so telemetry lands beside the psix runs either way.
    task_key = args.task_key
    if task_key and not args.instruction:
        looked_up = _lookup_instruction(task_key)
        if looked_up:
            TASK_INSTRUCTION = looked_up
            print(f"[MAIN] --task-key {task_key}: instruction from prompts.json: "
                  f"{TASK_INSTRUCTION!r}")
        else:
            print(f"[MAIN] WARNING: task key {task_key!r} not found in data/*/prompts.json; "
                  f"using the built-in default instruction")
    if not task_key:
        task_key = _resolve_task_key(TASK_INSTRUCTION)
        if task_key:
            print(f"[MAIN] Task key resolved from instruction: {task_key}")
        else:
            task_key = _fs_slug(TASK_INSTRUCTION, "unknown_task", limit=40)
            print(f"[MAIN] WARNING: instruction matches no prompts.json entry; "
                  f"grouping telemetry under {task_key!r}")

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
        task_key=task_key,
        method_name=args.method_name,
        dump_dir=args.dump_dir,
        obs_save_every=args.obs_save_every,
        enable_logging=not args.no_log,
        record_lerobot=args.record_lerobot,
        lerobot_root=args.lerobot_root,
        lerobot_src=args.lerobot_src,
    )
