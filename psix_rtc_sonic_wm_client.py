"""Dedicated RTC robot client with remote BAGEL world-model subgoals.

Kept separate from ``psix_rtc_sonic_client.py`` so the original disk-subgoal
workflow and its launch commands remain unchanged.
"""

import os
import sys
import time
import threading
import json
import signal
import struct
from base64 import b64encode, b64decode
from datetime import datetime

import cv2
import numpy as np
import zmq
import msgpack
import requests
from websocket import WebSocketApp

# Resolve imports and model assets relative to this checkout so the client can be
# copied between the VLA-machine paths without editing hard-coded home directories.
_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message,
    build_command_message,
)

# ---------------- Configuration ----------------
# TASK_INSTRUCTION = "grasp the pink chip can and place it into the orange plate"
TASK_INSTRUCTION = "pick up the green grapes and place it into the green bowl"
DEFAULT_BAGEL_IMAGE_ROOT = "/home/weiduoyuan/Desktop/psi/.logs/bagel_gen_images"
DEFAULT_WM_DUMP_DIR = os.path.join(
    DEFAULT_BAGEL_IMAGE_ROOT, datetime.now().strftime("%Y%m%d-%H%M%S")
)

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
from encoder_client import EncoderClient
ENCODER_MODEL = os.path.join(
    _GROOT_ROOT, "gear_sonic_deploy/policy/release/model_encoder.onnx"
)
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)
def _mujoco29_to_isaaclab29(qpos):
    return np.asarray(qpos, dtype=np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB_DOF].copy()

# ---------------- Serialization utilities ----------------
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


# ---------------- WM-backed subgoal provider ----------------
class WmSubgoalProvider:
    """Own prompt stages and asynchronously refresh the last-good WM goal.

    The camera is read only by the VLA send loop.  This worker receives immutable
    snapshots through :meth:`update_latest_ego`, so it never races the camera's ZMQ
    REQ socket.  Requests are serialized.  Each request is tagged with the prompt
    stage and epoch that produced it; an Enter/restart during an in-flight request
    makes that response stale and therefore unable to overwrite the current goal.
    """

    def __init__(self, base_url, subtasks, task="", period=3.0, timeout=120.0,
                 jpeg_quality=90, stale_warn=5.0,
                 dump_dir="/tmp/psix_wm_client"):
        subtasks = [str(s).strip() for s in (subtasks or [])]
        if not subtasks or any(not s for s in subtasks):
            raise ValueError("[wm] prompts.json must provide at least one non-empty subtask")
        if period <= 0:
            raise ValueError("[wm] period must be positive")
        if timeout <= 0:
            raise ValueError("[wm] timeout must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("[wm] jpeg quality must be in [1, 100]")

        self._base_url = base_url.rstrip("/")
        self._subtasks = subtasks
        self._task = str(task).strip()
        self._period = float(period)
        self._timeout = float(timeout)
        self._jpeg_quality = int(jpeg_quality)
        # Kept in the constructor for CLI/API compatibility. Prompt/goal mismatch
        # now gates immediately instead of waiting for a warning threshold.
        _ = float(stale_warn)
        self._dump_dir = os.path.abspath(os.path.expanduser(dump_dir))

        self._lock = threading.Lock()
        self._session = requests.Session()
        # Never route private-Wi-Fi WM traffic through HTTP(S)_PROXY.
        self._session.trust_env = False
        self._stop_evt = threading.Event()
        self._wake_evt = threading.Event()
        self._thread = None

        self._prompt_stage = 0
        self._prompt_epoch = 0
        self._prompt_changed_at = time.monotonic()
        self._latest_ego = None
        self._latest_ego_at = None
        self._last_good_goal = None
        self._goal_stage = None
        self._goal_updated_at = None
        self._pending_goal = None
        self._request_seq = 0
        self._goal_generation = 0
        self._last_wm_ms = None
        self._last_wm_error = None
        self._last_error_log_at = -float("inf")
        self._dumped_stage = None

    @staticmethod
    def _encode_jpeg(rgb, quality):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"ego must be RGB uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _decode_jpeg(value):
        encoded = np.frombuffer(b64decode(value, validate=True), dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode failed")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"bad decoded goal: {rgb.dtype} {rgb.shape}")
        return np.ascontiguousarray(rgb)

    def _save_bagel_goal(self, encoded_jpeg, request, generation):
        """Best-effort persistence of a newly accepted BAGEL response JPEG."""
        status = f"accepted-gen{int(generation):06d}"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        filename = (
            f"{stamp}_stage{int(request['stage']):02d}_"
            f"epoch{int(request['epoch']):04d}_"
            f"req{int(request['request_id']):06d}_{status}.jpg"
        )
        path = os.path.join(self._dump_dir, filename)
        try:
            os.makedirs(self._dump_dir, exist_ok=True)
            data = b64decode(encoded_jpeg, validate=True)
            with open(path, "wb") as f:
                f.write(data)
            print(f"[wm] saved BAGEL goal: {path}", flush=True)
            return path
        except Exception as exc:
            # Image logging must never interrupt WM refresh or robot control.
            print(f"[wm] WARNING: failed to save BAGEL goal ({exc})", flush=True)
            return None

    def update_latest_ego(self, rgb_uint8):
        """Atomically replace the latest camera frame without sharing camera sockets."""
        rgb = np.asarray(rgb_uint8)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"ego must be RGB uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
        rgb = np.ascontiguousarray(rgb).copy()
        with self._lock:
            first_frame = self._latest_ego is None
            self._latest_ego = rgb
            self._latest_ego_at = time.monotonic()
        if first_frame:
            # Do not wait up to one period before generating the startup goal.
            self._wake_evt.set()

    def get_current_condition(self):
        """Return ``(current_subtask, last_good_goal, goal_stale)`` atomically."""
        snap = self.snapshot()
        return snap["subtask"], snap["goal"], snap["goal_stale"]

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            stale = (self._last_good_goal is None or
                     self._goal_stage != self._prompt_stage)
            mismatch_age = (
                now - self._prompt_changed_at
                if self._last_good_goal is not None and stale else 0.0
            )
            goal_age = (
                None if self._goal_updated_at is None else now - self._goal_updated_at
            )
            return {
                "prompt_stage": self._prompt_stage,
                "prompt_epoch": self._prompt_epoch,
                "subtask": self._subtasks[self._prompt_stage],
                # Goals are never mutated after assignment, so returning this reference
                # is safe and avoids copying a full image at 30 Hz.
                "goal": self._last_good_goal,
                "goal_stage": self._goal_stage,
                "goal_stale": stale,
                "goal_generation": self._goal_generation,
                "goal_age_s": goal_age,
                "mismatch_age_s": mismatch_age,
                "pending_goal": self._pending_goal,
                "last_wm_ms": self._last_wm_ms,
                "last_wm_error": self._last_wm_error,
            }

    def status(self):
        snap = self.snapshot()
        snap.pop("goal")
        return snap

    def advance_prompt(self):
        """Advance language, invalidate the old-stage goal, and wake WM now.

        A prompt and goal image are one condition.  Once the prompt changes, the
        previous stage's image must not be paired with it, so consumers stay gated
        until a goal generated from the new prompt lands.
        """
        with self._lock:
            if self._prompt_stage >= len(self._subtasks) - 1:
                stage = self._prompt_stage
                changed = False
            else:
                self._prompt_stage += 1
                self._prompt_epoch += 1
                self._prompt_changed_at = time.monotonic()
                self._last_good_goal = None
                self._goal_stage = None
                self._goal_updated_at = None
                stage = self._prompt_stage
                changed = True
            subtask = self._subtasks[stage]
        if changed:
            print(
                f"[wm] prompt -> stage {stage}: {subtask!r}; "
                "gated until its WM goal lands",
                flush=True,
            )
        else:
            print(f"[wm] already at last prompt stage {stage}; refreshing WM now", flush=True)
        self._wake_evt.set()

    def restart(self):
        """Return to stage 0 and gate until a new epoch's first goal arrives."""
        with self._lock:
            self._prompt_stage = 0
            self._prompt_epoch += 1
            self._prompt_changed_at = time.monotonic()
            self._last_good_goal = None
            self._goal_stage = None
            self._goal_updated_at = None
            self._pending_goal = None
            self._dumped_stage = None
        print("[wm] restart -> prompt stage 0; gated until a fresh goal lands", flush=True)
        self._wake_evt.set()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._wake_evt.set()
        self._thread = threading.Thread(
            target=self._poll_loop, name="wm-subgoal-worker", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._wake_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._thread is not None and self._thread.is_alive():
            print("[wm] WARNING: worker still finishing an in-flight request", flush=True)
        else:
            self._session.close()

    def _set_error(self, message):
        now = time.monotonic()
        with self._lock:
            self._last_wm_error = str(message)
            should_log = now - self._last_error_log_at >= 2.0
            if should_log:
                self._last_error_log_at = now
        if should_log:
            print(f"[wm] /wm failed ({message}); retaining last-good goal", flush=True)

    def _request_snapshot(self):
        with self._lock:
            if self._latest_ego is None:
                return None
            self._request_seq += 1
            request = {
                "request_id": self._request_seq,
                "stage": self._prompt_stage,
                "epoch": self._prompt_epoch,
                "subtask": self._subtasks[self._prompt_stage],
                "ego": self._latest_ego,
            }
            self._pending_goal = {
                "requested_stage": request["stage"],
                "requested_subtask": request["subtask"],
                "request_epoch": request["epoch"],
                "request_id": request["request_id"],
            }
            return request

    def _poll_once(self):
        request = self._request_snapshot()
        if request is None:
            return

        t0 = time.perf_counter()
        try:
            body = {
                "jpeg": True,
                "ego_jpeg": self._encode_jpeg(request["ego"], self._jpeg_quality),
                "subtask": request["subtask"],
                "task": self._task,
                "req_id": request["request_id"],
            }
            response = self._session.post(
                f"{self._base_url}/wm", json=body, timeout=self._timeout
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or "subgoal_jpeg" not in payload:
                raise ValueError("response missing JPEG field 'subgoal_jpeg'")
            if int(payload.get("req_id", -1)) != request["request_id"]:
                raise ValueError(
                    f"response req_id {payload.get('req_id')!r} does not match "
                    f"request {request['request_id']}"
                )
            if str(payload.get("subtask", "")) != request["subtask"]:
                raise ValueError("response subtask does not match the requested stage")
            encoded_goal = payload["subgoal_jpeg"]
            goal = self._decode_jpeg(encoded_goal)
        except Exception as exc:
            with self._lock:
                if (self._pending_goal is not None and
                        self._pending_goal["request_id"] == request["request_id"]):
                    self._pending_goal = None
            self._set_error(exc)
            return

        wm_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            if self._stop_evt.is_set():
                if (self._pending_goal is not None and
                        self._pending_goal["request_id"] == request["request_id"]):
                    self._pending_goal = None
                return
            current_subtask = self._subtasks[self._prompt_stage]
            stale_response = (
                request["epoch"] != self._prompt_epoch or
                request["stage"] != self._prompt_stage or
                request["subtask"] != current_subtask
            )
            if (self._pending_goal is not None and
                    self._pending_goal["request_id"] == request["request_id"]):
                self._pending_goal = None
            if stale_response:
                current_stage = self._prompt_stage
                current_epoch = self._prompt_epoch
            else:
                self._last_good_goal = goal
                self._goal_stage = request["stage"]
                self._goal_updated_at = time.monotonic()
                self._goal_generation += 1
                self._last_wm_ms = wm_ms
                self._last_wm_error = None
                generation = self._goal_generation
                first_for_stage = request["stage"] != self._dumped_stage
                if first_for_stage:
                    self._dumped_stage = request["stage"]

        if stale_response:
            print(
                f"[wm] dropped stale response request={request['request_id']} "
                f"stage/epoch={request['stage']}/{request['epoch']} "
                f"current={current_stage}/{current_epoch}", flush=True
            )
            return

        self._save_bagel_goal(encoded_goal, request, generation)
        if first_for_stage:
            try:
                os.makedirs(self._dump_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(self._dump_dir, "sent_ego.jpg"),
                    cv2.cvtColor(request["ego"], cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(self._dump_dir, "sent_goal.jpg"),
                    cv2.cvtColor(goal, cv2.COLOR_RGB2BGR),
                )
            except Exception as exc:
                # Debug artifacts are best-effort and must never kill goal refresh.
                self._set_error(f"debug image dump failed: {exc}")
        print(
            f"[wm] goal landed stage={request['stage']} epoch={request['epoch']} "
            f"gen={generation} latency={wm_ms:.0f}ms", flush=True
        )

    def _poll_loop(self):
        print(
            f"[wm] worker started url={self._base_url} period={self._period:.2f}s "
            f"timeout={self._timeout:.1f}s JPEG={self._jpeg_quality} "
            f"image_log={self._dump_dir}", flush=True
        )
        try:
            next_due = time.monotonic()
            while not self._stop_evt.is_set():
                wait_for = max(0.0, next_due - time.monotonic())
                self._wake_evt.wait(wait_for)
                self._wake_evt.clear()
                if self._stop_evt.is_set():
                    break
                request_started = time.monotonic()
                try:
                    self._poll_once()
                except Exception as exc:
                    # A filesystem/logging/programming edge case must not permanently
                    # terminate the only WM refresh thread.
                    self._set_error(f"unexpected worker error: {exc}")
                # If Enter arrived during the serialized request, its Event remains set
                # and the next iteration starts the newest stage without another delay.
                next_due = request_started + self._period
        finally:
            self._session.close()
            print("[wm] worker stopped", flush=True)


# ---------------- RSCamera ----------------
DEFAULT_CAMERA_TIMEOUT_MS = 1000


class _ZmqReqCamera:
    """Timeout-safe camera REQ socket with recovery after a missing REP reply."""

    def __init__(self, address, timeout_ms=DEFAULT_CAMERA_TIMEOUT_MS):
        self._address = address
        self._timeout_ms = int(timeout_ms)
        if self._timeout_ms <= 0:
            raise ValueError("camera timeout must be positive")
        self.context = zmq.Context()
        self.socket = None
        self._last_frame_at = None
        self._frame_time_lock = threading.Lock()
        self._connect_socket()

    def _connect_socket(self):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self._address)
        self.socket = socket

    def _recover_socket(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self._connect_socket()

    def _request_parts(self):
        try:
            self.socket.send(b"get_frame")
            return self.socket.recv_multipart()
        except (zmq.Again, zmq.ZMQError) as exc:
            # A timed-out REQ socket cannot legally send another request until it
            # receives a reply. Recreate it so the next 30 Hz tick can recover.
            self._recover_socket()
            raise TimeoutError(
                f"camera request failed at {self._address}: {exc}"
            ) from exc

    def _mark_frame(self):
        with self._frame_time_lock:
            self._last_frame_at = time.monotonic()

    def age(self):
        with self._frame_time_lock:
            if self._last_frame_at is None:
                return float("inf")
            return time.monotonic() - self._last_frame_at

    def stop(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()


class RSCamera(_ZmqReqCamera):
    def __init__(self, address="tcp://192.168.123.164:5558",
                 timeout_ms=DEFAULT_CAMERA_TIMEOUT_MS):
        super().__init__(address, timeout_ms=timeout_ms)

    def get_frame(self):
        parts = self._request_parts()
        if len(parts) < 1 or not parts[0]:
            raise ValueError("camera returned no RGB frame")
        rgb_array = np.frombuffer(parts[0], np.uint8)
        rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        if rgb_image is None:
            raise ValueError("camera RGB JPEG decode failed")
        self._mark_frame()
        return rgb_image


# ---------------- ZedNeckCamera ----------------
class ZedNeckCamera(_ZmqReqCamera):
    """Neck-mounted ZED camera (--include-neck). Server reply is 4-part
    multipart [ego_rgb, ego_stereo, left_wrist, right_wrist]; only slot 0 used."""

    def __init__(self, address="tcp://192.168.123.164:5558",
                 timeout_ms=DEFAULT_CAMERA_TIMEOUT_MS):
        super().__init__(address, timeout_ms=timeout_ms)

    def get_frame(self):
        parts = self._request_parts()
        while len(parts) < 4:
            parts.append(b"")
        ego_rgb_jpeg = parts[0]
        if not ego_rgb_jpeg:
            raise ValueError("camera returned no ego RGB frame")
        arr = np.frombuffer(ego_rgb_jpeg, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("camera ego RGB JPEG decode failed")
        self._mark_frame()
        return image


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
        self._latest_state_at = None
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
                    self._latest_state_at = time.monotonic()
            except Exception as e:
                print(f"[StateSubscriber] Unpack error: {e}")

    def get_state(self):
        """Return the latest robot state dict, or None if not yet received."""
        with self._lock:
            return self._latest_state

    def get_state_with_timestamp(self):
        """Return an atomic ``(state, monotonic_receive_time)`` snapshot."""
        with self._lock:
            return self._latest_state, self._latest_state_at

    def age(self):
        with self._lock:
            if self._latest_state_at is None:
                return float("inf")
            return time.monotonic() - self._latest_state_at

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


# ---------------- Global state ----------------
running = threading.Event()
running.set()


# ---------------- RTCWebSocketClient ----------------
class RTCWebSocketClient:
    def __init__(self, server_url, state_subscriber, camera, token_publisher, wm_provider,
                 task_instruction, dry_run=False, observation_stale_timeout=0.5,
                 action_stale_timeout=0.5, wm_stale_warn=5.0,
                 include_neck=False, neck_publisher=None,
                 neck_state_reader=None):
        if include_neck:
            raise ValueError("This checkpoint is fixed to non-neck 43-D state / 78-D action")
        self.server_url = server_url
        self._running = True
        self._connected = threading.Event()
        self._ws = None
        self._send_thread_handle = None
        self._send_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self.start_time = time.monotonic()

        self._state_sub = state_subscriber
        self._camera = camera
        self._token_publisher = token_publisher
        self._wm = wm_provider
        self._task = task_instruction
        self._dry_run = bool(dry_run)
        self._observation_stale_timeout = float(observation_stale_timeout)
        if self._observation_stale_timeout <= 0:
            raise ValueError("observation stale timeout must be positive")
        self._action_stale_timeout = float(action_stale_timeout)
        if self._action_stale_timeout <= 0:
            raise ValueError("action stale timeout must be positive")
        # Kept for call-site compatibility; stale prompt/goal pairs now gate immediately.
        _ = float(wm_stale_warn)
        self._dbg_last_generation = -1
        self._include_neck = include_neck
        self._neck_publisher = neck_publisher
        self._neck_state_reader = neck_state_reader
        self._wbc_started = False
        self._last_observation_at = None
        self._last_observation_lock = threading.Lock()
        self._last_hold_reason = None
        self._last_problem_log_at = -float("inf")
        self._last_gate_log_at = -float("inf")
        self._send_count = 0
        self._send_rate_started_at = time.monotonic()
        self._action_count = 0
        self._action_rate_started_at = time.monotonic()
        self._last_action_log_at = -float("inf")
        self._action_state_lock = threading.Lock()
        self._last_received_version = -1
        self._last_accepted_action_at = None
        # The key identifies the exact prompt/goal pair most recently sent to VLA.
        # A changed key creates a conservative two-version barrier. The current
        # server does not echo an observation id, so this reduces queued old-
        # condition actions but is not a cryptographic proof of condition use.
        self._sent_condition_key = None
        self._condition_barrier_version = None
        self._last_barrier_log_at = -float("inf")
        # Encoder for the first-frame current-pose token (seeds the server first-chunk RTC).
        try:
            self._encoder = EncoderClient(ENCODER_MODEL, mode=0)
        except Exception as e:
            print(f"[init-prev] encoder load failed ({e}); first chunk falls back to unconditioned")
            self._encoder = None
        self._sent_init_prev = False

    def _throttled_problem(self, message, period=2.0):
        now = time.monotonic()
        if now - self._last_problem_log_at >= period:
            self._last_problem_log_at = now
            print(message, flush=True)

    def _stop_or_hold_wbc(self, reason):
        """Stop planner once on a safety transition; dry-run remains side-effect free."""
        if self._dry_run or self._token_publisher is None:
            if reason != self._last_hold_reason:
                print(f"[safety] dry-run hold: {reason}", flush=True)
                self._last_hold_reason = reason
            return
        with self._publish_lock:
            if self._wbc_started:
                try:
                    self._token_publisher.send_command(
                        start=False, stop=True, planner=True
                    )
                finally:
                    self._wbc_started = False
                print(f"[safety] WBC stopped/held: {reason}", flush=True)
            self._last_hold_reason = reason

    def _ensure_wbc_started(self):
        if self._dry_run or self._token_publisher is None:
            return
        if not self._wbc_started:
            self._token_publisher.send_command(start=True, stop=False, planner=True)
            self._wbc_started = True
            print("[safety] first fresh valid action -> WBC planner start", flush=True)
        self._last_hold_reason = None

    def _freshness(self):
        now = time.monotonic()
        with self._last_observation_lock:
            obs_at = self._last_observation_at
        obs_age = float("inf") if obs_at is None else now - obs_at
        state_age = self._state_sub.age()
        camera_age = self._camera.age()
        fresh = max(obs_age, state_age, camera_age) <= self._observation_stale_timeout
        return fresh, obs_age, state_age, camera_age

    def _record_condition_sent(self, wm):
        key = (int(wm["prompt_epoch"]), int(wm["goal_generation"]))
        with self._action_state_lock:
            if key != self._sent_condition_key:
                self._sent_condition_key = key
                # Hold the first version after the send as well as any version
                # already queued before it; accept only a later control tick.
                self._condition_barrier_version = self._last_received_version + 1

    def _accept_version_for_condition(self, version, wm):
        if isinstance(version, bool) or not isinstance(version, (int, np.integer)):
            raise ValueError(f"invalid action version {version!r}")
        version = int(version)
        current_key = (int(wm["prompt_epoch"]), int(wm["goal_generation"]))
        with self._action_state_lock:
            if version <= self._last_received_version:
                raise ValueError(
                    f"non-monotonic action version {version} <= {self._last_received_version}"
                )
            self._last_received_version = version
            sent_key = self._sent_condition_key
            barrier = self._condition_barrier_version
            if sent_key != current_key:
                return False, version, f"condition {current_key} not sent yet"
            if barrier is not None and version <= barrier:
                return False, version, f"waiting for version > condition barrier {barrier}"
            self._condition_barrier_version = None
        return True, version, None

    def _record_action_accepted(self, now):
        with self._action_state_lock:
            self._last_accepted_action_at = now

    def _check_action_liveness(self):
        if self._dry_run or self._token_publisher is None:
            return
        now = time.monotonic()
        # Serialize the check with start/publish and re-read under the same lock;
        # otherwise a watchdog tick could stop WBC immediately after a fresh publish.
        with self._publish_lock:
            if not self._wbc_started:
                return
            with self._action_state_lock:
                last = self._last_accepted_action_at
            age = float("inf") if last is None else now - last
            if age <= self._action_stale_timeout:
                return
            try:
                self._token_publisher.send_command(start=False, stop=True, planner=True)
            finally:
                self._wbc_started = False
            self._last_hold_reason = (
                f"VLA action stream stale ({age:.3f}s > "
                f"{self._action_stale_timeout:.3f}s)"
            )
            print(f"[safety] WBC stopped/held: {self._last_hold_reason}", flush=True)

    @staticmethod
    def _validated_action(action):
        if not isinstance(action, np.ndarray):
            raise ValueError(f"action is not a numpy array: {type(action).__name__}")
        if action.shape != (1, ACTION_DIM_DEFAULT):
            raise ValueError(
                f"action shape {action.shape}, expected (1, {ACTION_DIM_DEFAULT})"
            )
        if not np.issubdtype(action.dtype, np.number):
            raise ValueError(f"action dtype is not numeric: {action.dtype}")
        if not np.isfinite(action).all():
            raise ValueError("action contains NaN or Inf")
        return np.asarray(action, dtype=np.float32)

    def execute_action(self, action):
        """
        Map the server action -> robot command and publish via Protocol v4.

        Server action layout is [hand_joints(14) | body_token(64)] (78-D).
        publish_token expects [token(64) | left_hand(7) | right_hand(7)].
        """
        action = self._validated_action(action)[0]

        hand_joints = action[:HAND_DIM]
        token_ori = action[HAND_DIM:HAND_DIM + TOKEN_DIM]
        token_qtz = fsq_quantize(token_ori)

        action_out = np.concatenate([token_qtz, hand_joints])  # [token(64), LH(7), RH(7)]
        if action_out.shape != (ACTION_DIM_DEFAULT,) or not np.isfinite(action_out).all():
            raise ValueError("reordered publish action is not a finite 78-D vector")
        if self._dry_run:
            return
        if self._token_publisher is None:
            raise RuntimeError("live mode has no TokenPublisher")
        self._token_publisher.publish_token(action_out)

    def _on_open(self, ws):
        print("[client] Connected!")
        self._connected.set()

    def _on_message(self, ws, message):
        now = time.monotonic()
        interval = now - self.start_time
        self.start_time = now

        try:
            data = json.loads(message)
            action_data = data.get("action")
            version = data.get("version", -1)

            if action_data is not None:
                action = convert_numpy_in_dict(action_data, numpy_deserialize)
                try:
                    action = self._validated_action(action)
                except ValueError as exc:
                    self._stop_or_hold_wbc(f"invalid VLA action: {exc}")
                    print(f"[client] ERROR: rejected action version={version}: {exc}", flush=True)
                    return

                fresh, obs_age, state_age, camera_age = self._freshness()
                wm = self._wm.snapshot()
                wm_condition_ready = wm["goal"] is not None and not wm["goal_stale"]
                try:
                    condition_ready, version, barrier_reason = \
                        self._accept_version_for_condition(version, wm)
                except ValueError as exc:
                    self._stop_or_hold_wbc(f"invalid VLA action stream: {exc}")
                    print(f"[client] ERROR: rejected action: {exc}", flush=True)
                    return
                if not wm_condition_ready:
                    self._stop_or_hold_wbc("no WM goal for the current prompt")
                    return
                if not condition_ready:
                    if now - self._last_barrier_log_at >= 1.0:
                        self._last_barrier_log_at = now
                        print(
                            f"[safety] holding queued action version={version}: "
                            f"{barrier_reason}", flush=True
                        )
                    return
                if not fresh:
                    self._stop_or_hold_wbc(
                        "stale observation "
                        f"obs={obs_age:.3f}s state={state_age:.3f}s camera={camera_age:.3f}s "
                        f"limit={self._observation_stale_timeout:.3f}s"
                    )
                    return

                if self._dry_run:
                    self.execute_action(action)
                    self._record_action_accepted(now)
                else:
                    # Serialize planner start and token publish against a safety stop
                    # from the 30 Hz observation thread.
                    with self._publish_lock:
                        self._ensure_wbc_started()
                        self.execute_action(action)
                        self._record_action_accepted(now)

                self._action_count += 1
                elapsed = now - self._action_rate_started_at
                should_log = (
                    self._action_count == 1 or now - self._last_action_log_at >= 1.0
                )
                if should_log:
                    hz = self._action_count / max(elapsed, 1e-6)
                    mode = "dry-run validate" if self._dry_run else "published"
                    print(
                        f"[client] action {mode}: version={version} shape={action.shape} "
                        f"range=[{action.min():.3f},{action.max():.3f}] "
                        f"recv_interval={interval:.3f}s avg_hz={hz:.1f}", flush=True
                    )
                    self._last_action_log_at = now

        except Exception as e:
            self._stop_or_hold_wbc(f"action processing error: {e}")
            print(f"[client] Message processing error: {e}")

    def _on_error(self, ws, error):
        print(f"[client] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[client] Connection closed: {close_status_code} - {close_msg}")
        self._stop_or_hold_wbc("VLA WebSocket closed")
        self._running = False
        running.clear()

    @staticmethod
    def _build_state(state):
        required = ("body_q_measured", "left_hand_q", "right_hand_q")
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(f"robot state missing keys: {missing}")

        body_q = np.asarray(state["body_q_measured"], dtype=np.float32).reshape(-1)
        left_hand = np.asarray(state["left_hand_q"], dtype=np.float32).reshape(-1)
        right_hand = np.asarray(state["right_hand_q"], dtype=np.float32).reshape(-1)
        if body_q.shape != (29,):
            raise ValueError(f"body_q_measured shape {body_q.shape}, expected (29,)")
        if left_hand.shape != (7,) or right_hand.shape != (7,):
            raise ValueError(
                f"hand shapes left={left_hand.shape} right={right_hand.shape}, expected (7,)/(7,)"
            )

        leg = body_q[:15]
        arm = body_q[15:29]
        states = np.concatenate((left_hand, right_hand, arm, leg), axis=0)
        if states.shape != (43,):
            raise ValueError(f"model state shape {states.shape}, expected (43,)")
        if not np.isfinite(states).all():
            raise ValueError("model state contains NaN or Inf")
        return np.ascontiguousarray(states), left_hand, right_hand

    def _send_thread(self):
        print("[client] Send thread started, waiting for connection...")
        while self._running and running.is_set() and not self._connected.wait(0.1):
            pass
        if not self._connected.is_set():
            print("[client] Send thread stopped before connection")
            return
        print("[client] Connected, starting observation loop")

        while self._running and running.is_set():
            tick_started = time.monotonic()
            self._check_action_liveness()
            try:
                # Read one atomic robot-state snapshot and require it to be fresh.
                state, state_received_at = self._state_sub.get_state_with_timestamp()
                if state is None:
                    self._stop_or_hold_wbc("no robot state")
                    self._throttled_problem("[client] gated: waiting for robot state")
                    time.sleep(0.05)
                    continue
                state_age = time.monotonic() - state_received_at
                if state_age > self._observation_stale_timeout:
                    self._stop_or_hold_wbc(
                        f"robot state stale ({state_age:.3f}s > "
                        f"{self._observation_stale_timeout:.3f}s)"
                    )
                    self._throttled_problem(
                        f"[client] gated: robot state stale ({state_age:.3f}s)"
                    )
                    time.sleep(0.05)
                    continue

                states, left_hand_states, right_hand_states = self._build_state(state)

                # This is the sole camera socket reader. Publish an immutable RGB copy
                # to the WM cache before checking the startup goal gate.
                frame_bgr = self._camera.get_frame()
                if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8 or
                        frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3):
                    raise ValueError(
                        f"camera frame must be BGR uint8 HxWx3, got "
                        f"{getattr(frame_bgr, 'dtype', None)} "
                        f"{getattr(frame_bgr, 'shape', None)}"
                    )
                frame = np.ascontiguousarray(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                )
                self._wm.update_latest_ego(frame)

                wm = self._wm.snapshot()
                subgoal_frame = wm["goal"]
                if subgoal_frame is None or wm["goal_stale"]:
                    self._stop_or_hold_wbc("waiting for WM goal for the current prompt")
                    now = time.monotonic()
                    if now - self._last_gate_log_at >= 2.0:
                        self._last_gate_log_at = now
                        print(
                            f"[client] gated: waiting for current-prompt WM goal; "
                            f"stage={wm['prompt_stage']} epoch={wm['prompt_epoch']} "
                            f"goal_stage={wm['goal_stage']} "
                            f"error={wm['last_wm_error']!r}", flush=True
                        )
                    time.sleep(max(0.0, OBS_SEND_INTERVAL - (time.monotonic() - tick_started)))
                    continue
                if (subgoal_frame.dtype != np.uint8 or subgoal_frame.ndim != 3 or
                        subgoal_frame.shape[2] != 3):
                    raise ValueError(
                        f"WM goal must be RGB uint8 HxWx3, got "
                        f"{subgoal_frame.dtype} {subgoal_frame.shape}"
                    )

                if wm["goal_generation"] != self._dbg_last_generation:
                    self._dbg_last_generation = wm["goal_generation"]
                    print(
                        f"[client] condition prompt_stage={wm['prompt_stage']} "
                        f"goal_stage={wm['goal_stage']} stale={wm['goal_stale']} "
                        f"generation={wm['goal_generation']} subtask={wm['subtask']!r}",
                        flush=True,
                    )

                # Build observation payload. Image keys MUST match the server's repack:
                #   ego image  -> repack.image_keys[0]  == "video.egocentric"
                #   goal image -> repack.subgoal_key[0] == "subgoal.egocentric"
                img_obs = {
                    "video.egocentric": frame,
                    "subgoal.egocentric": subgoal_frame,
                }
                state_obs = {"states": states}
                init_prev_in_payload = False

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
                    if not np.isfinite(enc_token).all():
                        raise ValueError("encoder init token contains NaN or Inf")
                    init_prev_action = np.concatenate(
                        [left_hand_states, right_hand_states, enc_token]).astype(np.float32)  # (78,)
                    if (init_prev_action.shape != (ACTION_DIM_DEFAULT,) or
                            not np.isfinite(init_prev_action).all()):
                        raise ValueError("init_prev_action must be a finite 78-D vector")
                    state_obs["init_prev_action"] = init_prev_action
                    init_prev_in_payload = True

                # Assemble the instruction string here (server feeds it to the VLM verbatim).
                # Must match the training format: "Task: <task>. Subtask: <subtask>"
                # (task lowercased, subtask dropped when empty).

                task = str(self._task).strip().lower()
                subtask = str(wm["subtask"]).strip()
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
                        with self._last_observation_lock:
                            self._last_observation_at = time.monotonic()
                        self._record_condition_sent(wm)
                        if init_prev_in_payload:
                            self._sent_init_prev = True
                            print(
                                f"[init-prev] first-frame pseudo prev-action sent; "
                                f"token range=[{enc_token.min():.3f},{enc_token.max():.3f}]"
                            )
                        self._send_count += 1
                    else:
                        print("[client] WebSocket not connected, skipping send")
                        self._stop_or_hold_wbc("VLA WebSocket disconnected")
                        break

            except Exception as e:
                self._stop_or_hold_wbc(f"observation loop error: {e}")
                self._throttled_problem(f"[client] observation rejected: {e}")

            sleep_time = max(0, OBS_SEND_INTERVAL - (time.monotonic() - tick_started))
            time.sleep(sleep_time)

            now = time.monotonic()
            elapsed = now - self._send_rate_started_at
            if elapsed >= 1.0:
                print(
                    f"[client] observation send avg_hz={self._send_count / elapsed:.1f} "
                    f"prompt_stage={self._wm.status()['prompt_stage']}", flush=True
                )
                self._send_count = 0
                self._send_rate_started_at = now

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

        self._send_thread_handle = threading.Thread(
            target=self._send_thread, name="vla-observation-sender", daemon=True
        )
        self._send_thread_handle.start()

        self._ws.run_forever()

        self._running = False
        self._connected.set()
        self._send_thread_handle.join(timeout=2.0)
        if self._send_thread_handle.is_alive():
            print("[client] WARNING: observation sender did not stop within 2s", flush=True)
        print("[client] Client stopped")

    def stop(self):
        self._running = False
        self._stop_or_hold_wbc("client shutdown")
        self._connected.set()
        if self._ws:
            self._ws.close()
        if (self._send_thread_handle is not None and
                self._send_thread_handle is not threading.current_thread()):
            self._send_thread_handle.join(timeout=2.0)


# ---------------- Main ----------------
def main(server_url, zmq_host, zmq_pub_port, zmq_sub_port, zmq_topic, zmq_sub_topic,
         camera_address, camera_timeout_ms, task_instruction, subtasks,
         wm_base_url, wm_period, wm_timeout, jpeg_quality, wm_stale_warn,
         wm_dump_dir, observation_stale_timeout, action_stale_timeout,
         dry_run=False,
         include_neck=False):
    if include_neck:
        raise ValueError("fixed checkpoint requires 43-D state / 78-D action; do not use --include-neck")
    running.set()
    print("[MAIN] Initializing components...")

    # True dry-run: do not even construct TokenPublisher, which would bind a WBC port.
    token_publisher = None
    if dry_run:
        print("[MAIN] DRY-RUN: no TokenPublisher, WBC commands, token or neck publishes")
    else:
        token_publisher = TokenPublisher(host="*", port=zmq_pub_port, topic=zmq_topic)
        print(f"[MAIN] TokenPublisher bound on port {zmq_pub_port}, topic='{zmq_topic}'")

    state_sub = RobotStateSubscriber(host=zmq_host, port=zmq_sub_port, topic=zmq_sub_topic)
    print(f"[MAIN] State subscriber connected to {zmq_host}:{zmq_sub_port}, topic='{zmq_sub_topic}'")

    camera = RSCamera(address=camera_address, timeout_ms=camera_timeout_ms)
    print(
        f"[MAIN] Camera connected to {camera_address} "
        f"(request timeout={camera_timeout_ms}ms)"
    )

    wm_provider = WmSubgoalProvider(
        base_url=wm_base_url,
        subtasks=subtasks,
        task=task_instruction,
        period=wm_period,
        timeout=wm_timeout,
        jpeg_quality=jpeg_quality,
        stale_warn=wm_stale_warn,
        dump_dir=wm_dump_dir,
    )
    print(f"[MAIN] WM provider: {wm_base_url}; {len(subtasks)} prompt stages")
    print(f"[MAIN] Task instruction: {task_instruction!r}")

    # Let the live PUB subscription settle, but send no start command yet. The
    # receive callback starts WBC only after a fresh observation + valid 78-D action.
    if token_publisher is not None:
        time.sleep(1.0)

    print("[MAIN] Waiting for robot state...")
    for _ in range(30):
        state = state_sub.get_state()
        if state is not None:
            print(f"[MAIN] Got robot state with keys: {list(state.keys())}")
            body_q = np.array(state.get("body_q_measured", []))
            print(f"[MAIN] body_q_measured shape: {body_q.shape}")
            break
        time.sleep(0.5)
    else:
        print("[MAIN] WARNING: No robot state received after 15s, proceeding anyway...")

    wm_provider.start()

    client = RTCWebSocketClient(
        server_url=server_url,
        state_subscriber=state_sub,
        camera=camera,
        token_publisher=token_publisher,
        wm_provider=wm_provider,
        task_instruction=task_instruction,
        dry_run=dry_run,
        observation_stale_timeout=observation_stale_timeout,
        action_stale_timeout=action_stale_timeout,
        wm_stale_warn=wm_stale_warn,
        include_neck=include_neck,
    )

    def stdin_listener():
        print("[MAIN] Enter: advance prompt now | :restart: stage 0 + fresh startup goal")
        while running.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:  # EOF
                break
            command = line.strip()
            if command == "":
                wm_provider.advance_prompt()
            elif command == ":restart":
                wm_provider.restart()
            else:
                print(f"[MAIN] unknown command {command!r}")

    t_stdin = threading.Thread(target=stdin_listener, daemon=True)
    t_stdin.start()

    def websocket_thread():
        client.run()
        print("[WS] WebSocket thread stopped")

    t_ws = threading.Thread(target=websocket_thread, daemon=True)
    t_ws.start()

    print("[MAIN] Running. Ctrl+C to stop.")

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

    print("[MAIN] Shutting down...")
    client.stop()
    t_ws.join(timeout=3.0)
    if t_ws.is_alive():
        print("[MAIN] WARNING: WebSocket thread is still shutting down", flush=True)
    wm_provider.stop()
    state_sub.stop()
    camera.stop()
    if token_publisher is not None:
        token_publisher.stop()
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RTC VLA client orchestrating a remote BAGEL WM (fixed 43-D/78-D)"
    )
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
    parser.add_argument("--camera-timeout-ms", type=int, default=DEFAULT_CAMERA_TIMEOUT_MS,
                        help="Camera ZMQ send/receive timeout; timed-out REQ sockets recover")
    parser.add_argument("--episode-dir", type=str,
                        default="/mnt/data/weiduo/heng/GR00T-WholeBodyControl/data/real_clean_up_table/cleanup_table_1_episode_11",
                        help="Deprecated compatibility option; live goals now come from WM")
    parser.add_argument("--prompts-json", type=str,
                        default="/mnt/data/weiduo/heng/GR00T-WholeBodyControl/data/real_clean_up_table/prompts.json",
                        help="JSON mapping task-key -> {task_description, subtasks[]}")
    parser.add_argument("--task-key", type=str, default="cleanup_table_1_episode_11",
                        help="Key into prompts.json (e.g. pick_place_1); selects the "
                             "task_description and the per-stage subtask prompts.")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override task instruction (else taken from prompts.json[task-key])")
    parser.add_argument("--wm-host", type=str, default="192.168.123.240",
                        help="BAGEL WM server Wi-Fi address")
    parser.add_argument("--wm-port", type=int, default=8016,
                        help="BAGEL WM HTTP port")
    parser.add_argument("--wm-period", type=float, default=3.0,
                        help="Serialized WM refresh period in seconds")
    parser.add_argument("--wm-timeout", type=float, default=120.0,
                        help="Timeout for one POST /wm request")
    parser.add_argument("--jpeg-quality", type=int, default=90,
                        help="JPEG quality for ego/subgoal Wi-Fi transport (1-100)")
    parser.add_argument("--wm-stale-warn", type=float, default=5.0,
                        help="Deprecated compatibility option; mismatched prompt/goal now gates immediately")
    parser.add_argument("--wm-dump-dir", type=str, default=DEFAULT_WM_DUMP_DIR,
                        help="Exact directory for all BAGEL response images; default has a run timestamp")
    parser.add_argument("--observation-stale-timeout", type=float, default=0.5,
                        help="Stop/hold WBC if state, camera, or last VLA observation is older")
    parser.add_argument("--action-stale-timeout", type=float, default=0.5,
                        help="Stop/hold WBC if no fresh monotonic VLA action arrives")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run camera/state + WM + VLA validation without binding or publishing WBC")
    parser.add_argument("--include-neck", action="store_true",
                        help="Unsupported by the fixed psix_he_g1_sonic 43/78 checkpoint")

    args = parser.parse_args()

    if args.include_neck:
        parser.error("--include-neck is incompatible with the fixed 43-D/78-D checkpoint")
    if args.camera_timeout_ms <= 0:
        parser.error("--camera-timeout-ms must be positive")
    if args.wm_period <= 0 or args.wm_timeout <= 0:
        parser.error("--wm-period and --wm-timeout must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if args.wm_stale_warn < 0:
        parser.error("--wm-stale-warn must be non-negative")
    if args.observation_stale_timeout <= 0:
        parser.error("--observation-stale-timeout must be positive")
    if args.action_stale_timeout <= 0:
        parser.error("--action-stale-timeout must be positive")

    # Resolve task instruction + per-stage subtasks from prompts.json (--task-key).
    task_instruction = args.instruction
    subtasks = []
    if args.task_key:
        try:
            with open(args.prompts_json) as f:
                prompts = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot load --prompts-json {args.prompts_json!r}: {exc}")
        if not isinstance(prompts, dict):
            parser.error("--prompts-json top level must be an object keyed by task-key")
        if args.task_key not in prompts:
            parser.error(
                f"task-key {args.task_key!r} not found in {args.prompts_json}; "
                f"available: {list(prompts)}"
            )
        entry = prompts[args.task_key]
        if not isinstance(entry, dict):
            parser.error(f"task-key {args.task_key!r} must map to an object")
        if task_instruction is None:
            task_instruction = entry.get("task_description")
        subtasks = entry.get("subtasks", [])
    if task_instruction is None:
        task_instruction = TASK_INSTRUCTION
    if not subtasks:
        parser.error(
            f"task-key {args.task_key!r} has no subtasks; WM orchestration needs at least one"
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
        camera_timeout_ms=args.camera_timeout_ms,
        task_instruction=task_instruction,
        subtasks=subtasks,
        wm_base_url=f"http://{args.wm_host}:{args.wm_port}",
        wm_period=args.wm_period,
        wm_timeout=args.wm_timeout,
        jpeg_quality=args.jpeg_quality,
        wm_stale_warn=args.wm_stale_warn,
        wm_dump_dir=args.wm_dump_dir,
        observation_stale_timeout=args.observation_stale_timeout,
        action_stale_timeout=args.action_stale_timeout,
        dry_run=args.dry_run,
        include_neck=args.include_neck,
    )
