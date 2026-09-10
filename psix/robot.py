"""Robot I/O, body-token encoding, and VLA wire conversion."""

import cv2
import json
import msgpack
import numpy as np
import os
import requests
import threading
import time
import zmq
from base64 import b64decode, b64encode
from encoder_client import EncoderClient
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, pack_pose_message
from numpy.lib.format import descr_to_dtype, dtype_to_descr


_GROOT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FSQ_MIN = -0.625


FSQ_MAX = 0.625


FSQ_STEP = 0.0625  # = 1/16


OBS_SEND_INTERVAL = 1.0 / 30.0


HAND_DIM = 14


NECK_DIM = 2


TOKEN_DIM = 64


ACTION_DIM_DEFAULT = 78


ACTION_DIM_NECK = 80


DEFAULT_NECK_PUB_HOST = "*"


DEFAULT_NECK_PUB_PORT = 5570


DEFAULT_NECK_STATE_ZMQ = "tcp://192.168.123.164:5560"


def resolve_vla_embodiment(host, port, requested_tag=None,
                           include_neck_override=None, timeout=3.0, info=None, require_goal=True):
    """Read the served policy contract and select the client wire layout.

    The server remains authoritative for dimensions. ``requested_tag`` is an
    optional deployment pin: a mismatch fails before any robot publisher is
    constructed.  The client currently supports the two Sonic wire layouts,
    43/78 and 45/80.
    """
    if info is None:
        session = requests.Session()
        session.trust_env = False
        try:
            response = session.get(f"http://{host}:{int(port)}/info", timeout=timeout)
            response.raise_for_status()
            info = response.json()
        except Exception as exc:
            raise RuntimeError(f"cannot read VLA /info from {host}:{port}: {exc}") from exc
        finally:
            session.close()
    if not isinstance(info, dict):
        raise RuntimeError("VLA /info did not return a JSON object")
    served_tag = str(info.get("embodiment_tag", "")).strip()
    if not served_tag:
        raise RuntimeError("VLA /info is missing embodiment_tag")
    if requested_tag and served_tag != str(requested_tag).strip():
        raise RuntimeError(
            f"VLA embodiment mismatch: requested {requested_tag!r}, "
            f"server is {served_tag!r}"
        )
    # Model dimensions may be padded for mixed-embodiment training.  Only the
    # explicit wire contract describes the raw robot state and returned action.
    try:
        action_dim = int(info["wire"]["action_dim"])
        state_dim = int(info["wire"]["state_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "VLA /info is missing wire.state_dim/action_dim; deploy the matching "
            "serve_psix.py before using automatic embodiment selection"
        ) from exc
    layouts = {
        (43, ACTION_DIM_DEFAULT): False,
        (45, ACTION_DIM_NECK): True,
    }
    try:
        include_neck = layouts[(state_dim, action_dim)]
    except KeyError:
        raise RuntimeError(
            f"unsupported VLA state/action dims={state_dim}/{action_dim}; "
            "expected 43/78 or 45/80"
        )
    if (include_neck_override is not None and
            bool(include_neck_override) != include_neck):
        requested_layout = "45/80 neck" if include_neck_override else "43/78"
        raise RuntimeError(
            f"client layout override {requested_layout} conflicts with "
            f"VLA action_dim={action_dim}"
        )
    # Goal-conditioned runs must agree on the image slot used in the condition hash.
    goal_key = info.get("goal_key")
    if require_goal and info.get("goal_conditioned") is False:
        raise RuntimeError(
            "served checkpoint is not goal-conditioned (subgoal_key=null, trained "
            "with subgoal_prob<=0): the WM subgoal this client sends would be "
            "ignored and the condition hash would mismatch. Serve a goal-image "
            "checkpoint, or use --goal-source none."
        )
    if require_goal and goal_key is not None and goal_key != "subgoal.egocentric":
        raise RuntimeError(
            f"served checkpoint reads its goal under {goal_key!r}, but this client "
            "sends it under 'subgoal.egocentric'; align the client key before "
            "deploying."
        )
    return served_tag, state_dim, action_dim, include_neck


ENCODER_MODEL = os.path.join(
    _GROOT_ROOT, "gear_sonic_deploy/policy/release/model_encoder.onnx"
)


ENCODER_MODEL_V1_1 = os.path.join(
    _GROOT_ROOT, "gear_sonic_deploy/policy/sonic_v1_1/model_encoder.onnx"
)


ENCODER_MODELS = {"v1": ENCODER_MODEL, "v1_1": ENCODER_MODEL_V1_1}


def load_body_token_encoder(version="v1"):
    """EncoderClient for init-prev and hold tokens in the requested layout."""
    if version not in ENCODER_MODELS:
        raise ValueError(f"unknown encoder version {version!r} "
                         f"(expected one of {sorted(ENCODER_MODELS)})")
    return EncoderClient(ENCODER_MODELS[version], mode=0, version=version)


_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)


def _mujoco29_to_isaaclab29(qpos):
    return np.asarray(qpos, dtype=np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB_DOF].copy()


def goal_hw_from_info(vla_info):
    entry = next((t for t in vla_info.get("transforms", [])
                  if isinstance(t, dict) and t.get("name") == "resize"), {})
    size = entry.get("size")
    if isinstance(size, (list, tuple)) and len(size) == 2 and all(isinstance(v, int) and v > 0 for v in size):
        return tuple(size)
    return (270, 480)


def resize_goal_for_vla(rgb, size):
    """Match training: nearest resize to (H, W), without preserving aspect ratio.

    The training center crop uses the same dimensions and is a no-op.
    """
    if rgb.shape[:2] == size:
        return rgb
    return np.ascontiguousarray(
        cv2.resize(rgb, (size[1], size[0]),
                   interpolation=cv2.INTER_NEAREST)
    )


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


class TokenPublisher:
    """ZMQ publisher for token-only streaming (Protocol v4), same as g1_sonic_client."""

    def __init__(self, host="*", port=5556, topic="pose"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")
        self._topic = topic

    def send_command(self, start=False, stop=False, planner=False):
        msg = build_command_message(start=start, stop=stop, planner=planner)
        self._socket.send(msg)
        print(f"[TokenPublisher] Command: start={start} stop={stop} planner={planner}")

    def publish_token(self, action):
        """Publish Protocol v4 in wire order: token(64), left hand(7), right hand(7)."""
        action = action.astype(np.float32).reshape(1, -1)
        pose_data = {
            "token_state": action[:, :64],       # (1, 64)
            "left_hand_joints": action[:, 64:71],    # (1, 7)
            "right_hand_joints": action[:, 71:78], # (1, 7)
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=4)
        self._socket.send(msg)

    def stop(self):
        self._socket.close(linger=0)
        self._context.term()


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
        self._latest_at = None

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
            self._latest_at = time.monotonic()
        return self._latest

    def age(self):
        return (float("inf") if self._latest_at is None
                else time.monotonic() - self._latest_at)

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


def reset_neck_to_home(pub, reader, yaw=0.0, pitch=0.0, hold_s=3.0, tol=0.02,
                       stable_s=0.4):
    """Home to yaw/pitch (radians), publishing at 100 Hz until stable or timed out.

    Readback must stay within tol for stable_s. Return
    (converged, start_state, end_state, elapsed_s); missing readback cannot converge.
    """
    start = reader.get_latest() if reader is not None else None
    t0 = time.monotonic()
    in_band_since = None
    latest = start
    while time.monotonic() - t0 < hold_s:
        pub.publish(yaw, pitch)
        if reader is not None:
            latest = reader.get_latest()
            if latest is not None:
                in_band = (abs(latest[0] - yaw) <= tol and abs(latest[1] - pitch) <= tol)
                if not in_band:
                    in_band_since = None
                elif in_band_since is None:
                    in_band_since = time.monotonic()
                elif time.monotonic() - in_band_since >= stable_s:
                    return True, start, latest, time.monotonic() - t0
        time.sleep(0.01)
    # No state stream at all: we published for the full hold and cannot say more
    # than that. Report it as unconverged so the caller's policy decides.
    return False, start, latest, time.monotonic() - t0


def fsq_quantize(value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    """Telemetry only; policy body tokens are published without quantization."""
    return np.clip(np.round(np.clip(value, fsq_min, fsq_max) / fsq_step) * fsq_step, fsq_min, fsq_max)
