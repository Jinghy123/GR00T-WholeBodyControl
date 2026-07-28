"""Dedicated RTC robot client with remote world-model subgoals.

Kept separate from ``psix_rtc_sonic_client.py`` so the original disk-subgoal
workflow and its launch commands remain unchanged.
"""

import hashlib
import os
import queue
import subprocess
import sys
import time
import threading
import json
import signal
import struct
import uuid
from base64 import b64encode, b64decode
from collections import deque
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
from psix_wire_contracts import (
    ack_matches,
    build_condition,
    condition_hash,
    new_vla_session_id,
)

# ---------------- Configuration ----------------
DEFAULT_DATA_ROOT = os.path.join(
    _GROOT_ROOT, "data", "real_pick_place_0709_psix_train_val_general_prompt"
)
# Keep the deploy default on a short held-out episode (two manual stages, no
# locomotion). Operators can still select any prompts.json key.
DEFAULT_TASK_KEY = "real_pick_place_0709_psix_val_episode_0"
DEFAULT_EPISODE_DIR = os.path.join(DEFAULT_DATA_ROOT, DEFAULT_TASK_KEY)
DEFAULT_PROMPTS_JSON = os.path.join(DEFAULT_DATA_ROOT, "prompts.json")
TASK_INSTRUCTION = "Pick up the object and place it in the container."
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

ACTION_ACK_KEYS = (
    "action_vla_session_id",
    "action_condition_id",
    "action_condition_hash",
    "model_condition_hash",
    "action_version",
)

# Neck publisher configuration (to G1 NeckMotor, matches pose_publisher.py wire format)
DEFAULT_NECK_PUB_HOST = "*"
DEFAULT_NECK_PUB_PORT = 5570

# Neck state subscriber (ZMQ SUB <- realsense_server.py on the robot, port 5560)
# JSON `[yaw_rad, pitch_rad]` of the Dynamixel present-position read each tick.
DEFAULT_NECK_STATE_ZMQ = "tcp://192.168.123.164:5560"


def resolve_vla_embodiment(host, port, requested_tag=None,
                           include_neck_override=None, timeout=3.0):
    """Read the served policy contract and select the client wire layout.

    The server remains authoritative for dimensions. ``requested_tag`` is an
    optional deployment pin: a mismatch fails before any robot publisher is
    constructed.  The client currently supports the two Sonic wire layouts,
    43/78 and 45/80.
    """
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
    return served_tag, state_dim, action_dim, include_neck


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


# ---------------- Run event log / flight recorder ----------------
_EVENT_LOG_STOP = object()
_EVENT_LOG = None


class EventLog:
    """Append-only JSONL event writer with a dedicated I/O thread.

    Control threads only enqueue (non-blocking, drop-on-overflow); a single
    writer thread owns the file handle, so a slow disk can never stall the
    30 Hz observation/action paths.
    """

    def __init__(self, path, maxsize=4096):
        self._path = path
        self._queue = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._writer, name="event-log", daemon=True
        )
        self._thread.start()

    def emit(self, kind, **fields):
        record = {
            "kind": str(kind),
            "t_wall": datetime.now().isoformat(timespec="milliseconds"),
            "t_mono": time.monotonic(),
        }
        record.update(fields)
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def stop(self, timeout=2.0):
        try:
            self._queue.put(_EVENT_LOG_STOP, timeout=0.5)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def _writer(self):
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                while True:
                    item = self._queue.get()
                    if item is _EVENT_LOG_STOP:
                        if self._dropped:
                            f.write(json.dumps(
                                {"kind": "event_log_dropped",
                                 "count": int(self._dropped)}) + "\n")
                        f.flush()
                        return
                    f.write(json.dumps(item, default=str) + "\n")
                    if self._queue.empty():
                        f.flush()
        except Exception as exc:
            print(f"[events] WARNING: event log writer died ({exc})", flush=True)


def set_event_log(event_log):
    global _EVENT_LOG
    _EVENT_LOG = event_log


def log_event(kind, **fields):
    log = _EVENT_LOG
    if log is not None:
        log.emit(kind, **fields)


def _git_identity(repo_dir):
    """Best-effort {sha, dirty, dirty_diff_sha256} for one repo; never raises."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            text=True, timeout=5, stderr=subprocess.DEVNULL).strip()
        diff = subprocess.check_output(
            ["git", "-C", repo_dir, "diff", "HEAD"],
            text=True, timeout=15, stderr=subprocess.DEVNULL)
        return {
            "repo_dir": repo_dir,
            "sha": sha,
            "dirty": bool(diff),
            "dirty_diff_sha256": (
                hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None
            ),
        }
    except Exception as exc:
        return {"repo_dir": repo_dir, "error": str(exc)}


def _fetch_json(url, timeout=3.0):
    """Tolerant JSON GET for manifest identity; failures become error records."""
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"url": url, "error": str(exc)}
    finally:
        session.close()


def write_run_manifest(run_dir, config, vla_info, wm_state, episode_session_id):
    """Persist everything needed to regroup runs after the fact (plan P0.3)."""
    manifest = {
        "schema_version": "wm-run-manifest/1",
        "written_at": datetime.now().isoformat(timespec="milliseconds"),
        "written_at_monotonic": time.monotonic(),
        "argv": list(sys.argv),
        "config": config,
        "groot_repo": _git_identity(_GROOT_ROOT),
        "psi_repo": (
            _git_identity(os.environ["PSI_REPO_DIR"])
            if os.environ.get("PSI_REPO_DIR")
            else {"error": "PSI_REPO_DIR not set"}
        ),
        "vla_info": vla_info,
        "wm_state": wm_state,
        "robot_episode_session_id": episode_session_id,
        "fsq": {"min": FSQ_MIN, "max": FSQ_MAX, "step": FSQ_STEP},
    }
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    return path


class IncidentRecorder:
    """Rolling ~20 s ring of actions/states/frame thumbnails, dumped on demand.

    Appends are plain deque ops on the control threads (GIL-atomic); a dump
    snapshots the deques and does every file write on a fresh background
    thread, so neither recording nor dumping blocks control.
    """

    def __init__(self, out_dir, action_len=600, state_len=600, frame_len=100):
        self._out_dir = out_dir
        self._actions = deque(maxlen=action_len)
        self._states = deque(maxlen=state_len)
        self._frames = deque(maxlen=frame_len)
        self._last_dump_at = -float("inf")

    def record_action(self, mono, version, cid, chunk_id, chunk_tick,
                      repeat_last, action):
        self._actions.append((mono, version, cid, chunk_id, chunk_tick,
                              repeat_last, action))

    def record_state(self, mono, states):
        self._states.append((mono, states))

    def record_frame(self, mono, thumb_rgb):
        self._frames.append((mono, thumb_rgb))

    def dump(self, label, min_interval_s=10.0):
        now = time.monotonic()
        if now - self._last_dump_at < min_interval_s:
            return None
        self._last_dump_at = now
        actions = list(self._actions)
        states = list(self._states)
        frames = list(self._frames)
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_"
                             for c in str(label))[:40] or "mark"
        out = os.path.join(
            self._out_dir, "incidents",
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{safe_label}")

        def _write():
            try:
                os.makedirs(out, exist_ok=True)
                if actions:
                    np.savez_compressed(
                        os.path.join(out, "actions.npz"),
                        mono=np.array([a[0] for a in actions], np.float64),
                        version=np.array([a[1] for a in actions], np.int64),
                        cid=np.array([-1 if a[2] is None else a[2]
                                      for a in actions], np.int64),
                        chunk_id=np.array([-1 if a[3] is None else a[3]
                                           for a in actions], np.int64),
                        chunk_tick=np.array([-1 if a[4] is None else a[4]
                                             for a in actions], np.int64),
                        repeat_last=np.array([bool(a[5]) for a in actions]),
                        action=np.stack([a[6] for a in actions]),
                    )
                if states:
                    np.savez_compressed(
                        os.path.join(out, "states.npz"),
                        mono=np.array([s[0] for s in states], np.float64),
                        states=np.stack([s[1] for s in states]),
                    )
                if frames:
                    np.savez_compressed(
                        os.path.join(out, "frames.npz"),
                        mono=np.array([f[0] for f in frames], np.float64),
                        frames=np.stack([f[1] for f in frames]),
                    )
                with open(os.path.join(out, "meta.json"), "w",
                          encoding="utf-8") as f:
                    json.dump({
                        "schema_version": "wm-incident/1",
                        "label": str(label),
                        "dumped_at": datetime.now().isoformat(
                            timespec="milliseconds"),
                        "dumped_at_monotonic": now,
                        "counts": {"actions": len(actions),
                                   "states": len(states),
                                   "frames": len(frames)},
                        # Quantized tokens are reproducible offline via
                        # fsq_quantize(action[:, 14:78]); dq is not captured.
                    }, f, indent=2)
                    f.write("\n")
                print(f"[incident] dumped {label!r} -> {out}", flush=True)
                log_event("incident_dump", label=str(label), path=out,
                          actions=len(actions), states=len(states),
                          frames=len(frames))
            except Exception as exc:
                print(f"[incident] WARNING: dump failed ({exc})", flush=True)

        threading.Thread(target=_write, name="incident-dump",
                         daemon=True).start()
        return out


# ---------------- WM goal-content gate ----------------
# All metrics are mean absolute differences over grayscale 160x120 resizes of
# the paired ego/goal images (uint8 value scale). The initial thresholds came
# from 126 Cosmos 8-step pairs; a later 275-pair audit found both clear collapse
# catches and plausible walking false positives. This is therefore only a
# same-epoch gross-collapse stopgap, never a semantic or first-goal validator:
#   stay-put fixed point (27 s hover freeze): obs_motion 2.1-5.7 with
#     goal_vs_obs 17.0-20.5, persisting for 8 consecutive generations;
#   goal collapse examples: goal_vs_obs 52-92 and goal_jump 34-67 at low motion;
#   plausible walking false positives also reached gvo 61-63 at low motion.
# Re-calibrate when the WM backend, output size, or camera changes.
WM_GATE_GRAY_SIZE = (160, 120)
WM_STAYPUT_OBS_MOTION_MAX = 6.5
WM_STAYPUT_GOAL_VS_OBS_MAX = 25.0
WM_STAYPUT_CONSECUTIVE = 3
WM_COLLAPSE_OBS_MOTION_MAX = 15.0
WM_COLLAPSE_GOAL_VS_OBS_MIN = 48.0
WM_COLLAPSE_GOAL_JUMP_MIN = 30.0
WM_COLLAPSE_IMMEDIATE_RETRIES = 3
# A stay-put trial is HELD (not installed as the VLA condition) and resampled
# with a fresh seed; after this many consecutive holds the newest proposal is
# installed anyway (it may be a genuine completion frame) marked stayput_accept.
WM_STAYPUT_IMMEDIATE_RETRIES = 3
WM_GATE_VERSION = "wm-global-mad/3"
DEFAULT_WM_GOAL_HARD_AGE = 30.0

# Shadow stall detector (plan P0.4): PRINT-ONLY provisional thresholds; it
# never resets, rerolls, holds or publishes anything. Calibration against
# operator labels happens offline from events.jsonl before any action is tied
# to these numbers.
STALL_SHADOW_RAW_P95_MAX = 0.02   # per-tick raw body-token delta, 1 s p95
STALL_SHADOW_FSQ_RATIO_MIN = 0.35  # fraction of ticks with zero FSQ bin change
STALL_SHADOW_CAM_MOTION_MAX = 3.5  # 1 Hz gray-thumbnail mean abs diff
STALL_SHADOW_MIN_S = 4.0           # sustained same-condition low progress


# ---------------- WM-backed subgoal provider ----------------
class WmSubgoalProvider:
    """Own prompt stages and asynchronously refresh the last-good WM goal.

    The camera is read only by the VLA send loop.  This worker receives immutable
    snapshots through :meth:`update_latest_ego`, so it never races the camera's ZMQ
    REQ socket.  Requests are serialized.  Each request is tagged with the prompt
    stage and epoch that produced it; an Enter/restart during an in-flight request
    makes that response stale and therefore unable to overwrite the current goal.
    """

    def __init__(self, base_url, subtasks, task="", period=3.0, timeout=15.0,
                 jpeg_quality=90, stale_warn=5.0,
                 goal_hard_age=DEFAULT_WM_GOAL_HARD_AGE,
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
        if goal_hard_age < 0:
            raise ValueError("[wm] goal hard age must be non-negative")

        self._base_url = base_url.rstrip("/")
        self._subtasks = subtasks
        self._task = str(task).strip()
        self._period = float(period)
        self._timeout = float(timeout)
        self._jpeg_quality = int(jpeg_quality)
        # A same-prompt last-good goal remains usable through transient WM
        # failures/rejections.  Zero disables the final outage safety cutoff.
        self._goal_hard_age = float(goal_hard_age)
        # Kept in the constructor for CLI/API compatibility. Prompt/goal mismatch
        # now gates immediately instead of waiting for a warning threshold.
        _ = float(stale_warn)
        self._dump_dir = os.path.abspath(os.path.expanduser(dump_dir))

        self._lock = threading.Lock()
        self._session = requests.Session()
        # Never route private-Wi-Fi WM traffic through HTTP(S)_PROXY.
        self._session.trust_env = False
        # Cosmos keys idempotency by (episode session, req_id).  A process-local
        # UUID also remains harmless when talking to the legacy BAGEL server.
        self._episode_session_id = str(uuid.uuid4())
        self._stop_evt = threading.Event()
        self._wake_evt = threading.Event()
        self._thread = None

        self._prompt_stage = 0
        self._prompt_epoch = 0
        self._override_text = None
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
        # Goal-content gate state (all same-epoch; reset on prompt change).
        self._prev_gray_obs = None
        self._prev_gray_goal = None
        self._stayput_streak = 0
        self._stayput_rejects = 0
        self._collapse_rejects = 0
        self._reroll_seed = None
        self._reroll_parent_request_id = None
        self._previous_accepted_request_id = None

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

    @staticmethod
    def _gate_gray(rgb):
        """Grayscale thumbnail used by the goal-content gate metrics."""
        gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, WM_GATE_GRAY_SIZE).astype(np.float32)

    def _bump_reroll_seed_locked(self):
        self._reroll_seed = 1 if self._reroll_seed is None else self._reroll_seed + 1

    def _goal_gate_locked(self, gray_obs, gray_goal):
        """Content gate for a fresh same-epoch goal. Caller holds the lock.

        Returns ``(decision, metrics)``.  A suspected same-epoch collapse never
        replaces last-good; it is resampled in the background while the active
        condition keeps executing.  The first goal after an epoch change is
        deliberately monitor-only: global grayscale differences cannot tell a
        valid walking future from a hallucination without a same-epoch history,
        and waiting for a second WM image adds seconds to every prompt switch.

        ``stayput`` is installed before scheduling a different-seed refresh;
        completion, lost-view and a true fixed point are not yet separable by
        this lightweight metric.
        """
        goal_vs_obs = float(np.abs(gray_goal - gray_obs).mean())
        obs_motion = (None if self._prev_gray_obs is None else
                      float(np.abs(gray_obs - self._prev_gray_obs).mean()))
        goal_jump = (None if self._prev_gray_goal is None else
                     float(np.abs(gray_goal - self._prev_gray_goal).mean()))
        first_in_epoch = obs_motion is None or goal_jump is None
        collapse_like = (
            not first_in_epoch
            and obs_motion < WM_COLLAPSE_OBS_MOTION_MAX
            and goal_vs_obs > WM_COLLAPSE_GOAL_VS_OBS_MIN
            and goal_jump > WM_COLLAPSE_GOAL_JUMP_MIN
        )
        metrics = {
            "gate_version": WM_GATE_VERSION,
            "first_in_epoch": bool(first_in_epoch),
            "obs_motion": obs_motion,
            "goal_vs_obs": goal_vs_obs,
            "goal_jump": goal_jump,
            "collapse_like": bool(collapse_like),
            "reject_streak": 0,
        }
        if collapse_like:
            self._collapse_rejects += 1
            self._bump_reroll_seed_locked()
            metrics["reject_streak"] = self._collapse_rejects
            metrics["decision"] = "reject"
            return "reject", metrics

        decision = "accept_first_monitor" if first_in_epoch else "accept"
        stayput_like = (
            obs_motion is not None
            and obs_motion < WM_STAYPUT_OBS_MOTION_MAX
            and goal_vs_obs < WM_STAYPUT_GOAL_VS_OBS_MAX
        )
        if stayput_like:
            self._stayput_streak += 1
        else:
            self._stayput_streak = 0
            self._stayput_rejects = 0
        metrics["stayput_streak"] = self._stayput_streak
        if stayput_like and self._stayput_streak >= WM_STAYPUT_CONSECUTIVE:
            if self._stayput_rejects < WM_STAYPUT_IMMEDIATE_RETRIES:
                # A self-confirming hover proposal must not become the VLA
                # condition first (plan P0.2): keep last-good, resample with a
                # fresh seed.  Reference frames stay at the last ACCEPTED pair.
                self._stayput_rejects += 1
                self._bump_reroll_seed_locked()
                metrics["decision"] = "stayput_hold"
                metrics["stayput_rejects"] = self._stayput_rejects
                return "stayput_hold", metrics
            # Retry budget exhausted: the hover future may be a genuine
            # completion frame; install the newest proposal, plainly labeled,
            # rather than pinning an ever-staler last-good toward hard age.
            decision = "stayput_accept"
            self._stayput_rejects = 0
            self._stayput_streak = 0
        self._collapse_rejects = 0
        # Any install ends the current retry chain.
        self._reroll_seed = None
        self._prev_gray_obs = gray_obs
        self._prev_gray_goal = gray_goal
        metrics["decision"] = decision
        return decision, metrics

    @staticmethod
    def _format_gate_metrics(metrics):
        if not metrics:
            return "n/a"
        def fmt(value):
            return "n/a" if value is None else f"{float(value):.1f}"
        detail = (
            f"obs_motion={fmt(metrics.get('obs_motion'))} "
            f"goal_vs_obs={fmt(metrics.get('goal_vs_obs'))} "
            f"goal_jump={fmt(metrics.get('goal_jump'))}"
        )
        if metrics.get("reject_streak"):
            detail += f" rejects={int(metrics['reject_streak'])}"
        return detail

    def _save_bagel_goal(self, encoded_jpeg, request, generation, wm_ms,
                         status_label="accepted", gate_metrics=None,
                         wm_response=None, observation_jpeg=None):
        """Persist one WM goal together with its causal camera frame.

        Older rollout folders only contain a stage-level ``sent_ego.jpg`` that
        gets overwritten at the next stage. The per-generation ``.obs.jpg``
        and JSON sidecar make later audits frame-accurate without touching the
        control path when debug persistence fails. Content-gate rejections are
        persisted too (``status_label="rejected"``); the audit tooling matches
        only ``accepted-gen`` filenames, so they never pollute its timelines.
        """
        status = f"{status_label}-gen{int(generation):06d}"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        filename = (
            f"{stamp}_stage{int(request['stage']):02d}_"
            f"epoch{int(request['epoch']):04d}_"
            f"req{int(request['request_id']):06d}_{status}.jpg"
        )
        path = os.path.join(self._dump_dir, filename)
        stem = os.path.splitext(filename)[0]
        obs_filename = f"{stem}.obs.jpg"
        meta_filename = f"{stem}.json"
        obs_path = os.path.join(self._dump_dir, obs_filename)
        meta_path = os.path.join(self._dump_dir, meta_filename)
        try:
            os.makedirs(self._dump_dir, exist_ok=True)
            data = b64decode(encoded_jpeg, validate=True)
            with open(path, "wb") as f:
                f.write(data)
            if observation_jpeg is not None:
                # Persist the exact JPEG bytes sent to WM.  This is both more
                # causally precise and cheaper than re-encoding the RGB frame.
                with open(obs_path, "wb") as f:
                    f.write(b64decode(observation_jpeg, validate=True))
            else:
                wrote_obs = cv2.imwrite(
                    obs_path,
                    cv2.cvtColor(request["ego"], cv2.COLOR_RGB2BGR),
                )
                if not wrote_obs:
                    raise RuntimeError(f"cv2.imwrite failed for {obs_path}")
            response_keys = (
                "gen_id", "backend", "checkpoint", "prompt_schema", "seed",
                "num_inference_steps", "guidance_scale", "output_size",
                "inference_time_ms", "total_time_ms",
            )
            response_meta = {
                key: wm_response[key]
                for key in response_keys
                if isinstance(wm_response, dict) and key in wm_response
            }
            metadata = {
                "schema_version": "wm-rollout-pair/2",
                "saved_at": datetime.now().isoformat(timespec="milliseconds"),
                "robot_episode_session_id": self._episode_session_id,
                "task": self._task,
                "subtask": request["subtask"],
                "stage": int(request["stage"]),
                "prompt_epoch": int(request["epoch"]),
                "request_id": int(request["request_id"]),
                "goal_generation": int(generation),
                "wm_latency_ms": float(wm_ms),
                "observation_file": obs_filename,
                "goal_file": filename,
                "status": status_label,
                # Keep the legacy human-readable field so old audit scripts
                # continue to work; new analysis should use the numeric object.
                "gate_metrics": self._format_gate_metrics(gate_metrics),
                "gate": gate_metrics,
                "gate_thresholds": {
                    "stayput_obs_motion_max": WM_STAYPUT_OBS_MOTION_MAX,
                    "stayput_goal_vs_obs_max": WM_STAYPUT_GOAL_VS_OBS_MAX,
                    "stayput_consecutive": WM_STAYPUT_CONSECUTIVE,
                    "collapse_obs_motion_max": WM_COLLAPSE_OBS_MOTION_MAX,
                    "collapse_goal_vs_obs_min": WM_COLLAPSE_GOAL_VS_OBS_MIN,
                    "collapse_goal_jump_min": WM_COLLAPSE_GOAL_JUMP_MIN,
                    "collapse_immediate_retries": WM_COLLAPSE_IMMEDIATE_RETRIES,
                    "stayput_immediate_retries": WM_STAYPUT_IMMEDIATE_RETRIES,
                },
                "ego_captured_monotonic": request.get("ego_captured_mono"),
                "request_started_monotonic": request.get("request_started_mono"),
                "request_seed": request.get("seed"),
                "reroll_parent_request_id": request.get(
                    "reroll_parent_request_id"
                ),
                "previous_accepted_request_id": request.get(
                    "previous_accepted_request_id"
                ),
                "active_goal_age_before_s": request.get(
                    "active_goal_age_before_s"
                ),
                "wm_response": response_meta,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"[wm] saved paired obs/goal: {path}", flush=True)
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

    def _current_subtask_locked(self):
        return (self._override_text if self._override_text is not None
                else self._subtasks[self._prompt_stage])

    def _invalidate_goal_locked(self):
        self._prompt_epoch += 1
        self._prompt_changed_at = time.monotonic()
        self._last_good_goal = None
        self._goal_stage = None
        self._goal_updated_at = None
        self._pending_goal = None
        # Cross-epoch image diffs are meaningless; the gate restarts fresh.
        self._prev_gray_obs = None
        self._prev_gray_goal = None
        self._stayput_streak = 0
        self._stayput_rejects = 0
        self._collapse_rejects = 0
        self._reroll_seed = None
        self._reroll_parent_request_id = None
        self._previous_accepted_request_id = None

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
            goal_expired = bool(
                not stale and goal_age is not None and self._goal_hard_age > 0
                and goal_age > self._goal_hard_age
            )
            return {
                "prompt_stage": self._prompt_stage,
                "prompt_epoch": self._prompt_epoch,
                "subtask": self._current_subtask_locked(),
                "manual_override": self._override_text is not None,
                # Goals are never mutated after assignment, so returning this reference
                # is safe and avoids copying a full image at 30 Hz.
                "goal": self._last_good_goal,
                "goal_stage": self._goal_stage,
                "goal_stale": stale,
                "goal_generation": self._goal_generation,
                "goal_age_s": goal_age,
                "goal_expired": goal_expired,
                "goal_hard_age_s": self._goal_hard_age,
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
            had_override = self._override_text is not None
            self._override_text = None
            if self._prompt_stage >= len(self._subtasks) - 1:
                stage = self._prompt_stage
                changed = had_override
            else:
                self._prompt_stage += 1
                stage = self._prompt_stage
                changed = True
            if changed:
                self._invalidate_goal_locked()
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
        return changed

    def restart(self):
        """Return to stage 0 and gate until a new epoch's first goal arrives."""
        with self._lock:
            self._prompt_stage = 0
            self._override_text = None
            self._invalidate_goal_locked()
            self._dumped_stage = None
        print("[wm] restart -> prompt stage 0; gated until a fresh goal lands", flush=True)
        self._wake_evt.set()
        return True

    def takeover(self, text):
        """Use operator text verbatim as the current WM/VLA prompt."""
        text = str(text).strip()
        if not text:
            raise ValueError("manual prompt must not be empty")
        with self._lock:
            self._override_text = text
            self._invalidate_goal_locked()
        print(f"[wm] manual prompt -> {text!r}; gated until its WM goal lands",
              flush=True)
        self._wake_evt.set()
        return True

    def resume_scripted_prompt(self):
        """Leave manual takeover and regenerate the current episode stage."""
        with self._lock:
            if self._override_text is None:
                subtask = self._subtasks[self._prompt_stage]
                changed = False
            else:
                self._override_text = None
                self._invalidate_goal_locked()
                subtask = self._subtasks[self._prompt_stage]
                changed = True
        if changed:
            print(f"[wm] resumed episode prompt -> {subtask!r}; "
                  "gated until its WM goal lands", flush=True)
            self._wake_evt.set()
        else:
            print(f"[wm] already using episode prompt {subtask!r}", flush=True)
        return changed

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

    def _warn_if_goal_aging(self):
        """Loudly flag a same-prompt goal that keeps aging past the refresh cadence.

        The last-good goal is deliberately retained through /wm failures, but
        consumers keep conditioning on it through transient failures; make a
        silent outage visible before the configurable hard cutoff.
        """
        snap = self.snapshot()
        age = snap["goal_age_s"]
        if (snap["goal"] is None or snap["goal_stale"] or age is None
                or age <= 3.0 * self._period):
            return
        print(
            f"[wm] WARNING: last-good goal is {age:.1f}s old "
            f"(refresh period {self._period:.1f}s, hard age "
            + (f"{self._goal_hard_age:.1f}s" if self._goal_hard_age > 0
               else "disabled")
            + "); retaining it while /wm recovers "
            f"(last error: {snap['last_wm_error']!r})",
            flush=True,
        )

    def _request_snapshot(self):
        with self._lock:
            if self._latest_ego is None:
                return None
            now = time.monotonic()
            self._request_seq += 1
            request = {
                "request_id": self._request_seq,
                "stage": self._prompt_stage,
                "epoch": self._prompt_epoch,
                "subtask": self._current_subtask_locked(),
                "ego": self._latest_ego,
                "ego_captured_mono": self._latest_ego_at,
                "request_started_mono": now,
                # Bind retry provenance to this immutable request.  Reading the
                # shared seed later in _build_request_body raced prompt changes.
                "seed": self._reroll_seed,
                "reroll_parent_request_id": self._reroll_parent_request_id,
                "previous_accepted_request_id": self._previous_accepted_request_id,
                "active_goal_age_before_s": (
                    None if self._goal_updated_at is None
                    else now - self._goal_updated_at
                ),
            }
            self._pending_goal = {
                "requested_stage": request["stage"],
                "requested_subtask": request["subtask"],
                "request_epoch": request["epoch"],
                "request_id": request["request_id"],
            }
            return request

    def _build_request_body(self, request):
        """Build the cross-backend JPEG request contract.

        Cosmos consumes transport/session/prompt_gen; BAGEL consumes jpeg and
        ignores the extra provenance fields.
        """
        body = {
            "transport": "jpeg",
            "jpeg": True,
            "ego_jpeg": self._encode_jpeg(request["ego"], self._jpeg_quality),
            "subtask": request["subtask"],
            "task": self._task,
            "req_id": request["request_id"],
            "robot_episode_session_id": self._episode_session_id,
            "prompt_gen": request["epoch"],
        }
        if request["seed"] is not None:
            # Cosmos seeds its sampler (and idempotency digest) from this; a
            # bumped seed forces a genuinely resampled future for the same ego.
            # BAGEL ignores unknown fields, so the reroll degrades to a plain
            # immediate refresh there.
            body["seed"] = request["seed"]
        return body

    def _poll_once(self):
        request = self._request_snapshot()
        if request is None:
            return

        t0 = time.perf_counter()
        try:
            body = self._build_request_body(request)
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
            if ("robot_episode_session_id" in payload and
                    str(payload["robot_episode_session_id"]) != self._episode_session_id):
                raise ValueError("response episode session does not match request")
            if ("prompt_gen" in payload and
                    int(payload["prompt_gen"]) != request["epoch"]):
                raise ValueError("response prompt_gen does not match request")
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
        # Content-gate thumbnails are pure image math; keep them off the lock.
        gray_obs = self._gate_gray(request["ego"])
        gray_goal = self._gate_gray(goal)
        with self._lock:
            if self._stop_evt.is_set():
                if (self._pending_goal is not None and
                        self._pending_goal["request_id"] == request["request_id"]):
                    self._pending_goal = None
                return
            current_subtask = self._current_subtask_locked()
            stale_response = (
                request["epoch"] != self._prompt_epoch or
                request["stage"] != self._prompt_stage or
                request["subtask"] != current_subtask
            )
            if (self._pending_goal is not None and
                    self._pending_goal["request_id"] == request["request_id"]):
                self._pending_goal = None
            gate_decision = gate_metrics = None
            if stale_response:
                current_stage = self._prompt_stage
                current_epoch = self._prompt_epoch
            else:
                gate_decision, gate_metrics = self._goal_gate_locked(
                    gray_obs, gray_goal
                )
                if gate_decision == "reject":
                    rejected_generation = self._goal_generation
                    reroll_seed = self._reroll_seed
                    self._reroll_parent_request_id = request["request_id"]
                    immediate_retry = (
                        self._collapse_rejects
                        <= WM_COLLAPSE_IMMEDIATE_RETRIES
                    )
                    self._last_wm_ms = wm_ms
                    self._last_wm_error = "goal rejected by collapse monitor"
                elif gate_decision == "stayput_hold":
                    rejected_generation = self._goal_generation
                    reroll_seed = self._reroll_seed
                    self._reroll_parent_request_id = request["request_id"]
                    self._last_wm_ms = wm_ms
                    self._last_wm_error = "goal held by stay-put monitor"
                else:
                    self._last_good_goal = goal
                    self._goal_stage = request["stage"]
                    self._goal_updated_at = time.monotonic()
                    self._goal_generation += 1
                    self._last_wm_ms = wm_ms
                    self._last_wm_error = None
                    self._previous_accepted_request_id = request["request_id"]
                    self._reroll_parent_request_id = None
                    reroll_seed = self._reroll_seed
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
            log_event("wm_response", decision="stale_dropped",
                      request_id=request["request_id"], stage=request["stage"],
                      epoch=request["epoch"], wm_ms=wm_ms)
            return

        if gate_decision == "reject":
            gate_detail = self._format_gate_metrics(gate_metrics)
            print(
                f"[wm] goal REJECTED (collapse suspected): {gate_detail} "
                f"stage={request['stage']} epoch={request['epoch']} "
                f"req={request['request_id']} -> keeping last-good goal, "
                + (f"retrying now with seed={reroll_seed}"
                   if immediate_retry else
                   f"retrying on normal cadence with seed={reroll_seed}"),
                flush=True,
            )
            self._save_bagel_goal(encoded_goal, request, rejected_generation, wm_ms,
                                  status_label="rejected", gate_metrics=gate_metrics,
                                  wm_response=payload,
                                  observation_jpeg=body.get("ego_jpeg"))
            log_event("wm_response", decision="reject",
                      request_id=request["request_id"], stage=request["stage"],
                      epoch=request["epoch"], wm_ms=wm_ms,
                      seed=request["seed"], gate=gate_metrics)
            # Bound the immediate retry burst so a persistent gate false-positive
            # cannot monopolize the GPU.  The normal 3 s poll continues forever.
            if immediate_retry:
                self._wake_evt.set()
            return

        if gate_decision == "stayput_hold":
            gate_detail = self._format_gate_metrics(gate_metrics)
            print(
                f"[wm] STAY-PUT trial HELD (not installed): {gate_detail} "
                f"stage={request['stage']} epoch={request['epoch']} "
                f"req={request['request_id']} "
                f"hold={gate_metrics.get('stayput_rejects')}/"
                f"{WM_STAYPUT_IMMEDIATE_RETRIES} -> keeping last-good goal, "
                f"resampling now with seed={reroll_seed}", flush=True
            )
            self._save_bagel_goal(encoded_goal, request, rejected_generation, wm_ms,
                                  status_label="stayput-held",
                                  gate_metrics=gate_metrics,
                                  wm_response=payload,
                                  observation_jpeg=body.get("ego_jpeg"))
            log_event("wm_response", decision="stayput_hold",
                      request_id=request["request_id"], stage=request["stage"],
                      epoch=request["epoch"], wm_ms=wm_ms,
                      seed=request["seed"], gate=gate_metrics)
            self._wake_evt.set()
            return

        self._save_bagel_goal(encoded_goal, request, generation, wm_ms,
                              gate_metrics=gate_metrics, wm_response=payload,
                              observation_jpeg=body.get("ego_jpeg"))
        gate_detail = self._format_gate_metrics(gate_metrics)
        log_event("wm_response", decision=gate_decision,
                  request_id=request["request_id"], stage=request["stage"],
                  epoch=request["epoch"], generation=generation, wm_ms=wm_ms,
                  seed=request["seed"], gate=gate_metrics)
        if gate_decision == "stayput_accept":
            print(
                f"[wm] STAY-PUT retries exhausted; installing newest hover "
                f"proposal as gen={generation}: {gate_detail} "
                f"(possible completion frame)", flush=True
            )
        if first_for_stage:
            try:
                os.makedirs(self._dump_dir, exist_ok=True)
                stage_prefix = (
                    f"stage{int(request['stage']):02d}_"
                    f"epoch{int(request['epoch']):04d}_first"
                )
                for filename, rgb in (
                    ("sent_ego.jpg", request["ego"]),
                    ("sent_goal.jpg", goal),
                    (f"{stage_prefix}_ego.jpg", request["ego"]),
                    (f"{stage_prefix}_goal.jpg", goal),
                ):
                    if not cv2.imwrite(
                        os.path.join(self._dump_dir, filename),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    ):
                        raise RuntimeError(f"cv2.imwrite failed for {filename}")
            except Exception as exc:
                # Debug artifacts are best-effort and must never kill goal refresh.
                self._set_error(f"debug image dump failed: {exc}")
        print(
            f"[wm] goal landed stage={request['stage']} epoch={request['epoch']} "
            f"gen={generation} latency={wm_ms:.0f}ms gate[{gate_detail}]", flush=True
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
                self._warn_if_goal_aging()
                # If Enter arrived during the serialized request, its Event remains set
                # and the next iteration starts the newest stage without another delay.
                next_due = request_started + self._period
        finally:
            self._session.close()
            print("[wm] worker stopped", flush=True)


class EpisodeSubgoalProvider:
    """Serve fixed ground-truth goals from ``episode_dir/color_subgoal``.

    This implements the same snapshot/transition interface as
    :class:`WmSubgoalProvider`, so the production RTC condition handshake,
    freshness checks, flight recorder and WBC safety path remain identical.
    Only the source of the goal image changes; no WM HTTP request is made.
    """

    _IMG_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self, episode_dir, subtasks, task=""):
        self._episode_dir = os.path.abspath(os.path.expanduser(episode_dir))
        self._subtasks = [str(s).strip() for s in (subtasks or [])]
        self._task = str(task).strip()
        if not self._subtasks or any(not s for s in self._subtasks):
            raise ValueError(
                "[gt] prompts.json must provide at least one non-empty subtask"
            )

        goal_dir = os.path.join(self._episode_dir, "color_subgoal")
        if not os.path.isdir(goal_dir):
            raise ValueError(f"[gt] color_subgoal directory not found: {goal_dir}")
        self.paths = sorted(
            os.path.join(goal_dir, name)
            for name in os.listdir(goal_dir)
            if name.lower().endswith(self._IMG_EXTS)
        )
        if not self.paths:
            raise ValueError(f"[gt] no goal images found in {goal_dir}")
        if len(self.paths) != len(self._subtasks):
            raise ValueError(
                f"[gt] goal/subtask count mismatch: {len(self.paths)} images in "
                f"{goal_dir}, {len(self._subtasks)} subtasks"
            )

        self._images = []
        self._goal_records = []
        for path in self.paths:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"[gt] failed to decode goal image: {path}")
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            self._images.append(rgb)
            self._goal_records.append({
                "path": path,
                "sha256": digest,
                "shape": list(rgb.shape),
            })

        self._lock = threading.Lock()
        self._prompt_stage = 0
        self._prompt_epoch = 0
        self._goal_generation = 1
        self._goal_updated_at = time.monotonic()
        self._episode_session_id = str(uuid.uuid4())

    def provenance(self):
        return {
            "source": "episode_gt",
            "episode_dir": self._episode_dir,
            "goals": list(self._goal_records),
        }

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            stage = self._prompt_stage
            return {
                "prompt_stage": stage,
                "prompt_epoch": self._prompt_epoch,
                "subtask": self._subtasks[stage],
                "manual_override": False,
                "goal": self._images[stage],
                "goal_stage": stage,
                "goal_stale": False,
                "goal_generation": self._goal_generation,
                "goal_age_s": now - self._goal_updated_at,
                "goal_expired": False,
                "goal_hard_age_s": 0.0,
                "mismatch_age_s": 0.0,
                "pending_goal": None,
                "last_wm_ms": None,
                "last_wm_error": None,
                "goal_path": self.paths[stage],
                "goal_source": "episode_gt",
            }

    def status(self):
        snap = self.snapshot()
        snap.pop("goal")
        return snap

    def get_current_condition(self):
        snap = self.snapshot()
        return snap["subtask"], snap["goal"], False

    def update_latest_ego(self, _rgb):
        # The fixed GT provider never reads the live camera. The observation
        # sender remains the camera's sole ZMQ REQ owner.
        return None

    def start(self):
        print(
            f"[gt] fixed episode goals: {self._episode_dir} "
            f"({len(self.paths)} stages; no WM requests)",
            flush=True,
        )
        for idx, (path, subtask) in enumerate(zip(self.paths, self._subtasks)):
            print(f"[gt]   [{idx}] {path} | {subtask!r}", flush=True)
        log_event(
            "episode_goal_selected", stage=0, epoch=0, generation=1,
            path=self.paths[0], sha256=self._goal_records[0]["sha256"],
        )

    def stop(self):
        return None

    def advance_prompt(self):
        with self._lock:
            if self._prompt_stage >= len(self._images) - 1:
                stage = self._prompt_stage
                changed = False
            else:
                self._prompt_stage += 1
                self._prompt_epoch += 1
                self._goal_generation += 1
                self._goal_updated_at = time.monotonic()
                stage = self._prompt_stage
                changed = True
            epoch = self._prompt_epoch
            generation = self._goal_generation
        if not changed:
            print(f"[gt] already at last goal stage {stage}", flush=True)
            return False
        print(
            f"[gt] goal -> stage {stage}: {self.paths[stage]} | "
            f"{self._subtasks[stage]!r}",
            flush=True,
        )
        log_event(
            "episode_goal_selected", stage=stage, epoch=epoch,
            generation=generation, path=self.paths[stage],
            sha256=self._goal_records[stage]["sha256"],
        )
        return True

    def restart(self):
        with self._lock:
            self._prompt_stage = 0
            self._prompt_epoch += 1
            self._goal_generation += 1
            self._goal_updated_at = time.monotonic()
            epoch = self._prompt_epoch
            generation = self._goal_generation
        print(f"[gt] restart -> stage 0: {self.paths[0]}", flush=True)
        log_event(
            "episode_goal_selected", stage=0, epoch=epoch,
            generation=generation, path=self.paths[0],
            sha256=self._goal_records[0]["sha256"], restart=True,
        )
        return True

    def takeover(self, _text):
        print(
            "[gt] manual prompt ignored: fixed GT mode has no paired image; "
            "use --goal-source wm for manual prompts",
            flush=True,
        )
        return False

    def resume_scripted_prompt(self):
        print("[gt] already using the fixed episode prompt", flush=True)
        return False


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


# ---------------- Global state ----------------
running = threading.Event()
running.set()


# ---------------- RTCWebSocketClient ----------------
class RTCWebSocketClient:
    def __init__(self, server_url, state_subscriber, camera, token_publisher, wm_provider,
                 task_instruction, dry_run=False, observation_stale_timeout=0.5,
                 action_stale_timeout=0.5, condition_promote_timeout=6.0,
                 wm_stale_warn=5.0,
                 include_neck=False, neck_publisher=None,
                 neck_state_reader=None, incident_recorder=None):
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
        # Bound how long old-ACTIVE actions may keep executing after a newer
        # candidate was minted. A run-out server chunk repeats its last action
        # with a still-valid old ack, so arrival freshness alone cannot see a
        # frozen robot; an unacknowledged candidate older than this is treated
        # as a liveness failure and holds WBC.
        self._condition_promote_timeout = float(condition_promote_timeout)
        if self._condition_promote_timeout <= 0:
            raise ValueError("condition promote timeout must be positive")
        # Kept for call-site compatibility; stale prompt/goal pairs now gate immediately.
        _ = float(wm_stale_warn)
        self._dbg_last_generation = -1
        self._include_neck = include_neck
        self._neck_publisher = neck_publisher
        self._neck_state_reader = neck_state_reader
        if self._include_neck and self._neck_state_reader is None:
            raise ValueError("--include-neck requires a NeckStateReader")
        if (self._include_neck and not self._dry_run and
                self._neck_publisher is None):
            raise ValueError("live --include-neck requires a NeckPublisher")
        self._wbc_started = False
        self._last_observation_at = None
        self._last_observation_lock = threading.Lock()
        self._last_hold_reason = None
        self._last_problem_log_at = -float("inf")
        self._last_dry_hold_log_at = -float("inf")
        self._last_gate_log_at = -float("inf")
        self._send_count = 0
        self._send_rate_started_at = time.monotonic()
        self._action_count = 0
        self._action_rate_started_at = time.monotonic()
        self._last_action_log_at = -float("inf")
        self._telemetry_lock = threading.Lock()
        self._telemetry_epoch = None
        self._incidents = incident_recorder
        # ~3 Hz thumbnails for the flight recorder + ~1 Hz camera-motion
        # sample for the shadow stall detector (send-thread only).
        self._cam_tick = 0
        self._cam_prev_gray = None
        # Token-delta telemetry between consecutive accepted actions. A
        # live-but-frozen stream (run-out chunk repeating one action, or a
        # constant model output) passes every freshness/provenance check;
        # only content deltas expose it in the logs.
        self._prev_exec_action = None
        self._tok_delta_max = 0.0
        self._hand_delta_max = 0.0
        self._neck_delta_max = 0.0
        self._fsq_static_ticks = 0
        self._delta_ticks = 0
        # Distribution telemetry (plan P0.3): per-tick raw deltas for p50/p95,
        # FSQ changed-dims Hamming stats, and the longest zero-change run.
        self._tok_delta_window = []
        self._fsq_changed_sum = 0
        self._fsq_changed_max = 0
        self._fsq_static_run = 0
        self._fsq_static_run_max = 0
        self._rtc_infer_ms_last = None
        # Shadow stall detector state (print-only; plan P0.4).
        self._cam_motion_1s = None
        self._same_condition_stall_s = 0.0
        self._global_low_progress_s = 0.0
        self._stall_shadow_printed_at = -float("inf")
        # Constant-time, 1 Hz RTC transport provenance. These scalars add no
        # image copies or per-tick disk I/O and expose Event coalescing/run-out.
        self._recv_interval_max = 0.0
        self._telemetry_last_version = None
        self._version_gap_total = 0
        self._version_gap_max = 0
        self._rtc_chunk_id = None
        self._rtc_chunk_tick = None
        self._rtc_chunk_switches = 0
        self._rtc_repeat_last_ticks = 0
        self._action_state_lock = threading.Lock()
        self._last_received_version = -1
        self._last_accepted_action_at = None
        # Seamless condition rollover. A new WM image becomes CANDIDATE while
        # actions acknowledged for the old ACTIVE image keep executing. The
        # candidate is promoted only when the VLA echoes its exact id+hash.
        # This preserves continuous RTC actions without guessing from version
        # numbers or dropping control ticks at every periodic WM refresh.
        self._vla_session_id = None
        self._next_condition_id = 0
        self._active_condition = None
        self._candidate_condition = None
        self._pending_condition_since = None
        self._last_condition_log_at = -float("inf")
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
        with self._publish_lock:
            self._stop_or_hold_wbc_locked(reason)

    def _stop_or_hold_wbc_locked(self, reason):
        """Locked implementation; caller owns ``_publish_lock``."""
        if self._dry_run or self._token_publisher is None:
            if reason != self._last_hold_reason:
                self._last_hold_reason = reason
                # Hold reasons embed live ages, so plain reason-change logging
                # would print at the 30 Hz message rate during one hold.
                now = time.monotonic()
                if now - self._last_dry_hold_log_at >= 2.0:
                    self._last_dry_hold_log_at = now
                    print(f"[safety] dry-run hold: {reason}", flush=True)
                    log_event("hold", reason=str(reason), dry_run=True)
            return
        if self._wbc_started:
            try:
                self._token_publisher.send_command(
                    start=False, stop=True, planner=True
                )
            finally:
                self._wbc_started = False
            print(f"[safety] WBC stopped/held: {reason}", flush=True)
            log_event("wbc_stop", reason=str(reason))
        self._last_hold_reason = reason

    def _clear_action_conditions_locked(self):
        """Make every previously acknowledged condition non-executable.

        Caller owns ``_publish_lock``; taking the action-state lock in this
        order matches the liveness and publication paths.
        """
        with self._action_state_lock:
            self._active_condition = None
            self._candidate_condition = None
            self._pending_condition_since = None
            self._last_accepted_action_at = None

    def _expire_wm_goal_locked(self, wm):
        """Hold and poison ACTIVE when the same-prompt last-good goal expires."""
        if not wm["goal_expired"]:
            return False
        reason = (
            f"WM last-good goal expired ({wm['goal_age_s']:.1f}s > "
            f"{wm['goal_hard_age_s']:.1f}s)"
        )
        self._stop_or_hold_wbc_locked(reason)
        # Without this, a later fresh goal could restart WBC on an old ACTIVE
        # ack before the fresh candidate was ever acknowledged.
        self._clear_action_conditions_locked()
        return True

    def _hold_for_observed_wm_expiry(self, observed_wm):
        """Latch an observed expiry even if a fresh goal lands milliseconds later.

        Once hard age was observed, old ACTIVE must be poisoned and the fresh
        generation must earn a new exact ack. Otherwise a concurrent WM refresh
        could revive old actions before that candidate is acknowledged.
        """
        with self._publish_lock:
            current = self._wm.snapshot()
            if current["goal_expired"]:
                return self._expire_wm_goal_locked(current)
            if not observed_wm["goal_expired"]:
                return False
            self._stop_or_hold_wbc_locked(
                "WM last-good goal crossed hard age; requiring a fresh "
                "condition acknowledgement"
            )
            self._clear_action_conditions_locked()
            return True

    def apply_prompt_transition(self, label, transition, *args):
        """Linearize an operator prompt command against obs/action publication.

        A publication already holding its fence completes before the epoch bump;
        otherwise the bump wins and no old-epoch observation, body token, hand or
        neck command can be emitted after this method returns.
        """
        # No hot-loop path nests these locks in the reverse order.  Taking both
        # only on rare operator input also avoids adding work to the 30 Hz path.
        with self._send_lock:
            with self._publish_lock:
                before = self._wm.snapshot()
                transition(*args)
                after = self._wm.snapshot()
                changed = (
                    int(after["prompt_epoch"]) != int(before["prompt_epoch"])
                )
                if changed:
                    self._stop_or_hold_wbc_locked(
                        f"prompt transition {label}: epoch "
                        f"{before['prompt_epoch']} -> {after['prompt_epoch']}"
                    )
                    # Preserve the session/id counter, but make every old
                    # condition ack non-executable immediately.
                    self._clear_action_conditions_locked()
                    with self._telemetry_lock:
                        self._telemetry_epoch = int(after["prompt_epoch"])
                        self._prev_exec_action = None
                        self._tok_delta_max = 0.0
                        self._hand_delta_max = 0.0
                        self._neck_delta_max = 0.0
                        self._fsq_static_ticks = 0
                        self._delta_ticks = 0
                        self._tok_delta_window = []
                        self._fsq_changed_sum = 0
                        self._fsq_changed_max = 0
                        self._fsq_static_run = 0
                        self._fsq_static_run_max = 0
                        self._rtc_infer_ms_last = None
                        self._same_condition_stall_s = 0.0
                        self._global_low_progress_s = 0.0
                        self._recv_interval_max = 0.0
                        self._version_gap_total = 0
                        self._version_gap_max = 0
                        self._rtc_chunk_id = None
                        self._rtc_chunk_tick = None
                        self._rtc_chunk_switches = 0
                        self._rtc_repeat_last_ticks = 0
                    print(
                        f"[condition] atomic prompt transition {label}: "
                        f"epoch={after['prompt_epoch']}", flush=True
                    )
                    log_event(
                        "prompt_transition", label=str(label),
                        epoch_before=int(before["prompt_epoch"]),
                        epoch_after=int(after["prompt_epoch"]),
                        stage=after["prompt_stage"], subtask=after["subtask"],
                    )
                return changed

    def _ensure_wbc_started(self):
        if self._dry_run or self._token_publisher is None:
            return
        if not self._wbc_started:
            self._token_publisher.send_command(start=True, stop=False, planner=True)
            self._wbc_started = True
            print("[safety] first fresh valid action -> WBC planner start", flush=True)
            log_event("wbc_start")
        self._last_hold_reason = None

    def _freshness(self):
        now = time.monotonic()
        with self._last_observation_lock:
            obs_at = self._last_observation_at
        obs_age = float("inf") if obs_at is None else now - obs_at
        state_age = self._state_sub.age()
        camera_age = self._camera.age()
        neck_age = (self._neck_state_reader.age()
                    if self._include_neck else 0.0)
        fresh = max(obs_age, state_age, camera_age, neck_age) \
            <= self._observation_stale_timeout
        return fresh, obs_age, state_age, camera_age, neck_age

    def _reset_condition_session(self):
        """Start a fresh provenance namespace for this WebSocket connection."""
        with self._action_state_lock:
            self._vla_session_id = new_vla_session_id()
            self._next_condition_id = 0
            self._active_condition = None
            self._candidate_condition = None
            self._pending_condition_since = None
            self._last_received_version = -1
            sid = self._vla_session_id
        with self._telemetry_lock:
            self._telemetry_last_version = None
            self._telemetry_epoch = None
        return sid

    @staticmethod
    def _condition_content_key(condition):
        if condition is None:
            return None
        return (int(condition["prompt_epoch"]),
                int(condition["goal_generation"]))

    def _condition_for_send(self, wm, instruction, goal):
        """Return one atomic instruction/goal/provenance snapshot to send.

        Periodic WM refreshes mint a candidate but deliberately retain the old
        active condition. Until an action carrying the candidate's exact ack
        arrives, old-active actions remain executable with no version barrier.
        """
        key = (int(wm["prompt_epoch"]), int(wm["goal_generation"]))
        with self._action_state_lock:
            if self._vla_session_id is None:
                return None
            candidate = self._candidate_condition
            if (candidate is not None
                    and int(candidate["prompt_epoch"]) == key[0]):
                # Do not overwrite an in-flight same-epoch candidate when a
                # newer WM generation lands. The VLA may still be computing the
                # exact ack for this candidate; losing it makes valid actions
                # match neither CANDIDATE nor ACTIVE and recreates a periodic
                # discard/hold. Once it promotes, the next send mints only the
                # latest WM generation (intermediate generations may be skipped).
                return candidate
            if self._condition_content_key(self._active_condition) == key:
                return self._active_condition

            cid = self._next_condition_id
            self._next_condition_id += 1
            now = time.monotonic()
            candidate = {
                "sid": self._vla_session_id,
                "cid": cid,
                "hash": condition_hash(instruction, goal),
                "prompt_epoch": key[0],
                "goal_generation": key[1],
                "prompt_stage": int(wm["prompt_stage"]),
                "instruction": instruction,
                # WM goals are immutable after publication, so retaining the
                # array reference keeps image/hash/content atomically aligned.
                "goal": goal,
                "minted_at": now,
            }
            # One unbroken streak of unpromoted same-epoch candidates shares a
            # start time. If the streak outlives the promote timeout, the
            # server never switched to any of them and old-ACTIVE acceptance
            # must stop instead of masking a repeat-last frozen stream.
            self._pending_condition_since = now
            self._candidate_condition = candidate
        print(
            f"[condition] candidate cid={cid} stage={candidate['prompt_stage']} "
            f"epoch={key[0]} goal_gen={key[1]} "
            f"hash={candidate['hash'][:12]}", flush=True
        )
        log_event(
            "condition_candidate", cid=cid, epoch=key[0], generation=key[1],
            stage=candidate["prompt_stage"], hash=candidate["hash"][:16],
        )
        return candidate

    def _accept_action_for_condition(self, version, ack, wm):
        """Classify an action by exact provenance, without dropping old-active ticks.

        Returns ``(decision, version, reason, active_for_current_prompt)`` where
        decision is ``promoted``, ``active``, ``starved`` or ``discarded``.
        """
        if isinstance(version, bool) or not isinstance(version, (int, np.integer)):
            raise ValueError(f"invalid action version {version!r}")
        version = int(version)
        if not isinstance(ack, dict) or not ack:
            raise ValueError("VLA action is missing condition provenance ack")
        ack_version = ack.get("action_version")
        if (isinstance(ack_version, bool) or not isinstance(ack_version, int)
                or ack_version != version):
            raise ValueError(
                f"condition ack version {ack_version!r} != action version {version}"
            )
        current_epoch = int(wm["prompt_epoch"])
        now = time.monotonic()
        with self._action_state_lock:
            if version <= self._last_received_version:
                raise ValueError(
                    f"non-monotonic action version {version} <= {self._last_received_version}"
                )
            self._last_received_version = version
            candidate = self._candidate_condition
            active = self._active_condition
            active_for_current_prompt = (
                active is not None
                and int(active["prompt_epoch"]) == current_epoch
            )

            # Promote on an exact current-epoch candidate ack even if a newer
            # WM generation landed a moment ago: a same-epoch previous-goal
            # condition is exactly as executable as the old ACTIVE it replaces,
            # and the next send tick mints the newer generation as the next
            # candidate. Requiring generation equality here only converted that
            # benign race into a dropped ack and a watchdog stop.
            if (candidate is not None
                    and int(candidate["prompt_epoch"]) == current_epoch
                    and ack_matches(
                        ack,
                        vla_session_id=candidate["sid"],
                        condition_id=candidate["cid"],
                        supplied_condition_hash=candidate["hash"],
                    )):
                self._active_condition = candidate
                self._candidate_condition = None
                self._pending_condition_since = None
                return "promoted", version, None, True

            if (active_for_current_prompt
                    and ack_matches(
                        ack,
                        vla_session_id=active["sid"],
                        condition_id=active["cid"],
                        supplied_condition_hash=active["hash"],
                    )):
                # Old-ACTIVE continuation is only safe while the pending
                # candidate is young. A server that stopped installing new
                # chunks (superseded storm, wedged GPU) keeps streaming
                # valid old acks -- possibly a run-out chunk repeating one
                # frozen action -- so cap how long they stay acceptable.
                pending_since = self._pending_condition_since
                if (pending_since is not None
                        and now - pending_since > self._condition_promote_timeout):
                    reason = (
                        f"no candidate acknowledged for {now - pending_since:.1f}s "
                        f"(> {self._condition_promote_timeout:.1f}s); refusing to keep "
                        f"executing old-ACTIVE cid={active['cid']}"
                    )
                    return "starved", version, reason, True
                return "active", version, None, True

            reason = (
                f"ack cid={ack.get('action_condition_id')!r} does not match "
                f"candidate={None if candidate is None else candidate['cid']} "
                f"or current active={None if active is None else active['cid']}"
            )
            return "discarded", version, reason, active_for_current_prompt

    def _record_action_accepted(self, now):
        with self._action_state_lock:
            self._last_accepted_action_at = now

    def _ack_is_still_active(self, ack, prompt_epoch):
        """Final same-fence validation after any prompt/expiry poisoning."""
        with self._action_state_lock:
            active = self._active_condition
            return bool(
                active is not None
                and int(active["prompt_epoch"]) == int(prompt_epoch)
                and ack_matches(
                    ack,
                    vla_session_id=active["sid"],
                    condition_id=active["cid"],
                    supplied_condition_hash=active["hash"],
                )
            )

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

    def _validated_action(self, action):
        if not isinstance(action, np.ndarray):
            raise ValueError(f"action is not a numpy array: {type(action).__name__}")
        expected_dim = ACTION_DIM_NECK if self._include_neck else ACTION_DIM_DEFAULT
        if action.shape != (1, expected_dim):
            raise ValueError(
                f"action shape {action.shape}, expected (1, {expected_dim})"
            )
        if not np.issubdtype(action.dtype, np.number):
            raise ValueError(f"action dtype is not numeric: {action.dtype}")
        if not np.isfinite(action).all():
            raise ValueError("action contains NaN or Inf")
        return np.asarray(action, dtype=np.float32)

    def execute_action(self, action):
        """
        Map the server action -> robot command and publish via Protocol v4.

        Server action layout is [hand_joints(14) | body_token(64)] (78-D),
        with neck(2) appended for the 80-D neck policy.
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
        if self._include_neck:
            if self._neck_publisher is None:
                raise RuntimeError("live neck mode has no NeckPublisher")
            neck = action[HAND_DIM + TOKEN_DIM:HAND_DIM + TOKEN_DIM + NECK_DIM]
            self._neck_publisher.publish(neck[0], neck[1])

    def _note_condition_promoted(self):
        """Restart per-condition raw/FSQ aggregators at a promote boundary."""
        with self._telemetry_lock:
            self._prev_exec_action = None
            self._tok_delta_max = 0.0
            self._hand_delta_max = 0.0
            self._neck_delta_max = 0.0
            self._fsq_static_ticks = 0
            self._delta_ticks = 0
            self._tok_delta_window = []
            self._fsq_changed_sum = 0
            self._fsq_changed_max = 0
            self._fsq_static_run = 0
            self._fsq_static_run_max = 0
            self._same_condition_stall_s = 0.0

    def mark_incident(self, label):
        """Operator label (plan P0.3): event + flight-recorder dump, no control."""
        print(f"[mark] {label}", flush=True)
        log_event("operator_mark", label=str(label))
        if self._incidents is not None:
            self._incidents.dump(label, min_interval_s=0.0)

    def _record_received_version(self, version):
        """O(1) Event-coalescing counter; no logging or I/O on the hot path."""
        with self._telemetry_lock:
            if self._telemetry_last_version is not None:
                gap = max(
                    0, int(version) - int(self._telemetry_last_version) - 1
                )
                self._version_gap_total += gap
                self._version_gap_max = max(self._version_gap_max, gap)
            self._telemetry_last_version = int(version)

    def _update_action_telemetry(self, data, action, version, interval, now,
                                 prompt_epoch, active_cid=None):
        """Update scalar diagnostics; return ``(lines_to_print, shadow_fired)``.

        The epoch tag prevents an old callback that finished publication just
        before Enter from repopulating counters after the transition reset.
        """
        with self._telemetry_lock:
            prompt_epoch = int(prompt_epoch)
            if self._telemetry_epoch is None:
                self._telemetry_epoch = prompt_epoch
            elif int(self._telemetry_epoch) != prompt_epoch:
                return [], False

            self._recv_interval_max = max(
                self._recv_interval_max, float(interval)
            )
            chunk_id = data.get("rtc_chunk_id")
            chunk_tick = data.get("rtc_chunk_tick")
            if (isinstance(chunk_id, int) and not isinstance(chunk_id, bool)
                    and isinstance(chunk_tick, int)
                    and not isinstance(chunk_tick, bool)):
                if (self._rtc_chunk_id is not None
                        and int(chunk_id) != int(self._rtc_chunk_id)):
                    self._rtc_chunk_switches += 1
                self._rtc_chunk_id = int(chunk_id)
                self._rtc_chunk_tick = int(chunk_tick)
                if data.get("rtc_repeat_last") is True:
                    self._rtc_repeat_last_ticks += 1

            if "rtc_infer_ms" in data:
                try:
                    self._rtc_infer_ms_last = float(data["rtc_infer_ms"])
                except (TypeError, ValueError):
                    pass

            flat = action[0]
            if self._prev_exec_action is not None:
                tok_new = flat[HAND_DIM:HAND_DIM + TOKEN_DIM]
                tok_prev = self._prev_exec_action[HAND_DIM:HAND_DIM + TOKEN_DIM]
                tok_delta = float(np.abs(tok_new - tok_prev).max())
                self._tok_delta_max = max(self._tok_delta_max, tok_delta)
                self._tok_delta_window.append(tok_delta)
                self._hand_delta_max = max(
                    self._hand_delta_max,
                    float(np.abs(
                        flat[:HAND_DIM] - self._prev_exec_action[:HAND_DIM]
                    ).max()),
                )
                if self._include_neck:
                    self._neck_delta_max = max(
                        self._neck_delta_max,
                        float(np.abs(
                            flat[HAND_DIM + TOKEN_DIM:]
                            - self._prev_exec_action[HAND_DIM + TOKEN_DIM:]
                        ).max()),
                    )
                changed_dims = int(np.count_nonzero(
                    fsq_quantize(tok_new) != fsq_quantize(tok_prev)
                ))
                self._fsq_changed_sum += changed_dims
                self._fsq_changed_max = max(self._fsq_changed_max, changed_dims)
                if changed_dims == 0:
                    self._fsq_static_ticks += 1
                    self._fsq_static_run += 1
                    self._fsq_static_run_max = max(
                        self._fsq_static_run_max, self._fsq_static_run
                    )
                else:
                    self._fsq_static_run = 0
                self._delta_ticks += 1
            self._prev_exec_action = flat.copy()

            self._action_count += 1
            elapsed = now - self._action_rate_started_at
            should_log = (
                self._action_count == 1
                or now - self._last_action_log_at >= 1.0
            )
            if not should_log:
                return [], False

            hz = self._action_count / max(elapsed, 1e-6)
            window = self._tok_delta_window
            raw_p50 = float(np.median(window)) if window else 0.0
            raw_p95 = float(np.percentile(window, 95)) if window else 0.0
            static_ratio = (self._fsq_static_ticks / self._delta_ticks
                            if self._delta_ticks else 0.0)
            cam_motion = self._cam_motion_1s
            sample_dt = now - self._last_action_log_at
            if not 0.0 < sample_dt <= 3.0:
                sample_dt = 1.0
            low_progress = (
                self._delta_ticks >= 10
                and raw_p95 < STALL_SHADOW_RAW_P95_MAX
                and static_ratio >= STALL_SHADOW_FSQ_RATIO_MIN
                and cam_motion is not None
                and cam_motion < STALL_SHADOW_CAM_MOTION_MAX
            )
            if low_progress:
                self._same_condition_stall_s += sample_dt
                self._global_low_progress_s += sample_dt
            else:
                self._same_condition_stall_s = 0.0
                self._global_low_progress_s = 0.0
            shadow_fired = False
            lines = []
            if (low_progress
                    and self._same_condition_stall_s >= STALL_SHADOW_MIN_S
                    and now - self._stall_shadow_printed_at >= 5.0):
                self._stall_shadow_printed_at = now
                shadow_fired = True
                lines.append(
                    f"[stall-shadow] would_trigger "
                    f"same_condition_stall={self._same_condition_stall_s:.1f}s "
                    f"global_low_progress={self._global_low_progress_s:.1f}s "
                    f"raw_p95={raw_p95:.4f} fsq_static_ratio={static_ratio:.2f} "
                    f"cam_motion={cam_motion:.1f} cid={active_cid} (print-only)"
                )
                log_event(
                    "stall_shadow", cid=active_cid, epoch=prompt_epoch,
                    same_condition_stall_s=round(self._same_condition_stall_s, 2),
                    global_low_progress_s=round(self._global_low_progress_s, 2),
                    raw_p95=raw_p95, fsq_static_ratio=round(static_ratio, 3),
                    cam_motion=cam_motion,
                )
            mode = "dry-run validate" if self._dry_run else "published"
            lines.append(
                f"[client] action {mode}: "
                f"t={datetime.now().strftime('%H:%M:%S')} "
                f"version={version} cid={active_cid} shape={action.shape} "
                f"range=[{action.min():.3f},{action.max():.3f}] "
                f"recv_interval={interval:.3f}s "
                f"recv_imax={self._recv_interval_max:.3f}s avg_hz={hz:.1f} "
                f"version_gap={self._version_gap_total}/{self._version_gap_max} "
                f"rtc_chunk={self._rtc_chunk_id}/{self._rtc_chunk_tick} "
                f"chunk_switches={self._rtc_chunk_switches} "
                f"repeat_last={self._rtc_repeat_last_ticks} "
                f"infer_ms={self._rtc_infer_ms_last} "
                f"raw_tok_dmax={self._tok_delta_max:.4f} "
                f"raw_p95={raw_p95:.4f} "
                f"hand_dmax={self._hand_delta_max:.4f} "
                f"neck_dmax={self._neck_delta_max:.4f} "
                f"fsq_static={self._fsq_static_ticks}/{self._delta_ticks} "
                f"chg_max={self._fsq_changed_max} "
                f"static_run_max={self._fsq_static_run_max} "
                f"cam={'n/a' if cam_motion is None else f'{cam_motion:.1f}'} "
                f"scs={self._same_condition_stall_s:.1f}s "
                f"glp={self._global_low_progress_s:.1f}s"
            )
            log_event(
                "telemetry_1s", cid=active_cid, epoch=prompt_epoch,
                version=int(version), hz=round(hz, 2),
                recv_interval_max=round(self._recv_interval_max, 4),
                version_gap_total=self._version_gap_total,
                version_gap_max=self._version_gap_max,
                chunk_id=self._rtc_chunk_id, chunk_tick=self._rtc_chunk_tick,
                chunk_switches=self._rtc_chunk_switches,
                repeat_last_ticks=self._rtc_repeat_last_ticks,
                infer_ms=self._rtc_infer_ms_last,
                raw_p50=raw_p50, raw_p95=raw_p95,
                raw_max=self._tok_delta_max,
                hand_dmax=self._hand_delta_max,
                neck_dmax=self._neck_delta_max,
                fsq_static_ticks=self._fsq_static_ticks,
                ticks=self._delta_ticks,
                fsq_changed_sum=self._fsq_changed_sum,
                fsq_changed_max=self._fsq_changed_max,
                fsq_static_run_max=self._fsq_static_run_max,
                cam_motion=cam_motion, low_progress=low_progress,
                same_condition_stall_s=round(self._same_condition_stall_s, 2),
                global_low_progress_s=round(self._global_low_progress_s, 2),
            )
            self._last_action_log_at = now
            self._tok_delta_max = 0.0
            self._hand_delta_max = 0.0
            self._neck_delta_max = 0.0
            self._fsq_static_ticks = 0
            self._delta_ticks = 0
            self._tok_delta_window = []
            self._fsq_changed_sum = 0
            self._fsq_changed_max = 0
            # Keep an ongoing zero-change run visible across window boundaries.
            self._fsq_static_run_max = self._fsq_static_run
            self._recv_interval_max = 0.0
            self._version_gap_total = 0
            self._version_gap_max = 0
            self._rtc_chunk_switches = 0
            self._rtc_repeat_last_ticks = 0
            return lines, shadow_fired

    def _on_open(self, ws):
        sid = self._reset_condition_session()
        print(f"[client] Connected! condition_session={sid[:12]}")
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

                fresh, obs_age, state_age, camera_age, neck_age = self._freshness()
                wm = self._wm.snapshot()
                wm_condition_ready = (
                    wm["goal"] is not None
                    and not wm["goal_stale"]
                    and not wm["goal_expired"]
                )
                if not wm_condition_ready:
                    if wm["goal_expired"]:
                        self._hold_for_observed_wm_expiry(wm)
                    else:
                        self._stop_or_hold_wbc(
                            "no WM goal for the current prompt"
                        )
                    return
                if not fresh:
                    self._stop_or_hold_wbc(
                        "stale observation "
                        f"obs={obs_age:.3f}s state={state_age:.3f}s "
                        f"camera={camera_age:.3f}s neck={neck_age:.3f}s "
                        f"limit={self._observation_stale_timeout:.3f}s"
                    )
                    return

                ack = {key: data[key] for key in ACTION_ACK_KEYS if key in data}
                try:
                    decision, version, reject_reason, active_for_prompt = \
                        self._accept_action_for_condition(version, ack, wm)
                except ValueError as exc:
                    self._stop_or_hold_wbc(f"invalid VLA action stream: {exc}")
                    print(f"[client] ERROR: rejected action: {exc}", flush=True)
                    return
                self._record_received_version(version)
                if decision == "starved":
                    # The server kept streaming valid old-ACTIVE acks but never
                    # switched to any pending candidate: treat as a liveness
                    # failure (repeat-last chunk looks exactly like this) and
                    # hold instead of executing a possibly frozen stream.
                    self._stop_or_hold_wbc(
                        f"condition rollover starved: {reject_reason}"
                    )
                    if now - self._last_condition_log_at >= 1.0:
                        self._last_condition_log_at = now
                        print(
                            f"[condition] starved action version={version}: "
                            f"{reject_reason}", flush=True
                        )
                        log_event("condition_starved", version=int(version),
                                  reason=str(reject_reason))
                    return
                if decision == "discarded":
                    # During a same-stage candidate rollover the current active
                    # plan remains valid. A superseded/unknown ack is discarded,
                    # but must not stop the continuous old-active action stream.
                    if not active_for_prompt:
                        self._stop_or_hold_wbc(
                            "waiting for an acknowledged action for the current prompt"
                        )
                    if now - self._last_condition_log_at >= 1.0:
                        self._last_condition_log_at = now
                        print(
                            f"[condition] discarded action version={version}: "
                            f"{reject_reason}", flush=True
                        )
                        log_event("condition_discarded", version=int(version),
                                  reason=str(reject_reason),
                                  active_for_prompt=bool(active_for_prompt))
                    return
                if decision == "promoted":
                    with self._action_state_lock:
                        active = self._active_condition
                    # Enter may atomically clear ACTIVE immediately after the
                    # promotion. Publication will reject the old epoch below;
                    # the diagnostic must not dereference that mutable slot.
                    if active is None:
                        print(
                            f"[condition] promoted action version={version} was "
                            "superseded by a prompt transition before publication",
                            flush=True,
                        )
                    else:
                        print(
                            f"[condition] promoted cid={active['cid']} "
                            f"epoch={active['prompt_epoch']} "
                            f"goal_gen={active['goal_generation']} on version={version} "
                            f"latency={now - active['minted_at']:.3f}s",
                            flush=True,
                        )
                        log_event(
                            "condition_promoted", cid=active["cid"],
                            epoch=active["prompt_epoch"],
                            generation=active["goal_generation"],
                            version=int(version),
                            latency_s=round(now - active["minted_at"], 4),
                        )
                    # One telemetry sample must never mix two conditions
                    # (plan P0.4): restart the raw/FSQ aggregators here.
                    self._note_condition_promoted()

                # The epoch check and body/hand/neck publication share the same
                # fence as operator prompt mutation.  This closes the old TOCTOU
                # where Enter could return between a check and execute_action().
                with self._publish_lock:
                    latest_wm = self._wm.snapshot()
                    if (int(latest_wm["prompt_epoch"])
                            != int(wm["prompt_epoch"])):
                        self._stop_or_hold_wbc_locked(
                            "prompt changed before action publication"
                        )
                        return
                    if self._expire_wm_goal_locked(latest_wm):
                        return
                    if not self._ack_is_still_active(
                            ack, wm["prompt_epoch"]):
                        self._stop_or_hold_wbc_locked(
                            "accepted VLA condition was invalidated before publication"
                        )
                        return
                    if not self._dry_run:
                        self._ensure_wbc_started()
                    self.execute_action(action)
                    self._record_action_accepted(now)

                if self._incidents is not None:
                    self._incidents.record_action(
                        now, int(version), data.get("action_condition_id"),
                        data.get("rtc_chunk_id"), data.get("rtc_chunk_tick"),
                        bool(data.get("rtc_repeat_last")), action[0].copy(),
                    )
                action_lines, shadow_fired = self._update_action_telemetry(
                    data, action, version, interval, now, wm["prompt_epoch"],
                    active_cid=data.get("action_condition_id"),
                )
                for line in action_lines:
                    # Formatting/state aggregation happened under the tiny
                    # telemetry lock; terminal I/O deliberately stays outside.
                    print(line, flush=True)
                if shadow_fired and self._incidents is not None:
                    # Print-only detector, but the flight recorder still saves
                    # the surrounding window for offline label alignment.
                    self._incidents.dump("stall_shadow", min_interval_s=30.0)

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

    def _build_state(self, state):
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
        if self._include_neck:
            neck_latest = self._neck_state_reader.get_latest()
            if neck_latest is None:
                raise ValueError("no neck state yet")
            neck_age = self._neck_state_reader.age()
            if neck_age > self._observation_stale_timeout:
                raise ValueError(
                    f"neck state stale ({neck_age:.3f}s > "
                    f"{self._observation_stale_timeout:.3f}s)"
                )
            neck_state = np.asarray(neck_latest, dtype=np.float32).reshape(-1)
            if neck_state.shape != (NECK_DIM,) or not np.isfinite(neck_state).all():
                raise ValueError(
                    f"neck state must be a finite ({NECK_DIM},) vector, got "
                    f"{neck_state.shape}"
                )
            states = np.concatenate((states, neck_state), axis=0)
        expected_dim = 45 if self._include_neck else 43
        if states.shape != (expected_dim,):
            raise ValueError(
                f"model state shape {states.shape}, expected ({expected_dim},)"
            )
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
                if self._incidents is not None:
                    self._incidents.record_state(time.monotonic(), states)

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

                # ~3 Hz flight-recorder thumbnails and a ~1 Hz camera-motion
                # sample for the shadow stall detector; small resizes only.
                self._cam_tick += 1
                if self._cam_tick % 10 == 0:
                    thumb = cv2.resize(frame, WM_GATE_GRAY_SIZE)
                    if self._incidents is not None:
                        self._incidents.record_frame(time.monotonic(), thumb)
                    if self._cam_tick % 30 == 0:
                        gray = cv2.cvtColor(
                            thumb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                        if self._cam_prev_gray is not None:
                            motion = float(
                                np.abs(gray - self._cam_prev_gray).mean())
                            with self._telemetry_lock:
                                self._cam_motion_1s = motion
                        self._cam_prev_gray = gray

                wm = self._wm.snapshot()
                subgoal_frame = wm["goal"]
                if subgoal_frame is None or wm["goal_stale"] or wm["goal_expired"]:
                    if wm["goal_expired"]:
                        hold_reason = (
                            f"WM last-good goal expired ({wm['goal_age_s']:.1f}s > "
                            f"{wm['goal_hard_age_s']:.1f}s)"
                        )
                        self._hold_for_observed_wm_expiry(wm)
                    else:
                        hold_reason = "waiting for WM goal for the current prompt"
                        self._stop_or_hold_wbc(hold_reason)
                    now = time.monotonic()
                    if now - self._last_gate_log_at >= 2.0:
                        self._last_gate_log_at = now
                        print(
                            f"[client] gated: waiting for current-prompt WM goal; "
                            f"stage={wm['prompt_stage']} epoch={wm['prompt_epoch']} "
                            f"goal_stage={wm['goal_stage']} "
                            f"goal_age={wm['goal_age_s']} expired={wm['goal_expired']} "
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

                # One immutable condition snapshot owns the language, goal
                # image and provenance id/hash together. A periodic refresh
                # becomes a candidate; the previously active condition keeps
                # executing until an exactly matching candidate action arrives.
                # Training's PsixModelTransform lowercases BOTH the task text
                # and the subtask text before assembling the instruction
                # ("Task: {instruction.lower()}. Subtask: {sub_task.lower()}"),
                # so the served instruction must be lowercased the same way.
                # The /wm request keeps the original casing: the WM prompt
                # builder normalizes its own text.
                task = str(self._task).strip().lower()
                subtask = str(wm["subtask"]).strip().lower()
                instruction = (
                    f"Task: {task}. Subtask: {subtask}"
                    if subtask else f"Task: {task}"
                )
                send_condition = self._condition_for_send(
                    wm, instruction, subgoal_frame
                )
                if send_condition is None:
                    self._stop_or_hold_wbc("no VLA condition session")
                    continue
                subgoal_frame = send_condition["goal"]
                instruction = send_condition["instruction"]

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
                    if self._include_neck:
                        init_prev_action = np.concatenate(
                            [init_prev_action, states[-NECK_DIM:]]).astype(np.float32)
                    expected_dim = (ACTION_DIM_NECK if self._include_neck
                                    else ACTION_DIM_DEFAULT)
                    if (init_prev_action.shape != (expected_dim,) or
                            not np.isfinite(init_prev_action).all()):
                        raise ValueError(
                            f"init_prev_action must be a finite {expected_dim}-D vector"
                        )
                    state_obs["init_prev_action"] = init_prev_action
                    init_prev_in_payload = True

                payload = {
                    "image": img_obs,
                    "state": state_obs,
                    "gt_action": None,
                    "dataset_name": None,
                    "instruction": instruction,
                    "history": None,
                    "condition": build_condition(
                        send_condition["sid"],
                        send_condition["cid"],
                        send_condition["hash"],
                    ),
                    "timestamp": None,
                }
                payload = convert_numpy_in_dict(payload, numpy_serialize)
                message = json.dumps(payload)

                # Send (thread-safe). Operator prompt mutation takes this same
                # fence, so the final epoch check cannot race Enter/:ov/:resume.
                with self._send_lock:
                    latest_wm = self._wm.snapshot()
                    if (
                        int(latest_wm["prompt_epoch"])
                        != int(send_condition["prompt_epoch"])
                        or latest_wm["goal_expired"]
                    ):
                        continue
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send(message)
                        with self._last_observation_lock:
                            self._last_observation_at = time.monotonic()
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
         goal_source, episode_dir,
         wm_base_url, wm_period, wm_timeout, jpeg_quality, wm_stale_warn,
         wm_goal_hard_age, wm_dump_dir,
         observation_stale_timeout, action_stale_timeout,
         condition_promote_timeout=None,
         dry_run=False,
         include_neck=False, neck_pub_host=DEFAULT_NECK_PUB_HOST,
         neck_pub_port=DEFAULT_NECK_PUB_PORT,
         neck_state_zmq=DEFAULT_NECK_STATE_ZMQ):
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

    camera_cls = ZedNeckCamera if include_neck else RSCamera
    camera = camera_cls(address=camera_address, timeout_ms=camera_timeout_ms)
    print(
        f"[MAIN] Camera connected to {camera_address} "
        f"(include_neck={include_neck}, request timeout={camera_timeout_ms}ms)"
    )

    neck_publisher = None
    neck_state_reader = None
    if include_neck:
        neck_state_reader = NeckStateReader(neck_state_zmq)
        if not dry_run:
            neck_publisher = NeckPublisher(host=neck_pub_host, port=neck_pub_port)
        print(
            f"[MAIN] Neck state connected to {neck_state_zmq}; "
            + ("publication disabled by dry-run" if dry_run else
               f"publisher bound on {neck_pub_host}:{neck_pub_port}")
        )

    if goal_source == "episode":
        wm_provider = EpisodeSubgoalProvider(
            episode_dir=episode_dir,
            subtasks=subtasks,
            task=task_instruction,
        )
        print(
            f"[MAIN] GT episode provider: {episode_dir}; "
            f"{len(subtasks)} fixed prompt/image stages; WM disabled"
        )
    else:
        wm_provider = WmSubgoalProvider(
            base_url=wm_base_url,
            subtasks=subtasks,
            task=task_instruction,
            period=wm_period,
            timeout=wm_timeout,
            jpeg_quality=jpeg_quality,
            stale_warn=wm_stale_warn,
            goal_hard_age=wm_goal_hard_age,
            dump_dir=wm_dump_dir,
        )
        print(f"[MAIN] WM provider: {wm_base_url}; {len(subtasks)} prompt stages")
        print(
            f"[MAIN] WM last-good hard age: "
            + (f"{wm_goal_hard_age:.1f}s" if wm_goal_hard_age > 0 else "disabled")
        )
    print(f"[MAIN] Task instruction: {task_instruction!r}")
    if condition_promote_timeout is None:
        condition_promote_timeout = (
            2.0 if goal_source == "episode" else max(2.0 * wm_period, 2.0)
        )
    print(
        f"[MAIN] Condition promote timeout: {condition_promote_timeout:.1f}s "
        f"(unacknowledged candidates older than this hold WBC)"
    )

    # Flight recorder + run identity (plan P0.3). All best-effort: a failed
    # manifest field becomes an error record, never a startup failure.
    run_dir = os.path.abspath(os.path.expanduser(wm_dump_dir))
    os.makedirs(run_dir, exist_ok=True)
    event_log = EventLog(os.path.join(run_dir, "events.jsonl"))
    set_event_log(event_log)
    incident_recorder = IncidentRecorder(run_dir)
    vla_info = _fetch_json(
        server_url.replace("ws://", "http://").replace("/ws", "/info"))
    wm_state = (
        wm_provider.provenance()
        if goal_source == "episode"
        else _fetch_json(f"{wm_base_url}/state")
    )
    manifest_path = write_run_manifest(
        run_dir,
        config={
            "server_url": server_url,
            "goal_source": goal_source,
            "episode_dir": episode_dir if goal_source == "episode" else None,
            "wm_base_url": wm_base_url if goal_source == "wm" else None,
            "task_instruction": task_instruction,
            "subtasks": subtasks,
            "wm_period": wm_period,
            "wm_timeout": wm_timeout,
            "jpeg_quality": jpeg_quality,
            "wm_goal_hard_age": (
                wm_goal_hard_age if goal_source == "wm" else 0.0
            ),
            "condition_promote_timeout": condition_promote_timeout,
            "observation_stale_timeout": observation_stale_timeout,
            "action_stale_timeout": action_stale_timeout,
            "dry_run": bool(dry_run),
            "include_neck": bool(include_neck),
            "camera_address": camera_address,
            "gate_version": (
                WM_GATE_VERSION if goal_source == "wm"
                else "bypass-trusted-episode-gt"
            ),
            "stall_shadow": {
                "raw_p95_max": STALL_SHADOW_RAW_P95_MAX,
                "fsq_ratio_min": STALL_SHADOW_FSQ_RATIO_MIN,
                "cam_motion_max": STALL_SHADOW_CAM_MOTION_MAX,
                "min_s": STALL_SHADOW_MIN_S,
            },
        },
        vla_info=vla_info,
        wm_state=wm_state,
        episode_session_id=wm_provider._episode_session_id,
    )
    print(f"[MAIN] Run manifest: {manifest_path}")
    log_event("run_start", manifest=manifest_path, dry_run=bool(dry_run))

    # Let the live PUB subscription settle, but send no start command yet. The
    # receive callback starts WBC only after a fresh observation + valid action.
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
        condition_promote_timeout=condition_promote_timeout,
        wm_stale_warn=wm_stale_warn,
        include_neck=include_neck,
        neck_publisher=neck_publisher,
        neck_state_reader=neck_state_reader,
        incident_recorder=incident_recorder,
    )

    def stdin_listener():
        if goal_source == "episode":
            print(
                "[MAIN] Enter: next GT goal | :restart: stage 0 | "
                ":mark LABEL: flight-recorder dump"
            )
        else:
            print(
                "[MAIN] Enter: next episode prompt | text/:ov TEXT: manual prompt | "
                ":resume: current episode prompt | :restart: stage 0 | "
                ":mark LABEL: flight-recorder dump "
                "(stall_start/stall_end/empty_grasp/scene_lost)"
            )
        while running.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:  # EOF
                break
            command = line.strip()
            if command == "":
                client.apply_prompt_transition(
                    "next", wm_provider.advance_prompt
                )
            elif command == ":restart":
                client.apply_prompt_transition(
                    "restart", wm_provider.restart
                )
            elif command == ":resume":
                client.apply_prompt_transition(
                    "resume", wm_provider.resume_scripted_prompt
                )
            elif command.startswith(":ov "):
                client.apply_prompt_transition(
                    "manual", wm_provider.takeover, command[4:]
                )
            elif command.startswith(":mark"):
                client.mark_incident(command[5:].strip() or "manual")
            elif command.startswith(":"):
                print(f"[MAIN] unknown command {command!r}")
            else:
                client.apply_prompt_transition(
                    "manual", wm_provider.takeover, command
                )

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
    if neck_publisher is not None:
        neck_publisher.stop()
    if neck_state_reader is not None:
        neck_state_reader.stop()
    log_event("shutdown")
    event_log.stop()
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RTC VLA client with remote-WM or fixed episode GT goals"
    )
    parser.add_argument("--host", type=str, default="localhost",
                        help="VLA policy server host")
    parser.add_argument("--port", type=int, default=8014,
                        help="VLA policy server port")
    parser.add_argument(
        "--embodiment-tag", default=os.environ.get("EMBODIMENT_TAG"),
        help="Expected VLA embodiment tag; defaults to EMBODIMENT_TAG. The "
             "server /info remains authoritative for the wire dimensions."
    )
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
                        default=DEFAULT_EPISODE_DIR,
                        help="Episode folder; --goal-source episode loads its "
                             "color_subgoal images as fixed GT goals")
    parser.add_argument("--prompts-json", type=str,
                        default=DEFAULT_PROMPTS_JSON,
                        help="JSON mapping task-key -> {task_description, subtasks[]}")
    parser.add_argument("--task-key", type=str, default=DEFAULT_TASK_KEY,
                        help="Key into prompts.json (e.g. pick_place_1); selects the "
                             "task_description and the per-stage subtask prompts.")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override task instruction (else taken from prompts.json[task-key])")
    parser.add_argument(
        "--goal-source", choices=("wm", "episode"), default="wm",
        help="Goal image source: remote WM (default), or fixed "
             "episode-dir/color_subgoal GT images with no WM requests",
    )
    parser.add_argument("--wm-host", type=str, default="192.168.123.240",
                        help="WM server address on the direct G1-wired subnet")
    parser.add_argument("--wm-port", type=int, default=8016,
                        help="WM HTTP port")
    parser.add_argument("--wm-period", type=float, default=3.0,
                        help="Serialized WM refresh period in seconds")
    parser.add_argument("--wm-timeout", type=float, default=15.0,
                        help="Timeout for one POST /wm request; a hung WM blocks the "
                             "serialized refresh worker this long, so keep it short")
    parser.add_argument("--jpeg-quality", type=int, default=90,
                        help="JPEG quality for ego/subgoal wired transport (1-100)")
    parser.add_argument("--wm-stale-warn", type=float, default=5.0,
                        help="Deprecated compatibility option; mismatched prompt/goal now gates immediately")
    parser.add_argument(
        "--wm-goal-hard-age", type=float, default=DEFAULT_WM_GOAL_HARD_AGE,
        help="Stop/hold only when the current prompt's last-good WM goal exceeds "
             "this age; 0 disables (default: 30s)"
    )
    parser.add_argument("--wm-dump-dir", type=str, default=DEFAULT_WM_DUMP_DIR,
                        help="Exact directory for all BAGEL response images; default has a run timestamp")
    parser.add_argument("--observation-stale-timeout", type=float, default=0.5,
                        help="Stop/hold WBC if state, camera, or last VLA observation is older")
    parser.add_argument("--action-stale-timeout", type=float, default=0.5,
                        help="Stop/hold WBC if no fresh monotonic VLA action arrives")
    parser.add_argument("--condition-promote-timeout", type=float, default=None,
                        help="Hold WBC when a pending WM candidate stays unacknowledged "
                             "longer than this; guards against a run-out chunk repeating "
                             "one action with valid old acks (default: max(2 x wm-period, 2.0))")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run camera/state + WM + VLA validation without binding or publishing WBC")
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument(
        "--include-neck", dest="include_neck", action="store_true",
        help="Require the 45-D/80-D neck layout (normally inferred from VLA /info)"
    )
    layout.add_argument(
        "--no-include-neck", dest="include_neck", action="store_false",
        help="Require the legacy 43-D/78-D layout (normally inferred from VLA /info)"
    )
    parser.set_defaults(include_neck=None)
    parser.add_argument("--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST,
                        help=f"Neck command PUB bind host (default: {DEFAULT_NECK_PUB_HOST})")
    parser.add_argument("--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT,
                        help=f"Neck command PUB port (default: {DEFAULT_NECK_PUB_PORT})")
    parser.add_argument("--neck-state-zmq", type=str, default=DEFAULT_NECK_STATE_ZMQ,
                        help=f"Neck-state SUB address (default: {DEFAULT_NECK_STATE_ZMQ})")

    args = parser.parse_args()

    if args.camera_timeout_ms <= 0:
        parser.error("--camera-timeout-ms must be positive")
    if args.wm_period <= 0 or args.wm_timeout <= 0:
        parser.error("--wm-period and --wm-timeout must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if args.wm_stale_warn < 0:
        parser.error("--wm-stale-warn must be non-negative")
    if args.wm_goal_hard_age < 0:
        parser.error("--wm-goal-hard-age must be non-negative")
    if args.observation_stale_timeout <= 0:
        parser.error("--observation-stale-timeout must be positive")
    if args.action_stale_timeout <= 0:
        parser.error("--action-stale-timeout must be positive")
    if (args.condition_promote_timeout is not None
            and args.condition_promote_timeout <= 0):
        parser.error("--condition-promote-timeout must be positive")
    if args.goal_source == "episode":
        goal_dir = os.path.join(
            os.path.abspath(os.path.expanduser(args.episode_dir)),
            "color_subgoal",
        )
        if not os.path.isdir(goal_dir):
            parser.error(
                f"--goal-source episode requires color_subgoal/: {goal_dir}"
            )

    try:
        served_tag, state_dim, action_dim, include_neck = resolve_vla_embodiment(
            args.host, args.port, requested_tag=args.embodiment_tag,
            include_neck_override=args.include_neck)
    except RuntimeError as exc:
        parser.error(str(exc))
    print(
        f"[MAIN] VLA embodiment={served_tag} dims={state_dim}/{action_dim}; "
        f"client_layout={'45/80 neck' if include_neck else '43/78'}",
        flush=True,
    )

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
            f"task-key {args.task_key!r} has no subtasks; goal orchestration needs at least one"
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
        goal_source=args.goal_source,
        episode_dir=args.episode_dir,
        wm_base_url=f"http://{args.wm_host}:{args.wm_port}",
        wm_period=args.wm_period,
        wm_timeout=args.wm_timeout,
        jpeg_quality=args.jpeg_quality,
        wm_stale_warn=args.wm_stale_warn,
        wm_goal_hard_age=args.wm_goal_hard_age,
        wm_dump_dir=args.wm_dump_dir,
        observation_stale_timeout=args.observation_stale_timeout,
        action_stale_timeout=args.action_stale_timeout,
        condition_promote_timeout=args.condition_promote_timeout,
        dry_run=args.dry_run,
        include_neck=include_neck,
        neck_pub_host=args.neck_pub_host,
        neck_pub_port=args.neck_pub_port,
        neck_state_zmq=args.neck_state_zmq,
    )
