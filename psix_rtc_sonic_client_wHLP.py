"""RTC sonic client + HLP-backed instruction source (light client).

Based verbatim on psix_rtc_sonic_client.py — same robot I/O (78-D token+hand publish,
FSQ + encoder-freeze seed, ZMQ state/camera, WebSocket RTC transport). The ONLY added
concept: the text instruction comes from an HLP server (poll), not from a fixed
client-side subtask list. ALL subtask/memory/decision logic lives on the HLP server;
the client just forwards the `instruction` string verbatim and steps its (temporary,
client-held) goal-image list at the HLP's absolute subtask `stage`.

Controls (stdin):  :prev  previous subtask  |  :restart  clean restart  |  :resume  return to HLP
                   <text>  human takeover
                   (--no-hlp debug mode also accepts :adv to step the local subtask)

The robot-only deps (gear_sonic, encoder_client) are imported lazily so the HLP
integration (HlpController, SubgoalManager) is importable/testable without them.
"""

import os
import sys
import time
import threading
import json
import signal

import numpy as np
import requests

# robot I/O deps (present on the robot; cv2/zmq/msgpack/websocket are light enough to import here)
import cv2
import zmq
import msgpack
from websocket import WebSocketApp

_GROOT_ROOT = os.path.expanduser("~/hsc/GR00T-WholeBodyControl")
if _GROOT_ROOT not in sys.path:
    sys.path.insert(0, _GROOT_ROOT)

# gear_sonic is robot-only — import lazily so this module is importable for testing the HLP path.
try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        pack_pose_message,
        build_command_message,
    )
except Exception as _e:  # pragma: no cover - only on non-robot machines
    pack_pose_message = build_command_message = None
    print(f"[wHLP] gear_sonic unavailable ({_e}); TokenPublisher disabled (HLP path still testable)")

# ---------------- Configuration ----------------
TASK_INSTRUCTION = "pick up the green grapes and place it into the green bowl"

# FSQ configuration (must match g1_sonic_client / encoder)
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16

# cap obs send rate; server control loop runs at 30Hz, sending faster just floods it
OBS_SEND_INTERVAL = 1.0 / 30.0

ENCODER_MODEL = "gear_sonic_deploy/policy/release/model_encoder.onnx"
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)


def fsq_quantize(continuous_value, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(continuous_value, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    quantized = np.clip(quantized, fsq_min, fsq_max)
    return quantized


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


# ---------------- SubgoalManager (client-held goal-image list; temporary) ----------------
class SubgoalManager:
    """Fixed goal-image sequence, stepped by the HLP `decision` (advance) or the control
    features (:prev -> retreat, :restart -> reset). Index is clamped to [0, last]."""

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

        # Per-stage subtasks: DEBUG/--no-hlp only. In HLP mode the subtask comes from the HLP.
        subtasks = list(subtasks or [])
        self._subtasks = [(subtasks[i] if i < len(subtasks) else "") for i in range(len(self.paths))]

        print(f"[Subgoal] Loaded {len(self._images)} goal images from {episode_dir}:")
        for i, p in enumerate(self.paths):
            print(f"  [{i}] {p}  | subtask: {self._subtasks[i]!r}")

    @classmethod
    def _list_images(cls, d):
        if not os.path.isdir(d):
            return []
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(cls._IMG_EXTS))

    @staticmethod
    def _load(path):
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"[Subgoal] Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

    def get(self):
        with self._lock:
            return self._images[self._idx]

    def get_subtask(self):
        with self._lock:
            return self._subtasks[self._idx]

    def get_stage(self):
        with self._lock:
            return self._idx, self.paths[self._idx], self._subtasks[self._idx]

    def advance(self):
        with self._lock:
            if self._idx < len(self._images) - 1:
                self._idx += 1
                print(f"[Subgoal] -> advance to [{self._idx}]: {os.path.basename(self.paths[self._idx])}")
            else:
                print(f"[Subgoal] advance: already at last goal image [{self._idx}] (kept)")
            return self._idx

    def retreat(self):
        with self._lock:
            if self._idx > 0:
                self._idx -= 1
            print(f"[Subgoal] -> retreat to [{self._idx}]: {os.path.basename(self.paths[self._idx])}")
            return self._idx

    def reset(self):
        with self._lock:
            self._idx = 0
            print(f"[Subgoal] -> reset to [0]: {os.path.basename(self.paths[0])}")
            return self._idx

    def goto(self, idx):
        """Render the goal image at an absolute stage index (clamped). Used to mirror the HLP's
        subtask index so the goal image can't desync (HLP mode)."""
        with self._lock:
            self._idx = max(0, min(int(idx), len(self._images) - 1))
            return self._idx


# ---------------- HlpController (the only added concept) ----------------
class HlpController:
    """Polls the HLP server for the verbatim `instruction` string + an absolute subtask
    `stage`, renders the client-held goal image at that stage, and applies the 3 control
    features. The robot client reads current_instruction() (None => gate the VLA) + the image.

    Goal image: render goal_image[clamp(stage)] from the HLP's ABSOLUTE subtask index every
    reply (poll + control endpoints) — so it can never desync, even if the HLP is reset
    externally. (`decision` is only used to gate the VLA on "done".) A feature command bumps
    `_gen` so an in-flight poll can't clobber it. memory_items / timestamp are NOT sent
    (server owns memory/history + its own clock).
    """

    def __init__(self, hlp_url, camera, subgoal_manager, task, *, period=0.7, timeout=30.0,
                 no_hlp=False):
        self._url = (hlp_url or "").rstrip("/")
        self._camera = camera
        self._sg = subgoal_manager
        self._task = task
        self._period = period
        self._timeout = timeout
        self._no_hlp = no_hlp
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._instruction = None     # latest verbatim instruction (None => gate VLA)
        self._is_initial = True
        self._gen = 0                # feature commands bump this so in-flight polls don't clobber
        self._running = False
        self._thread = None
        self._manual_override = None  # --no-hlp local takeover text

    # ---- helpers ----
    def _ego_rgb(self):
        frame = self._camera.get_frame()
        if frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(frame).astype(np.uint8)

    def _post(self, path, body):
        r = self._session.post(f"{self._url}{path}",
                               json=convert_numpy_in_dict(body, numpy_serialize),
                               timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def current_instruction(self):
        if self._no_hlp:
            return self._local_instruction()
        with self._lock:
            return self._instruction

    def _local_instruction(self):
        if self._manual_override is not None:
            sub = self._manual_override
        else:
            sub = str(self._sg.get_subtask()).strip()
        task = str(self._task).strip().lower()
        return f"Task: {task}. Subtask: {sub}" if sub else f"Task: {task}"

    def instruction_and_goal(self):
        """Atomic (instruction, goal_image) snapshot so a sent payload never pairs a fresh
        instruction with a stale goal image (or vice versa). (None, None) while gated."""
        with self._lock:
            instr = self._local_instruction() if self._no_hlp else self._instruction
            if not instr:
                return None, None
            return instr, self._sg.get()

    # ---- poll loop ----
    def _poll_once(self):
        ego = self._ego_rgb()
        with self._lock:
            gen, is_initial = self._gen, self._is_initial
        body = {"task": self._task, "is_initial": is_initial}
        if ego is not None:
            body["ego_image"] = ego
        try:
            out = self._post("/hlp", body)
        except Exception as e:
            print(f"[HLP] poll failed (ignored): {e}")
            return
        instr, decision, stage = out.get("instruction"), out.get("decision"), out.get("stage")
        with self._lock:
            if gen != self._gen:
                return  # a feature command superseded this in-flight poll
            # Clear is_initial ONLY once the server actually establishes a subtask. A model reply
            # with no parseable next_subtask (task-only, no "Subtask:") must NOT consume the
            # is_initial signal, else the first subtask could never get established (keep retrying).
            if decision == "switch" or (instr and "Subtask:" in instr):
                self._is_initial = False
            if decision == "done":
                self._instruction = None        # gate the VLA at episode end (avoid OOD task-only)
            elif instr:
                self._instruction = instr
            if stage is not None:
                self._sg.goto(stage)            # goal image = absolute server stage (can't desync)
        if decision == "switch":
            print(f"[HLP] switch -> {instr!r}  goal_stage={self._sg.get_stage()[0]}")

    def _loop(self):
        print(f"[HLP] poller started (url={self._url}, period={self._period}s)")
        while self._running:
            self._poll_once()
            time.sleep(self._period)
        print("[HLP] poller stopped")

    def start(self):
        if self._no_hlp:
            print("[HLP] --no-hlp: using local SubgoalManager subtask as the instruction")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    # ---- control features ----
    def prev(self):
        if self._no_hlp:
            self._manual_override = None
            self._sg.retreat()
            return
        with self._lock:
            self._gen += 1
        try:
            out = self._post("/prev", {})
        except Exception as e:
            print(f"[HLP] :prev failed: {e}"); return
        with self._lock:
            if out.get("clean_start"):
                self._instruction = None       # gate until the next poll re-establishes subtask 0
                self._is_initial = True
            elif out.get("instruction"):
                self._instruction = out["instruction"]
            if out.get("stage") is not None:
                self._sg.goto(out["stage"])    # goal image follows the server's absolute stage
        print(f"[HLP] :prev -> {out.get('instruction')!r}  clean_start={out.get('clean_start')}")

    def restart(self):
        if self._no_hlp:
            self._manual_override = None
            self._sg.reset()
            return
        with self._lock:
            self._gen += 1                     # invalidate any in-flight poll before /reset
            self._instruction = None
            self._is_initial = True
        try:
            out = self._post("/reset", {})
        except Exception as e:
            print(f"[HLP] :restart failed: {e}"); return
        with self._lock:
            if out.get("stage") is not None:
                self._sg.goto(out["stage"])    # -> 0 (goal image follows the server's absolute stage)
            else:
                self._sg.reset()
        print("[HLP] :restart -> clean initial state (VLA gated until subtask 0 re-established)")

    def takeover(self, text):
        if self._no_hlp:
            self._manual_override = text
            return
        with self._lock:
            self._gen += 1
        try:
            out = self._post("/override", {"subtask": text})
        except Exception as e:
            print(f"[HLP] takeover failed: {e}"); return
        with self._lock:
            if out.get("instruction"):
                self._instruction = out["instruction"]
            if out.get("stage") is not None:
                self._sg.goto(out["stage"])    # unchanged (override freezes the subtask index)
        print(f"[HLP] takeover -> {out.get('instruction')!r}  (sticky until :resume/:prev/:restart)")

    def resume(self):
        if self._no_hlp:
            self._manual_override = None
            print("[HLP] :resume -> local override cleared")
            return
        with self._lock:
            self._gen += 1
        try:
            out = self._post("/resume", {})
        except Exception as e:
            print(f"[HLP] :resume failed: {e}"); return
        with self._lock:
            if out.get("clean_start"):
                self._instruction = None
                self._is_initial = True
            elif out.get("instruction"):
                self._instruction = out["instruction"]
            if out.get("stage") is not None:
                self._sg.goto(out["stage"])
        print(f"[HLP] :resume -> {out.get('instruction')!r}  (model polling resumed)")

    def adv_local(self):
        """--no-hlp only: step the local subtask + goal image forward."""
        self._manual_override = None
        self._sg.advance()


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
        return cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)

    def close(self):
        try:
            self.socket.close(linger=0)
            self.context.term()
        except Exception:
            pass


# ---------------- RobotStateSubscriber ----------------
class RobotStateSubscriber:
    """Subscribe to robot state published by g1_deploy_onnx_ref on ZMQ PUB port."""

    def __init__(self, host="localhost", port=5557, topic="g1_debug"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
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
            payload = msg[len(topic_bytes):] if msg.startswith(topic_bytes) else msg
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
        action = action.astype(np.float32).reshape(1, -1)
        pose_data = {
            "token_state": action[:, :64],
            "left_hand_joints": action[:, 64:71],
            "right_hand_joints": action[:, 71:78],
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=4)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close()
        self._context.term()


# ---------------- Global state ----------------
running = threading.Event()
running.set()


# ---------------- RTCWebSocketClient ----------------
class RTCWebSocketClient:
    def __init__(self, server_url, state_subscriber, camera, token_publisher, subgoal_manager, hlp):
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
        self._hlp = hlp
        self._dbg_last_idx = -1
        self._wbc_started = False   # delay WBC start until the first HLP instruction (robot safety)
        self._last_gate_log = 0.0   # throttle the "still gated" log
        try:
            from encoder_client import EncoderClient
            self._encoder = EncoderClient(ENCODER_MODEL, mode=0)
        except Exception as e:
            print(f"[init-prev] encoder load failed ({e}); first chunk falls back to unconditioned")
            self._encoder = None
        self._sent_init_prev = False

    def execute_action(self, action):
        if action.ndim > 1:
            action = action[0]
        hand_joints = action[:14]
        token_ori = action[14:78]
        token_qtz = fsq_quantize(token_ori)
        action_out = np.concatenate([token_qtz, hand_joints])  # [token(64), LH(7), RH(7)]
        self._token_publisher.publish_token(action_out)

    def _on_open(self, ws):
        print("[client] Connected!")
        self._connected.set()

    def _on_message(self, ws, message):
        self.start_time = time.time()
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

        while self._running and running.is_set():
            start = time.time()
            try:
                # --- gate: don't send obs / don't start WBC until the HLP gives an instruction.
                # instruction + goal image are read atomically so they can never mismatch. ---
                instruction, subgoal_frame = self._hlp.instruction_and_goal()
                if instruction is None:
                    now = time.time()
                    if now - self._last_gate_log > 5.0:
                        print("[client] gated: waiting for HLP instruction (HLP slow/down?)...")
                        self._last_gate_log = now
                    time.sleep(0.05)
                    continue
                if not self._wbc_started:
                    self._token_publisher.send_command(start=True, stop=False, planner=True)
                    self._wbc_started = True
                    print(f"[client] first HLP instruction received -> WBC start sent: {instruction!r}")

                state = self._state_sub.get_state()
                if state is None:
                    print("[client] No robot state yet, waiting...")
                    time.sleep(0.1)
                    continue

                body_q = np.array(state["body_q_measured"], dtype=np.float32)        # (29,)
                left_hand_states = np.array(state["left_hand_q"], dtype=np.float32)   # (7,)
                right_hand_states = np.array(state["right_hand_q"], dtype=np.float32)  # (7,)
                leg_states = body_q[:15]
                arm_states = body_q[15:29]
                states = np.concatenate(
                    (left_hand_states, right_hand_states, arm_states, leg_states), axis=0)  # (43,)

                frame = self._camera.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
                # subgoal_frame was captured atomically with `instruction` above.

                stage_idx, stage_path, stage_subtask = self._subgoal_manager.get_stage()
                if stage_idx != self._dbg_last_idx:
                    self._dbg_last_idx = stage_idx
                    cv2.imwrite("/tmp/sent_ego.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    cv2.imwrite("/tmp/sent_goal.jpg", cv2.cvtColor(subgoal_frame, cv2.COLOR_RGB2BGR))
                    print(f"[DEBUG] goal stage idx={stage_idx} | goal={os.path.basename(stage_path)} "
                          f"| instruction={instruction!r}")

                img_obs = {"video.egocentric": frame, "subgoal.egocentric": subgoal_frame}
                state_obs = {"states": states}

                if not self._sent_init_prev and self._encoder is not None:
                    qpos = _mujoco29_to_isaaclab29(state["body_q_measured"])
                    base_quat = np.asarray(state.get("base_quat_measured", [1, 0, 0, 0]),
                                           dtype=np.float32).reshape(4)
                    jp = np.tile(qpos, (10, 1)).astype(np.float32)
                    jv = np.zeros((10, 29), dtype=np.float32)
                    bq = np.tile(base_quat, (10, 1)).astype(np.float32)
                    enc_token = np.asarray(self._encoder.encode(jp, jv, bq), dtype=np.float32).reshape(64)
                    init_prev_action = np.concatenate(
                        [left_hand_states, right_hand_states, enc_token]).astype(np.float32)  # (78,)
                    state_obs["init_prev_action"] = init_prev_action
                    self._sent_init_prev = True
                    print(f"[init-prev] first-frame pseudo prev-action sent")

                # Instruction comes from the HLP (verbatim) — no client-side assembly.
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

                with self._send_lock:
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send(message)
                    else:
                        print("[client] WebSocket not connected, skipping send")
                        break
            except Exception as e:
                print(f"[client] Send error: {e}")
                break

            time.sleep(max(0, OBS_SEND_INTERVAL - (time.time() - start)))

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
         hlp_url=None, hlp_period=0.7, hlp_timeout=30.0, no_hlp=False):
    print("[MAIN] Initializing components...")

    token_publisher = TokenPublisher(host="*", port=zmq_pub_port, topic=zmq_topic)
    print(f"[MAIN] TokenPublisher bound on port {zmq_pub_port}, topic='{zmq_topic}'")

    state_sub = RobotStateSubscriber(host=zmq_host, port=zmq_sub_port, topic=zmq_sub_topic)
    print(f"[MAIN] State subscriber connected to {zmq_host}:{zmq_sub_port}, topic='{zmq_sub_topic}'")

    camera = RSCamera(address=camera_address)
    print(f"[MAIN] Camera connected to {camera_address}")

    subgoal_manager = SubgoalManager(episode_dir=episode_dir, subtasks=subtasks)
    print(f"[MAIN] Task instruction: {task_instruction!r}")

    # HLP controller — its OWN camera so the ~1s poll never shares the control-loop REQ socket.
    hlp_camera = RSCamera(address=camera_address) if not no_hlp else None
    hlp = HlpController(hlp_url, hlp_camera, subgoal_manager, task_instruction,
                        period=hlp_period, timeout=hlp_timeout, no_hlp=no_hlp)

    time.sleep(1.0)  # let ZMQ PUB establish

    # NOTE: WBC start is NOT sent here — the send thread sends it on the first HLP instruction.

    print("[MAIN] Waiting for robot state...")
    for _ in range(30):
        if state_sub.get_state() is not None:
            print("[MAIN] Got robot state")
            break
        time.sleep(0.5)
    else:
        print("[MAIN] WARNING: No robot state after 15s, proceeding anyway...")

    hlp.start()

    client = RTCWebSocketClient(
        server_url=server_url, state_subscriber=state_sub, camera=camera,
        token_publisher=token_publisher, subgoal_manager=subgoal_manager, hlp=hlp,
    )

    def stdin_listener():
        if no_hlp:
            print("[MAIN] (--no-hlp) commands: ':adv' next | ':prev' back | ':restart' reset | ':resume' HLP | '<text>' override")
        else:
            print("[MAIN] commands: ':prev' previous subtask | ':restart' restart | ':resume' return to HLP | '<text>' human takeover")
        while running.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break
            cmd = line.strip()
            if cmd == "":
                continue
            if cmd == ":prev":
                hlp.prev()
            elif cmd == ":restart":
                hlp.restart()
            elif cmd == ":resume":
                hlp.resume()
            elif cmd == ":adv" and no_hlp:
                hlp.adv_local()
            elif cmd.startswith(":"):
                print(f"[MAIN] unknown command {cmd!r}")
            else:
                hlp.takeover(cmd)

    threading.Thread(target=stdin_listener, daemon=True).start()

    def websocket_thread():
        client.run()
        print("[WS] WebSocket thread stopped")

    threading.Thread(target=websocket_thread, daemon=True).start()
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
        running.clear()

    print("[MAIN] Shutting down...")
    hlp.stop()
    client.stop()
    try:
        token_publisher.send_command(start=False, stop=True, planner=True)
    except Exception as e:
        print(f"[MAIN] Error sending stop command: {e}")
    state_sub.stop()
    token_publisher.stop()
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RTC sonic client + HLP-backed instruction")
    parser.add_argument("--host", type=str, default="localhost", help="VLA policy server host")
    parser.add_argument("--port", type=int, default=8014, help="VLA policy server port")
    parser.add_argument("--zmq-host", type=str, default="localhost")
    parser.add_argument("--zmq-pub-port", type=int, default=5556)
    parser.add_argument("--zmq-sub-port", type=int, default=5557)
    parser.add_argument("--zmq-topic", type=str, default="pose")
    parser.add_argument("--zmq-sub-topic", type=str, default="g1_debug")
    parser.add_argument("--camera-address", type=str, default="tcp://192.168.123.164:5558")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="task folder (e.g. .data/hongyi_ood_data/put_chip_can_into_plate); "
                             "task = folder name, goal images = episode_<eps-idx>/color_subgoal/")
    parser.add_argument("--eps-idx", type=int, default=0, help="episode index inside --data-dir")
    parser.add_argument("--task", type=str, default=None,
                        help="override task string (default: from --data-dir folder name)")
    parser.add_argument("--subtasks", type=str, default=None,
                        help="optional ';'-separated subtask text (only used in --no-hlp / goal-stage labels)")
    parser.add_argument("--hlp-host", type=str, default="localhost", help="HLP server host")
    parser.add_argument("--hlp-port", type=int, default=8015, help="HLP server port")
    parser.add_argument("--hlp-period", type=float, default=0.7, help="HLP poll period (s)")
    parser.add_argument("--hlp-timeout", type=float, default=30.0)
    parser.add_argument("--no-hlp", action="store_true",
                        help="Debug: no HLP; use the local SubgoalManager subtask + :adv stepping")

    args = parser.parse_args()

    # One folder path supplies task (folder name) + goal images (episode_<eps-idx>/color_subgoal/).
    data_dir = os.path.normpath(args.data_dir)
    task_instruction = args.task or os.path.basename(data_dir).replace("_", " ").strip() or TASK_INSTRUCTION
    episode_dir = os.path.join(data_dir, f"episode_{args.eps_idx}")
    subtasks = [s.strip() for s in args.subtasks.split(";") if s.strip()] if args.subtasks else []

    server_url = f"ws://{args.host}:{args.port}/ws"
    hlp_url = None if args.no_hlp else f"http://{args.hlp_host}:{args.hlp_port}"
    main(
        server_url=server_url, zmq_host=args.zmq_host, zmq_pub_port=args.zmq_pub_port,
        zmq_sub_port=args.zmq_sub_port, zmq_topic=args.zmq_topic, zmq_sub_topic=args.zmq_sub_topic,
        camera_address=args.camera_address, episode_dir=episode_dir,
        task_instruction=task_instruction, subtasks=subtasks,
        hlp_url=hlp_url, hlp_period=args.hlp_period, hlp_timeout=args.hlp_timeout, no_hlp=args.no_hlp,
    )
