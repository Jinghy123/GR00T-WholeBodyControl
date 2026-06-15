import os
import sys
import time
import threading
import json
import signal

import cv2
import numpy as np
import zmq
import msgpack
import requests
import json_numpy

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message,
    build_command_message,
)
from encoder_client import EncoderClient

# json_numpy.patch() monkey-patches stdlib json so np.ndarray round-trips over HTTP.
# The PsiX non-RTC server (psix_serve_sonic.py) (de)serializes with helpers.py's
# base64 "__numpy__" scheme, which is byte-identical to json_numpy — so requests'
# json=payload / resp.json() interoperate with it directly.
json_numpy.patch()

# ---------------- Configuration ----------------
TASK_INSTRUCTION = "grasp the pink chip can and place it into the orange plate"
# TASK_INSTRUCTION = "pick up the grapes and place in the bowl"

# FSQ configuration (must match g1_sonic_client / encoder)
FSQ_MIN = -0.625
FSQ_MAX = 0.625
FSQ_STEP = 0.0625  # = 1/16

# Control frequency
FREQ_POLICY = 30  # Hz

# Action layout: token(64) + hand_joints(14) = 78
ACTION_DIM = 78

# Encoder model path (for frozen token between chunks)
ENCODER_MODEL = "gear_sonic_deploy/policy/release/model_encoder.onnx"

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


# ---------------- SubgoalManager ----------------
class SubgoalManager:
    """
    Holds the current subgoal image and advances through a fixed sequence via advance()
    (driven by the `:adv` / `:goal` stdin commands or an HLP `switch`).

    Sequence: every image in color_subgoal/ in order, then color[-1] as the final one.
    advance() past the last stage is a no-op (index caps at len-1).
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

        # Per-stage subtask prompts, parallel to self.paths — DEBUG/LOG ONLY (printed below
        # and in advance()/get_stage()). They are NOT sent to the server: the VLA's subtask
        # comes from the client's inline state (manual stdin / HLP) via _make_instruction().
        # Missing entries -> "".
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
        print(f"[Subgoal] Current index: 0  (advance via :adv / :goal <n> or an HLP switch)")

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

    def get_stage(self):
        """(idx, subgoal_path, subtask) for the current stage — for debug logging."""
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
        """
        Publish action token message (Protocol v4).

        Args:
            action: np.ndarray of shape (78,) — token(64) + left_hand(7) + right_hand(7)
        """
        action = action.astype(np.float32).reshape(1, -1)
        pose_data = {
            "token_state": action[:, :64],          # (1, 64)
            "left_hand_joints": action[:, 64:71],   # (1, 7)
            "right_hand_joints": action[:, 71:78],  # (1, 7)
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=4)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close(linger=0)
        self._context.term()


# ---------------- Main Client ----------------
class PsixSonicClient:
    """
    Chunk-based PsiX (subtask + goal-image) client for the NON-RTC HTTP server
    (psix_serve_sonic.py, POST /act). Each inference returns a chunk of actions;
    the publish loop iterates it at 30 Hz, and the encoder freezes the body token
    between chunks. Identical transport/playback to psi_sonic_client.py — only the
    observation payload (two images + dict instruction + single-frame state) and the
    server's action layout differ.
    """

    def __init__(self, server_url, state_subscriber, camera, token_publisher,
                 subgoal_manager, task_instruction, http_timeout=30.0,
                 hlp_url=None, hlp_timeout=30.0, hlp_camera=None, hlp_period=0.7,
                 hlp_auto_threshold=1.5):
        self._server_url = server_url
        self._state_sub = state_subscriber
        self._camera = camera
        self._token_publisher = token_publisher
        self._subgoal_manager = subgoal_manager
        self._task = task_instruction
        self._http_timeout = http_timeout
        self._dbg_last_idx = -1  # debug: track stage changes for image/idx dumps

        # ---- Inline subtask state (self-contained; no external subtask driver) ----
        # The VLA's subtask comes from manual stdin / HLP instead of the fixed per-stage list.
        # Start with NO subtask + EMPTY memory: the FIRST HLP call (is_initial=True, empty
        # memory) PREDICTS the first subtask, which then enters memory; until that first reply
        # (or any manual input) the VLA gets task-only conditioning. We do NOT seed the canned
        # stage-0 subtask, because that would (a) feed the HLP a non-empty memory at the
        # initial frame (OOD vs training, where the initial prediction sees Memory:none) and
        # (b) make the initial "switch to <first>" a no-op (next==current).
        self._subtask_lock = threading.Lock()
        self._subtask = ""
        self._subtask_source = "init"
        self._manual_override = False
        self._subtask_memory = []  # list of (text, started_at), oldest-first

        # ---- HLP polling (separate HTTP process; None => disabled, e.g. --no-hlp) ----
        self._hlp_url = hlp_url
        self._hlp_timeout = hlp_timeout
        self._hlp_camera = hlp_camera
        self._hlp_period = hlp_period
        self._hlp_session = requests.Session()  # independent of the VLA self._session
        self._hlp_first = True                  # is_initial until first successful HLP reply
        self._hlp_thread = None
        # Predicted-time auto-transition: when HLP says 'continue' with a predicted
        # seconds_to_subgoal < this threshold, switch to the (already-predicted) next subtask at
        # the predicted time instead of waiting for a future 'switch' reply — hides the ~1s HLP
        # latency for imminent transitions. 0 disables it.
        self._auto_threshold = hlp_auto_threshold
        self._pending_transition = None         # (next_subtask, fire_at, from_subtask), under _subtask_lock
        self._transition_thread = None

        # Encoder for frozen token
        self._encoder = EncoderClient(ENCODER_MODEL, mode=0)

        # Threading / synchronization (mirrors psi_sonic_client / g1_sonic_client)
        self._running = threading.Event()
        self._sequence_done_event = threading.Event()
        self._pending_chunk = None  # latest chunk from inference worker
        self._chunk_lock = threading.Lock()

        # HTTP session (reuses TCP connection)
        self._session = requests.Session()

        self._inference_thread = None
        self._publish_thread = None

    # ---------- Policy request ----------
    def _build_observation_payload(self):
        """Capture current state + ego/goal frames, build PsiX payload dict.

        Image keys MUST match the server's repack:
          ego image  -> repack.image_keys[0]  == "video.egocentric"
          goal image -> repack.subgoal_key[0] == "subgoal.egocentric"
        State is a SINGLE frame (the server reshapes (D,) -> (1, 1, D)).
        Instruction is a dict {"task", "subtask"}; the server assembles
        "Task: <task>. Subtask: <subtask>" (subtask dropped when empty).
        """
        state = self._state_sub.get_state()
        assert state is not None, "Robot state not available"

        body_q = np.array(state["body_q_measured"], dtype=np.float32)        # (29,) = [leg/base(15) | arm(14)]
        left_hand_states = np.array(state["left_hand_q_measured"], dtype=np.float32)   # (7,)
        right_hand_states = np.array(state["right_hand_q_measured"], dtype=np.float32)  # (7,)

        # Model expects state.joint_positions = [hand(L7,R7) | arm(14) | leg(15)].
        leg_states = body_q[:15]    # base/leg
        arm_states = body_q[15:29]  # arm
        states = np.concatenate(
            (left_hand_states, right_hand_states, arm_states, leg_states), axis=0
        )  # (43,)

        frame = self._camera.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.uint8)

        subgoal_frame = self._subgoal_manager.get()

        # --- DEBUG: dump ego + goal to /tmp whenever the subgoal stage changes ---
        stage_idx, stage_path, stage_subtask = self._subgoal_manager.get_stage()
        if stage_idx != self._dbg_last_idx:
            self._dbg_last_idx = stage_idx
            cv2.imwrite("/tmp/sent_ego.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite("/tmp/sent_goal.jpg", cv2.cvtColor(subgoal_frame, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] stage idx={stage_idx} | goal={os.path.basename(stage_path)} "
                  f"| subtask={stage_subtask!r} | dumped /tmp/sent_ego.jpg /tmp/sent_goal.jpg")

        payload = {
            "image": {
                "video.egocentric": frame,
                "subgoal.egocentric": subgoal_frame,
            },
            "state": {"states": states},  # single (43,) frame
            "gt_action": None,
            "dataset_name": None,
            # subtask now from the inline state (manual stdin now; HLP later); the goal
            # image above is still the fixed SubgoalManager sequence.
            "instruction": self._make_instruction(),
            "history": None,
            "condition": None,
            "timestamp": None,
        }
        return payload

    def _get_policy_chunk(self):
        """
        POST current observation to the HTTP policy server and return a chunk of
        actions, shape (N, 78), repacked to publish layout [token(64) | hand(14)]
        with FSQ quantization applied to the token. Returns None on failure.

        The PsiX server returns each action in repack.action_key order
            [action.hand_joints(14) | action.body_token(64)]
        i.e. hand_joints FIRST, then the 64-D sonic token. We reorder to
        [token(64) | hand(14)] so the downstream publish loop + encoder-freeze
        logic match psi_sonic_client exactly, and publish_token receives
        [token(64) | left_hand(7) | right_hand(7)].
        """
        payload = self._build_observation_payload()

        try:
            resp = self._session.post(self._server_url, json=payload,
                                      timeout=self._http_timeout)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Inference] HTTP request error: {e}")
            return None

        try:
            data = resp.json()
            action = data["action"] if isinstance(data, dict) else data
            chunk = np.asarray(action, dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
        except Exception as e:
            print(f"[Inference] Response parse error: {e}")
            return None

        if chunk.shape[-1] != ACTION_DIM:
            print(f"[Inference] Unexpected action dim: {chunk.shape}, expected (*, {ACTION_DIM})")
            return None

        # Server layout is [hand_joints(14) | body_token(64)] -> reorder to
        # [token(64) | hand(14)] and FSQ-quantize the token part.
        hand_joints = chunk[:, :14]    # [left_hand(7), right_hand(7)]
        token_ori = chunk[:, 14:78]    # 64-D sonic body token
        token_qtz = fsq_quantize(token_ori)
        chunk_out = np.concatenate([token_qtz, hand_joints], axis=-1).astype(np.float32)

        print(f"[Inference] Chunk received: shape={chunk_out.shape}, "
              f"token range=[{token_ori.min():.4f},{token_ori.max():.4f}] → "
              f"[{token_qtz.min():.4f},{token_qtz.max():.4f}]")
        return chunk_out

    # ---------- Threads ----------
    def _inference_worker(self):
        """Wait for request, run policy, post result. Matches psi_sonic_client."""
        # Wait for first robot state
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
        """
        Main 30 Hz control loop.

        State machine:
          - EXECUTING: iterate through action chunk, one token per tick (1/30 s)
          - WAITING:   chunk exhausted; send encoder-derived frozen token until
                       next chunk arrives
        Internal action layout is [token(64) | hand(14)] (set by _get_policy_chunk).
        """
        dt = 1.0 / FREQ_POLICY

        # Send start command and request first chunk
        self._token_publisher.send_command(start=True, stop=False, planner=True)
        print("[PublishLoop] Requesting first policy inference...")
        self._sequence_done_event.set()
        while self._sequence_done_event.is_set() and self._running.is_set():
            time.sleep(0.05)
        if not self._running.is_set():
            return
        with self._chunk_lock:
            chunk = self._pending_chunk
        # Guard: normally non-None (the inference worker sets _pending_chunk before clearing
        # the done-event, and we'd have returned above if _running was cleared). Defensive
        # against a None/empty first chunk (e.g. a 0-row chunk would crash the WAITING branch
        # on the first tick via last_action=None).
        if chunk is None or len(chunk) == 0:
            print("[PublishLoop] No/empty first chunk; aborting loop.")
            return

        idx = 0
        frozen_action = None
        last_action = None
        print(f"[PublishLoop] First chunk: shape={chunk.shape}. Starting execution.")

        while self._running.is_set():
            t_start = time.perf_counter()

            if idx < len(chunk):
                # ── EXECUTING ──────────────────────────────────────────────
                action = chunk[idx]
                last_action = action.copy()
                idx += 1
            else:
                # ── WAITING for next chunk ─────────────────────────────────

                # First tick after chunk exhausted: read robot state → run encoder → freeze token
                if idx == len(chunk):
                    state = self._state_sub.get_state()
                    if state is not None:
                        qpos = _mujoco29_to_isaaclab29(state["body_q_measured"])  # (29,)
                        base_quat = np.array(state["base_quat_measured"], dtype=np.float32)  # (4,) wxyz

                        joint_pos = np.tile(qpos, (10, 1)).astype(np.float32)       # (10, 29)
                        joint_vel = np.zeros((10, 29), dtype=np.float32)
                        body_quat = np.tile(base_quat, (10, 1)).astype(np.float32)  # (10, 4)

                        enc_token = self._encoder.encode(joint_pos, joint_vel, body_quat)  # (64,)
                        # internal layout: token(64) + hand_joints(14)
                        # keep hand joints from last action, replace body token
                        frozen_action = np.concatenate([enc_token, last_action[64:78]])
                        print(f"[PublishLoop] Chunk done ({len(chunk)} tokens), "
                              f"encoder freeze token computed.")
                    else:
                        frozen_action = last_action.copy()
                        print(f"[PublishLoop] Chunk done, no robot state — repeating last action.")
                    self._sequence_done_event.set()
                    idx += 1

                if not self._sequence_done_event.is_set():  # means inference done
                    with self._chunk_lock:
                        chunk = self._pending_chunk
                    frozen_action = None
                    idx = 0
                    print(f"[PublishLoop] New chunk received: shape={chunk.shape}. "
                          f"Resuming execution.")

                    # directly execute the first action
                    action = chunk[idx]
                    last_action = action.copy()
                    idx += 1
                else:
                    action = frozen_action

            self._token_publisher.publish_token(action)

            # Maintain 30 Hz
            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ---------- Subtask state (inline; self-contained) ----------
    def _get_subtask(self):
        with self._subtask_lock:
            return self._subtask

    def _make_instruction(self):
        """{"task","subtask"} for the payload; server assembles 'Task: X. Subtask: Y'."""
        with self._subtask_lock:
            return {"task": self._task, "subtask": self._subtask}

    def _render_memory(self):
        """Most-recent-first [{text, seconds_ago}] for the HLP prompt's Memory block."""
        with self._subtask_lock:
            now = time.time()  # inside the lock so no entry has started_at > now (neg seconds)
            return [{"text": t, "seconds_ago": now - ts}
                    for t, ts in reversed(self._subtask_memory)]

    def _set_subtask_locked(self, text, source):
        # caller holds _subtask_lock. Push memory on a real change to a real subtask
        # (skip "" task-only and the "__done__" terminal sentinel).
        changed = text != self._subtask
        self._subtask = text
        self._subtask_source = source
        if changed and text and text != "__done__":
            self._subtask_memory.append((text, time.time()))

    def _set_manual_subtask(self, text):
        with self._subtask_lock:
            self._manual_override = True
            self._set_subtask_locked(text, "manual")

    def _release_to_hlp(self):
        with self._subtask_lock:
            self._manual_override = False

    def _apply_hlp(self, decision, next_subtask):
        """Apply an HLP decision unless a manual override is sticky.
        switch->next_subtask, done->__done__, continue/unknown->keep.
        Returns True iff this call actually switched the subtask to a NEW value — the caller
        gates the goal-image advance on this atomic result (not a before/after compare), which
        avoids a TOCTOU race where stdin changes the subtask between the two reads."""
        with self._subtask_lock:
            if self._manual_override:
                return False  # manual wins
            if decision == "switch" and next_subtask:
                changed = next_subtask != self._subtask
                self._set_subtask_locked(next_subtask, "hlp")
                return changed
            elif decision == "done":
                self._set_subtask_locked("__done__", "hlp")
            return False

    def _hlp_switch_to(self, next_subtask):
        """Switch to next_subtask (HLP source; respects manual override) and advance the fixed
        goal image in lockstep iff it actually switched. Shared by the HLP 'switch' decision and
        the predicted-time auto-transition. Returns True iff it switched."""
        applied = self._apply_hlp("switch", next_subtask)
        if applied:
            idx, _, _ = self._subgoal_manager.get_stage()
            if idx >= len(self._subgoal_manager.paths) - 1:
                print("[HLP] switch but goal image already at last stage; image/subtask decoupled")
            self._subgoal_manager.advance()
        return applied

    def _maybe_schedule_transition(self, next_subtask, secs, was_initial):
        """On an HLP 'continue': schedule an auto-transition to next_subtask at now+secs iff it
        is imminent (0<=secs<threshold), non-initial, not manual, and a real change. Records the
        subtask we transition FROM so a STALE schedule can't overwrite a newer switch (the timer
        only fires while the current subtask is still that one). Returns True iff scheduled."""
        with self._subtask_lock:
            if (not was_initial) and self._auto_threshold > 0 and next_subtask \
                    and isinstance(secs, (int, float)) and 0 <= float(secs) < self._auto_threshold \
                    and not self._manual_override and next_subtask != self._subtask:
                self._pending_transition = (next_subtask, time.time() + float(secs), self._subtask)
                return True
            self._pending_transition = None
            return False

    def _clear_pending_transition(self):
        with self._subtask_lock:
            self._pending_transition = None

    def _transition_timer(self):
        """Fire a due pending auto-transition (~0.1 s resolution) so an imminent switch happens at
        the HLP-predicted time instead of waiting for the next ~1 s poll. Fires ONLY if the current
        subtask is still the one it was scheduled FROM — otherwise a newer HLP 'switch' superseded
        it (stale) and we keep the latest. The staleness check + subtask update are atomic under
        the lock; the goal-image advance follows. Also skipped under manual override."""
        print("[HLP] transition timer started")
        while self._running.is_set():
            fire = None
            with self._subtask_lock:
                pend = self._pending_transition
                if pend is not None and time.time() >= pend[1]:
                    next_sub, _, from_sub = pend
                    if (not self._manual_override) and self._subtask == from_sub \
                            and next_sub and next_sub != self._subtask:
                        self._set_subtask_locked(next_sub, "hlp")
                        fire = next_sub
                    self._pending_transition = None   # consume (whether fired or stale)
            if fire is not None:
                idx, _, _ = self._subgoal_manager.get_stage()
                if idx >= len(self._subgoal_manager.paths) - 1:
                    print("[HLP] auto-transition but goal image already at last stage; decoupled")
                self._subgoal_manager.advance()
                print(f"[HLP] auto-transition at predicted time -> {fire!r}")
            time.sleep(0.1)
        print("[HLP] transition timer stopped")

    # ---------- HLP poller (separate process; never blocks the 30 Hz loop) ----------
    def _hlp_worker(self):
        """Poll the HLP server (default 0.7 s; --hlp-period 0 = as fast as possible) and
        apply its decision. Own thread + HTTP session + camera, so the slow 2B model (in
        the HLP server process) never stalls the publish loop. Failures are logged and
        ignored (control keeps running)."""
        print(f"[HLP] worker started (url={self._hlp_url}, period={self._hlp_period}s)")
        while self._running.is_set():
            try:
                frame = self._hlp_camera.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
                with self._subtask_lock:
                    task = self._task          # snapshot under lock (stdin :task writes it)
                mem = self._render_memory()     # takes the lock internally
                resp = self._hlp_session.post(self._hlp_url, timeout=self._hlp_timeout, json={
                    "ego_image": frame,
                    "task": task,
                    "memory_items": mem,
                    "is_initial": self._hlp_first,
                })
                resp.raise_for_status()
                out = resp.json()
            except Exception as e:
                print(f"[HLP] request failed (ignored): {e}")
                time.sleep(self._hlp_period)
                continue
            # The HLP server returns HTTP 200 even on failure ({"error": ...}, like the VLA
            # server). Treat any reply without a valid decision (error / unparseable / truncated
            # JSON -> decision None) as a no-op: keep the current subtask AND keep is_initial
            # True so the next call still flags the episode start.
            decision = out.get("decision")
            if "error" in out or decision not in ("continue", "switch", "done"):
                print(f"[HLP] no valid decision (ignored): {out.get('error') or out.get('raw_text') or out}")
                time.sleep(self._hlp_period)
                continue
            was_initial = self._hlp_first
            self._hlp_first = False  # only after a VALID reply
            next_subtask = out.get("next_subtask")
            secs = out.get("seconds_to_subgoal")
            if decision == "switch":
                # Initial 'switch' only ESTABLISHES the first subtask (goal already at stage 0);
                # a later switch advances the goal image in lockstep (_hlp_switch_to).
                if was_initial:
                    self._apply_hlp("switch", next_subtask)
                else:
                    self._hlp_switch_to(next_subtask)
                self._clear_pending_transition()  # a real switch supersedes any pending schedule
            elif decision == "done":
                self._apply_hlp("done", next_subtask)  # passthrough -> __done__
                self._clear_pending_transition()
            else:  # 'continue': maybe schedule an imminent predicted-time auto-transition.
                self._maybe_schedule_transition(next_subtask, secs, was_initial)
            print(f"[HLP] decision={decision!r} subtask={self._get_subtask()!r} secs={secs}")
            time.sleep(self._hlp_period)
        print("[HLP] worker stopped")

    # ---------- stdin manual control ----------
    def _print_subtask_state(self):
        with self._subtask_lock:
            st, src, mo = self._subtask, self._subtask_source, self._manual_override
        idx, _, _ = self._subgoal_manager.get_stage()
        print(f"[stdin] subtask={st!r} source={src} manual={mo} | goal_stage={idx}")

    def _goto_goal(self, rest):
        try:
            n = int(rest)
        except (TypeError, ValueError):
            print("[stdin] usage: :goal <index>")
            return
        while True:
            idx, _, _ = self._subgoal_manager.get_stage()
            if idx >= n or idx >= len(self._subgoal_manager.paths) - 1:
                break
            self._subgoal_manager.advance()

    def _stdin_loop(self):
        """Manual subtask control + goal-image stepping over stdin (blocks on readline):
          <text>     set current subtask (manual, sticky — HLP won't override)
          :adv       advance the fixed goal image one stage (old Enter behavior)
          :goal <n>  advance the goal image up to stage n
          :hlp       release manual override (hand subtask back to HLP)
          :task <t>  change the task string
          :clear     subtask = "" (task-only)    |   :done   subtask = "__done__"
          :show      print state                 |   <blank> reprint state
        """
        print("[stdin] type a subtask to steer; :adv goal+1, :hlp release, :show, :task <t>")
        self._print_subtask_state()
        while self._running.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:  # EOF
                break
            cmd = line.rstrip("\n")
            if cmd == "":
                self._print_subtask_state()
            elif cmd.startswith(":"):
                head, _, rest = cmd.partition(" ")
                rest = rest.strip()
                if head == ":adv":
                    self._subgoal_manager.advance()
                elif head == ":goal":
                    self._goto_goal(rest)
                elif head == ":hlp":
                    self._release_to_hlp()
                    print("[stdin] released manual override -> HLP controls subtask")
                elif head == ":task":
                    with self._subtask_lock:
                        self._task = rest
                    print(f"[stdin] task = {rest!r}")
                elif head == ":clear":
                    self._set_manual_subtask("")
                    print("[stdin] subtask cleared (task-only)")
                elif head == ":done":
                    self._set_manual_subtask("__done__")
                    print("[stdin] subtask = __done__")
                elif head == ":show":
                    self._print_subtask_state()
                else:
                    print(f"[stdin] unknown command {head!r}")
            else:
                self._set_manual_subtask(cmd)
                print(f"[stdin] subtask = {cmd!r} (manual, sticky)")
        print("[stdin] loop stopped")

    # ---------- Lifecycle ----------
    def start(self):
        print("[PsixSonicClient] Starting...")
        self._running.set()

        # Start inference worker
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()

        # Start publish loop
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()

        # Start HLP poller (only if an HLP url was given; --no-hlp => disabled).
        if self._hlp_url:
            self._hlp_thread = threading.Thread(target=self._hlp_worker, daemon=True)
            self._hlp_thread.start()
            # Predicted-time auto-transition timer (fires imminent transitions on schedule).
            if self._auto_threshold > 0:
                self._transition_thread = threading.Thread(target=self._transition_timer, daemon=True)
                self._transition_thread.start()

        print("[PsixSonicClient] Started successfully!")
        return True

    def stop(self):
        print("[PsixSonicClient] Stopping...")
        self._running.clear()

        # Send stop command to WBC
        try:
            self._token_publisher.send_command(start=False, stop=True, planner=True)
        except Exception as e:
            print(f"[PsixSonicClient] Error sending stop command: {e}")

        # Close HTTP session
        try:
            self._session.close()
        except Exception:
            pass

        # Clean up
        try:
            self._camera.close()
        except Exception:
            pass
        try:
            self._hlp_session.close()
        except Exception:
            pass
        if self._hlp_camera is not None:
            try:
                self._hlp_camera.close()
            except Exception:
                pass
        self._state_sub.stop()
        self._token_publisher.stop()

        print("[PsixSonicClient] Stopped.")


# ---------------- Main ----------------
def main(server_url, zmq_host, zmq_pub_port, zmq_sub_port, zmq_topic, zmq_sub_topic,
         camera_address, episode_dir, task_instruction, subtasks,
         hlp_url=None, hlp_timeout=30.0, hlp_period=0.7, hlp_auto_threshold=1.5):
    print("[MAIN] Initializing components...")

    # 1. Initialize token publisher (ZMQ PUB, Protocol v4)
    token_publisher = TokenPublisher(host="*", port=zmq_pub_port, topic=zmq_topic)
    print(f"[MAIN] TokenPublisher bound on port {zmq_pub_port}, topic='{zmq_topic}'")

    # 2. Initialize robot state subscriber (ZMQ SUB)
    state_sub = RobotStateSubscriber(host=zmq_host, port=zmq_sub_port, topic=zmq_sub_topic)
    print(f"[MAIN] State subscriber connected to {zmq_host}:{zmq_sub_port}, topic='{zmq_sub_topic}'")

    # 3. Initialize camera
    camera = RSCamera(address=camera_address)
    print(f"[MAIN] Camera connected to {camera_address}")

    # 3b. Initialize subgoal manager (subgoal images + per-stage subtask prompts)
    subgoal_manager = SubgoalManager(episode_dir=episode_dir, subtasks=subtasks)
    print(f"[MAIN] Task instruction: {task_instruction!r}")

    # 4. Wait briefly for ZMQ PUB socket to establish connections
    time.sleep(1.0)

    # 5. Wait for first robot state
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

    # 5b. Second camera for the HLP poller (own REQ socket on the same camera server), so
    # it never shares the control-loop camera socket. Only created when HLP is enabled.
    hlp_camera = RSCamera(address=camera_address) if hlp_url else None

    # 6. Create and start client
    client = PsixSonicClient(
        server_url=server_url,
        state_subscriber=state_sub,
        camera=camera,
        token_publisher=token_publisher,
        subgoal_manager=subgoal_manager,
        task_instruction=task_instruction,
        hlp_url=hlp_url,
        hlp_timeout=hlp_timeout,
        hlp_camera=hlp_camera,
        hlp_period=hlp_period,
        hlp_auto_threshold=hlp_auto_threshold,
    )

    if not client.start():
        print("[MAIN] Client failed to start")
        client.stop()
        return

    # 6b. Start stdin command loop: the client owns the subtask text now (type to set it,
    # manual+sticky). `:adv` / `:goal <n>` step the FIXED goal-image sequence (old Enter
    # behavior); `:hlp` releases manual override; `:task`/`:show`/`:clear`/`:done`. Lets a
    # --no-hlp manual run drive BOTH the subtask text and the goal image.
    t_stdin = threading.Thread(target=client._stdin_loop, daemon=True)
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

    parser = argparse.ArgumentParser(
        description="Chunk-based PsiX (subtask + goal-image) Policy Client, non-RTC HTTP"
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
    parser.add_argument("--episode-dir", type=str,
                        default="/home/xiawei/data/multi-task/put_chip_can_into_plate/episode_0",
                        help="Episode folder containing color/ and color_subgoal/ for subgoal images")
    parser.add_argument("--prompts-json", type=str,
                        default="/home/xiawei/data/multi-task/prompts.json",
                        help="JSON mapping task-key -> {task_description, subtasks[]}")
    parser.add_argument("--task-key", type=str, default="put_chip_can_into_plate",
                        help="Key into prompts.json; selects the task_description and the "
                             "per-stage subtask prompts.")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override task instruction (else taken from prompts.json[task-key])")
    parser.add_argument("--hlp-host", type=str, default="localhost",
                        help="HLP server host")
    parser.add_argument("--hlp-port", type=int, default=8015,
                        help="HLP server port")
    parser.add_argument("--hlp-timeout", type=float, default=30.0,
                        help="HLP HTTP request timeout (s)")
    parser.add_argument("--hlp-period", type=float, default=0.7,
                        help="HLP poll period (s); 0 = as fast as possible")
    parser.add_argument("--hlp-auto-transition-threshold", type=float, default=1.5,
                        help="When HLP says continue with seconds_to_subgoal < this (s), "
                             "auto-transition to the predicted next subtask at the predicted "
                             "time instead of waiting for an HLP 'switch' (hides HLP latency). "
                             "0 disables.")
    parser.add_argument("--no-hlp", action="store_true",
                        help="Disable the HLP poller — manual stdin steering only")

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

    server_url = f"http://{args.host}:{args.port}/act"
    hlp_url = None if args.no_hlp else f"http://{args.hlp_host}:{args.hlp_port}/hlp"
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
        hlp_url=hlp_url,
        hlp_timeout=args.hlp_timeout,
        hlp_period=args.hlp_period,
        hlp_auto_threshold=args.hlp_auto_transition_threshold,
    )
