"""Shared action conditions and robot publication, with RTC and HTTP transports."""

import cv2
import json
import numpy as np
import requests
import threading
import time
from dataclasses import dataclass
from .recording import log_event
from .robot import (
    ACTION_DIM_DEFAULT,
    ACTION_DIM_NECK,
    ENCODER_MODELS,
    HAND_DIM,
    NECK_DIM,
    OBS_SEND_INTERVAL,
    TOKEN_DIM,
    _mujoco29_to_isaaclab29,
    convert_numpy_in_dict,
    fsq_quantize,
    load_body_token_encoder,
    numpy_deserialize,
    numpy_serialize,
    resize_goal_for_vla,
)
from .telemetry import ActionTelemetry, RateCounter
from websocket import WebSocketApp
import os

_GOAL_WINDOW_NAME = "VLA goal (as sent)"


def display_available():
    """True when cv2's GUI can safely be used.

    Checked BEFORE any highgui call: with no DISPLAY the Qt plugin aborts from
    C++, which try/except around imshow cannot catch.
    """
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@dataclass
class Observation:
    """One control tick's inputs, shared by both transports."""
    frame: np.ndarray
    state: dict
    states: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    wm: dict
    goal: object
    instruction: str
    epoch: int


class VlaClientBase:
    def __init__(self, server_url, state_subscriber, camera, token_publisher, wm_provider,
                 task_instruction, dry_run=False, observation_stale_timeout=0.5,
                 action_stale_timeout=0.5, include_neck=False, neck_publisher=None,
                 neck_state_reader=None, rollout_recorder=None, encoder_version="v1",
                 frozen_action=True, subtask_prompt=False, goal_hw=(270, 480), run_event=None,
                 show_goal=False):
        self.server_url = server_url
        # Live cv2 window on the goal actually sent to the VLA (post-resize).
        self._show_goal = bool(show_goal)
        self._goal_window_ready = False
        if self._show_goal and not display_available():
            self._show_goal = False
            print("[goal-window] --show-goal ignored: no DISPLAY/WAYLAND_DISPLAY. Run from a "
                  "desktop session or `ssh -X`.", flush=True)
        elif self._show_goal:
            print("[goal-window] enabled", flush=True)
        self._running = True
        self._connected = threading.Event()
        self._send_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self.start_time = time.monotonic()

        self._state_sub = state_subscriber
        self._camera = camera
        self._token_publisher = token_publisher
        self._wm = wm_provider
        self._task = task_instruction
        self._subtask = None
        self._instruction_hold_reason = None
        self._execution_started_at = None
        self._dry_run = bool(dry_run)
        self._observation_stale_timeout = float(observation_stale_timeout)
        if self._observation_stale_timeout <= 0:
            raise ValueError("observation stale timeout must be positive")
        self._action_stale_timeout = float(action_stale_timeout)
        if self._action_stale_timeout <= 0:
            raise ValueError("action stale timeout must be positive")
        self._dbg_last_generation = -1
        self._include_neck = include_neck
        self._neck_publisher = neck_publisher
        self._neck_state_reader = neck_state_reader
        if self._include_neck and self._neck_state_reader is None:
            raise ValueError("--include-neck requires a NeckStateReader")
        if (self._include_neck and not self._dry_run and
                self._neck_publisher is None):
            raise ValueError("live --include-neck requires a NeckPublisher")
        # Frozen-action fallback. Loaded lazily on the first starved tick so an
        # RTC rollout (which should never starve) pays nothing, and a missing
        # onnxruntime/model degrades to the plain repeat instead of failing the run.
        self._frozen_action_enabled = bool(frozen_action)
        self._encoder = None
        # Init-prev and hold must use the same selected body-token space.
        self._encoder_version = encoder_version
        self._subtask_prompt = bool(subtask_prompt)
        self._goal_hw = tuple(goal_hw)
        self._run_event = run_event if run_event is not None else threading.Event()
        if run_event is None:
            self._run_event.set()
        self._hold_encoder = None
        self._encoder_failed = False
        self._frozen_ticks = 0
        self._wbc_started = False
        self._holding = False
        self._hold_action = None
        self._last_published_action = None
        self._last_observation_at = None
        self._last_observation_lock = threading.Lock()
        self._last_hold_reason = None
        self._last_problem_log_at = -float("inf")
        self._last_dry_hold_log_at = -float("inf")
        self._last_gate_log_at = -float("inf")
        self._rollout = rollout_recorder
        self._cam_tick = 0
        self._telemetry = ActionTelemetry(include_neck=include_neck, dry_run=dry_run)
        self._obs_rate = RateCounter("observation")
        self._action_state_lock = threading.Lock()
        self._last_version = -1
        self._last_accepted_action_at = None
        # An action is executable when it was computed for the prompt now in
        # effect. `_action_floor` is the newest version seen when the first
        # observation for this prompt went out; the server answers observations
        # in order, so anything above it cannot come from an earlier prompt.
        # None means no such observation has been sent yet: execute nothing.
        self._action_floor = None
        self._last_drop_log_at = -float("inf")
        # Encoder for the first-frame current-pose token (seeds the server first-chunk RTC).
        try:
            self._encoder = load_body_token_encoder(self._encoder_version)
        except Exception as e:
            print(f"[init-prev] encoder load failed ({e}); first chunk falls back to unconditioned")
            self._encoder = None

    def _throttled_problem(self, message, period=2.0):
        now = time.monotonic()
        if now - self._last_problem_log_at >= period:
            self._last_problem_log_at = now
            print(message, flush=True)

    def _hold_wbc(self, reason):
        with self._publish_lock:
            self._hold_wbc_locked(reason)

    def _hold_wbc_locked(self, reason):
        """Recoverable wait: keep the WBC process alive and freeze the measured pose."""
        if not self._holding:
            self._hold_action = None
            log_event("hold", reason=reason, dry_run=self._dry_run)
        self._holding = True
        self._last_hold_reason = reason

    def _terminate_wbc(self, reason):
        """Terminal stop is reserved for shutdown or a failed frozen-pose path."""
        with self._publish_lock:
            self._terminate_wbc_locked(reason)

    def _terminate_wbc_locked(self, reason):
        self._running = False
        if self._wbc_started and not self._dry_run and self._token_publisher is not None:
            try:
                self._token_publisher.send_command(start=False, stop=True, planner=True)
            finally:
                self._wbc_started = False
        self._last_hold_reason = reason
        self._block_actions_locked()

    def hold_tick(self):
        with self._publish_lock:
            if not self._running or self._dry_run or not self._holding or not self._wbc_started:
                return
            try:
                if self._state_sub.age() > self._observation_stale_timeout:
                    raise RuntimeError("robot state is stale during hold")
                if self._last_published_action is None:
                    raise RuntimeError("no executed action to retain hands/neck during hold")
                if self._hold_action is None:
                    self._hold_action = self._freeze_action(self._last_published_action, strict=True)
                self.execute_action(self._hold_action[None, :])
                if self._rollout is not None:
                    self._rollout.record_action(time.monotonic(), -1, -1, -1, True, self._hold_action)
            except Exception as exc:
                print(f"[client] ERROR: frozen hold failed: {exc}; stopping WBC", flush=True)
                log_event("hold_failure", error=str(exc))
                self._terminate_wbc_locked(f"hold failure: {exc}")

    def hold_loop(self):
        while self._running and self._run_event.is_set():
            started = time.monotonic()
            self.hold_tick()
            time.sleep(max(0, OBS_SEND_INTERVAL - (time.monotonic() - started)))

    def _block_actions_locked(self):
        """Make everything now in flight non-executable until an observation for
        the current prompt has gone out. Caller owns ``_publish_lock``."""
        with self._action_state_lock:
            self._action_floor = None
            self._last_accepted_action_at = None

    def _expire_wm_goal_locked(self, wm):
        """Hold and poison ACTIVE when the same-prompt last-good goal expires."""
        if not wm["goal_expired"]:
            return False
        reason = (
            f"WM last-good goal expired ({wm['goal_age_s']:.1f}s > "
            f"{wm['goal_hard_age_s']:.1f}s)"
        )
        self._hold_wbc_locked(reason)
        # Without this, a later fresh goal could restart WBC on an old ACTIVE
        # ack before the fresh candidate was ever acknowledged.
        self._block_actions_locked()
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
            self._hold_wbc_locked(
                "WM last-good goal crossed hard age; requiring a fresh "
                "condition acknowledgement"
            )
            self._block_actions_locked()
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
                    self._execution_started_at = None
                    self._hold_wbc_locked(
                        f"prompt transition {label}: epoch "
                        f"{before['prompt_epoch']} -> {after['prompt_epoch']}"
                    )
                    # Preserve the session/id counter, but make every old
                    # condition ack non-executable immediately.
                    self._block_actions_locked()
                    self._telemetry.reset(after["prompt_epoch"])
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

    def set_instruction(self, text, *, reason, task=None):
        """Commit the VLA prompt and the WM task under one publication fence.

        `text` is what the planner wants executed now and is what the WM
        conditions its goal image on. `task` is the instruction-level sentence
        the VLA was trained on as its Task; while the planner is stepping
        through atomic subtasks the two differ, and the step then rides in the
        optional Subtask clause instead of replacing the Task.
        """
        def transition():
            self._wm.set_task(text)
            self._task = task or text
            self._subtask = text if task and task != text else None
            self._instruction_hold_reason = None
            if hasattr(self._wm, "set_enabled"):
                self._wm.set_enabled(True)
        return self.apply_prompt_transition(reason, transition)

    def hold_instructions(self, reason):
        def transition():
            self._instruction_hold_reason = reason
            if hasattr(self._wm, "set_enabled"):
                self._wm.set_enabled(False)
            self._wm.restart()
        return self.apply_prompt_transition(reason, transition)

    def condition_snapshot(self):
        with self._send_lock, self._publish_lock, self._action_state_lock:
            return {
                "executing": self._action_floor is not None and not self._holding,
                "wm": self._wm.snapshot(), "instruction": self._task, "subtask": self._subtask,
                "instruction_hold": self._instruction_hold_reason,
                "hold_reason": self._last_hold_reason, "wbc_started": self._wbc_started,
                "connected": self._connected.is_set() and self._running, "dry_run": self._dry_run,
                "execution_started_at": self._execution_started_at, "holding": self._holding,
            }

    def _ensure_wbc_started(self):
        if not self._running:
            raise RuntimeError("client has stopped")
        self._holding, self._hold_action, self._last_hold_reason = False, None, None
        if self._dry_run or self._token_publisher is None:
            return
        if not self._wbc_started:
            self._token_publisher.send_command(start=True, stop=False, planner=True)
            self._wbc_started = True
            log_event("wbc_start")

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
        """A new connection: nothing from the previous one may execute."""
        epoch = self._wm.snapshot()["prompt_epoch"]
        with self._action_state_lock:
            self._action_floor = None
            self._last_accepted_action_at = None
        self._telemetry.reset(epoch)

    def _show_goal_window(self, rgb, caption=""):
        """Mirror the goal image the VLA is about to receive into a local window.

        Display-only: a failure here degrades to a one-time warning and never
        touches control. Checked against DISPLAY before any highgui call, since
        with no display the Qt plugin aborts from C++ and try/except cannot catch it.
        """
        if not self._show_goal or rgb is None:
            return
        try:
            frame = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
            if caption:
                cv2.putText(frame, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            if not self._goal_window_ready:
                cv2.namedWindow(_GOAL_WINDOW_NAME, cv2.WINDOW_NORMAL)
                self._goal_window_ready = True
            cv2.imshow(_GOAL_WINDOW_NAME, frame)
            cv2.waitKey(1)
        except Exception as exc:
            self._show_goal = False
            print(f"[goal-window] disabled after an error: {exc}", flush=True)

    def _observe(self, on_gate=lambda: None):
        """One control tick's inputs: robot state, ego frame, the WM goal for the
        current prompt, and the prompt epoch they belong to.

        Returns None when the tick is gated. The robot is already held and
        `on_gate` has run, so the caller only has to skip to its next tick.
        """
        # Match collection: capture the image, then sample the latest state.
        frame_bgr = self._camera.get_frame()
        state, state_received_at = self._state_sub.get_state_with_timestamp()
        if state is None:
            self._hold_wbc("no robot state")
            self._throttled_problem("[client] gated: waiting for robot state")
            on_gate()
            return None
        state_age = time.monotonic() - state_received_at
        if state_age > self._observation_stale_timeout:
            self._hold_wbc(f"robot state stale ({state_age:.3f}s > "
                           f"{self._observation_stale_timeout:.3f}s)")
            self._throttled_problem(f"[client] gated: robot state stale ({state_age:.3f}s)")
            on_gate()
            return None

        states, left_hand, right_hand = self._build_state(state)
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3):
            raise ValueError(f"camera frame must be BGR uint8 HxWx3, got "
                             f"{getattr(frame_bgr, 'dtype', None)} "
                             f"{getattr(frame_bgr, 'shape', None)}")
        frame = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._wm.update_latest_ego(frame)
        self._cam_tick += 1
        if self._rollout is not None:
            self._rollout.record_state(time.monotonic(), states)
            # Full-resolution ego frame; the recorder rate-limits internally.
            self._rollout.record_video_frame(time.monotonic(), frame)

        wm = self._wm.snapshot()
        goal = wm["goal"]
        # no-goal mode has no goal by construction, so "goal is None" is the
        # steady state there, not a reason to hold.
        no_goal_mode = wm.get("goal_source") == "no_goal"
        if (goal is None and not no_goal_mode) or wm["goal_stale"] or wm["goal_expired"]:
            if wm["goal_expired"]:
                self._hold_for_observed_wm_expiry(wm)
            else:
                self._hold_wbc("waiting for WM goal for the current prompt")
            on_gate()
            now = time.monotonic()
            if now - self._last_gate_log_at >= 2.0:
                self._last_gate_log_at = now
                print(f"[client] gated: waiting for current-prompt WM goal; "
                      f"stage={wm['prompt_stage']} epoch={wm['prompt_epoch']} "
                      f"goal_stage={wm['goal_stage']} goal_age={wm['goal_age_s']} "
                      f"expired={wm['goal_expired']} error={wm['last_wm_error']!r}", flush=True)
            return None
        if goal is not None:
            if goal.dtype != np.uint8 or goal.ndim != 3 or goal.shape[2] != 3:
                raise ValueError(f"WM goal must be RGB uint8 HxWx3, got {goal.dtype} {goal.shape}")
            goal = resize_goal_for_vla(goal, self._goal_hw)
        if self._instruction_hold_reason:
            on_gate()
            return None

        if wm["goal_generation"] != self._dbg_last_generation:
            self._dbg_last_generation = wm["goal_generation"]
            print(f"[client] condition prompt_stage={wm['prompt_stage']} "
                  f"goal_stage={wm['goal_stage']} stale={wm['goal_stale']} "
                  f"generation={wm['goal_generation']} subtask={wm['subtask']!r}", flush=True)

        # Match training capitalization; language and image share one condition.
        task = str(self._task).strip().lower()
        subtask = str(self._subtask or wm["subtask"] or "").strip().lower()
        instruction = (f"Task: {task}. Subtask: {subtask}"
                       if subtask and self._subtask_prompt else f"Task: {task}")
        self._show_goal_window(goal, f"stage {wm['prompt_stage']} gen {wm['goal_generation']} "
                                     f"{wm.get('goal_source', '')}".strip())
        return Observation(frame=frame, state=state, states=states, left_hand=left_hand,
                           right_hand=right_hand, wm=wm, goal=goal,
                           instruction=instruction, epoch=int(wm["prompt_epoch"]))

    def _accept_action(self, version, epoch):
        """(ok, reason) — is this action executable for the prompt now in effect?

        HTTP pairs a chunk with the observation record it was requested for, so
        that record's epoch is exact and is passed in. RTC streams actions back
        with no such pairing and passes epoch=None; the version floor stands in,
        because the server answers observations in order and numbers them
        monotonically.
        """
        version = int(version)
        with self._action_state_lock:
            if version <= self._last_version:
                raise ValueError(f"non-monotonic action version {version} <= {self._last_version}")
            self._last_version = version
            floor = self._action_floor
        if epoch is not None:
            current = int(self._wm.snapshot()["prompt_epoch"])
            if int(epoch) != current:
                return False, f"computed for prompt epoch {epoch}, now on {current}"
            return True, None
        if floor is None:
            return False, "no observation sent for the current prompt yet"
        if version <= floor:
            return False, f"version {version} <= floor {floor}: computed before the prompt changed"
        return True, None

    def _observation_sent(self):
        """First observation for this prompt: anything the server numbers above
        the versions seen so far was computed from it or something newer."""
        with self._action_state_lock:
            if self._action_floor is None:
                self._action_floor = self._last_version

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
            # Instruction transitions clear policy timestamps while hold_loop
            # publishes frozen actions. Its state/encoder checks own this wait.
            if not self._wbc_started or self._holding:
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
            log_event("wbc_stop", source="action_watchdog", reason=self._last_hold_reason)
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

    def _freeze_action(self, action, *, strict=False):
        """Encode measured body pose; retain the last commanded hands and neck.

        A transient interlock uses strict=True: a failed encoder cannot replay motion.
        """
        if self._encoder_failed:
            if strict:
                raise RuntimeError("hold encoder unavailable")
            return action
        if self._hold_encoder is None:
            try:
                if self._encoder is not None:
                    # Same layout as the init-prev encoder; no second ONNX session.
                    self._hold_encoder = self._encoder
                else:
                    self._hold_encoder = load_body_token_encoder(self._encoder_version)
                print(f"[frozen] hold encoder {self._encoder_version} loaded from "
                      f"{ENCODER_MODELS[self._encoder_version]}", flush=True)
            except Exception as exc:
                self._encoder_failed = True
                if strict:
                    raise RuntimeError("hold encoder unavailable") from exc
                print(f"[frozen] WARNING: encoder unavailable ({type(exc).__name__}: {exc}); "
                      f"falling back to plain repeat-last", flush=True)
                return action
        try:
            state = self._state_sub.get_state()
            if state is None:
                raise RuntimeError("no robot state for hold")
            qpos = _mujoco29_to_isaaclab29(state["body_q_measured"])          # (29,)
            base_quat = np.asarray(state["base_quat_measured"], dtype=np.float32)  # (4,) wxyz
            joint_pos = np.tile(qpos, (10, 1)).astype(np.float32)             # (10,29)
            joint_vel = np.zeros((10, 29), dtype=np.float32)
            body_quat = np.tile(base_quat, (10, 1)).astype(np.float32)        # (10,4)
            enc_token = np.asarray(
                self._hold_encoder.encode(joint_pos, joint_vel, body_quat), dtype=np.float32
            ).reshape(-1)
            if enc_token.shape != (TOKEN_DIM,) or not np.isfinite(enc_token).all():
                raise RuntimeError("invalid frozen body token")
            frozen = np.array(action, dtype=np.float32, copy=True)
            frozen[HAND_DIM:HAND_DIM + TOKEN_DIM] = enc_token
            self._frozen_ticks += 1
            if self._frozen_ticks == 1 or self._frozen_ticks % 30 == 0:
                print(f"[frozen] holding pose via encoder token "
                      f"({self._frozen_ticks} starved ticks so far)", flush=True)
            return frozen
        except Exception as exc:
            if strict:
                raise
            print(f"[frozen] WARNING: freeze failed ({type(exc).__name__}: {exc}); "
                  f"using repeat-last", flush=True)
            return action

    def execute_action(self, action):
        """
        Map the server action -> robot command and publish via Protocol v4.

        Server action layout is [hand_joints(14) | body_token(64)] (78-D),
        with neck(2) appended for the 80-D neck policy.
        publish_token expects [token(64) | left_hand(7) | right_hand(7)].
        """
        if not self._running:
            raise RuntimeError("cannot publish after client stop")
        action = self._validated_action(action)[0]

        hand_joints = action[:HAND_DIM]
        # Raw policy token passed straight through: the model trained on FSQ-grid
        # tokens, so its output is already near-grid, and the sonic decoder takes
        # continuous floats. No fsq_quantize (and no [-0.625, 0.625] clip) here.
        token_ori = action[HAND_DIM:HAND_DIM + TOKEN_DIM]

        action_out = np.concatenate([token_ori, hand_joints])  # [token(64), LH(7), RH(7)]
        if action_out.shape != (ACTION_DIM_DEFAULT,) or not np.isfinite(action_out).all():
            raise ValueError("reordered publish action is not a finite 78-D vector")
        if self._dry_run:
            return
        if self._token_publisher is None:
            raise RuntimeError("live mode has no TokenPublisher")
        self._token_publisher.publish_token(action_out)
        self._last_published_action = action.copy()
        if self._include_neck:
            if self._neck_publisher is None:
                raise RuntimeError("live neck mode has no NeckPublisher")
            neck = action[HAND_DIM + TOKEN_DIM:HAND_DIM + TOKEN_DIM + NECK_DIM]
            self._neck_publisher.publish(neck[0], neck[1])

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


    def _publish_action_for_prompt(self, action, version, epoch, data, now, *, freeze_repeat=False):
        """Validate, gate and publish one action. `epoch` is the prompt epoch the
        action was computed for when the transport knows it exactly (HTTP), else
        None (RTC), where the version floor decides."""
        interval = now - self.start_time
        self.start_time = now
        try:
            action = self._validated_action(action)
        except ValueError as exc:
            self._hold_wbc(f"invalid VLA action: {exc}")
            print(f"[client] ERROR: rejected action version={version}: {exc}", flush=True)
            return "dropped"

        fresh, obs_age, state_age, camera_age, neck_age = self._freshness()
        wm = self._wm.snapshot()
        _no_goal_mode = wm.get("goal_source") == "no_goal"
        wm_condition_ready = (
            (wm["goal"] is not None or _no_goal_mode)
            and not wm["goal_stale"]
            and not wm["goal_expired"]
        )
        if not wm_condition_ready:
            if wm["goal_expired"]:
                self._hold_for_observed_wm_expiry(wm)
            else:
                self._hold_wbc("no WM goal for the current prompt")
            return "held"
        if not fresh:
            self._hold_wbc(
                "stale observation "
                f"obs={obs_age:.3f}s state={state_age:.3f}s "
                f"camera={camera_age:.3f}s neck={neck_age:.3f}s "
                f"limit={self._observation_stale_timeout:.3f}s"
            )
            return "held"

        try:
            ok, reason = self._accept_action(version, epoch)
        except ValueError as exc:
            self._hold_wbc(f"invalid VLA action stream: {exc}")
            print(f"[client] ERROR: rejected action: {exc}", flush=True)
            return "dropped"
        self._telemetry.received(version)
        if not ok:
            self._hold_wbc(f"no action for the current prompt: {reason}")
            if now - self._last_drop_log_at >= 1.0:
                self._last_drop_log_at = now
                print(f"[client] dropped action version={version}: {reason}", flush=True)
                log_event("action_dropped", version=int(version), reason=str(reason))
            return "dropped"

        with self._publish_lock:
            latest_wm = self._wm.snapshot()
            if int(latest_wm["prompt_epoch"]) != int(wm["prompt_epoch"]):
                self._hold_wbc_locked("prompt changed before action publication")
                return "dropped"
            if self._expire_wm_goal_locked(latest_wm):
                return "dropped"
            self._ensure_wbc_started()
            if freeze_repeat and self._frozen_action_enabled and data.get("rtc_repeat_last") is True:
                action = self._freeze_action(action[0])[None, :]
            self.execute_action(action)
            self._record_action_accepted(now)
            if self._execution_started_at is None:
                self._execution_started_at = now
                log_event("execution_started", epoch=int(wm["prompt_epoch"]),
                          version=int(version), started_at_mono=now)

        if self._rollout is not None:
            self._rollout.record_action(
                now, int(version), data.get("rtc_chunk_id"), data.get("rtc_chunk_tick"),
                bool(data.get("rtc_repeat_last")), action[0].copy())
        line = self._telemetry.update(data, action, version, interval, now,
                                          wm["prompt_epoch"], None)
        if line is not None:
            print(line, flush=True)
        return "ok"



class RTCWebSocketClient(VlaClientBase):
    """RTC observation streaming and per-action ACKs over WebSocket."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ws = None
        self._send_thread_handle = None
        self._sent_init_prev = False

    def _on_open(self, ws):
        sid = self._reset_condition_session()
        print(f"[client] Connected! condition_session={sid[:12]}")
        self._connected.set()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get("action") is None:
                return
            action = convert_numpy_in_dict(data["action"], numpy_deserialize)
            self._publish_action_for_prompt(action, data.get("version", -1), None, data,
                                            time.monotonic(), freeze_repeat=True)
        except Exception as exc:
            self._hold_wbc(f"action processing error: {exc}")
            print(f"[client] Message processing error: {exc}", flush=True)

    def _on_error(self, ws, error):
        print(f"[client] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[client] Connection closed: {close_status_code} - {close_msg}")
        self._hold_wbc("VLA WebSocket closed")
        self._running = False
        self._run_event.clear()

    def _send_thread(self):
        print("[client] Send thread started, waiting for connection...")
        while self._running and self._run_event.is_set() and not self._connected.wait(0.1):
            pass
        if not self._connected.is_set():
            print("[client] Send thread stopped before connection")
            return
        print("[client] Connected, starting observation loop")

        while self._running and self._run_event.is_set():
            tick_started = time.monotonic()
            self._check_action_liveness()
            try:
                obs = self._observe()
                if obs is None:
                    time.sleep(max(0, OBS_SEND_INTERVAL - (time.monotonic() - tick_started)))
                    continue
                frame, state, states, wm = obs.frame, obs.state, obs.states, obs.wm
                subgoal_frame, instruction, obs_epoch = obs.goal, obs.instruction, obs.epoch
                left_hand_states, right_hand_states = obs.left_hand, obs.right_hand

                # Build observation payload. Image keys MUST match the server's repack:
                #   ego image  -> repack.image_keys[0]  == "video.egocentric"
                #   goal image -> repack.subgoal_key[0] == "subgoal.egocentric"
                img_obs = {"video.egocentric": frame}
                if subgoal_frame is not None:
                    img_obs["subgoal.egocentric"] = subgoal_frame
                # else: no-goal mode ships no goal key at all. The server's
                # _goal_slot() then puts a copy of the observation in the slot and
                # flags goal_dropped, reproducing training's placeholder path.
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
                    "condition": None,
                    "timestamp": None,
                }
                payload = convert_numpy_in_dict(payload, numpy_serialize)
                message = json.dumps(payload)

                # Send (thread-safe). Operator prompt mutation takes this same
                # fence, so the final epoch check cannot race Enter/:ov/:resume.
                with self._send_lock:
                    latest_wm = self._wm.snapshot()
                    if int(latest_wm["prompt_epoch"]) != obs_epoch or latest_wm["goal_expired"]:
                        continue
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send(message)
                        self._observation_sent()
                        with self._last_observation_lock:
                            self._last_observation_at = time.monotonic()
                        if init_prev_in_payload:
                            self._sent_init_prev = True
                            print(
                                f"[init-prev] first-frame pseudo prev-action sent; "
                                f"token range=[{enc_token.min():.3f},{enc_token.max():.3f}]"
                            )
                        self._obs_rate.tick()
                    else:
                        print("[client] WebSocket not connected, skipping send")
                        self._hold_wbc("VLA WebSocket disconnected")
                        break

            except Exception as e:
                self._hold_wbc(f"observation loop error: {e}")
                self._throttled_problem(f"[client] observation rejected: {e}")

            sleep_time = max(0, OBS_SEND_INTERVAL - (time.monotonic() - tick_started))
            time.sleep(sleep_time)

            line = self._obs_rate.report(time.monotonic(),
                                         prompt_stage=self._wm.status()["prompt_stage"])
            if line is not None:
                print(line, flush=True)

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
        self._terminate_wbc("client shutdown")
        self._connected.set()
        if self._ws:
            self._ws.close()
        if (self._send_thread_handle is not None and
                self._send_thread_handle is not threading.current_thread()):
            self._send_thread_handle.join(timeout=2.0)


CTRL_DT = OBS_SEND_INTERVAL  # 30 Hz playback / capture cadence


DEFAULT_HTTP_TIMEOUT = 30.0


class HttpChunkClient(VlaClientBase):
    """Capture observations, request a chunk, and play it at 30 Hz with frozen gaps."""

    def __init__(self, *args, http_timeout=DEFAULT_HTTP_TIMEOUT, **kwargs):
        super().__init__(*args, **kwargs)
        self._http_timeout = float(http_timeout)
        if self._http_timeout <= 0:
            raise ValueError("http timeout must be positive")
        # A chunk that has not landed this long after it was requested is a
        # liveness failure (hung VLA); the hold-in-place stream must not mask it.
        # A chunk that never lands is a stalled server; the observation and action
        # staleness watchdogs are what actually stop the robot.
        self._chunk_overdue_timeout = max(2.0, 2 * OBS_SEND_INTERVAL * 30)
        self._session = requests.Session()
        self._session.trust_env = False
        self._obs_lock = threading.Lock()
        self._latest_obs = None
        self._chunk_lock = threading.Lock()
        self._pending_chunk = None
        self._request_event = threading.Event()
        self._request_seq = 0
        self._chunk_seq = 0
        self._action_version = 0
        self._obs_thread = None
        self._infer_thread = None
        self._last_http_error_log_at = -float("inf")
        self._last_overdue_log_at = -float("inf")

    # ------------------------------------------------------------------ obs
    def _observation_thread(self):
        """30 Hz capture of (state, ego frame, WM goal, condition) records.

        Mirrors the RTC client's send thread minus the WebSocket send: the same
        gating, the same condition minting, the same recorder feeds. The newest
        record is what the inference thread POSTs when a chunk is requested.
        """
        print("[client] Observation capture thread started")
        while self._running and self._run_event.is_set():
            tick_started = time.monotonic()
            self._check_action_liveness()
            try:
                obs = self._observe(self._drop_latest_obs)
                if obs is None:
                    time.sleep(max(0.0, CTRL_DT - (time.monotonic() - tick_started)))
                    continue
                frame, states, wm = obs.frame, obs.states, obs.wm
                subgoal_frame, instruction, obs_epoch = obs.goal, obs.instruction, obs.epoch

                img_obs = {"video.egocentric": frame}
                if subgoal_frame is not None:
                    img_obs["subgoal.egocentric"] = subgoal_frame
                payload = {
                    "image": img_obs,
                    "state": {"states": states},
                    "gt_action": None,
                    "dataset_name": None,
                    "instruction": instruction,
                    "history": None,
                    # No wire provenance on /act: the reply is paired with this
                    # request by construction, and the server pins its session id
                    # per WebSocket connection only, which a restarted HTTP client
                    # would trip. The condition is tracked locally instead.
                    "condition": None,
                    "timestamp": None,
                }
                now = time.monotonic()
                record = {"payload": payload, "epoch": obs_epoch, "wm": wm, "captured_at": now}
                with self._obs_lock:
                    self._latest_obs = record
                self._observation_sent()
                with self._last_observation_lock:
                    self._last_observation_at = now
                self._obs_rate.tick()

            except Exception as exc:
                self._hold_wbc(f"observation loop error: {exc}")
                self._throttled_problem(f"[client] observation rejected: {exc}")
                self._drop_latest_obs()

            time.sleep(max(0.0, CTRL_DT - (time.monotonic() - tick_started)))

            line = self._obs_rate.report(time.monotonic(),
                                         prompt_stage=self._wm.status()["prompt_stage"])
            if line is not None:
                print(line, flush=True)
        print("[client] Observation capture thread stopped")

    def _drop_latest_obs(self):
        with self._obs_lock:
            self._latest_obs = None

    # ------------------------------------------------------------ inference
    def _post_act(self, payload):
        """POST one observation; return the chunk as float32 (rows, action_dim)."""
        body = json.dumps(convert_numpy_in_dict(payload, numpy_serialize))
        response = self._session.post(
            self.server_url, data=body,
            headers={"Content-Type": "application/json"},
            timeout=self._http_timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or "action" not in data:
            raise RuntimeError(f"VLA /act returned no action: {str(data)[:200]}")
        chunk = convert_numpy_in_dict(data["action"], numpy_deserialize)
        if not isinstance(chunk, np.ndarray):
            raise RuntimeError(f"VLA /act action is not an array: {type(chunk).__name__}")
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        expected_dim = ACTION_DIM_NECK if self._include_neck else ACTION_DIM_DEFAULT
        if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != expected_dim:
            raise RuntimeError(
                f"VLA /act chunk shape {chunk.shape}, expected (>=1, {expected_dim})"
            )
        if not np.isfinite(chunk).all():
            raise RuntimeError("VLA /act chunk contains NaN or Inf")
        return np.ascontiguousarray(chunk)

    def _inference_thread(self):
        """Serve chunk requests from the playback loop, one at a time."""
        print("[client] HTTP inference thread started")
        while self._running and self._run_event.is_set():
            if not self._request_event.wait(0.05):
                continue
            request_seq = self._request_seq

            with self._obs_lock:
                record = self._latest_obs
            now = time.monotonic()
            if record is None or now - record["captured_at"] > self._observation_stale_timeout:
                self._throttled_problem("[http] waiting for a fresh observation to request a chunk")
                time.sleep(CTRL_DT)
                continue
            # Same fence as the RTC send: an operator prompt transition that
            # landed after capture must not be sent as an old-epoch request.
            with self._send_lock:
                latest_wm = self._wm.snapshot()
                if int(latest_wm["prompt_epoch"]) != record["epoch"] or latest_wm["goal_expired"]:
                    time.sleep(CTRL_DT)
                    continue

            t0 = time.monotonic()
            try:
                chunk = self._post_act(record["payload"])
            except Exception as exc:
                self._hold_wbc(f"VLA /act failed: {exc}")
                if time.monotonic() - self._last_http_error_log_at >= 1.0:
                    self._last_http_error_log_at = time.monotonic()
                    print(f"[http] ERROR: /act failed: {exc}", flush=True)
                    log_event("http_act_error", error=str(exc))
                time.sleep(0.2)
                continue
            infer_ms = (time.monotonic() - t0) * 1000.0
            obs_age_ms = (t0 - record["captured_at"]) * 1000.0

            self._chunk_seq += 1
            pending = {
                "seq": self._chunk_seq,
                "request_seq": request_seq,
                "chunk": chunk,
                "epoch": record["epoch"],
                "infer_ms": infer_ms,
                "obs_age_ms": obs_age_ms,
                "landed_at": time.monotonic(),
            }
            with self._chunk_lock:
                self._pending_chunk = pending
            # Only the request this reply answers is cleared; a newer request
            # (prompt transition mid-flight) stays pending and is served next.
            if self._request_seq == request_seq:
                self._request_event.clear()
            print(
                f"[http] chunk seq={pending['seq']} rows={chunk.shape[0]} "
                f"epoch={record['epoch']} infer_ms={infer_ms:.1f} "
                f"obs_age_ms={obs_age_ms:.1f}", flush=True
            )
            log_event(
                "http_chunk", seq=pending["seq"], rows=int(chunk.shape[0]),
                epoch=record["epoch"], infer_ms=round(infer_ms, 2),
                obs_age_ms=round(obs_age_ms, 2),
            )
        print("[client] HTTP inference thread stopped")

    def _request_chunk(self):
        self._request_seq += 1
        self._request_event.set()
        return time.monotonic()

    # ------------------------------------------------------------- playback
    def _publish_action(self, action, meta, chunk_tick, repeat_last, now):
        """/act pairs a chunk with the record it was requested for, so that
        record's prompt epoch is exact."""
        self._action_version += 1
        data = {"rtc_chunk_id": meta["seq"], "rtc_chunk_tick": chunk_tick,
                "rtc_repeat_last": bool(repeat_last), "rtc_infer_ms": meta["infer_ms"]}
        return self._publish_action_for_prompt(
            np.asarray(action, dtype=np.float32)[None, :], self._action_version,
            meta["epoch"], data, now)

    def _publish_loop(self):
        """30 Hz: play the chunk, then hold in place until the next one lands."""
        chunk = None
        meta = None
        idx = 0
        last_action = None
        frozen = None
        requested_at = None
        next_tick = time.monotonic()
        print("[client] Playback loop started; requesting first chunk")
        requested_at = self._request_chunk()

        while self._running and self._run_event.is_set():
            now = time.monotonic()

            with self._chunk_lock:
                pending = self._pending_chunk
                self._pending_chunk = None
            if pending is not None:
                current_epoch = int(self._wm.snapshot()["prompt_epoch"])
                if pending["epoch"] != current_epoch:
                    print(
                        f"[http] chunk seq={pending['seq']} discarded: epoch "
                        f"{pending['prompt_epoch']} != current {current_epoch}; "
                        "re-requesting", flush=True
                    )
                    log_event("http_chunk_discarded", seq=pending["seq"],
                              epoch=pending["epoch"], current_epoch=current_epoch)
                    requested_at = self._request_chunk()
                else:
                    wait_ms = (now - requested_at) * 1000.0 if requested_at else 0.0
                    chunk = pending["chunk"]
                    meta = pending
                    idx = 0
                    frozen = None
                    requested_at = None
                    print(
                        f"[http] playing chunk seq={meta['seq']} rows={chunk.shape[0]} "
                        f"epoch={meta['epoch']} freeze_ms={wait_ms:.0f}", flush=True
                    )
                    log_event("http_chunk_start", seq=meta["seq"],
                              rows=int(chunk.shape[0]), epoch=meta["epoch"],
                              freeze_ms=round(wait_ms, 1))

            if chunk is not None and idx < len(chunk):
                action = chunk[idx]
                tick = idx
                idx += 1
                repeat_last = False
            else:
                # Exhausted (or nothing yet). First exhausted tick: request the
                # next chunk from the observation captured now, and compute the
                # hold-in-place action once.
                if requested_at is None:
                    requested_at = self._request_chunk()
                if last_action is None or meta is None:
                    self._pace(next_tick)
                    next_tick = self._next_tick(next_tick)
                    continue
                if frozen is None:
                    if self._frozen_action_enabled:
                        frozen = self._freeze_action(last_action)
                    else:
                        frozen = np.array(last_action, dtype=np.float32, copy=True)
                    print(
                        f"[http] chunk seq={meta['seq']} done ({len(chunk) if chunk is not None else 0} rows); "
                        f"holding {'encoder token' if self._frozen_action_enabled else 'last action'} "
                        "until the next chunk", flush=True
                    )
                if now - requested_at > self._chunk_overdue_timeout:
                    self._hold_wbc(
                        f"VLA chunk overdue ({now - requested_at:.1f}s > "
                        f"{self._chunk_overdue_timeout:.1f}s)")
                    if now - self._last_overdue_log_at >= 2.0:
                        self._last_overdue_log_at = now
                        print(f"[http] chunk overdue: {now - requested_at:.1f}s since request",
                              flush=True)
                        log_event("http_chunk_overdue", seconds=round(now - requested_at, 2))
                    self._pace(next_tick)
                    next_tick = self._next_tick(next_tick)
                    continue
                action = frozen
                tick = len(chunk) if chunk is not None else -1
                repeat_last = True

            outcome = self._publish_action(action, meta, tick, repeat_last, now)
            if outcome == "ok":
                last_action = np.array(action, dtype=np.float32, copy=True)
            elif outcome == "dropped":
                # The chunk's condition died (prompt transition, expiry, starved
                # rollover): stop playing it and fetch a chunk for the new one.
                chunk = None
                meta = None
                idx = 0
                frozen = None
                last_action = None
                requested_at = self._request_chunk()

            self._pace(next_tick)
            next_tick = self._next_tick(next_tick)
        print("[client] Playback loop stopped")

    @staticmethod
    def _pace(next_tick):
        sleep_time = next_tick + CTRL_DT - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

    @staticmethod
    def _next_tick(next_tick):
        nxt = next_tick + CTRL_DT
        now = time.monotonic()
        # A missed tick restarts the schedule instead of bursting to catch up.
        return nxt if nxt > now - CTRL_DT else now

    # ------------------------------------------------------------ lifecycle
    def run(self):
        print(f"[client] HTTP chunk client -> {self.server_url}")
        sid = self._reset_condition_session()
        print(f"[client] condition_session={sid[:12]} (acks synthesised per /act reply)")
        self._connected.set()
        self._obs_thread = threading.Thread(
            target=self._observation_thread, name="vla-observation-capture", daemon=True)
        self._infer_thread = threading.Thread(
            target=self._inference_thread, name="vla-http-act", daemon=True)
        self._obs_thread.start()
        self._infer_thread.start()
        try:
            self._publish_loop()
        finally:
            self._running = False
            self._request_event.set()
            for th in (self._obs_thread, self._infer_thread):
                if th is not None and th is not threading.current_thread():
                    th.join(timeout=2.0)
                    if th.is_alive():
                        print(f"[client] WARNING: {th.name} did not stop within 2s", flush=True)
            print("[client] Client stopped")

    def stop(self):
        self._running = False
        self._terminate_wbc("client shutdown")
        self._connected.set()
        self._request_event.set()
        for th in (self._obs_thread, self._infer_thread):
            if th is not None and th is not threading.current_thread():
                th.join(timeout=2.0)
        try:
            self._session.close()
        except Exception:
            pass
