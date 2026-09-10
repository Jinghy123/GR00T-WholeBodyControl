"""World-model prediction and instruction-only goals."""

import cv2
import numpy as np
import os
import requests
import threading
import time
import uuid
from .recording import InferenceRecorder, log_event
from base64 import b64decode, b64encode


WM_GATE_GRAY_SIZE = (160, 120)


WM_STAYPUT_OBS_MOTION_MAX = 6.5


WM_STAYPUT_GOAL_VS_OBS_MAX = 25.0


WM_STAYPUT_CONSECUTIVE = 3


WM_COLLAPSE_OBS_MOTION_MAX = 15.0


WM_COLLAPSE_GOAL_VS_OBS_MIN = 48.0


WM_COLLAPSE_GOAL_JUMP_MIN = 30.0


WM_COLLAPSE_IMMEDIATE_RETRIES = 3


WM_STAYPUT_IMMEDIATE_RETRIES = 3


WM_GATE_VERSION = "wm-global-mad/3"


DEFAULT_WM_GOAL_HARD_AGE = 30.0


class WmClient:
    """Own prompt stages and asynchronously refresh the last-good WM goal.

    The camera is read only by the VLA send loop.  This worker receives immutable
    snapshots through :meth:`update_latest_ego`, so it never races the camera's ZMQ
    REQ socket.  Requests are serialized.  Each request is tagged with the prompt
    stage and epoch that produced it; an Enter/restart during an in-flight request
    makes that response stale and therefore unable to overwrite the current goal.
    """

    def __init__(self, base_url, subgoal="", task="", period=1.6, timeout=15.0,
                 jpeg_quality=90,
                 goal_hard_age=DEFAULT_WM_GOAL_HARD_AGE,
                 dump_dir="/tmp/psix_wm_client", mode="future", seconds=None,
                 request_recorder=None, collapse_gate=False):
        subgoal = str(subgoal or "").strip()
        if mode == "subgoal" and not subgoal:
            raise ValueError("subgoal WM mode requires target text")
        if period <= 0 or timeout <= 0:
            raise ValueError("WM period and timeout must be positive")
        if not 1 <= jpeg_quality <= 100 or goal_hard_age < 0:
            raise ValueError("invalid WM JPEG quality or goal age limit")
        if mode not in ("future", "subgoal"):
            raise ValueError(f"[wm] mode must be 'future' or 'subgoal', got {mode!r}")
        if seconds is not None and float(seconds) != 0 and not 0 < float(seconds) <= 60:
            raise ValueError(f"[wm] seconds out of sane range: {seconds!r}")

        self._collapse_gate = bool(collapse_gate)
        self._base_url = base_url.rstrip("/")
        self._subgoal = subgoal
        self._task = str(task).strip()
        self._period = float(period)
        self._timeout = float(timeout)
        self._jpeg_quality = int(jpeg_quality)
        # A same-prompt last-good goal remains usable through transient WM
        # failures/rejections.  Zero disables the final outage safety cutoff.
        self._goal_hard_age = float(goal_hard_age)
        # Which prediction path the WM server is serving. The zedmini deployment
        # checkpoints are trained FUTURE-only, so that is the default; "subgoal"
        # is for the annotated-completion servers. Sent to the server, which
        # rejects a mismatch instead of silently answering with the wrong kind
        # of prediction.
        self._mode = mode
        # Future-mode prediction horizon, sent explicitly on every request so the
        # deployed horizon never depends on how the server happened to be
        # launched. None (or 0) means "defer to the server's default". The
        # server clamps to its trained window and quantizes to 0.1 s.
        self._seconds = None if seconds in (None, 0) else float(seconds)
        self._dump_dir = os.path.abspath(os.path.expanduser(dump_dir))
        self.request_recorder = (request_recorder if request_recorder is not None
                                 else InferenceRecorder(self._dump_dir))

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
        self._enabled = True
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
        """Return (decision, metrics) for a fresh goal; caller holds the lock.

        The first goal in an epoch is accepted without a content history.
        Collapse rejection is opt-in. Stay-put proposals are retried before
        installation, then accepted after the retry limit to allow completion.
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
            "collapse_suppressed": bool(
                collapse_like and not self._collapse_gate),
            "reject_streak": 0,
        }
        if collapse_like and self._collapse_gate:
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

    def latest_ego(self):
        """The cached camera frame and when it was captured, shared by WM and
        HLP; neither of them opens the camera."""
        with self._lock:
            return self._latest_ego, self._latest_ego_at

    def set_enabled(self, enabled):
        with self._lock:
            self._enabled = bool(enabled)
        self._wake_evt.set()

    def _current_subtask_locked(self):
        return self._subgoal

    def _invalidate_goal_locked(self):
        self._prompt_epoch += 1
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
                "seconds": self._seconds,
                # Goals are never mutated after assignment, so returning this reference
                # is safe and avoids copying a full image at 30 Hz.
                "goal": self._last_good_goal,
                "goal_stage": self._goal_stage,
                "goal_stale": stale,
                "goal_generation": self._goal_generation,
                "goal_age_s": goal_age,
                "goal_expired": goal_expired,
                "goal_hard_age_s": self._goal_hard_age,
                "pending_goal": self._pending_goal,
                "last_wm_ms": self._last_wm_ms,
                "last_wm_error": self._last_wm_error,
            }

    def status(self):
        snap = self.snapshot()
        snap.pop("goal")
        return snap


    def set_seconds(self, seconds):
        """Change the future horizon at runtime and regenerate immediately.

        ``None`` defers to the server's own default. Unlike a prompt change this
        does NOT invalidate the last-good goal: the scene and the instruction are
        unchanged, only how far ahead the next prediction looks, so the existing
        goal stays valid to serve until the new one lands.
        """
        if seconds is not None:
            seconds = float(seconds)
            if not 0 < seconds <= 60:
                raise ValueError(f"[wm] seconds out of sane range: {seconds!r}")
        if self._mode != "future":
            print(f"[wm] mode={self._mode!r}: horizon is ignored (future mode only)", flush=True)
        with self._lock:
            self._seconds = seconds
        shown = "server default" if seconds is None else f"{seconds}s"
        print(f"[wm] horizon -> {shown}; regenerating now "
              "(server clamps to its trained window)", flush=True)
        self._wake_evt.set()


    def set_task(self, text):
        """Replace the task instruction sent to the WM; regenerate under a new epoch."""
        text = str(text).strip()
        if not text:
            raise ValueError("task prompt must not be empty")
        with self._lock:
            self._task = text
            self._invalidate_goal_locked()
        print(f"[wm] task -> {text!r}; gated until its WM goal lands", flush=True)
        self._wake_evt.set()
        return True

    def restart(self):
        """Invalidate previous and pending predictions under a new epoch."""
        with self._lock:
            self._invalidate_goal_locked()
            self._dumped_stage = None
        print("[wm] restart -> prompt stage 0; gated until a fresh goal lands", flush=True)
        self._wake_evt.set()
        return True


    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._wake_evt.set()
        with self._lock:
            stage = self._prompt_stage
            subtask = self._subgoal
        # Same "[wm] prompt -> stage" shape as advance_prompt so the launcher's
        # quiet terminal filter shows the starting subtask too.
        print(f"[wm] prompt -> stage {stage}: {subtask!r} (start)", flush=True)
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
            if self._latest_ego is None or not self._enabled:
                return None
            now = time.monotonic()
            self._request_seq += 1
            request = {
                "request_id": self._request_seq,
                "stage": self._prompt_stage,
                "epoch": self._prompt_epoch,
                "task": self._task,
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
                # snapshot under the lock: :sec can change it from the stdin thread
                "seconds": self._seconds,
            }
            self._pending_goal = {
                "requested_stage": request["stage"],
                "requested_subtask": request["subtask"],
                "request_epoch": request["epoch"],
                "request_id": request["request_id"],
            }
            return request

    def _build_request_body(self, request):
        """Build a WM request entirely from the captured prompt and frame."""
        body = {
            "transport": "jpeg",
            "jpeg": True,
            "ego_jpeg": self._encode_jpeg(request["ego"], self._jpeg_quality),
            "task": request["task"],
            "req_id": request["request_id"],
            "robot_episode_session_id": self._episode_session_id,
            "prompt_gen": request["epoch"],
            "mode": self._mode,
        }
        # The envelope requires non-empty subtask even for task-only future
        # predictions. This fallback does not create a semantic subtask stage.
        body["subtask"] = request["subtask"] or request["task"]
        if self._mode == "future" and request["seconds"] is not None:
            body["seconds"] = request["seconds"]
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
        record = None
        response = None
        try:
            body = self._build_request_body(request)
            record = self.request_recorder.begin(
                "wm", f"{self._base_url}/wm", body,
                context={**{key: value for key, value in request.items() if key != "ego"},
                         "wm_mode": self._mode, "refresh_period_s": self._period,
                         "gate_version": WM_GATE_VERSION},
            )
            response = self._session.post(
                f"{self._base_url}/wm", json=body, timeout=self._timeout
            )
            try:
                payload = response.json()
            except ValueError:
                self.request_recorder.response(record, response.status_code,
                                               {"raw_text": response.text})
                raise ValueError(f"non-JSON WM response (http {response.status_code})")
            self.request_recorder.response(record, response.status_code, payload)
            if response.status_code != 200:
                raise ValueError(f"http {response.status_code}: {str(payload)[:200]}")
            if not isinstance(payload, dict) or "subgoal_jpeg" not in payload:
                raise ValueError("response missing JPEG field 'subgoal_jpeg'")
            if int(payload.get("req_id", -1)) != request["request_id"]:
                raise ValueError(
                    f"response req_id {payload.get('req_id')!r} does not match "
                    f"request {request['request_id']}"
                )
            # Future servers may omit the compatibility subtask echo. Request
            # identity and the local prompt recheck still reject stale responses.
            if (self._mode != "future" and
                    str(payload.get("subtask", "")) != request["subtask"]):
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
            outcome = ("transport_error" if response is None else
                       "http_error" if response.status_code != 200 else "invalid_response")
            self.request_recorder.finish(record, outcome, error=str(exc))
            with self._lock:
                if (self._pending_goal is not None and
                        self._pending_goal["request_id"] == request["request_id"]):
                    self._pending_goal = None
            self._set_error(exc)
            return

        wm_ms = (time.perf_counter() - t0) * 1000.0
        if self._stop_evt.is_set():
            self.request_recorder.finish(record, "cancelled")
            return
        # Content-gate thumbnails are pure image math; keep them off the lock.
        gray_obs = self._gate_gray(request["ego"])
        gray_goal = self._gate_gray(goal)
        with self._lock:
            if self._stop_evt.is_set():
                self.request_recorder.finish(record, "cancelled")
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
            generation = self._goal_generation
            if stale_response:
                current_stage = self._prompt_stage
                current_epoch = self._prompt_epoch
            else:
                gate_decision, gate_metrics = self._goal_gate_locked(
                    gray_obs, gray_goal
                )
                if gate_decision == "reject":
                    reroll_seed = self._reroll_seed
                    self._reroll_parent_request_id = request["request_id"]
                    immediate_retry = (
                        self._collapse_rejects
                        <= WM_COLLAPSE_IMMEDIATE_RETRIES
                    )
                    self._last_wm_ms = wm_ms
                    self._last_wm_error = "goal rejected by collapse monitor"
                elif gate_decision == "stayput_hold":
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

        self.request_recorder.finish(
            record, "stale_dropped" if stale_response else gate_decision,
            gate=gate_metrics, wm_latency_ms=wm_ms, goal_generation=generation,
        )
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
            log_event("wm_response", decision="stayput_hold",
                      request_id=request["request_id"], stage=request["stage"],
                      epoch=request["epoch"], wm_ms=wm_ms,
                      seed=request["seed"], gate=gate_metrics)
            self._wake_evt.set()
            return

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
        if first_for_stage and self.request_recorder.enabled:
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


class NoGoalProvider:
    """Instruction-only condition, with no WM or dataset dependency."""

    def __init__(self, task=""):
        self._task = task
        self._prompt_epoch = 0
        self._lock = threading.Lock()
        self._episode_session_id = str(uuid.uuid4())

    def snapshot(self):
        with self._lock:
            return {"prompt_stage": 0, "prompt_epoch": self._prompt_epoch, "subtask": "",
                    "goal": None, "goal_stage": 0, "goal_stale": False,
                    "goal_generation": self._prompt_epoch, "goal_age_s": None,
                    "goal_expired": False, "goal_hard_age_s": 0, "pending_goal": None,
                    "last_wm_ms": None, "last_wm_error": None, "goal_source": "no_goal"}

    def status(self):
        return self.snapshot()

    def set_task(self, text):
        with self._lock:
            self._task = text
            self._prompt_epoch += 1

    def restart(self):
        with self._lock:
            self._prompt_epoch += 1

    def update_latest_ego(self, rgb):
        pass

    def start(self):
        pass

    def stop(self):
        pass
