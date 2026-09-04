"""HTTP chunk client with remote-WM / episode-GT goals for the PsiX server.

Transport-only sibling of ``psix_rtc_sonic_wm_client.py``: the robot I/O stack,
the 45/80 (or 43/78) state/action layout, the WM / episode / no-goal providers,
the prompt controls (Enter, :ov, :resume, :restart, :sec, :mark), the safety
holds, the condition bookkeeping, the telemetry, the flight recorder and the
whole command line are imported from that client unchanged.  Only the way
actions reach the robot differs:

* one observation is POSTed to ``serve_psix.py`` ``/act`` (served with
  ``rtc_mode=off``) and answers with a chunk of ``action_exec_horizon`` rows;
* the chunk is played back one row per control tick (30 Hz);
* when it is exhausted the next chunk is requested from the observation
  captured at that moment, and until it lands a hold-in-place action is
  published: the body token is re-encoded from the measured pose while hand
  and neck keep their last commanded values (``psix_sonic_client.py``).

Requests are strictly sequential, so every chunk boundary carries a freeze of
one HTTP round trip.  That is the intended open-loop baseline, not a bug.

Condition provenance: ``/act`` has no ack fields, but an HTTP reply is by
construction the chunk computed for the condition that was sent, so the ack
is synthesised locally from the request's own (session, cid, hash).  The
candidate/active/promote state machine, its logs and its promote timeout
therefore behave exactly as in the RTC client.
"""
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import requests

_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _GROOT_ROOT not in sys.path:
    sys.path.insert(0, _GROOT_ROOT)

import psix_rtc_sonic_wm_client as base  # noqa: E402
from psix_rtc_sonic_wm_client import (  # noqa: E402
    ACTION_DIM_DEFAULT,
    ACTION_DIM_NECK,
    DEFAULT_NECK_PUB_HOST,
    DEFAULT_NECK_PUB_PORT,
    DEFAULT_NECK_STATE_ZMQ,
    OBS_SEND_INTERVAL,
    STALL_SHADOW_CAM_MOTION_MAX,
    STALL_SHADOW_FSQ_RATIO_MIN,
    STALL_SHADOW_MIN_S,
    STALL_SHADOW_RAW_P95_MAX,
    WM_GATE_GRAY_SIZE,
    WM_GATE_VERSION,
    EpisodeSubgoalProvider,
    EventLog,
    IncidentRecorder,
    NeckPublisher,
    NeckStateReader,
    NoGoalProvider,
    RSCamera,
    RobotStateSubscriber,
    TokenPublisher,
    WmSubgoalProvider,
    ZedNeckCamera,
    _fetch_json,
    apply_runtime_flags,
    build_arg_parser,
    convert_numpy_in_dict,
    log_event,
    numpy_deserialize,
    numpy_serialize,
    reset_neck_to_home,
    resize_goal_for_vla,
    resolve_task_prompts,
    resolve_vla_embodiment,
    running,
    save_init_frame,
    set_event_log,
    show_goal_window,
    validate_args,
    write_run_manifest,
)
from rollout_recorder import maybe_rollout_recorder  # noqa: E402

CTRL_DT = OBS_SEND_INTERVAL  # 30 Hz playback / capture cadence
DEFAULT_HTTP_TIMEOUT = 30.0


class HttpChunkClient(base.RTCWebSocketClient):
    """Chunk-and-hold playback over POST /act.

    Threads:
      * observation capture (30 Hz): robot state + camera + WM snapshot ->
        one immutable observation record; feeds the WM ego cache, the flight
        recorder and the freshness watchdog exactly like the RTC send thread.
      * inference: when the playback loop asks for a chunk, POSTs the latest
        observation record and parks the returned chunk.
      * playback (the thread that calls ``run()``): publishes one action per
        tick, freezes between chunks.
    """

    def __init__(self, server_url, state_subscriber, camera, token_publisher, wm_provider,
                 task_instruction, dry_run=False, observation_stale_timeout=0.5,
                 action_stale_timeout=0.5, condition_promote_timeout=6.0,
                 wm_stale_warn=5.0, include_neck=False, neck_publisher=None,
                 neck_state_reader=None, incident_recorder=None,
                 http_timeout=DEFAULT_HTTP_TIMEOUT):
        super().__init__(
            server_url, state_subscriber, camera, token_publisher, wm_provider,
            task_instruction, dry_run=dry_run,
            observation_stale_timeout=observation_stale_timeout,
            action_stale_timeout=action_stale_timeout,
            condition_promote_timeout=condition_promote_timeout,
            wm_stale_warn=wm_stale_warn, include_neck=include_neck,
            neck_publisher=neck_publisher, neck_state_reader=neck_state_reader,
            incident_recorder=incident_recorder,
        )
        self._http_timeout = float(http_timeout)
        if self._http_timeout <= 0:
            raise ValueError("http timeout must be positive")
        # A chunk that has not landed this long after it was requested is a
        # liveness failure (hung VLA); the hold-in-place stream must not mask it.
        self._chunk_overdue_timeout = self._condition_promote_timeout
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
        enc_token = None
        while self._running and running.is_set():
            tick_started = time.monotonic()
            self._check_action_liveness()
            try:
                state, state_received_at = self._state_sub.get_state_with_timestamp()
                if state is None:
                    self._stop_or_hold_wbc("no robot state")
                    self._throttled_problem("[client] gated: waiting for robot state")
                    self._drop_latest_obs()
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
                    self._drop_latest_obs()
                    time.sleep(0.05)
                    continue

                states, _left_hand, _right_hand = self._build_state(state)
                if self._incidents is not None:
                    self._incidents.record_state(time.monotonic(), states)

                # Sole camera socket reader; the WM cache gets an immutable RGB copy.
                frame_bgr = self._camera.get_frame()
                if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8 or
                        frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3):
                    raise ValueError(
                        f"camera frame must be BGR uint8 HxWx3, got "
                        f"{getattr(frame_bgr, 'dtype', None)} "
                        f"{getattr(frame_bgr, 'shape', None)}"
                    )
                frame = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                self._wm.update_latest_ego(frame)
                if (self._incidents is not None
                        and getattr(self._incidents, "_rollout", None) is not None):
                    self._incidents._rollout.record_video_frame(time.monotonic(), frame)

                self._cam_tick += 1
                if self._cam_tick % 10 == 0:
                    thumb = cv2.resize(frame, WM_GATE_GRAY_SIZE)
                    if self._incidents is not None:
                        self._incidents.record_frame(time.monotonic(), thumb)
                    if self._cam_tick % 30 == 0:
                        gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                        if self._cam_prev_gray is not None:
                            motion = float(np.abs(gray - self._cam_prev_gray).mean())
                            with self._telemetry_lock:
                                self._cam_motion_1s = motion
                        self._cam_prev_gray = gray

                wm = self._wm.snapshot()
                subgoal_frame = wm["goal"]
                _no_goal_mode = wm.get("goal_source") == "no_goal"
                if ((subgoal_frame is None and not _no_goal_mode)
                        or wm["goal_stale"] or wm["goal_expired"]):
                    if wm["goal_expired"]:
                        self._hold_for_observed_wm_expiry(wm)
                    else:
                        self._stop_or_hold_wbc("waiting for WM goal for the current prompt")
                    self._drop_latest_obs()
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
                    time.sleep(max(0.0, CTRL_DT - (time.monotonic() - tick_started)))
                    continue
                if subgoal_frame is not None and (
                        subgoal_frame.dtype != np.uint8 or subgoal_frame.ndim != 3 or
                        subgoal_frame.shape[2] != 3):
                    raise ValueError(
                        f"WM goal must be RGB uint8 HxWx3, got "
                        f"{subgoal_frame.dtype} {subgoal_frame.shape}"
                    )
                if subgoal_frame is not None:
                    subgoal_frame = resize_goal_for_vla(subgoal_frame)

                if wm["goal_generation"] != self._dbg_last_generation:
                    self._dbg_last_generation = wm["goal_generation"]
                    print(
                        f"[client] condition prompt_stage={wm['prompt_stage']} "
                        f"goal_stage={wm['goal_stage']} stale={wm['goal_stale']} "
                        f"generation={wm['goal_generation']} subtask={wm['subtask']!r}",
                        flush=True,
                    )

                task = str(self._task).strip().lower()
                subtask = str(wm["subtask"]).strip().lower()
                instruction = (
                    f"Task: {task}. Subtask: {subtask}"
                    if subtask and base.USE_SUBTASK_PROMPT else f"Task: {task}"
                )
                send_condition = self._condition_for_send(wm, instruction, subgoal_frame)
                if send_condition is None:
                    self._stop_or_hold_wbc("no VLA condition session")
                    self._drop_latest_obs()
                    continue
                subgoal_frame = send_condition["goal"]
                instruction = send_condition["instruction"]

                if base.SHOW_GOAL_WINDOW:
                    show_goal_window(
                        subgoal_frame,
                        f"stage {wm['prompt_stage']} gen {wm['goal_generation']} "
                        f"{wm.get('goal_source', '')}".strip(),
                    )

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
                record = {
                    "payload": payload,
                    "condition": send_condition,
                    "wm": wm,
                    "captured_at": now,
                }
                with self._obs_lock:
                    self._latest_obs = record
                with self._last_observation_lock:
                    self._last_observation_at = now
                self._send_count += 1

            except Exception as exc:
                self._stop_or_hold_wbc(f"observation loop error: {exc}")
                self._throttled_problem(f"[client] observation rejected: {exc}")
                self._drop_latest_obs()

            time.sleep(max(0.0, CTRL_DT - (time.monotonic() - tick_started)))

            now = time.monotonic()
            elapsed = now - self._send_rate_started_at
            if elapsed >= 1.0:
                print(
                    f"[client] observation capture avg_hz={self._send_count / elapsed:.1f} "
                    f"prompt_stage={self._wm.status()['prompt_stage']}", flush=True
                )
                self._send_count = 0
                self._send_rate_started_at = now
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
        while self._running and running.is_set():
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
                if (int(latest_wm["prompt_epoch"]) != int(record["condition"]["prompt_epoch"])
                        or latest_wm["goal_expired"]):
                    time.sleep(CTRL_DT)
                    continue

            t0 = time.monotonic()
            try:
                chunk = self._post_act(record["payload"])
            except Exception as exc:
                self._stop_or_hold_wbc(f"VLA /act failed: {exc}")
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
                "condition": record["condition"],
                "prompt_epoch": int(record["condition"]["prompt_epoch"]),
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
            cond = record["condition"]
            print(
                f"[http] chunk seq={pending['seq']} rows={chunk.shape[0]} "
                f"cid={cond['cid']} epoch={cond['prompt_epoch']} "
                f"goal_gen={cond['goal_generation']} infer_ms={infer_ms:.1f} "
                f"obs_age_ms={obs_age_ms:.1f}", flush=True
            )
            log_event(
                "http_chunk", seq=pending["seq"], rows=int(chunk.shape[0]),
                cid=cond["cid"], epoch=cond["prompt_epoch"],
                generation=cond["goal_generation"], infer_ms=round(infer_ms, 2),
                obs_age_ms=round(obs_age_ms, 2),
            )
        print("[client] HTTP inference thread stopped")

    def _request_chunk(self):
        self._request_seq += 1
        self._request_event.set()
        return time.monotonic()

    # ------------------------------------------------------------- playback
    def _synth_ack(self, condition, version):
        """The ack /ws would have carried; on /act the pairing is exact by construction."""
        return {
            "action_vla_session_id": condition["sid"],
            "action_condition_id": condition["cid"],
            "action_condition_hash": condition["hash"],
            "model_condition_hash": "",
            "action_version": int(version),
        }

    def _publish_action(self, action, meta, chunk_tick, repeat_last, now):
        """Per-tick safety + publication, the RTC client's _on_message in order.

        Returns "ok" (published), "held" (skipped this tick, keep the chunk) or
        "dropped" (the chunk's condition is no longer executable).
        """
        interval = now - self.start_time
        self.start_time = now
        self._action_version += 1
        version = self._action_version
        try:
            action = self._validated_action(np.asarray(action, dtype=np.float32)[None, :])
        except ValueError as exc:
            self._stop_or_hold_wbc(f"invalid VLA action: {exc}")
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
                self._stop_or_hold_wbc("no WM goal for the current prompt")
            return "held"
        if not fresh:
            self._stop_or_hold_wbc(
                "stale observation "
                f"obs={obs_age:.3f}s state={state_age:.3f}s "
                f"camera={camera_age:.3f}s neck={neck_age:.3f}s "
                f"limit={self._observation_stale_timeout:.3f}s"
            )
            return "held"

        condition = meta["condition"]
        ack = self._synth_ack(condition, version)
        try:
            decision, version, reject_reason, active_for_prompt = \
                self._accept_action_for_condition(version, ack, wm)
        except ValueError as exc:
            self._stop_or_hold_wbc(f"invalid VLA action stream: {exc}")
            print(f"[client] ERROR: rejected action: {exc}", flush=True)
            return "dropped"
        self._record_received_version(version)
        if decision == "starved":
            self._stop_or_hold_wbc(f"condition rollover starved: {reject_reason}")
            if now - self._last_condition_log_at >= 1.0:
                self._last_condition_log_at = now
                print(f"[condition] starved action version={version}: {reject_reason}",
                      flush=True)
                log_event("condition_starved", version=int(version), reason=str(reject_reason))
            return "dropped"
        if decision == "discarded":
            if not active_for_prompt:
                self._stop_or_hold_wbc(
                    "waiting for an acknowledged action for the current prompt")
            if now - self._last_condition_log_at >= 1.0:
                self._last_condition_log_at = now
                print(f"[condition] discarded action version={version}: {reject_reason}",
                      flush=True)
                log_event("condition_discarded", version=int(version),
                          reason=str(reject_reason),
                          active_for_prompt=bool(active_for_prompt))
            return "dropped"
        if decision == "promoted":
            with self._action_state_lock:
                active = self._active_condition
            if active is None:
                print(f"[condition] promoted action version={version} was superseded "
                      "by a prompt transition before publication", flush=True)
            else:
                print(
                    f"[condition] promoted cid={active['cid']} "
                    f"epoch={active['prompt_epoch']} "
                    f"goal_gen={active['goal_generation']} on version={version} "
                    f"latency={now - active['minted_at']:.3f}s", flush=True
                )
                log_event(
                    "condition_promoted", cid=active["cid"], epoch=active["prompt_epoch"],
                    generation=active["goal_generation"], version=int(version),
                    latency_s=round(now - active["minted_at"], 4),
                )
            self._note_condition_promoted()

        with self._publish_lock:
            latest_wm = self._wm.snapshot()
            if int(latest_wm["prompt_epoch"]) != int(wm["prompt_epoch"]):
                self._stop_or_hold_wbc_locked("prompt changed before action publication")
                return "dropped"
            if self._expire_wm_goal_locked(latest_wm):
                return "dropped"
            if not self._ack_is_still_active(ack, wm["prompt_epoch"]):
                self._stop_or_hold_wbc_locked(
                    "accepted VLA condition was invalidated before publication")
                return "dropped"
            if not self._dry_run:
                self._ensure_wbc_started()
            self.execute_action(action)
            self._record_action_accepted(now)

        cid = condition["cid"]
        if self._incidents is not None:
            self._incidents.record_action(
                now, int(version), cid, meta["seq"], chunk_tick,
                bool(repeat_last), action[0].copy(),
            )
        data = {
            "rtc_chunk_id": int(meta["seq"]),
            "rtc_chunk_tick": int(chunk_tick),
            "rtc_repeat_last": bool(repeat_last),
            "rtc_infer_ms": float(meta["infer_ms"]),
            "action_condition_id": cid,
        }
        action_lines, shadow_fired = self._update_action_telemetry(
            data, action, version, interval, now, wm["prompt_epoch"], active_cid=cid,
        )
        for line in action_lines:
            print(line, flush=True)
        if shadow_fired and self._incidents is not None:
            self._incidents.dump("stall_shadow", min_interval_s=30.0)
        return "ok"

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

        while self._running and running.is_set():
            now = time.monotonic()

            with self._chunk_lock:
                pending = self._pending_chunk
                self._pending_chunk = None
            if pending is not None:
                current_epoch = int(self._wm.snapshot()["prompt_epoch"])
                if pending["prompt_epoch"] != current_epoch:
                    print(
                        f"[http] chunk seq={pending['seq']} discarded: epoch "
                        f"{pending['prompt_epoch']} != current {current_epoch}; "
                        "re-requesting", flush=True
                    )
                    log_event("http_chunk_discarded", seq=pending["seq"],
                              epoch=pending["prompt_epoch"], current_epoch=current_epoch)
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
                        f"cid={meta['condition']['cid']} "
                        f"freeze_ms={wait_ms:.0f}", flush=True
                    )
                    log_event("http_chunk_start", seq=meta["seq"],
                              rows=int(chunk.shape[0]), cid=meta["condition"]["cid"],
                              freeze_ms=round(wait_ms, 1))

            if chunk is not None and idx < len(chunk):
                action = chunk[idx]
                tick = idx
                idx += 1
                repeat_last = False
            else:
                # Exhausted (or nothing yet). First exhausted tick: request the
                # next chunk from the observation captured now, and compute the
                # hold-in-place action once (psix_sonic_client semantics).
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
                    self._stop_or_hold_wbc(
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
        self._stop_or_hold_wbc("client shutdown")
        self._connected.set()
        self._request_event.set()
        for th in (self._obs_thread, self._infer_thread):
            if th is not None and th is not threading.current_thread():
                th.join(timeout=2.0)
        try:
            self._session.close()
        except Exception:
            pass


# ---------------- Main ----------------
def main(server_url, zmq_host, zmq_pub_port, zmq_sub_port, zmq_topic, zmq_sub_topic,
         camera_address, camera_timeout_ms, task_instruction, subtasks,
         goal_source, episode_dir,
         wm_base_url, wm_period, wm_timeout, jpeg_quality, wm_stale_warn,
         wm_goal_hard_age, wm_dump_dir, wm_mode, wm_seconds,
         observation_stale_timeout, action_stale_timeout,
         condition_promote_timeout=None,
         dry_run=False,
         include_neck=False, neck_pub_host=DEFAULT_NECK_PUB_HOST,
         neck_pub_port=DEFAULT_NECK_PUB_PORT,
         neck_state_zmq=DEFAULT_NECK_STATE_ZMQ,
         task_key=None, method_name=None,
         http_timeout=DEFAULT_HTTP_TIMEOUT, allow_rtc_server=False,
         rollout_record=True, rollout_video_fps=10.0, rollout_video_scale=1.0):
    running.set()
    print("[MAIN] Initializing components...")

    info_url = server_url.rsplit("/", 1)[0] + "/info"
    vla_info = _fetch_json(info_url)
    served_rtc_mode = vla_info.get("rtc_mode") if isinstance(vla_info, dict) else None
    if served_rtc_mode not in (None, "off"):
        message = (
            f"[MAIN] VLA serves rtc_mode={served_rtc_mode!r}: /act would run the RTC "
            "sampler conditioned on the server's previous chunk, not the plain "
            "open-loop chunk this client is for. Serve with RTC_MODE=off "
            "(scripts/deploy/serve_psix_rtc.sh) or pass --allow-rtc-server."
        )
        if not allow_rtc_server:
            raise SystemExit(message)
        print(message + " Continuing (--allow-rtc-server).", flush=True)
    try:
        chunk_rows = int(vla_info["action"]["action_exec_horizon"])
    except (KeyError, TypeError, ValueError):
        chunk_rows = None
    chunk_seconds = (chunk_rows * CTRL_DT) if chunk_rows else 0.0
    print(
        f"[MAIN] HTTP chunk mode: POST {server_url}; server rtc_mode={served_rtc_mode!r}; "
        f"chunk rows={chunk_rows} ({chunk_seconds:.2f}s at 30 Hz); "
        f"http timeout={http_timeout:.1f}s"
    )

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

    if goal_source == "none":
        wm_provider = NoGoalProvider(task=task_instruction)
        print("[MAIN] no-goal provider: task instruction only, "
              "no goal image / no subtask; WM disabled")
    elif goal_source == "episode":
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
            mode=wm_mode,
            seconds=wm_seconds,
        )
        print(f"[MAIN] WM provider: {wm_base_url}; {len(subtasks)} prompt stages")
        print(
            f"[MAIN] WM last-good hard age: "
            + (f"{wm_goal_hard_age:.1f}s" if wm_goal_hard_age > 0 else "disabled")
        )
        print(
            f"[MAIN] WM mode: {wm_mode}"
            + (f"; horizon {wm_seconds}s" if wm_mode == "future" and wm_seconds else "")
        )
    print(f"[MAIN] Task instruction: {task_instruction!r}")
    if condition_promote_timeout is None:
        # A candidate minted right after a request went out is only used by the
        # NEXT request, i.e. after the current chunk plays out plus one round
        # trip, so the RTC default gains one chunk duration here.
        condition_promote_timeout = (
            2.0 if goal_source == "episode" else max(2.0 * wm_period, 2.0)
        ) + chunk_seconds
    print(
        f"[MAIN] Condition promote timeout: {condition_promote_timeout:.1f}s "
        f"(unacknowledged candidates older than this hold WBC; also the chunk-overdue limit)"
    )

    run_dir = os.path.abspath(os.path.expanduser(wm_dump_dir))
    os.makedirs(run_dir, exist_ok=True)
    event_log = EventLog(os.path.join(run_dir, "events.jsonl"))
    set_event_log(event_log)

    if base.NECK_RESET and include_neck:
        if neck_publisher is None:
            print("[neck-reset] skipped: dry-run does not bind the neck publisher", flush=True)
        else:
            ok, before, after, dt = reset_neck_to_home(
                neck_publisher, neck_state_reader,
                yaw=base.NECK_RESET_YAW, pitch=base.NECK_RESET_PITCH,
                hold_s=base.NECK_RESET_HOLD, tol=base.NECK_RESET_TOL)
            print(f"[neck-reset] {'converged' if ok else 'NOT converged'} in {dt:.2f}s | "
                  f"start={before} end={after} "
                  f"target=({base.NECK_RESET_YAW:+.3f},{base.NECK_RESET_PITCH:+.3f}) rad "
                  f"tol={base.NECK_RESET_TOL}", flush=True)
            log_event("neck_reset", converged=bool(ok), start=before, end=after,
                      elapsed_s=round(dt, 3), target=[base.NECK_RESET_YAW, base.NECK_RESET_PITCH])
            if not ok:
                msg = ("[neck-reset] neck did not reach home. A live-but-frozen state "
                       "stream looks exactly like this: check that realsense_server.py "
                       "started with --enable-neck-motor and that its serial port opened "
                       "(tail ~/realsense_server.log on the robot).")
                if base.NECK_RESET_ON_FAIL == "abort":
                    raise SystemExit(msg + " Aborting (--neck-reset-on-fail=abort).")
                print(msg, flush=True)
            if base.NECK_RESET_SETTLE > 0:
                print(f"[neck-reset] settling {base.NECK_RESET_SETTLE:.1f}s before start",
                      flush=True)
                time.sleep(base.NECK_RESET_SETTLE)

    rollout_recorder = maybe_rollout_recorder(
        os.path.join(run_dir, "rollout"), enabled=rollout_record,
        video_fps=rollout_video_fps, video_scale=rollout_video_scale)
    incident_recorder = IncidentRecorder(run_dir, rollout=rollout_recorder)
    wm_state = (
        wm_provider.provenance()
        if goal_source == "episode"
        else _fetch_json(f"{wm_base_url}/state")
    )
    manifest_path = write_run_manifest(
        run_dir,
        config={
            "server_url": server_url,
            "transport": "http",
            "http_timeout": http_timeout,
            "chunk_rows": chunk_rows,
            "goal_source": goal_source,
            "episode_dir": episode_dir if goal_source == "episode" else None,
            "wm_base_url": wm_base_url if goal_source == "wm" else None,
            "task_key": task_key,
            "method_name": method_name,
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
    log_event("run_start", manifest=manifest_path, dry_run=bool(dry_run), transport="http")

    save_init_frame(
        run_dir, camera, task_instruction, task_key, method_name,
        camera_address, include_neck, wm_provider._episode_session_id,
    )

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

    client = HttpChunkClient(
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
        http_timeout=http_timeout,
    )

    def stdin_listener():
        if goal_source == "none":
            print("[MAIN] no-goal mode: single stage | :restart: new epoch | "
                  ":mark LABEL: flight-recorder dump")
        elif goal_source == "episode":
            print(
                "[MAIN] Enter: next GT goal | :restart: stage 0 | "
                ":mark LABEL: flight-recorder dump"
            )
        else:
            print(
                "[MAIN] Enter: next episode prompt | text/:ov TEXT: manual prompt | "
                ":resume: current episode prompt | :restart: stage 0 | "
                ":sec X: future horizon seconds (':sec server' = server default) | "
                ":mark LABEL: flight-recorder dump "
                "(stall_start/stall_end/empty_grasp/scene_lost)"
            )
        while running.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break
            command = line.strip()
            if command == "":
                client.apply_prompt_transition("next", wm_provider.advance_prompt)
            elif command == ":restart":
                client.apply_prompt_transition("restart", wm_provider.restart)
            elif command == ":resume":
                client.apply_prompt_transition("resume", wm_provider.resume_scripted_prompt)
            elif command == ":sec" or command.startswith(":sec "):
                arg = command[len(":sec"):].strip()
                if not arg:
                    print(f"[MAIN] horizon = {wm_provider.get_seconds()}")
                    continue
                try:
                    wm_provider.set_seconds(None if arg == "server" else float(arg))
                except (ValueError, AttributeError) as error:
                    print(f"[MAIN] {error}")
            elif command.startswith(":ov "):
                client.apply_prompt_transition("manual", wm_provider.takeover, command[4:])
            elif command.startswith(":mark"):
                client.mark_incident(command[5:].strip() or "manual")
            elif command.startswith(":"):
                print(f"[MAIN] unknown command {command!r}")
            else:
                client.apply_prompt_transition("manual", wm_provider.takeover, command)

    t_stdin = threading.Thread(target=stdin_listener, daemon=True)
    t_stdin.start()

    def client_thread():
        client.run()
        print("[HTTP] client thread stopped")

    t_client = threading.Thread(target=client_thread, daemon=True)
    t_client.start()

    print("[MAIN] Running. Ctrl+C to stop.")

    def signal_handler(sig, frame):
        print("\n[MAIN] Caught signal, shutting down...")
        running.clear()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running.is_set() and t_client.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[MAIN] Caught Ctrl+C, exiting...")
        running.clear()

    print("[MAIN] Shutting down...")
    running.clear()
    client.stop()
    t_client.join(timeout=3.0)
    if t_client.is_alive():
        print("[MAIN] WARNING: client thread is still shutting down", flush=True)
    wm_provider.stop()
    state_sub.stop()
    camera.stop()
    if token_publisher is not None:
        token_publisher.stop()
    if neck_publisher is not None:
        neck_publisher.stop()
    if neck_state_reader is not None:
        neck_state_reader.stop()
    if rollout_recorder is not None:
        rollout_recorder.close()
        print(f"[rollout-recorder] closed (dropped={rollout_recorder.dropped})", flush=True)
    log_event("shutdown")
    event_log.stop()
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    parser = build_arg_parser(
        description="HTTP chunk VLA client (POST /act, chunk playback + hold) "
                    "with remote-WM or fixed episode GT goals")
    parser.add_argument("--http-timeout", type=float, default=DEFAULT_HTTP_TIMEOUT,
                        help="Timeout for one POST /act round trip in seconds "
                             f"(default: {DEFAULT_HTTP_TIMEOUT:.0f})")
    parser.add_argument("--allow-rtc-server", action="store_true",
                        help="Do not refuse a VLA whose /info reports rtc_mode != off "
                             "(its /act then runs the RTC sampler on the server's "
                             "previous chunk instead of a plain open-loop chunk).")
    args = parser.parse_args()

    apply_runtime_flags(args)
    validate_args(parser, args)
    if args.http_timeout <= 0:
        parser.error("--http-timeout must be positive")

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

    task_instruction, subtasks = resolve_task_prompts(parser, args)

    server_url = f"http://{args.host}:{args.port}/act"
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
        wm_mode=args.wm_mode,
        wm_seconds=args.wm_seconds,
        observation_stale_timeout=args.observation_stale_timeout,
        action_stale_timeout=args.action_stale_timeout,
        condition_promote_timeout=args.condition_promote_timeout,
        dry_run=args.dry_run,
        include_neck=include_neck,
        neck_pub_host=args.neck_pub_host,
        neck_pub_port=args.neck_pub_port,
        neck_state_zmq=args.neck_state_zmq,
        task_key=args.task_key,
        method_name=args.method_name,
        http_timeout=args.http_timeout,
        allow_rtc_server=args.allow_rtc_server,
        rollout_record=not args.no_rollout_record,
        rollout_video_fps=args.rollout_video_fps,
        rollout_video_scale=args.rollout_video_scale,
    )
