"""HLP x WM x RTC-VLA robot orchestrator (plan_next.md v2, Phase A).

This is the new deployment client that replaces the manual-stage WM client for
the HLP-driven stack. Design contract (plan §0/§3.4/§5/§6/§7):

  * Five kinds of state are kept apart: HLP authoritative state, the DESIRED
    condition (canonical stage the robot should pursue), the WM PENDING request,
    the CANDIDATE condition (instruction+goal atomically paired, keyed, sent to
    the VLA, awaiting its ack), and the ACTIVE condition (the only condition
    whose actions may be executed).
  * Hard gate is the default: any stage change enters a recoverable transient
    hold IMMEDIATELY and releases only when an action arrives whose provenance
    ack equals the candidate's condition_key — promote candidate->active
    atomically, execute that same action, exit hold.
  * The HoldController is the ONLY gate primitive: none of the legacy client's
    12 stop sources exist here. stop=1 is terminal on the WBC wire, so it is
    reserved for :quit / emergency / HOLD_PATH_FAILURE.
  * Fail closed everywhere: raw HLP text that doesn't canonicalize does NOT
    reach the WM or the VLA; malformed replies/acks never mutate state; every
    watchdog lands in hold, and a hard deadline latches (ABORT_LATCHED) — the
    operator's :ack only returns to HOLD, release needs a fresh handshake.

Modes (--hlp-mode):
  off      no HLP requests; stages driven manually (Enter) through the scene
           trajectory. This is Phase D level-1 (manual stage + WM).
  shadow   DEFAULT. HLP is polled with a lease and full logging, but its
           committed switches only produce would_* log events; stages remain
           manual. This is the G9 shadow stage.
  active   HLP committed events drive the desired condition (hard gate).
           Requires an explicit flag; G1 must pass before robot use.

Wire peers (all contracts from Phase 0):
  HLP  http://…:8015  /reset/acquire /hlp /prev /reset /override /resume
  WM   http://…:8016  /wm /ready     (BAGEL local or Cosmos tunnel — same core)
  VLA  ws://…:8014/ws                (condition provenance + action acks)

Run headless smoke (no robot, no GPU) against the psi mock stack:
  python3 mock_g1_obs.py &                                  # fake camera/state
  (psi) python src/psi/deploy/mock_psix_hlp_serve.py --subtasks "a;b" &
  (psi) python src/psi/deploy/mock_psix_wm_serve.py --delay 0.3 &
  (psi) python src/psi/deploy/mock_psix_serve.py --action-dim 78 --tick-hz 30 &
  python3 psix_rtc_sonic_hlpwm_client.py --dry-run --hlp-mode active \
      --profile profiles/cleanup_table_fine_v1.json --scene "…;…"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _GROOT_ROOT not in sys.path:
    sys.path.insert(0, _GROOT_ROOT)

from psix_wire_contracts import (
    GateState,
    ack_matches,
    build_condition,
    canonical_json,
    condition_hash,
    new_vla_session_id,
    validate_hlp_reply,
)
from semantic_profile import SemanticProfile, TrajectoryTracker
from hold_controller import (
    ACTION_DIM,
    HAND_DIM,
    TOKEN_DIM,
    ActionParts,
    HoldController,
    HoldState,
    PublicationAdapter,
    TokenPublisherAdapter,
    make_g1_hold_token_encoder,
)

# Robot I/O + wire codecs come verbatim from the proven WM client — importing
# (not copying) keeps one implementation of the camera/state/publisher stack.
from psix_rtc_sonic_wm_client import (  # noqa: E402
    DEFAULT_CAMERA_TIMEOUT_MS,
    ENCODER_MODEL,
    OBS_SEND_INTERVAL,
    RSCamera,
    RobotStateSubscriber,
    TokenPublisher,
    convert_numpy_in_dict,
    fsq_quantize,
    numpy_deserialize,
    numpy_serialize,
    _mujoco29_to_isaaclab29,
)

CTRL_DT = OBS_SEND_INTERVAL  # 30 Hz control/obs cadence


# =========================================================================
# Small shared pieces
# =========================================================================

class EventLog:
    """Thread-safe JSONL event log (structured, NO images — plan P1-9).
    Every state transition that matters for G6b/G9 replay lands here."""

    def __init__(self, path: Optional[str]):
        self._lock = threading.Lock()
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "a", buffering=1)

    def emit(self, ev: str, **fields):
        rec = {"t": time.time(), "mono": time.monotonic(), "ev": ev}
        rec.update(fields)
        line = json.dumps(rec, default=str)
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
        print(f"[{ev}] " + " ".join(f"{k}={v}" for k, v in fields.items()
                                    if k not in ("mono",)), flush=True)

    def close(self):
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class EgoCache:
    """Latest camera frame, stamped with a monotonic frame id. The capture loop
    is the ONLY camera-socket reader; HLP poller / WM worker / VLA sender all
    take immutable snapshots here (same frame => same capture_id, which is what
    lets the HLP server dedup confirmation votes on a frozen frame)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._lock = threading.Lock()
        self._clock = clock
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._at: Optional[float] = None

    def update(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"ego must be RGB uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
        with self._lock:
            self._frame = rgb
            self._frame_id += 1
            self._at = self._clock()

    def get(self):
        """-> (frame|None, frame_id, age_s)"""
        with self._lock:
            if self._frame is None:
                return None, 0, float("inf")
            return self._frame, self._frame_id, self._clock() - self._at

    def age(self) -> float:
        with self._lock:
            return float("inf") if self._at is None else self._clock() - self._at


class NullAdapter(PublicationAdapter):
    """--dry-run publication sink: full state machine, zero robot side effects."""

    def __init__(self):
        self.published = 0
        self.starts = 0

    def publish(self, parts: ActionParts) -> None:
        self.published += 1

    def send_planner_start(self) -> None:
        self.starts += 1


@dataclass
class WatchdogConfig:
    obs_stale_s: float = 0.5          # camera/state/ego freshness for execution
    action_stale_s: float = 0.5       # RUN: no fresh executed action -> hold
    hlp_stale_gate_s: float = 8.0     # active mode: no HLP success for this long -> hold
    max_pending_s: float = 30.0       # WM request lifetime bound (P1-5)
    ack_timeout_s: float = 10.0       # candidate sent but no matching ack (G6b tunes)
    max_gate_s: float = 120.0         # continuous hold -> ABORT_LATCHED


# =========================================================================
# Condition orchestrator — the four-condition state machine (plan §5)
# =========================================================================

@dataclass
class Desired:
    prompt_gen: int
    label: Optional[str]              # canonical subtask; None <=> done
    instruction: Optional[str]
    done: bool
    source: str
    set_at: float


@dataclass
class PendingGen:
    prompt_gen: int
    req_id: int
    started_at: float


@dataclass
class AcceptedGoal:
    prompt_gen: int
    goal_gen: int
    image: np.ndarray                 # decoded RGB uint8 (the condition content)
    meta: Dict[str, Any]
    at: float


@dataclass
class Cond:
    """A keyed condition snapshot (candidate or active)."""
    sid: str
    cid: int
    supplied_hash: str
    prompt_gen: int
    goal_gen: int
    label: str
    instruction: str
    image: np.ndarray
    minted_at: float
    first_sent_at: Optional[float] = None
    ack_escalated: bool = False

    @property
    def key(self):
        return (self.sid, self.cid)


class ConditionOrchestrator:
    """Owns desired/wm_pending/candidate/active + the gate. Every mutation and
    every action-release decision happens under one lock, so 'promote then
    execute the same action' is atomic with respect to the hold tick."""

    def __init__(self, holdctl: HoldController, adapter: PublicationAdapter, *,
                 watchdogs: WatchdogConfig, log: EventLog,
                 state_age_fn: Callable[[], float],
                 ego_age_fn: Callable[[], float],
                 hlp_blocker_fn: Callable[[], Optional[str]] = lambda: None,
                 allow_periodic_refresh: bool = False,
                 refresh_period_s: float = 0.0,
                 clock: Callable[[], float] = time.monotonic):
        self._lock = threading.RLock()
        self._holdctl = holdctl
        self._adapter = adapter
        self._wd = watchdogs
        self._log = log
        self._clock = clock
        self._state_age_fn = state_age_fn
        self._ego_age_fn = ego_age_fn
        self._hlp_blocker_fn = hlp_blocker_fn
        self._allow_periodic = bool(allow_periodic_refresh)
        self._refresh_period_s = float(refresh_period_s)

        self.gate = GateState.HOLD
        self.hold_reason = "startup: waiting for first condition"
        self._hold_since: Optional[float] = self._clock()
        self.latched_cause: Optional[str] = None
        self.latched_at: Optional[float] = None

        self.desired: Optional[Desired] = None
        self.prompt_gen = 0
        self.pending: Optional[PendingGen] = None
        self.goal: Optional[AcceptedGoal] = None
        self.goal_gen = 0
        self.candidate: Optional[Cond] = None
        self.active: Optional[Cond] = None
        self._vla_sid: Optional[str] = None
        self._next_cid = 0

        self._wbc_started = False
        self._last_executed_at: Optional[float] = None
        self.executed_actions = 0
        self.discarded_actions = 0
        self._wm_wake = threading.Event()
        self._last_mismatch_log = -float("inf")

    # ----------------------------------------------------------- gate core

    def _enter_hold(self, reason: str) -> None:
        """Single funnel for every recoverable interlock (the legacy client's
        12 stop sources all map here; none may reach the WBC stop byte)."""
        if self.gate in (GateState.ABORT_LATCHED, GateState.HOLD_PATH_FAILURE):
            return
        st = self._holdctl.enter_hold(reason)
        if st == HoldState.HOLD_PATH_FAILURE:
            self.gate = GateState.HOLD_PATH_FAILURE
            self.hold_reason = self._holdctl.hold_reason
            return
        if self.gate != GateState.HOLD:
            self.gate = GateState.HOLD
            self._hold_since = self._clock()
            self._log.emit("hold_enter", reason=reason)
        self.hold_reason = reason

    def enter_hold(self, reason: str) -> None:
        with self._lock:
            self._enter_hold(reason)

    def _standing_blockers(self) -> List[str]:
        out = []
        if self._state_age_fn() > self._wd.obs_stale_s:
            out.append("robot state stale")
        if self._ego_age_fn() > self._wd.obs_stale_s:
            out.append("camera stale")
        hlp = self._hlp_blocker_fn()
        if hlp:
            out.append(hlp)
        return out

    # ----------------------------------------------------------- desired

    def set_desired(self, label: str, instruction: str, *, source: str) -> None:
        """Committed canonical stage change — HARD GATE: hold immediately, drop
        the old-stage goal/candidate, wake the WM (plan §5.2 steps 2-3)."""
        with self._lock:
            self.prompt_gen += 1
            self.desired = Desired(self.prompt_gen, label, instruction, False,
                                   source, self._clock())
            self.goal = None
            self.candidate = None
            self._enter_hold(f"stage change -> {label!r}")
            self._log.emit("desired_set", prompt_gen=self.prompt_gen, label=label,
                           source=source)
        self._wm_wake.set()

    def set_done(self, *, source: str) -> None:
        with self._lock:
            self.prompt_gen += 1
            self.desired = Desired(self.prompt_gen, None, None, True, source,
                                   self._clock())
            self.goal = None
            self.candidate = None
            self._enter_hold("episode done (recoverable; :prev to reopen)")
            self._log.emit("desired_done", prompt_gen=self.prompt_gen, source=source)

    def clear_desired(self, *, source: str) -> None:
        """Back to 'awaiting establishment' (restart / prev past stage 0)."""
        with self._lock:
            self.prompt_gen += 1
            self.desired = None
            self.goal = None
            self.candidate = None
            self._enter_hold("awaiting stage establishment")
            self._log.emit("desired_cleared", prompt_gen=self.prompt_gen, source=source)

    # ----------------------------------------------------------- WM side

    def wm_wait(self, timeout: float) -> None:
        self._wm_wake.wait(timeout)
        self._wm_wake.clear()

    def wm_want(self) -> Optional[Dict[str, Any]]:
        """What the WM worker should generate now (None = nothing)."""
        with self._lock:
            if self.gate in (GateState.ABORT_LATCHED, GateState.HOLD_PATH_FAILURE):
                return None
            d = self.desired
            if d is None or d.done:
                return None
            if self.pending is not None and \
                    self._clock() - self.pending.started_at <= self._wd.max_pending_s:
                return None
            if self.goal is not None and self.goal.prompt_gen == self.prompt_gen:
                refresh_due = (self._allow_periodic and self._refresh_period_s > 0
                               and self._clock() - self.goal.at >= self._refresh_period_s)
                if not refresh_due:
                    return None
            return {"prompt_gen": d.prompt_gen, "label": d.label,
                    "instruction": d.instruction}

    def wm_begin(self, prompt_gen: int, req_id: int) -> None:
        with self._lock:
            self.pending = PendingGen(prompt_gen, req_id, self._clock())

    def wm_result(self, prompt_gen: int, req_id: int, image: np.ndarray,
                  meta: Dict[str, Any]) -> str:
        """Adopt a finished generation; superseded results never overwrite."""
        with self._lock:
            if self.pending is None or self.pending.req_id != req_id:
                self._log.emit("wm_result_dropped", why="abandoned", req_id=req_id)
                return "stale"
            self.pending = None
            if prompt_gen != self.prompt_gen or self.desired is None \
                    or self.desired.done:
                self._log.emit("wm_result_dropped", why="prompt_gen changed",
                               req_id=req_id, gen=prompt_gen, now=self.prompt_gen)
                return "stale"
            self.goal_gen += 1
            self.goal = AcceptedGoal(prompt_gen, self.goal_gen, image, meta,
                                     self._clock())
            self._log.emit("wm_goal_accepted", prompt_gen=prompt_gen,
                           goal_gen=self.goal_gen, shape=str(image.shape),
                           wm_ms=meta.get("inference_time_ms"))
            if self._vla_sid is not None:
                self._mint_candidate_locked()
            return "accepted"

    def wm_failed(self, prompt_gen: int, req_id: int, why: str) -> None:
        with self._lock:
            if self.pending is not None and self.pending.req_id == req_id:
                self.pending = None
            self._log.emit("wm_failed", req_id=req_id, why=str(why)[:200])

    # ----------------------------------------------------------- VLA side

    def _mint_candidate_locked(self) -> None:
        d, g = self.desired, self.goal
        if d is None or d.done or g is None or g.prompt_gen != self.prompt_gen:
            return
        h = condition_hash(d.instruction, g.image)
        self.candidate = Cond(self._vla_sid, self._next_cid, h, g.prompt_gen,
                              g.goal_gen, d.label, d.instruction, g.image,
                              self._clock())
        self._next_cid += 1
        self._log.emit("candidate_minted", cid=self.candidate.cid,
                       prompt_gen=g.prompt_gen, goal_gen=g.goal_gen,
                       label=d.label, hash=h[:12])

    def on_ws_connected(self, sid: str) -> None:
        with self._lock:
            self._vla_sid = sid
            # a new session voids every key from the old one (plan §6.3)
            self.candidate = None
            self.active = None
            self._log.emit("vla_connected", sid=sid[:12])
            self._mint_candidate_locked()

    def on_ws_closed(self, why: str = "") -> None:
        with self._lock:
            if self._vla_sid is None:
                return
            self._vla_sid = None
            self.candidate = None
            self.active = None
            self._enter_hold(f"vla ws closed {why}".strip())

    def current_send_content(self) -> Optional[Dict[str, Any]]:
        """The (instruction, goal, condition-key) triple every obs must carry —
        always self-consistent because it comes from ONE Cond snapshot."""
        with self._lock:
            if self.gate in (GateState.ABORT_LATCHED, GateState.HOLD_PATH_FAILURE):
                return None
            c = self.candidate or self.active
            if c is None or self._vla_sid is None or c.sid != self._vla_sid:
                return None
            return {"sid": c.sid, "cid": c.cid, "hash": c.supplied_hash,
                    "instruction": c.instruction, "image": c.image,
                    "is_candidate": c is self.candidate}

    def note_condition_sent(self, cid: int) -> None:
        with self._lock:
            if self.candidate is not None and self.candidate.cid == cid \
                    and self.candidate.first_sent_at is None:
                self.candidate.first_sent_at = self._clock()

    # ----------------------------------------------------------- actions

    @staticmethod
    def _split_action(action: np.ndarray) -> ActionParts:
        """VLA layout [hand14 | token64] -> typed parts (wire mapping is the
        adapter's job; fsq quantization happens here like the proven client)."""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (ACTION_DIM,) or not np.isfinite(a).all():
            raise ValueError(f"action is not a finite {ACTION_DIM}-D vector")
        return ActionParts(a[:HAND_DIM], a[HAND_DIM:2 * HAND_DIM],
                           fsq_quantize(a[2 * HAND_DIM:]))

    def _execute_locked(self, parts: ActionParts) -> None:
        if not self._wbc_started:
            self._adapter.send_planner_start()
            self._wbc_started = True
            self._holdctl.note_wbc_started()
            self._log.emit("wbc_started")
        self._adapter.publish(parts)
        self._holdctl.note_executed(parts)
        self._last_executed_at = self._clock()
        self.executed_actions += 1

    def on_action(self, action: Any, ack: Any) -> str:
        """Every VLA action message lands here. Returns the decision string
        (executed / promoted / discarded / held) for logging/tests."""
        with self._lock:
            if self.gate in (GateState.ABORT_LATCHED, GateState.HOLD_PATH_FAILURE):
                self.discarded_actions += 1
                return "discarded:latched"
            try:
                parts = self._split_action(action)
            except ValueError as e:
                self._enter_hold(f"invalid VLA action: {e}")
                return "held:invalid-action"
            if not isinstance(ack, dict) or ack.get("action_vla_session_id") is None:
                # a server without provenance acks is incompatible (plan §6.1)
                self._enter_hold("action without provenance ack (incompatible VLA server)")
                return "held:no-ack"

            cand, act = self.candidate, self.active
            if cand is not None and ack_matches(
                    ack, vla_session_id=cand.sid, condition_id=cand.cid,
                    supplied_condition_hash=cand.supplied_hash):
                blockers = self._standing_blockers()
                if blockers:
                    now = self._clock()
                    if now - self._last_mismatch_log >= 2.0:
                        self._last_mismatch_log = now
                        self._log.emit("promote_blocked", blockers=blockers)
                    self.discarded_actions += 1
                    return "held:blockers"
                # atomic promote THEN execute this same action (plan §5.2-6)
                self.active, self.candidate = cand, None
                self.gate = GateState.RUN
                self.hold_reason = None
                self._hold_since = None
                self._holdctl.resume_run()
                try:
                    self._execute_locked(parts)
                except Exception as e:
                    self._enter_hold(f"publish failed: {e}")
                    return "held:publish-failed"
                self._log.emit("promoted", cid=act.cid if act else None,
                               new_cid=self.active.cid, label=self.active.label,
                               prompt_gen=self.active.prompt_gen)
                return "promoted"

            if act is not None and ack_matches(
                    ack, vla_session_id=act.sid, condition_id=act.cid,
                    supplied_condition_hash=act.supplied_hash):
                if self.gate != GateState.RUN:
                    # hard gate: old-condition actions never execute during hold
                    self.discarded_actions += 1
                    return "discarded:old-active-during-hold"
                blockers = self._standing_blockers()
                if blockers:
                    self._enter_hold("; ".join(blockers))
                    return "held:blockers"
                try:
                    self._execute_locked(parts)
                except Exception as e:
                    self._enter_hold(f"publish failed: {e}")
                    return "held:publish-failed"
                return "executed"

            # unknown / superseded / missing key
            if self.gate == GateState.RUN:
                self._enter_hold("condition ack mismatch while RUN (causality violation)")
                return "held:ack-mismatch"
            self.discarded_actions += 1
            return "discarded:mismatch-during-hold"

    # ----------------------------------------------------------- control tick

    def control_tick(self) -> None:
        """Call at the control rate (30 Hz): watchdogs + hold publication."""
        now = self._clock()
        with self._lock:
            if self.gate == GateState.HOLD_PATH_FAILURE:
                return
            if self.gate == GateState.HOLD and self._wd.max_gate_s > 0 \
                    and self._hold_since is not None \
                    and now - self._hold_since > self._wd.max_gate_s:
                self.gate = GateState.ABORT_LATCHED
                self.latched_cause = self.hold_reason
                self.latched_at = now
                self._log.emit("abort_latched", cause=self.hold_reason,
                               held_s=round(now - self._hold_since, 1))
            if self.gate == GateState.RUN:
                blockers = self._standing_blockers()
                if blockers:
                    self._enter_hold("; ".join(blockers))
                elif self._last_executed_at is not None and \
                        now - self._last_executed_at > self._wd.action_stale_s:
                    self._enter_hold(
                        f"VLA action stream stale "
                        f"({now - self._last_executed_at:.2f}s)")
            if self.pending is not None and \
                    now - self.pending.started_at > self._wd.max_pending_s:
                self._log.emit("wm_pending_timeout", req_id=self.pending.req_id)
                self.pending = None
                self._enter_hold("wm pending timeout")
                self._wm_wake.set()
            c = self.candidate
            if c is not None and not c.ack_escalated and c.first_sent_at is not None \
                    and now - c.first_sent_at > self._wd.ack_timeout_s:
                c.ack_escalated = True
                self._log.emit("ack_timeout", cid=c.cid,
                               waited_s=round(now - c.first_sent_at, 1))
            # Stale-recovery re-key (plan §6.3): a hold entered from RUN (action
            # liveness, transient blocker, ack mismatch) consumed no candidate,
            # and on_change mode won't regenerate a goal that already exists —
            # so the release path would starve. Re-mint a FRESH condition_key
            # over the current (still-valid) content; its ack releases the hold.
            if self.gate == GateState.HOLD and self.candidate is None \
                    and self._vla_sid is not None and self.goal is not None \
                    and self.goal.prompt_gen == self.prompt_gen \
                    and self.desired is not None and not self.desired.done:
                self._mint_candidate_locked()
            if self.gate in (GateState.HOLD, GateState.ABORT_LATCHED):
                self._holdctl.tick()   # publishes only in TRANSIENT_HOLD

    # ----------------------------------------------------------- operator

    def manual_ack(self) -> bool:
        """ABORT_LATCHED -> HOLD only; forces a fresh WM/VLA handshake."""
        with self._lock:
            if self.gate != GateState.ABORT_LATCHED:
                return False
            self.gate = GateState.HOLD
            self._hold_since = self._clock()
            self.hold_reason = f"acked latch ({self.latched_cause}); fresh handshake required"
            self.latched_cause = None
            self.prompt_gen += 1        # invalidates goal/candidate lineage
            if self.desired is not None:
                self.desired.prompt_gen = self.prompt_gen
            self.goal = None
            self.candidate = None
            self._log.emit("latch_acked")
        self._wm_wake.set()
        return True

    def hold_path_failed(self, reason: str) -> None:
        with self._lock:
            self.gate = GateState.HOLD_PATH_FAILURE
            self.hold_reason = reason

    def status(self) -> Dict[str, Any]:
        with self._lock:
            d, g, c, a = self.desired, self.goal, self.candidate, self.active
            return {
                "gate": self.gate.value,
                "hold_reason": self.hold_reason,
                "hold_s": (None if self._hold_since is None
                           else round(self._clock() - self._hold_since, 1)),
                "latched_cause": self.latched_cause,
                "prompt_gen": self.prompt_gen,
                "desired": (None if d is None else
                            ("__done__" if d.done else d.label)),
                "pending": None if self.pending is None else self.pending.req_id,
                "goal_gen": None if g is None else g.goal_gen,
                "candidate": None if c is None else
                             {"cid": c.cid, "label": c.label},
                "active": None if a is None else
                          {"cid": a.cid, "label": a.label},
                "vla_sid": (self._vla_sid or "")[:12] or None,
                "executed": self.executed_actions,
                "discarded": self.discarded_actions,
                "hold": self._holdctl.status(),
            }


# =========================================================================
# Stage director — canonical adoption + grammar/scene validation (plan §4.2)
# =========================================================================

class StageDirector:
    """Turns HLP authoritative state (or manual commands) into canonical
    desired conditions. Fail-closed: canonical miss, grammar violation and
    scene mismatch never set a desired condition in active mode."""

    def __init__(self, profile: SemanticProfile, scene: Optional[List[str]],
                 orch: ConditionOrchestrator, mode: str, log: EventLog,
                 allow_unsafe_override: bool = False):
        assert mode in ("off", "shadow", "active")
        self.profile = profile
        self.scene = list(scene) if scene else None
        self.orch = orch
        self.mode = mode
        self.log = log
        self.allow_unsafe_override = allow_unsafe_override
        self.tracker = TrajectoryTracker(profile)
        self.shadow_tracker = TrajectoryTracker(profile)
        self._lock = threading.RLock()
        self._hlp_current_raw: Optional[str] = None
        self._hlp_done = False
        self._lineage_trusted = True   # False after an unsafe override

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _norm(text: Optional[str]) -> str:
        return " ".join(str(text or "").split()).casefold()

    def _scene_expected(self, tracker: TrajectoryTracker) -> Optional[str]:
        if not self.scene:
            return None
        i = len(tracker.committed)
        return self.scene[i] if i < len(self.scene) else None

    def _checked_commit(self, label: str, *, source: str) -> bool:
        """Grammar + scene validation, then tracker.commit. False = rejected."""
        if self._lineage_trusted and not self.tracker.admissible(label):
            self.log.emit("grammar_violation", label=label,
                          after=self.tracker.current, source=source)
            return False
        exp = self._scene_expected(self.tracker)
        if self._lineage_trusted and exp is not None and label != exp:
            self.log.emit("scene_mismatch", label=label, expected=exp,
                          source=source)
            return False
        if self._lineage_trusted:
            self.tracker.commit(label)
        return True

    def _drive(self, label: str, *, source: str) -> None:
        self.orch.set_desired(label, self.profile.canonical_instruction(label),
                              source=source)

    # ---------------------------------------------------------- HLP sync

    def sync_from_hlp(self, reply: Dict[str, Any], source: str) -> None:
        """source: poll | acquire | prev | reset | override | resume"""
        with self._lock:
            done = bool(reply.get("done"))
            cur = reply.get("next_subtask")
            if source == "poll":
                self._sync_poll(cur, done, reply)
            else:
                self._sync_command(cur, done, source,
                                   instruction=reply.get("instruction"))

    def _sync_poll(self, cur, done, reply) -> None:
        if done and not self._hlp_done:
            self._hlp_done = True
            self.log.emit("hlp_done_committed",
                          revision=reply.get("state_revision"))
            if self.mode == "active":
                self.orch.set_done(source="hlp")
            else:
                self.log.emit("would_done", mode=self.mode)
            return
        if not done:
            self._hlp_done = False
        if cur is None or self._norm(cur) == self._norm(self._hlp_current_raw):
            return
        # the server committed an establish/switch
        prev_raw = self._hlp_current_raw
        self._hlp_current_raw = cur
        m = self.profile.match(cur)
        self.log.emit("hlp_committed", raw=cur, prev=prev_raw,
                      match=m.kind, canonical=m.canonical,
                      stage=reply.get("stage"),
                      revision=reply.get("state_revision"))
        if self.mode != "active":
            # shadow: log the full verdict chain, mutate nothing
            if m.ok:
                admissible = self.shadow_tracker.admissible(m.canonical)
                exp = self._scene_expected(self.shadow_tracker)
                if admissible:
                    self.shadow_tracker.commit(m.canonical)
                self.log.emit("would_switch", canonical=m.canonical,
                              admissible=admissible,
                              scene_expected=exp, mode=self.mode)
            else:
                self.log.emit("would_fail_closed", raw=cur, mode=self.mode)
            return
        if not m.ok:
            # fail closed: no raw pass-through (P1-2)
            self.orch.enter_hold(f"canonical miss: {cur!r} (需 :ov 或 :prev)")
            return
        if not self._checked_commit(m.canonical, source="hlp"):
            self.orch.enter_hold(
                f"rejected commit {m.canonical!r} (grammar/scene); 需人工处理")
            return
        self._drive(m.canonical, source="hlp")

    def _sync_command(self, cur, done, source: str,
                      instruction: Optional[str] = None) -> None:
        """Explicit transitions from command replies — validated as commands,
        not against the natural monotonic lineage (plan §3.4)."""
        self._hlp_done = done
        self._hlp_current_raw = cur
        if source in ("reset", "acquire"):
            self.tracker = TrajectoryTracker(self.profile)
            self.shadow_tracker = TrajectoryTracker(self.profile)
            self._lineage_trusted = True
            if self.mode == "active":
                self.orch.clear_desired(source=source)
            return
        if source == "prev":
            if self.mode == "active":
                self.tracker.retreat()
                if cur is None:
                    self.orch.clear_desired(source="prev")
                    return
                m = self.profile.match(cur)
                if not m.ok:
                    self.orch.enter_hold(f"canonical miss after prev: {cur!r}")
                    return
                # tracker.current should equal m.canonical when lineage is in
                # sync; if not, resync loudly but keep going (prev is manual).
                if self._lineage_trusted and self.tracker.current != m.canonical:
                    self.log.emit("lineage_resync", tracker=self.tracker.current,
                                  server=m.canonical)
                    self._lineage_trusted = False
                self._drive(m.canonical, source="prev")
            return
        if source == "override":
            if self.mode == "active" and cur is not None:
                m = self.profile.match(cur)
                if not m.ok:
                    if not self.allow_unsafe_override:
                        self.orch.enter_hold(f"override text not canonical: {cur!r}")
                        return
                    # legacy real-robot takeover semantics (wHLP client): the
                    # operator's raw text drives WM+VLA verbatim, bypassing the
                    # vocabulary safety layer. Explicitly flag-gated; operator
                    # owns the consequences (plan P1-2 deviation, user-approved).
                    self._lineage_trusted = False
                    self.log.emit("canonical_bypass", raw=cur)
                    instr = instruction or f"Task: {self.profile.task_text.strip().lower()}. Subtask: {cur}"
                    self.orch.set_desired(cur, instr, source="override-raw")
                    return
                if not self.tracker.admissible(m.canonical):
                    self._lineage_trusted = False
                    self.log.emit("override_off_grammar", label=m.canonical)
                else:
                    self.tracker.commit(m.canonical)
                self._drive(m.canonical, source="override")
            return
        if source == "resume":
            if self.mode == "active" and cur is not None:
                m = self.profile.match(cur)
                if m.ok:
                    self._drive(m.canonical, source="resume")
                else:
                    self.orch.enter_hold(f"canonical miss after resume: {cur!r}")
            elif self.mode == "active":
                self.orch.clear_desired(source="resume")
            return

    # ---------------------------------------------------------- manual mode

    def manual_advance(self) -> None:
        """off/shadow: drive the scene trajectory by hand (Phase D level 1)."""
        with self._lock:
            if self.mode == "active":
                print("[director] Enter 无效：active 模式由 HLP 驱动 (:prev/:ov 可用)")
                return
            if not self.scene:
                print("[director] 没有 --scene/--task-key 序列，无法手动推进")
                return
            i = len(self.tracker.committed)
            if i >= len(self.scene):
                self.log.emit("manual_done")
                self.orch.set_done(source="manual")
                return
            label = self.scene[i]
            if not self.tracker.admissible(label):
                self.log.emit("grammar_violation", label=label, source="manual")
                return
            self.tracker.commit(label)
            self._drive(label, source="manual")

    def manual_prev(self) -> None:
        with self._lock:
            if self.mode == "active":
                return  # active-mode prev goes through the HLP endpoint
            self.tracker.retreat()
            cur = self.tracker.current
            if cur is None:
                self.orch.clear_desired(source="manual-prev")
            else:
                self._drive(cur, source="manual-prev")

    def manual_restart(self) -> None:
        with self._lock:
            self.tracker = TrajectoryTracker(self.profile)
            self.shadow_tracker = TrajectoryTracker(self.profile)
            self._lineage_trusted = True
            if self.mode != "active":
                self.orch.clear_desired(source="manual-restart")

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"mode": self.mode,
                    "hlp_current": self._hlp_current_raw,
                    "hlp_done": self._hlp_done,
                    "committed": list(self.tracker.committed),
                    "scene_next": self._scene_expected(self.tracker),
                    "lineage_trusted": self._lineage_trusted}


# =========================================================================
# HLP poller (plan §3.4)
# =========================================================================

class HlpPoller:
    def __init__(self, base_url: str, mode: str, *, task_text: str,
                 director: StageDirector, ego_cache: EgoCache, log: EventLog,
                 profile_hash_hex: Optional[str] = None,
                 scene_manifest_hash: Optional[str] = None,
                 period: float = 1.0, connect_timeout: float = 2.0,
                 read_timeout: float = 5.0, max_capture_age_s: float = 1.0,
                 post_fn: Optional[Callable] = None,
                 clock: Callable[[], float] = time.monotonic):
        assert mode in ("off", "shadow", "active")
        self.mode = mode
        self._url = base_url.rstrip("/")
        self._task = task_text
        self._director = director
        self._ego = ego_cache
        self._log = log
        self._profile_hash = profile_hash_hex
        self._scene_hash = scene_manifest_hash
        self._period = float(period)
        self._timeout = (float(connect_timeout), float(read_timeout))
        self._max_capture_age_s = float(max_capture_age_s)
        self._clock = clock
        if post_fn is None:
            import requests
            s = requests.Session()
            s.trust_env = False    # never route robot traffic through proxies
            def post_fn(path, body, timeout):
                r = s.post(f"{self._url}{path}",
                           json=convert_numpy_in_dict(body, numpy_serialize),
                           timeout=timeout)
                try:
                    return r.status_code, r.json()
                except ValueError:
                    return r.status_code, {"error": f"non-JSON http {r.status_code}"}
        self._post = post_fn

        self._lock = threading.RLock()
        self.session_id: Optional[str] = None
        self.task_fingerprint: Optional[str] = None
        self._control_generation = 0
        self._last_revision = -1
        self._req_seq = 0
        self.last_success_at: Optional[float] = None
        self.last_reply_at: Optional[float] = None
        self.consecutive_failures = 0
        self.polls_ok = 0
        self.polls_failed = 0
        self._paused = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_fail_log = -float("inf")

    # ---------------------------------------------------------- health

    def health_blocker(self) -> Optional[str]:
        """None = healthy. Only gates in active mode (plan §3.4 wall-clock).

        LOCK-FREE on purpose: this is called from control_tick while the
        ORCHESTRATOR lock is held, and the poller thread holds the POLLER lock
        while adopting into the orchestrator — taking the poller lock here
        would close a lock-order cycle (observed deadlock). A bare attribute
        read of the timestamp is atomic in CPython and is all we need."""
        if self.mode != "active":
            return None
        last = self.last_success_at
        if last is None:
            return "hlp: no successful poll yet"
        age = self._clock() - last
        if age > self._stale_gate_s:
            return f"hlp stale ({age:.1f}s)"
        return None

    _stale_gate_s = 8.0   # overwritten from WatchdogConfig by the client glue

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused

    # ---------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self.mode == "off":
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="hlp-poller")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def acquire(self, *, force: bool = False, retries: int = 0) -> bool:
        """POST /reset/acquire: clean episode + lease. Retries on 409."""
        body = {"task": self._task, "profile_hash": self._profile_hash,
                "scene_manifest_hash": self._scene_hash, "force": force,
                "request_id": self._next_req()}
        attempt = 0
        while True:
            try:
                status, reply = self._post("/reset/acquire", body, self._timeout)
            except Exception as e:
                self._log.emit("hlp_acquire_failed", why=str(e)[:200])
                status, reply = -1, None
            if status == 200 and isinstance(reply, dict) \
                    and reply.get("robot_episode_session_id"):
                with self._lock:
                    self.session_id = reply["robot_episode_session_id"]
                    self.task_fingerprint = reply.get("task_fingerprint")
                    self._last_revision = int(reply.get("state_revision", -1))
                self._log.emit("hlp_lease_acquired", session=self.session_id[:12],
                               fingerprint=(self.task_fingerprint or "")[:12])
                self._director.sync_from_hlp(reply, "acquire")
                return True
            if status == 409:
                self._log.emit("hlp_lease_held",
                               expires=reply.get("lease_expires_at"))
            attempt += 1
            if attempt > retries:
                return False
            time.sleep(2.0)

    # ---------------------------------------------------------- polling

    def _next_req(self) -> str:
        with self._lock:
            self._req_seq += 1
            return f"{os.getpid()}-{self._req_seq}"

    def _loop(self) -> None:
        self._log.emit("hlp_poller_started", mode=self.mode, url=self._url,
                       period=self._period)
        while not self._stop.is_set():
            started = self._clock()
            with self._lock:
                paused = self._paused
            if not paused:
                try:
                    self.poll_once()
                except Exception as e:
                    self._note_failure(f"poller error: {e}")
            elapsed = self._clock() - started
            self._stop.wait(max(0.05, self._period - elapsed))

    def poll_once(self) -> Optional[Dict[str, Any]]:
        frame, frame_id, age = self._ego.get()
        if frame is None or age > self._max_capture_age_s:
            self._note_failure(f"no fresh ego frame (age={age:.2f}s)")
            return None
        with self._lock:
            gen = self._control_generation
            body = {
                "task": self._task,
                "is_initial": self._director._hlp_current_raw is None,
                "robot_episode_session_id": self.session_id,
                "capture_id": int(frame_id),
                "capture_age_s": round(float(age), 3),
                "request_id": self._next_req(),
                "ego_image": frame,
            }
        try:
            status, reply = self._post("/hlp", body, self._timeout)
        except Exception as e:
            self._note_failure(f"http: {e}")
            return None
        now = self._clock()
        with self._lock:
            self.last_reply_at = now
        if status == 409:
            self._note_failure("session mismatch (lease lost?)")
            return None
        if status != 200:
            self._note_failure(f"http {status}")
            return None
        kind, why = validate_hlp_reply(reply)
        if kind != "ok":
            self._note_failure(f"reply {kind}: {why}")
            return None
        with self._lock:
            if self.session_id is not None and \
                    reply.get("robot_episode_session_id") != self.session_id:
                self._note_failure("reply session != lease session")
                return None
            rev = int(reply["state_revision"])
            if rev < self._last_revision:
                self._note_failure(
                    f"revision regressed {rev} < {self._last_revision}")
                return None
            self._last_revision = rev
            if gen != self._control_generation:
                self._log.emit("hlp_poll_dropped", why="superseded by control cmd")
                return None
            self.last_success_at = now
            self.consecutive_failures = 0
            self.polls_ok += 1
            # Adopt INSIDE the poller lock: a control command bumps the
            # generation under this same lock BEFORE posting, so a pre-command
            # poll can never re-adopt stale state after the command reply did.
            self._director.sync_from_hlp(reply, "poll")
        return reply

    def _note_failure(self, why: str) -> None:
        with self._lock:
            self.consecutive_failures += 1
            self.polls_failed += 1
        now = self._clock()
        if now - self._last_fail_log >= 2.0:
            self._last_fail_log = now
            self._log.emit("hlp_poll_failed", why=str(why)[:200],
                           consecutive=self.consecutive_failures)

    # ---------------------------------------------------------- controls

    def post_control(self, path: str, body: Optional[Dict[str, Any]] = None,
                     source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Control command: bump control_generation FIRST (in-flight polls get
        dropped at adoption), then post, then sync via the command handler."""
        with self._lock:
            self._control_generation += 1
            payload = dict(body or {})
            payload.setdefault("robot_episode_session_id", self.session_id)
            payload.setdefault("request_id", self._next_req())
        try:
            status, reply = self._post(path, payload, self._timeout)
        except Exception as e:
            self._log.emit("hlp_control_failed", path=path, why=str(e)[:200])
            return None
        if status != 200:
            self._log.emit("hlp_control_failed", path=path, status=status,
                           why=str((reply or {}).get("error"))[:120])
            return None
        kind, why = validate_hlp_reply(reply)
        if kind != "ok":
            self._log.emit("hlp_control_failed", path=path, why=f"{kind}: {why}")
            return None
        with self._lock:
            self._last_revision = max(self._last_revision,
                                      int(reply["state_revision"]))
        self._director.sync_from_hlp(reply, source or path.strip("/"))
        return reply

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"mode": self.mode, "session": (self.session_id or "")[:12] or None,
                    "ok": self.polls_ok, "failed": self.polls_failed,
                    "consecutive_failures": self.consecutive_failures,
                    "last_success_age_s": (
                        None if self.last_success_at is None
                        else round(self._clock() - self.last_success_at, 1)),
                    "paused": self._paused}


# =========================================================================
# WM worker (plan §5.4 + §8.4 client side)
# =========================================================================

class WmWorker:
    def __init__(self, base_url: str, orch: ConditionOrchestrator,
                 ego_cache: EgoCache, log: EventLog, *,
                 task_text: str, episode_session_fn: Callable[[], Optional[str]],
                 connect_timeout: float = 3.0, read_timeout: float = 10.0,
                 jpeg_quality: int = 90, max_capture_age_s: float = 1.0,
                 post_fn: Optional[Callable] = None,
                 ready_fn: Optional[Callable] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._url = base_url.rstrip("/")
        self._orch = orch
        self._ego = ego_cache
        self._log = log
        self._task = task_text
        self._episode_session_fn = episode_session_fn
        self._timeout = (float(connect_timeout), float(read_timeout))
        self._jpeg_quality = int(jpeg_quality)
        self._max_capture_age_s = float(max_capture_age_s)
        self._clock = clock
        if post_fn is None or ready_fn is None:
            import requests
            s = requests.Session()
            s.trust_env = False
            if post_fn is None:
                def post_fn(body, timeout):
                    r = s.post(f"{self._url}/wm", json=body, timeout=timeout)
                    try:
                        return r.status_code, r.json()
                    except ValueError:
                        return r.status_code, {"error": f"non-JSON http {r.status_code}"}
            if ready_fn is None:
                def ready_fn():
                    try:
                        r = s.get(f"{self._url}/ready", timeout=(2.0, 3.0))
                        return bool(r.json().get("ready", r.status_code == 200))
                    except Exception:
                        return False
        self._post = post_fn
        self._ready = ready_fn

        self._req_seq = int(time.time()) % 100000 * 1000
        self._fail_streak = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_wm_ms: Optional[float] = None

    @staticmethod
    def _encode_jpeg(rgb: np.ndarray, quality: int) -> str:
        import cv2
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return b64encode(enc.tobytes()).decode("ascii")

    @staticmethod
    def _decode_jpeg(value: str) -> np.ndarray:
        import cv2
        arr = np.frombuffer(b64decode(value, validate=True), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("subgoal JPEG decode failed")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="wm-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._orch._wm_wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _backoff(self) -> float:
        import random
        return min(5.0, 0.5 * (1.5 ** min(self._fail_streak, 6))) \
            + random.uniform(0.0, 0.3)

    def _loop(self) -> None:
        self._log.emit("wm_worker_started", url=self._url)
        ready_logged = False
        while not self._stop.is_set():
            if not self._ready():
                if not ready_logged:
                    self._log.emit("wm_not_ready", url=self._url)
                    ready_logged = True
                self._stop.wait(2.0)
                continue
            if ready_logged:
                self._log.emit("wm_ready")
                ready_logged = False
            plan = self._orch.wm_want()
            if plan is None:
                self._orch.wm_wait(0.5)
                continue
            self._generate_once(plan)

    def _generate_once(self, plan: Dict[str, Any]) -> None:
        frame, frame_id, age = self._ego.get()
        if frame is None or age > self._max_capture_age_s:
            self._stop.wait(0.2)
            return
        self._req_seq += 1
        req_id = self._req_seq
        body = {
            "jpeg": True,
            "ego_jpeg": self._encode_jpeg(frame, self._jpeg_quality),
            "subtask": plan["label"],
            "task": self._task,
            "req_id": req_id,
            "robot_episode_session_id": self._episode_session_fn(),
            "prompt_gen": plan["prompt_gen"],
        }
        self._orch.wm_begin(plan["prompt_gen"], req_id)
        t0 = self._clock()
        try:
            status, reply = self._post(body, self._timeout)
        except Exception as e:
            self._fail_streak += 1
            self._orch.wm_failed(plan["prompt_gen"], req_id, f"http: {e}")
            self._stop.wait(self._backoff())
            return
        wm_ms = (self._clock() - t0) * 1000.0
        if status == 429:
            self._fail_streak += 1
            self._orch.wm_failed(plan["prompt_gen"], req_id, "busy (429)")
            self._stop.wait(self._backoff())
            return
        if status != 200 or not isinstance(reply, dict) or "error" in reply:
            self._fail_streak += 1
            why = (reply or {}).get("error", f"http {status}")
            self._orch.wm_failed(plan["prompt_gen"], req_id, str(why))
            self._stop.wait(self._backoff())
            return
        try:
            if int(reply.get("req_id", -1)) != req_id:
                raise ValueError("response req_id mismatch")
            if str(reply.get("subtask", "")) != plan["label"]:
                raise ValueError("response subtask mismatch")
            if "subgoal_jpeg" in reply:
                goal = self._decode_jpeg(reply["subgoal_jpeg"])
            elif "subgoal_image" in reply:
                goal = np.asarray(
                    numpy_deserialize(reply["subgoal_image"])
                    if isinstance(reply["subgoal_image"], dict)
                    else reply["subgoal_image"]).astype(np.uint8)
            else:
                raise ValueError("response missing subgoal payload")
            if goal.dtype != np.uint8 or goal.ndim != 3 or goal.shape[2] != 3:
                raise ValueError(f"bad subgoal: {goal.dtype} {goal.shape}")
        except Exception as e:
            self._fail_streak += 1
            self._orch.wm_failed(plan["prompt_gen"], req_id, f"payload: {e}")
            self._stop.wait(self._backoff())
            return
        self._fail_streak = 0
        self.last_wm_ms = wm_ms
        meta = {"gen_id": reply.get("gen_id"), "backend": reply.get("backend"),
                "inference_time_ms": reply.get("inference_time_ms"),
                "wm_total_ms": round(wm_ms, 1)}
        self._orch.wm_result(plan["prompt_gen"], req_id, goal, meta)


# =========================================================================
# VLA link — WebSocket with reconnect + provenance acks (plan §6)
# =========================================================================

_ACK_KEYS = ("action_vla_session_id", "action_condition_id",
             "action_condition_hash", "model_condition_hash", "action_version")


class VlaLink:
    def __init__(self, ws_url: str, orch: ConditionOrchestrator, log: EventLog,
                 *, reconnect_max_backoff: float = 10.0):
        self._url = ws_url
        self._orch = orch
        self._log = log
        self._reconnect_max = float(reconnect_max_backoff)
        self._ws = None
        self._ws_lock = threading.Lock()
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connection_epoch = 0
        self.last_ack_version: Optional[int] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="vla-link")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def connected(self) -> bool:
        return self._connected.is_set()

    def send_text(self, text: str) -> bool:
        with self._ws_lock:
            ws = self._ws
            if ws is None or not self._connected.is_set():
                return False
            try:
                ws.send(text)
                return True
            except Exception as e:
                self._log.emit("vla_send_failed", why=str(e)[:120])
                return False

    def _run_loop(self) -> None:
        from websocket import WebSocketApp
        backoff = 1.0
        while not self._stop.is_set():
            sid = new_vla_session_id()
            epoch_open = threading.Event()

            def on_open(ws):
                self.connection_epoch += 1
                self._orch.on_ws_connected(sid)
                self._connected.set()
                epoch_open.set()

            def on_message(ws, message):
                self._on_message(message)

            def on_close(ws, code, msg):
                self._connected.clear()
                self._orch.on_ws_closed(f"({code} {msg})")

            def on_error(ws, error):
                self._log.emit("vla_ws_error", why=str(error)[:120])

            ws = WebSocketApp(self._url, on_open=on_open, on_message=on_message,
                              on_close=on_close, on_error=on_error)
            with self._ws_lock:
                self._ws = ws
            ws.run_forever()
            self._connected.clear()
            with self._ws_lock:
                self._ws = None
            if self._stop.is_set():
                break
            self._orch.on_ws_closed("(run_forever returned)")
            backoff = 1.0 if epoch_open.is_set() else min(backoff * 2, self._reconnect_max)
            self._log.emit("vla_reconnect_in", seconds=round(backoff, 1))
            self._stop.wait(backoff)

    def _on_message(self, message: str) -> None:
        try:
            data = json.loads(message)
        except ValueError:
            return
        action_data = data.get("action")
        if action_data is None:
            return
        try:
            action = convert_numpy_in_dict(action_data, numpy_deserialize)
        except Exception as e:
            self._log.emit("vla_action_decode_failed", why=str(e)[:120])
            self._orch.enter_hold(f"undecodable VLA action: {e}")
            return
        ack = {k: data[k] for k in _ACK_KEYS if k in data}
        if "action_version" in ack:
            self.last_ack_version = ack["action_version"]
        self._orch.on_action(action, ack if ack else None)


# =========================================================================
# Client glue: capture loop + control loop + stdin + main
# =========================================================================

class HlpwmClient:
    def __init__(self, args):
        self.args = args
        run_stamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = None
        if args.log_dir:
            log_path = os.path.join(args.log_dir, f"hlpwm_{run_stamp}.jsonl")
        self.log = EventLog(log_path)
        self.clock = time.monotonic

        # --- semantic profile + scene trajectory ---------------------------
        self.profile = SemanticProfile.load(args.profile,
                                            expected_hash=args.profile_hash or None)
        scene = _resolve_scene(args, self.profile)
        self.scene = scene
        scene_hash = None
        if scene:
            import hashlib
            scene_hash = hashlib.sha256(
                canonical_json(scene).encode("utf-8")).hexdigest()
        self.task_text = self.profile.task_text

        # --- robot I/O ------------------------------------------------------
        self.ego_cache = EgoCache(self.clock)
        self.camera = RSCamera(address=args.camera_address,
                               timeout_ms=args.camera_timeout_ms)
        self.state_sub = RobotStateSubscriber(host=args.zmq_host,
                                              port=args.zmq_sub_port,
                                              topic=args.zmq_sub_topic)
        if args.dry_run:
            self.adapter: PublicationAdapter = NullAdapter()
            self.token_publisher = None
            print("[MAIN] DRY-RUN: NullAdapter — no WBC binds/commands/publishes")
        else:
            self.token_publisher = TokenPublisher(host="*", port=args.zmq_pub_port,
                                                  topic=args.zmq_topic)
            self.adapter = TokenPublisherAdapter(self.token_publisher)

        try:
            from encoder_client import EncoderClient
            self.encoder = EncoderClient(ENCODER_MODEL, mode=0)
        except Exception as e:
            print(f"[MAIN] encoder load failed ({e}); hold path UNAVAILABLE")
            self.encoder = None

        # --- gate primitives -------------------------------------------------
        def get_state_with_t():
            st, at = self.state_sub.get_state_with_timestamp()
            if st is None:
                return None
            return dict(st, t=at)

        if self.encoder is not None:
            encode_hold = make_g1_hold_token_encoder(
                self.encoder, get_state_with_t,
                token_postprocess=fsq_quantize,
                state_max_age_s=args.observation_stale_timeout,
                clock=self.clock)
        else:
            def encode_hold():
                raise RuntimeError("no encoder — hold path unavailable")

        self.holdctl = HoldController(
            self.adapter, encode_hold_token=encode_hold,
            reencode_period_s=args.hold_reencode_period,
            on_failure=self._on_hold_path_failure, clock=self.clock)

        wd = WatchdogConfig(
            obs_stale_s=args.observation_stale_timeout,
            action_stale_s=args.action_stale_timeout,
            hlp_stale_gate_s=args.hlp_stale_gate,
            max_pending_s=args.max_pending_s,
            ack_timeout_s=args.ack_timeout_s,
            max_gate_s=args.max_gate_s)

        self.orch = ConditionOrchestrator(
            self.holdctl, self.adapter, watchdogs=wd, log=self.log,
            state_age_fn=self.state_sub.age, ego_age_fn=self.ego_cache.age,
            hlp_blocker_fn=self._hlp_blocker,
            allow_periodic_refresh=args.allow_periodic_refresh,
            refresh_period_s=args.wm_refresh_period, clock=self.clock)

        self.director = StageDirector(self.profile, scene, self.orch,
                                      args.hlp_mode, self.log,
                                      allow_unsafe_override=args.allow_raw_override)

        # --- service legs -----------------------------------------------------
        self.poller = HlpPoller(
            f"http://{args.hlp_host}:{args.hlp_port}", args.hlp_mode,
            task_text=self.task_text, director=self.director,
            ego_cache=self.ego_cache, log=self.log,
            profile_hash_hex=self.profile.hash, scene_manifest_hash=scene_hash,
            period=args.hlp_period, connect_timeout=args.hlp_connect_timeout,
            read_timeout=args.hlp_read_timeout,
            max_capture_age_s=args.capture_max_age, clock=self.clock)
        self.poller._stale_gate_s = wd.hlp_stale_gate_s

        self._client_session_id = uuid.uuid4().hex

        self.wm = WmWorker(
            f"http://{args.wm_host}:{args.wm_port}", self.orch, self.ego_cache,
            self.log, task_text=self.task_text,
            episode_session_fn=lambda: self.poller.session_id
            or self._client_session_id,
            connect_timeout=args.wm_connect_timeout,
            read_timeout=args.wm_read_timeout,
            jpeg_quality=args.wm_jpeg_quality,
            max_capture_age_s=args.capture_max_age, clock=self.clock)

        self.vla = VlaLink(f"ws://{args.host}:{args.port}/ws", self.orch, self.log)

        self.running = threading.Event()
        self.running.set()
        self._threads: List[threading.Thread] = []
        self._init_prev_sent_epoch = -1
        self._terminal_sent = False

    # ---------------------------------------------------------------- hooks

    def _hlp_blocker(self) -> Optional[str]:
        return self.poller.health_blocker()

    def _on_hold_path_failure(self, reason: str) -> None:
        self.orch.hold_path_failed(reason)
        self._terminal(f"HOLD_PATH_FAILURE: {reason}")

    def _terminal(self, reason: str) -> None:
        """The ONLY place that may emit the terminal WBC stop (plan §7)."""
        if self._terminal_sent:
            return
        self._terminal_sent = True
        self.log.emit("terminal_stop", reason=reason)
        if self.token_publisher is not None and self.orch._wbc_started:
            try:
                self.token_publisher.send_command(start=False, stop=True,
                                                  planner=True)
            except Exception as e:
                print(f"[terminal] stop send failed: {e}")
        self.running.clear()

    # ---------------------------------------------------------------- loops

    def _capture_loop(self) -> None:
        import cv2
        while self.running.is_set():
            t0 = self.clock()
            try:
                frame_bgr = self.camera.get_frame()
                rgb = np.ascontiguousarray(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                self.ego_cache.update(rgb)
            except Exception:
                pass   # staleness watchdogs own the consequences
            time.sleep(max(0.0, CTRL_DT - (self.clock() - t0)))

    def _control_loop(self) -> None:
        while self.running.is_set():
            t0 = self.clock()
            self.orch.control_tick()
            if self.orch.gate == GateState.HOLD_PATH_FAILURE:
                self._terminal(self.orch.hold_reason or "hold path failure")
                return
            self._maybe_send_obs()
            time.sleep(max(0.0, CTRL_DT - (self.clock() - t0)))

    def _maybe_send_obs(self) -> None:
        if not self.vla.connected():
            return
        content = self.orch.current_send_content()
        if content is None:
            return
        state, state_at = self.state_sub.get_state_with_timestamp()
        if state is None or \
                self.clock() - state_at > self.args.observation_stale_timeout:
            return
        frame, _, age = self.ego_cache.get()
        if frame is None or age > self.args.observation_stale_timeout:
            return
        try:
            states, lh, rh = _build_state_43(state)
        except ValueError:
            return
        state_obs: Dict[str, Any] = {"states": states}
        if self._init_prev_sent_epoch != self.vla.connection_epoch \
                and self.encoder is not None:
            try:
                qpos = _mujoco29_to_isaaclab29(state["body_q_measured"])
                quat = np.asarray(state.get("base_quat_measured", [1, 0, 0, 0]),
                                  dtype=np.float32).reshape(4)
                jp = np.tile(qpos, (10, 1)).astype(np.float32)
                jv = np.zeros((10, 29), dtype=np.float32)
                bq = np.tile(quat, (10, 1)).astype(np.float32)
                tok = np.asarray(self.encoder.encode(jp, jv, bq),
                                 dtype=np.float32).reshape(TOKEN_DIM)
                init_prev = np.concatenate([lh, rh, tok]).astype(np.float32)
                if init_prev.shape == (ACTION_DIM,) and np.isfinite(init_prev).all():
                    state_obs["init_prev_action"] = init_prev
            except Exception as e:
                self.log.emit("init_prev_failed", why=str(e)[:120])
        payload = {
            "image": {"video.egocentric": frame,
                      "subgoal.egocentric": content["image"]},
            "state": state_obs,
            "gt_action": None,
            "dataset_name": None,
            "instruction": content["instruction"],
            "history": None,
            "condition": build_condition(content["sid"], content["cid"],
                                         content["hash"]),
            "timestamp": None,
        }
        text = json.dumps(convert_numpy_in_dict(payload, numpy_serialize))
        if self.vla.send_text(text):
            if "init_prev_action" in state_obs:
                self._init_prev_sent_epoch = self.vla.connection_epoch
            self.orch.note_condition_sent(content["cid"])

    # ---------------------------------------------------------------- stdin

    def _stdin_loop(self) -> None:
        help_line = (":show 状态 | Enter 手动推进(off/shadow) | :prev | :restart | "
                     ":ov <text> | :resume | :ack | :quit")
        print(f"[MAIN] {help_line}")
        try:
            for line in sys.stdin:
                if not self.running.is_set():
                    break
                cmd = line.strip()
                try:
                    self._handle_cmd(cmd)
                except Exception as e:
                    print(f"[cmd] error: {e}")
                if cmd == ":quit":
                    break
        except (OSError, ValueError):
            pass   # no interactive stdin (tests / detached run)

    def _handle_cmd(self, cmd: str) -> None:
        if cmd == "":
            self.director.manual_advance()
        elif cmd == ":show":
            print(json.dumps({"orch": self.orch.status(),
                              "director": self.director.status(),
                              "hlp": self.poller.status(),
                              "wm_ms": self.wm.last_wm_ms}, indent=1))
        elif cmd == ":prev":
            self.orch.enter_hold("manual :prev")
            if self.args.hlp_mode == "active":
                self.poller.post_control("/prev", source="prev")
            else:
                self.director.manual_prev()
        elif cmd == ":restart":
            self.orch.enter_hold("manual :restart")
            if self.args.hlp_mode != "off":
                self.poller.post_control("/reset", source="reset")
            self.director.manual_restart()
        elif cmd.startswith(":ov "):
            text = cmd[4:].strip()
            m = self.profile.match(text)
            if not m.ok and not self.args.allow_raw_override:
                print(f"[cmd] 非 canonical 文本，拒绝 override（--allow-raw-override 可放行）。可用标签:\n  "
                      + "\n  ".join(self.profile.labels))
                return
            self.orch.enter_hold("manual override")
            send_text = m.canonical if m.ok else text
            if self.args.hlp_mode == "active":
                self.poller.post_control("/override", {"subtask": send_text},
                                         source="override")
            elif m.ok:
                if self.director.tracker.admissible(m.canonical):
                    self.director.tracker.commit(m.canonical)
                self.director._drive(m.canonical, source="manual-override")
            else:
                self.log.emit("canonical_bypass", raw=text, mode=self.args.hlp_mode)
                self.orch.set_desired(
                    text, f"Task: {self.profile.task_text.strip().lower()}. Subtask: {text}",
                    source="manual-override-raw")
        elif cmd == ":resume":
            if self.args.hlp_mode == "active":
                self.poller.post_control("/resume", source="resume")
        elif cmd == ":ack":
            if self.orch.manual_ack():
                self.poller.set_paused(False)
                print("[cmd] latch acked -> HOLD；等待全新 HLP/WM/VLA 握手后才会 RUN")
            else:
                print("[cmd] 当前不在 ABORT_LATCHED")
        elif cmd == ":quit":
            self._terminal("operator :quit")
        else:
            print(f"[cmd] unknown {cmd!r}")

    # ---------------------------------------------------------------- run

    def run(self) -> None:
        self.log.emit("client_started", mode=self.args.hlp_mode,
                      dry_run=self.args.dry_run, profile=self.profile.name,
                      profile_hash=self.profile.hash[:12],
                      scene_len=len(self.scene or []))
        for target, name in ((self._capture_loop, "capture"),
                             (self._control_loop, "control"),
                             (self._stdin_loop, "stdin")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

        if self.args.hlp_mode != "off":
            ok = self.poller.acquire(force=self.args.force_lease, retries=3)
            if not ok and self.args.hlp_mode == "active":
                print("[MAIN] FATAL: 无法获取 HLP lease（active 模式必须持有）")
                self.running.clear()
                return
            self.poller.start()

        self.wm.start()
        self.vla.start()

        # ABORT_LATCHED pauses automatic recovery (obs sends stop via
        # current_send_content=None; pause the poller too).
        try:
            while self.running.is_set():
                if self.orch.gate == GateState.ABORT_LATCHED:
                    self.poller.set_paused(True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        self.shutdown()

    def shutdown(self) -> None:
        self.running.clear()
        self.log.emit("client_stopping")
        self.vla.stop()
        self.wm.stop()
        self.poller.stop()
        self.state_sub.stop()
        self.camera.stop()
        if self.token_publisher is not None:
            self.token_publisher.stop()
        self.log.close()
        print("[MAIN] shutdown complete")


def _build_state_43(state: Dict[str, Any]):
    """43-D model state (verbatim semantics from the proven RTC client)."""
    body_q = np.asarray(state["body_q_measured"], dtype=np.float32).reshape(-1)
    lh = np.asarray(state["left_hand_q"], dtype=np.float32).reshape(-1)
    rh = np.asarray(state["right_hand_q"], dtype=np.float32).reshape(-1)
    if body_q.shape != (29,) or lh.shape != (7,) or rh.shape != (7,):
        raise ValueError("bad robot state shapes")
    leg, arm = body_q[:15], body_q[15:29]
    states = np.concatenate((lh, rh, arm, leg), axis=0)
    if states.shape != (43,) or not np.isfinite(states).all():
        raise ValueError("bad 43-D state")
    return np.ascontiguousarray(states), lh, rh


def _resolve_scene(args, profile: SemanticProfile) -> Optional[List[str]]:
    """--scene 'a;;b' or --prompts-json/--task-key; every entry must
    canonicalize against the profile (fail at startup, not on the robot)."""
    raw: List[str] = []
    if args.scene:
        raw = [s.strip() for s in args.scene.split(";") if s.strip()]
    elif args.prompts_json and args.task_key:
        with open(args.prompts_json) as f:
            prompts = json.load(f)
        entry = prompts[args.task_key]
        raw = [str(s).strip() for s in entry.get("subtasks", [])]
    if not raw:
        return None
    out = []
    misses = []
    for s in raw:
        m = profile.match(s)
        if m.ok:
            out.append(m.canonical)
        else:
            misses.append(s)
    if misses:
        raise SystemExit("[MAIN] scene 序列含非 canonical 条目（先修 prompts/scene）：\n  "
                         + "\n  ".join(misses))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="HLP x WM x RTC-VLA orchestrator (hard-gate, shadow-first)")
    # VLA
    p.add_argument("--host", default="localhost", help="VLA server host")
    p.add_argument("--port", type=int, default=8014)
    # robot I/O (same wires as the proven clients)
    p.add_argument("--zmq-host", default="localhost")
    p.add_argument("--zmq-pub-port", type=int, default=5556)
    p.add_argument("--zmq-sub-port", type=int, default=5557)
    p.add_argument("--zmq-topic", default="pose")
    p.add_argument("--zmq-sub-topic", default="g1_debug")
    p.add_argument("--camera-address", default="tcp://192.168.123.164:5558")
    p.add_argument("--camera-timeout-ms", type=int, default=DEFAULT_CAMERA_TIMEOUT_MS)
    # HLP
    p.add_argument("--hlp-host", default="localhost")
    p.add_argument("--hlp-port", type=int, default=8015)
    p.add_argument("--hlp-mode", choices=["off", "shadow", "active"],
                   default="shadow")
    p.add_argument("--hlp-period", type=float, default=1.0)
    p.add_argument("--hlp-connect-timeout", type=float, default=2.0)
    p.add_argument("--hlp-read-timeout", type=float, default=5.0)
    p.add_argument("--hlp-stale-gate", type=float, default=8.0)
    p.add_argument("--force-lease", action="store_true",
                   help="admin takeover of a held HLP lease")
    # WM
    p.add_argument("--wm-host", default="localhost")
    p.add_argument("--wm-port", type=int, default=8016)
    p.add_argument("--wm-connect-timeout", type=float, default=3.0)
    p.add_argument("--wm-read-timeout", type=float, default=10.0)
    p.add_argument("--wm-jpeg-quality", type=int, default=90)
    p.add_argument("--wm-refresh-period", type=float, default=0.0,
                   help="periodic same-stage goal refresh period; needs "
                        "--allow-periodic-refresh (paired-safe, post-G6b)")
    p.add_argument("--allow-periodic-refresh", action="store_true")
    # vocabulary / scene
    p.add_argument("--profile", default=os.path.join(
        _GROOT_ROOT, "profiles/cleanup_table_fine_v1.json"))
    p.add_argument("--profile-hash", default=None,
                   help="pin the exact profile hash (recommended for active)")
    p.add_argument("--scene", default=None,
                   help="';'-separated ordered canonical stage list")
    p.add_argument("--prompts-json", default=None)
    p.add_argument("--task-key", default=None)
    # watchdogs
    p.add_argument("--observation-stale-timeout", type=float, default=0.5)
    p.add_argument("--action-stale-timeout", type=float, default=0.5)
    p.add_argument("--capture-max-age", type=float, default=1.0)
    p.add_argument("--max-pending-s", type=float, default=30.0)
    p.add_argument("--ack-timeout-s", type=float, default=10.0)
    p.add_argument("--max-gate-s", type=float, default=120.0)
    p.add_argument("--hold-reencode-period", type=float, default=0.0,
                   help="0 = entry-snapshot hold token (G0 decides)")
    # misc
    p.add_argument("--allow-raw-override", action="store_true",
                   help="legacy takeover: :ov accepts NON-canonical free text and "
                        "drives WM+VLA verbatim (wHLP real-robot semantics; bypasses "
                        "the vocabulary safety layer — operator owns the risk)")
    p.add_argument("--log-dir", default=os.path.join(_GROOT_ROOT, "hlpwm_logs"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main():
    args = parse_args()
    client = HlpwmClient(args)

    import signal

    def _sig(_s, _f):
        client.running.clear()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    client.run()


if __name__ == "__main__":
    main()
