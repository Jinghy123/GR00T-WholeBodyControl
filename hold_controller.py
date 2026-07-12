"""Recoverable transient hold for the G1 WBC — the ONLY gate primitive active
clients may use (plan_next.md §7).

Why this exists: the ZMQ planner wire's stop byte is TERMINAL — the receiver maps
it to operator_state.stop, the C++ control loop exits into damping and the process
ends; a later start cannot resume it (g1_deploy_onnx_ref.cpp:4030/4714-4725, and
`operator_state.stop = false` appears nowhere). So a "gate" implemented as
stop→start actually kills WBC. The recoverable gate instead keeps WBC running and
publishes a FROZEN pose action at control rate, following the proven
g1_sonic_client_rtc.py::_freeze_action recipe — minus its all-zero fallback, which
is banned here (P0-12).

States:
  PRE_START          WBC/planner not started and no valid action yet. Gate means
                     "keep not starting": publish NOTHING (never a zero action).
  RUN                normal VLA-driven execution (the client publishes VLA actions
                     itself and feeds note_executed()).
  TRANSIENT_HOLD     WBC keeps running; tick() publishes the frozen ActionParts at
                     the control rate. Hands = last executed desired hands; body
                     token = current pose re-encoded at gate entry (optionally
                     re-encoded every reencode_period_s — G0 decides the default).
  HOLD_PATH_FAILURE  the hold path itself is unhealthy (stale robot state, encoder
                     or publisher failure, no last action while running). NOT
                     recoverable in-band: on_failure() fires and the caller must
                     apply the G0-frozen terminal/emergency policy.

Type-level safety: the PublicationAdapter interface has NO stop capability at all —
this module cannot express "stop WBC" even by bug. Terminal stop stays with the
client's explicit :quit/emergency path, outside any gate.

Active profile is no-neck 78-D: ActionParts(left_hand7, right_hand7, body_token64);
the single TokenPublisherAdapter is the only place the wire order/quantization
([token64 | LH7 | RH7], fsq on the token) lives. neck/heading/80-D is a future
profile with its own adapter + gate (plan §7.2-4).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np

HAND_DIM = 7
TOKEN_DIM = 64
ACTION_DIM = 2 * HAND_DIM + TOKEN_DIM  # 78

# mujoco -> isaaclab DOF order for body_q_measured(29); the ONNX encoder expects
# isaaclab order (psix_rtc_sonic_wm_client.py:81-86).
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24,
     18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)


def mujoco29_to_isaaclab29(qpos) -> np.ndarray:
    return np.asarray(qpos, dtype=np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB_DOF].copy()


@dataclass(frozen=True)
class ActionParts:
    """Typed, semantics-labeled action — never a bare 78-D flat array (P0-10)."""
    left_hand7: np.ndarray
    right_hand7: np.ndarray
    body_token64: np.ndarray   # wire-ready (already quantized) body token

    def __post_init__(self):
        for name, arr, dim in (("left_hand7", self.left_hand7, HAND_DIM),
                               ("right_hand7", self.right_hand7, HAND_DIM),
                               ("body_token64", self.body_token64, TOKEN_DIM)):
            a = np.asarray(arr, dtype=np.float32)
            if a.shape != (dim,):
                raise ValueError(f"{name} must be shape ({dim},), got {a.shape}")
            if not np.isfinite(a).all():
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, a)


class PublicationAdapter:
    """The ONE component that knows the WBC wire. Deliberately has no stop API."""

    def publish(self, parts: ActionParts) -> None:
        raise NotImplementedError

    def send_planner_start(self) -> None:
        raise NotImplementedError


class TokenPublisherAdapter(PublicationAdapter):
    """Adapter over the existing TokenPublisher (Protocol v4): wire layout
    [token64 | LH7 | RH7]; planner start is (start=True, stop=False, planner=True)."""

    def __init__(self, token_publisher):
        self._pub = token_publisher

    def publish(self, parts: ActionParts) -> None:
        wire = np.concatenate([parts.body_token64, parts.left_hand7, parts.right_hand7])
        if wire.shape != (ACTION_DIM,) or not np.isfinite(wire).all():
            raise ValueError("wire action is not a finite 78-D vector")
        self._pub.publish_token(wire)

    def send_planner_start(self) -> None:
        self._pub.send_command(start=True, stop=False, planner=True)


class HoldState(str, Enum):
    PRE_START = "PRE_START"
    RUN = "RUN"
    TRANSIENT_HOLD = "TRANSIENT_HOLD"
    HOLD_PATH_FAILURE = "HOLD_PATH_FAILURE"


def make_g1_hold_token_encoder(encoder, get_state: Callable[[], Optional[Dict[str, Any]]],
                               *, token_postprocess: Optional[Callable] = None,
                               state_max_age_s: float = 0.5,
                               clock: Callable[[], float] = time.monotonic):
    """Compose the verified hold-still token recipe (g1_sonic_client_rtc.py:342-367):
    fresh body_q_measured(29, mujoco order) -> isaaclab reorder -> tile 10 frames,
    zero velocities, tiled base_quat(4, wxyz) -> EncoderClient.encode -> (64,) token
    -> token_postprocess (fsq quantization on the live wire; identity in tests).

    Raises RuntimeError on stale/missing state or encoder failure — the caller
    (HoldController) turns that into HOLD_PATH_FAILURE. Never returns zeros."""

    def encode() -> np.ndarray:
        state = get_state()
        if state is None:
            raise RuntimeError("no robot state for hold-token encode")
        t = state.get("t")
        if t is not None and (clock() - float(t)) > state_max_age_s:
            raise RuntimeError(f"robot state stale (> {state_max_age_s}s) for hold-token encode")
        qpos = mujoco29_to_isaaclab29(state["body_q_measured"])
        quat = np.asarray(state.get("base_quat_measured", [1, 0, 0, 0]),
                          dtype=np.float32).reshape(4)
        joint_pos = np.tile(qpos, (10, 1)).astype(np.float32)
        joint_vel = np.zeros((10, 29), dtype=np.float32)
        body_quat = np.tile(quat, (10, 1)).astype(np.float32)
        token = np.asarray(encoder.encode(joint_pos, joint_vel, body_quat),
                           dtype=np.float32).reshape(TOKEN_DIM)
        if not np.isfinite(token).all():
            raise RuntimeError("encoder produced non-finite hold token")
        return token_postprocess(token) if token_postprocess is not None else token

    return encode


class HoldController:
    """All 12 legacy stop sources route through enter_hold(); none may stop WBC."""

    def __init__(self, adapter: PublicationAdapter, *,
                 encode_hold_token: Callable[[], np.ndarray],
                 reencode_period_s: float = 0.0,
                 on_failure: Optional[Callable[[str], None]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._adapter = adapter
        self._encode = encode_hold_token
        self._reencode_period_s = float(reencode_period_s)
        self._on_failure = on_failure
        self._clock = clock
        self.state = HoldState.PRE_START
        self.hold_reason: Optional[str] = None
        self.hold_entered_at: Optional[float] = None
        self._last_executed: Optional[ActionParts] = None
        self._frozen: Optional[ActionParts] = None
        self._last_encode_at: float = 0.0
        self.hold_ticks_published = 0

    # ------------------------------------------------------------- RUN path
    def note_wbc_started(self) -> None:
        """First planner start happened (client sends it on the first valid action)."""
        if self.state == HoldState.PRE_START:
            self.state = HoldState.RUN

    def note_executed(self, parts: ActionParts) -> None:
        """The client executed a VLA action — remember it as the hold target."""
        self._last_executed = parts
        if self.state == HoldState.PRE_START:
            self.state = HoldState.RUN

    # ------------------------------------------------------------- gate
    def enter_hold(self, reason: str) -> HoldState:
        self.hold_reason = reason
        if self.state == HoldState.HOLD_PATH_FAILURE:
            return self.state
        if self.state == HoldState.PRE_START:
            # gate before start = keep not starting; publish NOTHING (no zeros).
            return self.state
        if self.state == HoldState.TRANSIENT_HOLD:
            return self.state
        if self._last_executed is None:
            return self._fail("hold requested while running but no last executed action")
        try:
            token = self._encode()
        except Exception as e:
            return self._fail(f"hold-token encode failed: {e}")
        self._frozen = ActionParts(self._last_executed.left_hand7,
                                   self._last_executed.right_hand7, token)
        self._last_encode_at = self._clock()
        self.hold_entered_at = self._last_encode_at
        self.state = HoldState.TRANSIENT_HOLD
        return self.state

    def tick(self) -> bool:
        """Call at the control rate. Publishes the frozen action while holding.
        Returns True when a hold action was published."""
        if self.state != HoldState.TRANSIENT_HOLD:
            return False
        if (self._reencode_period_s > 0
                and self._clock() - self._last_encode_at >= self._reencode_period_s):
            try:
                token = self._encode()
                self._frozen = ActionParts(self._frozen.left_hand7,
                                           self._frozen.right_hand7, token)
                self._last_encode_at = self._clock()
            except Exception as e:
                self._fail(f"hold-token re-encode failed: {e}")
                return False
        try:
            self._adapter.publish(self._frozen)
        except Exception as e:
            self._fail(f"hold publish failed: {e}")
            return False
        self.hold_ticks_published += 1
        return True

    def resume_run(self) -> None:
        """Exit hold after the condition-ack equality check passed (§5.2-6)."""
        if self.state == HoldState.TRANSIENT_HOLD:
            self.state = HoldState.RUN
            self.hold_reason = None
            self._frozen = None

    # ------------------------------------------------------------- failure
    def _fail(self, reason: str) -> HoldState:
        self.state = HoldState.HOLD_PATH_FAILURE
        self.hold_reason = reason
        if self._on_failure is not None:
            try:
                self._on_failure(reason)
            except Exception:
                pass
        return self.state

    def status(self) -> Dict[str, Any]:
        return {"state": self.state.value, "reason": self.hold_reason,
                "hold_entered_at": self.hold_entered_at,
                "hold_ticks_published": self.hold_ticks_published,
                "has_last_executed": self._last_executed is not None}
