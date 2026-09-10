"""Diagnostics for the action publication path.

Nothing here can hold or stop the robot: the acting guards are the observation
and action staleness timeouts and the condition-promote timeout in vla.py. This
module exists to make one failure legible in the log — a live-but-frozen action
stream, where a run-out server chunk keeps replaying a single action with a
still-valid acknowledgement, so every freshness and provenance check passes
while the robot does not move. Only the content deltas expose that.
"""

import numpy as np
import threading
from .recording import log_event
from .robot import HAND_DIM, TOKEN_DIM, fsq_quantize


class RateCounter:
    """Average rate of a periodic loop, reported at most once a second."""

    def __init__(self, label):
        self.label = label
        self.count = 0
        self.started_at = None

    def tick(self):
        self.count += 1

    def report(self, now, **fields):
        if self.started_at is None:
            self.started_at = now
            return None
        elapsed = now - self.started_at
        if elapsed < 1.0:
            return None
        line = f"[client] {self.label} avg_hz={self.count / elapsed:.1f}" + "".join(
            f" {k}={v}" for k, v in fields.items())
        self.count, self.started_at = 0, now
        return line


class ActionTelemetry:
    """Per-published-action counters, summarised once a second.

    Every counter is scoped to a prompt epoch: an in-flight callback that lands
    just after an operator switch must not repopulate the new epoch's window.
    """

    def __init__(self, *, include_neck=False, dry_run=False):
        # Reset comes from the operator/HLP thread, received/update from the
        # transport thread. This lock is innermost: nothing here calls out.
        self._lock = threading.Lock()
        self.include_neck = bool(include_neck)
        self.dry_run = bool(dry_run)
        self.epoch = None
        self.last_version = None
        self.prev_action = None
        self.last_log_at = None
        self.static_run = 0
        self.chunk_id = self.chunk_tick = self.infer_ms = None
        self._clear()

    def _clear(self):
        """The one-second window. `static_run` and `prev_action` deliberately
        survive it so a zero-change run stays visible across the boundary."""
        self.count = 0
        self.started_at = None
        self.recv_interval_max = 0.0
        self.version_gap_total = 0
        self.version_gap_max = 0
        self.chunk_switches = 0
        self.repeat_last_ticks = 0
        self.tok_delta_max = 0.0
        self.hand_delta_max = 0.0
        self.neck_delta_max = 0.0
        self.tok_deltas = []
        self.fsq_static_ticks = 0
        self.fsq_changed_max = 0
        self.static_run_max = self.static_run
        self.ticks = 0

    def reset(self, epoch):
        with self._lock:
            self._reset(epoch)

    def _reset(self, epoch):
        """A prompt transition: the previous epoch's deltas mean nothing now."""
        self.epoch = int(epoch)
        self.last_version = None
        self.prev_action = None
        self.chunk_id = self.chunk_tick = self.infer_ms = None
        self.static_run = 0
        self._clear()

    def received(self, version):
        with self._lock:
            self._received(version)

    def _received(self, version):
        """Count versions the transport coalesced away, in O(1) off the log path."""
        if self.last_version is not None:
            gap = max(0, int(version) - int(self.last_version) - 1)
            self.version_gap_total += gap
            self.version_gap_max = max(self.version_gap_max, gap)
        self.last_version = int(version)

    def _transport(self, data):
        chunk_id, chunk_tick = data.get("rtc_chunk_id"), data.get("rtc_chunk_tick")
        if isinstance(chunk_id, int) and not isinstance(chunk_id, bool):
            if self.chunk_id is not None and chunk_id != self.chunk_id:
                self.chunk_switches += 1
            self.chunk_id, self.chunk_tick = chunk_id, chunk_tick
            if data.get("rtc_repeat_last") is True:
                self.repeat_last_ticks += 1
        try:
            self.infer_ms = float(data["rtc_infer_ms"])
        except (KeyError, TypeError, ValueError):
            pass

    def _deltas(self, flat):
        previous, self.prev_action = self.prev_action, flat.copy()
        if previous is None:
            return
        token, was = flat[HAND_DIM:HAND_DIM + TOKEN_DIM], previous[HAND_DIM:HAND_DIM + TOKEN_DIM]
        delta = float(np.abs(token - was).max())
        self.tok_delta_max = max(self.tok_delta_max, delta)
        self.tok_deltas.append(delta)
        self.hand_delta_max = max(self.hand_delta_max,
                                  float(np.abs(flat[:HAND_DIM] - previous[:HAND_DIM]).max()))
        if self.include_neck:
            self.neck_delta_max = max(
                self.neck_delta_max,
                float(np.abs(flat[HAND_DIM + TOKEN_DIM:] - previous[HAND_DIM + TOKEN_DIM:]).max()))
        changed = int(np.count_nonzero(fsq_quantize(token) != fsq_quantize(was)))
        self.fsq_changed_max = max(self.fsq_changed_max, changed)
        if changed:
            self.static_run = 0
        else:
            self.fsq_static_ticks += 1
            self.static_run += 1
            self.static_run_max = max(self.static_run_max, self.static_run)
        self.ticks += 1

    def update(self, data, action, version, interval, now, epoch, cid):
        """Fold one published action in; return the 1 Hz summary line, or None."""
        with self._lock:
            return self._update(data, action, version, interval, now, epoch, cid)

    def _update(self, data, action, version, interval, now, epoch, cid):
        if self.epoch is None:
            self.epoch = int(epoch)
        elif self.epoch != int(epoch):
            return None
        if self.started_at is None:
            self.started_at = now
        self.recv_interval_max = max(self.recv_interval_max, float(interval))
        self._transport(data)
        self._deltas(action[0])
        self.count += 1
        if self.last_log_at is not None and now - self.last_log_at < 1.0:
            return None
        self.last_log_at = now
        hz = self.count / max(now - self.started_at, 1e-6)
        p95 = float(np.percentile(self.tok_deltas, 95)) if self.tok_deltas else 0.0
        line = (
            f"[client] action {'validated' if self.dry_run else 'published'}: "
            f"version={version} cid={cid} hz={hz:.1f} "
            f"recv_imax={self.recv_interval_max:.3f}s "
            f"version_gap={self.version_gap_total}/{self.version_gap_max} "
            f"chunk={self.chunk_id}/{self.chunk_tick} switches={self.chunk_switches} "
            f"repeat_last={self.repeat_last_ticks} infer_ms={self.infer_ms} "
            f"tok_dmax={self.tok_delta_max:.4f} tok_p95={p95:.4f} "
            f"hand_dmax={self.hand_delta_max:.4f} neck_dmax={self.neck_delta_max:.4f} "
            f"fsq_static={self.fsq_static_ticks}/{self.ticks} "
            f"chg_max={self.fsq_changed_max} static_run_max={self.static_run_max}")
        log_event("action_telemetry", cid=cid, epoch=self.epoch, version=int(version),
                  hz=round(hz, 2), recv_interval_max=round(self.recv_interval_max, 4),
                  version_gap_total=self.version_gap_total, version_gap_max=self.version_gap_max,
                  chunk_id=self.chunk_id, chunk_tick=self.chunk_tick,
                  chunk_switches=self.chunk_switches, repeat_last_ticks=self.repeat_last_ticks,
                  infer_ms=self.infer_ms, tok_dmax=self.tok_delta_max, tok_p95=p95,
                  hand_dmax=self.hand_delta_max, neck_dmax=self.neck_delta_max,
                  fsq_static_ticks=self.fsq_static_ticks, ticks=self.ticks,
                  fsq_changed_max=self.fsq_changed_max, static_run_max=self.static_run_max)
        self._clear()
        self.started_at = now
        return line
