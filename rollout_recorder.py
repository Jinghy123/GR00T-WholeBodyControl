"""Continuous rollout recording: states, executed actions, and the ego video.

The existing IncidentRecorder keeps a ~20 s ring and only lands it when
something trips, so a finished rollout retains just the world-model query images
-- not enough to find a stall afterwards, let alone see what was commanded
during it. This writes the whole episode instead: state and action rows at
control rate, and the ego camera as an mp4.

Everything runs behind a bounded queue on one writer thread. The queue DROPS on
overflow; telemetry is expendable and the 30 Hz control loop is not.

Predicted chunks are NOT here: the wire carries one action row per tick, so the
planned chunk only exists inside the server. Set PSIX_CHUNK_DUMP_DIR there
(psi/src/psi/deploy/chunk_recorder.py) to capture it.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import signal
import threading
import time
from datetime import datetime

import cv2
import numpy as np

_QUEUE = 4096
_SHARD = 3000          # rows per npz shard (~100 s of 30 Hz control)
_FLUSH_EVERY_S = 10.0  # land whatever is buffered this often, so a hard kill
                       # costs at most this many seconds instead of everything


class RolloutRecorder:
    _STOP = object()

    def __init__(self, out_dir, video_fps=10.0, video_scale=1.0, shard=_SHARD,
                 video_format="avi"):
        self.dir = out_dir
        os.makedirs(self.dir, exist_ok=True)
        self._q = queue.Queue(maxsize=_QUEUE)
        self._shard = int(shard)
        self._video_fps = float(video_fps)
        self._video_scale = float(video_scale)
        self._last_video_at = -float("inf")
        self.dropped = 0
        self._t0 = time.monotonic()
        # MJPG/AVI by default: every frame is independently decodable, so a
        # truncated file loses only its tail. An mp4 without its trailer is not
        # readable at all -- which is exactly what happened on 2026-08-23, when
        # all 25 recorded episodes came back as unreadable files.
        self._video_format = video_format
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._writer, name="rollout-recorder",
                                        daemon=True)
        self._thread.start()
        # The control path is not guaranteed to reach a graceful shutdown, so the
        # flush must also be reachable from atexit and from the usual signals.
        atexit.register(self.close)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.getsignal(sig)
                if prev == signal.SIG_IGN:
                    # The process was started with this signal ignored (a
                    # background launch inherits that from its shell). Leaving
                    # SIG_IGN in place is the only correct option: chaining had
                    # no branch for it, so the handler used to close the
                    # recorder and then return -- the run carried on with
                    # recording silently switched off for the rest of the
                    # episode. atexit still lands the data at exit.
                    continue
                signal.signal(sig, self._make_handler(sig, prev))
            except (ValueError, OSError):
                pass          # not on the main thread; atexit still covers us

    def _make_handler(self, sig, prev):
        def _handler(signum, frame):
            self.close()
            if callable(prev):
                prev(signum, frame)
            elif prev == signal.SIG_DFL:
                signal.signal(sig, signal.SIG_DFL)
                os.kill(os.getpid(), sig)
        return _handler

    # -- producers (control threads) -----------------------------------------
    def _put(self, item):
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def record_action(self, mono, version, cid, chunk_id, chunk_tick,
                      repeat_last, action):
        self._put(("action", (mono, int(version),
                              -1 if cid is None else int(cid),
                              -1 if chunk_id is None else int(chunk_id),
                              -1 if chunk_tick is None else int(chunk_tick),
                              bool(repeat_last),
                              np.asarray(action, np.float32).copy())))

    def record_state(self, mono, states):
        self._put(("state", (mono, np.asarray(states, np.float32).copy())))

    def record_video_frame(self, mono, frame_rgb):
        """Full-resolution ego frame, rate-limited to video_fps here so the
        camera thread never pays for a frame that would be dropped anyway."""
        if mono - self._last_video_at < 1.0/max(self._video_fps, 1e-6):
            return
        self._last_video_at = mono
        self._put(("frame", (mono, np.ascontiguousarray(frame_rgb).copy())))

    def close(self, timeout=15.0):
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._q.put_nowait(self._STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    # -- consumer -------------------------------------------------------------
    def _flush_actions(self, buf, n):
        if not buf:
            return
        np.savez_compressed(
            os.path.join(self.dir, f"actions_{n:04d}.npz"),
            mono=np.array([b[0] for b in buf], np.float64),
            version=np.array([b[1] for b in buf], np.int64),
            cid=np.array([b[2] for b in buf], np.int64),
            chunk_id=np.array([b[3] for b in buf], np.int64),
            chunk_tick=np.array([b[4] for b in buf], np.int64),
            repeat_last=np.array([b[5] for b in buf], bool),
            action=np.stack([b[6] for b in buf]))

    def _flush_states(self, buf, n):
        if not buf:
            return
        np.savez_compressed(
            os.path.join(self.dir, f"states_{n:04d}.npz"),
            mono=np.array([b[0] for b in buf], np.float64),
            states=np.stack([b[1] for b in buf]))

    def _write_meta(self, na, ns, nframes, tstamps, partial=False):
        try:
            with open(os.path.join(self.dir, "rollout_meta.json"), "w") as f:
                json.dump({"schema_version": "rollout-record/1",
                           "partial": bool(partial),
                           "written_at": datetime.now().isoformat(timespec="milliseconds"),
                           "duration_s": time.monotonic()-self._t0,
                           "action_shards": na, "state_shards": ns,
                           "video": {"fps": self._video_fps, "frames": nframes,
                                     "scale": self._video_scale},
                           "video_mono": tstamps,
                           "dropped": self.dropped}, f)
        except Exception as exc:
            print(f"[rollout-recorder] meta write failed: {exc}", flush=True)

    def _writer(self):
        acts, sts, vw = [], [], None
        na = ns = nframes = 0
        ext = "avi" if self._video_format == "avi" else "mp4"
        fourcc = "MJPG" if ext == "avi" else "mp4v"
        vpath = os.path.join(self.dir, f"ego.{ext}")
        tstamps = []
        last_flush = time.monotonic()
        try:
            while True:
                try:
                    item = self._q.get(timeout=1.0)
                except queue.Empty:
                    item = None
                now = time.monotonic()
                if now - last_flush >= _FLUSH_EVERY_S:
                    # Time-based landing: shard boundaries alone mean a short
                    # episode that is killed keeps nothing at all.
                    if acts:
                        self._flush_actions(acts, na); acts = []; na += 1
                    if sts:
                        self._flush_states(sts, ns); sts = []; ns += 1
                    self._write_meta(na, ns, nframes, tstamps, partial=True)
                    last_flush = now
                if item is None:
                    continue
                if item is self._STOP:
                    break
                kind, payload = item
                if kind == "action":
                    acts.append(payload)
                    if len(acts) >= self._shard:
                        self._flush_actions(acts, na); acts = []; na += 1
                elif kind == "state":
                    sts.append(payload)
                    if len(sts) >= self._shard:
                        self._flush_states(sts, ns); sts = []; ns += 1
                else:
                    mono, frame = payload
                    if self._video_scale != 1.0:
                        frame = cv2.resize(frame, None, fx=self._video_scale,
                                           fy=self._video_scale)
                    if vw is None:
                        h, w = frame.shape[:2]
                        vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*fourcc),
                                             self._video_fps, (w, h))
                        if not vw.isOpened():
                            print("[rollout-recorder] WARNING: VideoWriter failed to open",
                                  flush=True)
                            vw = False
                    if vw:
                        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        tstamps.append(mono); nframes += 1
        finally:
            if acts:
                self._flush_actions(acts, na); na += 1
            if sts:
                self._flush_states(sts, ns); ns += 1
            if vw:
                vw.release()
            self._write_meta(na, ns, nframes, tstamps, partial=False)


def maybe_rollout_recorder(out_dir, enabled=True, video_fps=10.0, video_scale=1.0):
    if not enabled or not out_dir:
        return None
    try:
        return RolloutRecorder(out_dir, video_fps=video_fps, video_scale=video_scale)
    except Exception as exc:
        print(f"[rollout-recorder] disabled: {type(exc).__name__}: {exc}", flush=True)
        return None
