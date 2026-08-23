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

import json
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np

_QUEUE = 4096
_SHARD = 3000          # rows per npz shard (~100 s of 30 Hz control)


class RolloutRecorder:
    _STOP = object()

    def __init__(self, out_dir, video_fps=10.0, video_scale=1.0, shard=_SHARD):
        self.dir = out_dir
        os.makedirs(self.dir, exist_ok=True)
        self._q = queue.Queue(maxsize=_QUEUE)
        self._shard = int(shard)
        self._video_fps = float(video_fps)
        self._video_scale = float(video_scale)
        self._last_video_at = -float("inf")
        self.dropped = 0
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._writer, name="rollout-recorder",
                                        daemon=True)
        self._thread.start()

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

    def _writer(self):
        acts, sts, vw = [], [], None
        na = ns = nframes = 0
        vpath = os.path.join(self.dir, "ego.mp4")
        tstamps = []
        try:
            while True:
                try:
                    item = self._q.get(timeout=1.0)
                except queue.Empty:
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
                        vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"),
                                             self._video_fps, (w, h))
                        if not vw.isOpened():
                            print("[rollout-recorder] WARNING: VideoWriter failed to open",
                                  flush=True)
                            vw = False
                    if vw:
                        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        tstamps.append(mono); nframes += 1
        finally:
            self._flush_actions(acts, na)
            self._flush_states(sts, ns)
            if vw:
                vw.release()
            try:
                with open(os.path.join(self.dir, "rollout_meta.json"), "w") as f:
                    json.dump({"schema_version": "rollout-record/1",
                               "closed_at": datetime.now().isoformat(timespec="milliseconds"),
                               "duration_s": time.monotonic()-self._t0,
                               "action_shards": na + (1 if acts else 0),
                               "state_shards": ns + (1 if sts else 0),
                               "video": {"path": "ego.mp4", "fps": self._video_fps,
                                         "frames": nframes, "scale": self._video_scale},
                               # mono is the same clock as actions/states, so the video
                               # can be aligned to them without guessing a start offset.
                               "video_mono": tstamps,
                               "dropped": self.dropped}, f)
            except Exception as exc:
                print(f"[rollout-recorder] meta write failed: {exc}", flush=True)


def maybe_rollout_recorder(out_dir, enabled=True, video_fps=10.0, video_scale=1.0):
    if not enabled or not out_dir:
        return None
    try:
        return RolloutRecorder(out_dir, video_fps=video_fps, video_scale=video_scale)
    except Exception as exc:
        print(f"[rollout-recorder] disabled: {type(exc).__name__}: {exc}", flush=True)
        return None
