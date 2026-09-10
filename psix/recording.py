"""Request artifacts, events, and continuous-rollout integration."""

import cv2
import hashlib
import json
import numpy as np
import os
import queue
import requests
import subprocess
import sys
import threading
import time
from .robot import FSQ_MAX, FSQ_MIN, FSQ_STEP, _GROOT_ROOT
from base64 import b64decode
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path


_EVENT_LOG_STOP = object()


_EVENT_LOG = None


class EventLog:
    """Append-only JSONL event writer with a dedicated I/O thread.

    Control threads only enqueue (non-blocking, drop-on-overflow); a single
    writer thread owns the file handle, so a slow disk can never stall the
    30 Hz observation/action paths.
    """

    def __init__(self, path, maxsize=4096):
        self._path = path
        self._queue = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._writer, name="event-log", daemon=True
        )
        self._thread.start()

    def emit(self, kind, **fields):
        record = {
            "kind": str(kind),
            "t_wall": datetime.now().isoformat(timespec="milliseconds"),
            "t_mono": time.monotonic(),
        }
        record.update(fields)
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def stop(self, timeout=2.0):
        try:
            self._queue.put(_EVENT_LOG_STOP, timeout=0.5)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def _writer(self):
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                while True:
                    item = self._queue.get()
                    if item is _EVENT_LOG_STOP:
                        if self._dropped:
                            f.write(json.dumps(
                                {"kind": "event_log_dropped",
                                 "count": int(self._dropped)}) + "\n")
                        f.flush()
                        return
                    f.write(json.dumps(item, default=str) + "\n")
                    if self._queue.empty():
                        f.flush()
        except Exception as exc:
            print(f"[events] WARNING: event log writer died ({exc})", flush=True)


def set_event_log(event_log):
    global _EVENT_LOG
    _EVENT_LOG = event_log


def log_event(kind, **fields):
    log = _EVENT_LOG
    if log is not None:
        log.emit(kind, **fields)


def _git_identity(repo_dir):
    """Best-effort {sha, dirty, dirty_diff_sha256} for one repo; never raises."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            text=True, timeout=5, stderr=subprocess.DEVNULL).strip()
        diff = subprocess.check_output(
            ["git", "-C", repo_dir, "diff", "HEAD"],
            text=True, timeout=15, stderr=subprocess.DEVNULL)
        return {
            "repo_dir": repo_dir,
            "sha": sha,
            "dirty": bool(diff),
            "dirty_diff_sha256": (
                hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None
            ),
        }
    except Exception as exc:
        return {"repo_dir": repo_dir, "error": str(exc)}


def _fetch_json(url, timeout=3.0):
    """Tolerant JSON GET for manifest identity; failures become error records."""
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"url": url, "error": str(exc)}
    finally:
        session.close()


def write_run_manifest(run_dir, config, vla_info, wm_state, episode_session_id):
    """Persist everything needed to regroup runs after the fact (plan P0.3)."""
    manifest = {
        "schema_version": "wm-run-manifest/1",
        "written_at": datetime.now().isoformat(timespec="milliseconds"),
        "written_at_monotonic": time.monotonic(),
        "argv": list(sys.argv),
        "config": config,
        "groot_repo": _git_identity(_GROOT_ROOT),
        "psi_repo": _git_identity(os.environ.get(
            "PSI_REPO_DIR", os.path.expanduser("~/Desktop/psi"))),
        "vla_info": vla_info,
        "wm_state": wm_state,
        "robot_episode_session_id": episode_session_id,
        "fsq": {"min": FSQ_MIN, "max": FSQ_MAX, "step": FSQ_STEP},
    }
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    return path


def _fs_slug(value, fallback):
    """Filesystem-safe slug for embedding free-text labels in a filename."""
    text = str(value) if value else fallback
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
    return slug[:60] or fallback


def save_init_frame(run_dir, camera, task_instruction, method_name,
                     camera_address, include_neck, episode_session_id):
    """Save the initial frame before the observation loop takes camera ownership."""
    try:
        frame = camera.get_frame()  # BGR uint8, ready for cv2.imwrite
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        method_slug = _fs_slug(method_name, "nolabel")
        filename = f"init_ego_{method_slug}_{stamp}.jpg"
        path = os.path.join(run_dir, filename)
        os.makedirs(run_dir, exist_ok=True)
        if not cv2.imwrite(path, frame):
            raise RuntimeError(f"cv2.imwrite failed for {path}")
        meta = {
            "schema_version": "wm-init-frame/1",
            "saved_at": datetime.now().isoformat(timespec="milliseconds"),
            "task": task_instruction,
            "method_name": method_name,
            "robot_episode_session_id": episode_session_id,
            "image_file": filename,
            "camera_address": camera_address,
            "include_neck": bool(include_neck),
        }
        meta_path = os.path.join(run_dir, "init_frame.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[MAIN] Saved init frame: {path}", flush=True)
        return path
    except Exception as exc:
        # Telemetry failures must not interrupt startup.
        print(f"[MAIN] WARNING: failed to save init frame ({exc})", flush=True)
        return None


class InferenceRecorder:
    def __init__(self, run_dir, *, enabled=True):
        self.root = Path(run_dir) / "requests"
        self.enabled = enabled
        self.errors = 0

    def _error(self, exc):
        self.errors += 1
        print(f"[request-recorder] WARNING: {exc}", flush=True)

    @staticmethod
    def _image(encoded, path):
        data = b64decode(encoded, validate=True)
        path.write_bytes(data)
        return {"file": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data)}

    @staticmethod
    def _write(record):
        path, data = record
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)

    def begin(self, service, endpoint, payload, *, context=None):
        """Persist before POST; an interrupted request remains marked pending."""
        if not self.enabled:
            return None
        try:
            directory = self.root / service
            directory.mkdir(parents=True, exist_ok=True)
            request_id = payload.get("req_id", payload.get("request_id", 0))
            stem = f"{time.time_ns()}_req-{request_id}"
            request = deepcopy(payload)
            image_path = directory / f"{stem}.obs.jpg"
            if "ego_jpeg" in request:
                request["ego_jpeg"] = self._image(request["ego_jpeg"], image_path)
            elif isinstance(request.get("ego_image"), dict) and "jpeg_b64" in request["ego_image"]:
                request["ego_image"]["jpeg_b64"] = self._image(request["ego_image"]["jpeg_b64"], image_path)
            record = (directory / f"{stem}.json", {
                "schema_version": "inference-record/1", "service": service,
                "endpoint": endpoint, "request_id": request_id,
                "started_at": time.time(), "started_monotonic": time.monotonic(),
                "request": request, "context": deepcopy(context or {}),
                "outcome": "pending",
            })
            self._write(record)
            return record
        except Exception as exc:
            self._error(exc)
            return None

    def response(self, record, status, payload):
        if record is None:
            return
        try:
            path, data = record
            response = deepcopy(payload)
            if isinstance(response, dict) and isinstance(response.get("subgoal_jpeg"), str):
                try:
                    response["subgoal_jpeg"] = self._image(
                        response["subgoal_jpeg"], path.with_suffix(".goal.jpg"))
                except Exception as exc:
                    # Retain the original malformed field for response validation audits.
                    data["image_recording_error"] = str(exc)
            data.update(http_status=int(status), response=response,
                        replied_monotonic=time.monotonic(), outcome="received")
            self._write(record)
        except Exception as exc:
            self._error(exc)

    def finish(self, record, outcome, **details):
        if record is None:
            return
        try:
            record[1].update(outcome=outcome, finished_monotonic=time.monotonic(), **details)
            self._write(record)
        except Exception as exc:
            self._error(exc)
