"""
Background ZED-mini recorder for deploy runs.

Subscribes to realsense_server.py's viewer PUB stream (2-part multipart:
[ego_jpeg, stereo_jpeg], default port 5559) and writes MP4 video plus a
timestamps CSV. The PUB socket is SEPARATE from the REP inference socket
(5558) and drops frames for a slow subscriber instead of backpressuring,
so recording never slows down the policy client.

Meant to be launched in the background by client.sh:

  python zed_recorder.py --out-dir zed_recordings/my_run &

Stop with SIGINT/SIGTERM (client.sh does this on exit) — the MP4 is
finalized cleanly before the process exits.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import queue
import re
import signal
import threading
import time

import cv2
import numpy as np
import zmq


def _decode(jpeg_bytes):
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class VideoSink:
    """Lazy-open MP4 writer (frame size is only known at the first frame)."""

    def __init__(self, path: str, fps: float):
        self.path = path
        self.fps = fps
        self.writer = None
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f"cannot open VideoWriter for {self.path}")
            print(f"[recorder] writing {self.path} ({w}x{h} @ {self.fps} fps)")
        self.writer.write(frame)
        self.count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            print(f"[recorder] finalized {self.path}: {self.count} frames")


def next_run_index(out_dir: str) -> int:
    """Runs accumulate in one folder as ego_001.mp4, ego_002.mp4, ...
    Pick the next free index so a new run never overwrites an old one."""
    max_idx = 0
    for path in glob.glob(os.path.join(out_dir, "ego_*.mp4")):
        m = re.match(r"ego_(\d+)\.mp4$", os.path.basename(path))
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def writer_loop(q: "queue.Queue", args, stop: threading.Event) -> None:
    """Decode + disk I/O off the receive path so slow disk never makes the
    SUB socket fall behind (PUB would silently drop frames if it did)."""
    run = f"{next_run_index(args.out_dir):03d}"
    ego_sink = VideoSink(os.path.join(args.out_dir, f"ego_{run}.mp4"), args.fps)
    stereo_sink = (
        VideoSink(os.path.join(args.out_dir, f"stereo_{run}.mp4"), args.fps)
        if args.record_stereo else None
    )
    ts_path = os.path.join(args.out_dir, f"timestamps_{run}.csv")
    with open(ts_path, "w", newline="") as ts_file:
        ts_writer = csv.writer(ts_file)
        ts_writer.writerow(["frame_idx", "recv_unix_time"])
        idx = 0
        while True:
            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                if stop.is_set():
                    break
                continue
            if item is None:
                break
            recv_t, ego_jpeg, stereo_jpeg = item
            ego = _decode(ego_jpeg)
            if ego is None:
                continue
            ego_sink.write(ego)
            if stereo_sink is not None:
                stereo = _decode(stereo_jpeg)
                if stereo is not None:
                    stereo_sink.write(stereo)
            ts_writer.writerow([idx, f"{recv_t:.6f}"])
            idx += 1
    ego_sink.close()
    if stereo_sink is not None:
        stereo_sink.close()
    print(f"[recorder] timestamps saved to {ts_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="192.168.123.164",
                   help="robot IP running realsense_server.py")
    p.add_argument("--port", type=int, default=5559,
                   help="viewer PUB port (NOT the 5558 inference REP port)")
    p.add_argument("--out-dir", required=True,
                   help="output directory for ego.mp4 / timestamps.csv")
    p.add_argument("--fps", type=float, default=30.0,
                   help="nominal FPS stamped into the MP4 (server default 30)")
    p.add_argument("--record-stereo", action="store_true",
                   help="also record the stereo L|R stream to stereo.mp4")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    stop = threading.Event()

    def _on_signal(signum, _frame):
        print(f"[recorder] got signal {signum}, finalizing...")
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.setsockopt(zmq.RCVHWM, 4)
    sock.setsockopt(zmq.LINGER, 0)
    addr = f"tcp://{args.server}:{args.port}"
    sock.connect(addr)
    print(f"[recorder] subscribed to {addr}, out_dir={args.out_dir}")

    q: "queue.Queue" = queue.Queue(maxsize=120)
    wt = threading.Thread(target=writer_loop, args=(q, args, stop), daemon=True)
    wt.start()

    frames = 0
    dropped = 0
    last_report = time.time()
    try:
        while not stop.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue
            while len(parts) < 2:
                parts.append(b"")
            try:
                q.put_nowait((time.time(), parts[0], parts[1]))
                frames += 1
            except queue.Full:
                dropped += 1
            now = time.time()
            if now - last_report > 5.0:
                print(f"[recorder] {frames} frames queued"
                      + (f", {dropped} dropped (slow disk)" if dropped else ""))
                last_report = now
    finally:
        q.put(None)
        wt.join(timeout=10.0)
        sock.close()
        ctx.term()
        print(f"[recorder] done: {frames} frames, {dropped} dropped")


if __name__ == "__main__":
    main()
