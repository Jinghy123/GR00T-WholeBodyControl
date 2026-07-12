"""Fake G1 observation streams — lets the UNMODIFIED robot clients run off-robot.

Serves the two ZMQ endpoints the clients consume, byte-compatible with the real
robot stack (wire formats read off psix_rtc_sonic_wm_client.py):

  * camera REQ/REP (default :5558): request b"get_frame" -> multipart reply whose
    part[0] is a JPEG-encoded BGR frame (RSCamera decodes exactly that; pass
    --multipart 4 to emulate the ZedNeck 4-part reply, slot 0 used).
    Frames come from --frames-dir (sorted images, looped) or a synthetic moving
    pattern with a frame counter burned in.
  * robot-state PUB (default :5557, topic "g1_debug"): topic-prefixed msgpack dict
    at --state-hz with the keys the clients read: body_q_measured(29, gentle
    sinusoid), left_hand_q(7), right_hand_q(7), base_quat_measured(4, identity).

Fault injection for the §10.1 matrix — stdin commands (or POST-free, single char + Enter):
    pc / rc   pause / resume the CAMERA (REQ times out -> client's recovery path)
    ps / rs   pause / resume the STATE stream (staleness watchdogs / frozen-frame drop)
    q         quit

Launch:
    python3 mock_g1_obs.py --camera-port 5558 --state-port 5557
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

_IMG_EXTS = (".jpg", ".jpeg", ".png")


class MockG1Obs:
    def __init__(self, *, camera_port: int, state_port: int, topic: str = "g1_debug",
                 frames_dir: str | None = None, state_hz: float = 50.0,
                 multipart: int = 1, jpeg_quality: int = 90):
        self.ctx = zmq.Context()
        self.camera_port = camera_port
        self.state_port = state_port
        self.topic = topic.encode("utf-8")
        self.state_hz = float(state_hz)
        self.multipart = max(1, int(multipart))
        self.jpeg_quality = int(jpeg_quality)
        self.camera_paused = threading.Event()
        self.state_paused = threading.Event()
        self._stop = threading.Event()
        self._frame_idx = 0
        self._frames = self._load_frames(frames_dir)

    # ------------------------------------------------------------- frames
    def _load_frames(self, frames_dir):
        if not frames_dir:
            return None
        paths = sorted(p for p in glob.glob(os.path.join(frames_dir, "*"))
                       if p.lower().endswith(_IMG_EXTS))
        if not paths:
            raise SystemExit(f"[mock-g1] no images under {frames_dir}")
        print(f"[mock-g1] replaying {len(paths)} frames from {frames_dir}")
        return paths

    def _next_frame_bgr(self) -> np.ndarray:
        i = self._frame_idx
        self._frame_idx += 1
        if self._frames is not None:
            img = cv2.imread(self._frames[i % len(self._frames)], cv2.IMREAD_COLOR)
            if img is not None:
                return img
        # synthetic: moving gradient + frame counter (640x480 like the RS cam)
        h, w = 480, 640
        base = (np.arange(w, dtype=np.int32)[None, :] * 255 // w + i * 3) % 256
        img = np.repeat(base[:, :, None], 3, axis=2).astype(np.uint8)
        img = np.repeat(img, h // img.shape[0] + 1, axis=0)[:h]
        cv2.putText(img, f"mock frame {i}", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return img

    # ------------------------------------------------------------- camera REP
    def _camera_loop(self):
        sock = self.ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://*:{self.camera_port}")
        print(f"[mock-g1] camera REP on :{self.camera_port} (multipart={self.multipart})")
        while not self._stop.is_set():
            try:
                _ = sock.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            if self.camera_paused.is_set():
                # a REQ client with a timeout treats a missing REP as a camera
                # fault and recovers its socket — emulate by never replying.
                # (REP must still be reset: close+rebind to clear the state machine.)
                sock.close(linger=0)
                sock = self.ctx.socket(zmq.REP)
                sock.setsockopt(zmq.RCVTIMEO, 200)
                sock.setsockopt(zmq.LINGER, 0)
                sock.bind(f"tcp://*:{self.camera_port}")
                continue
            okflag, buf = cv2.imencode(
                ".jpg", self._next_frame_bgr(),
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            payload = buf.tobytes() if okflag else b""
            parts = [payload] + [b""] * (self.multipart - 1)
            try:
                sock.send_multipart(parts)
            except zmq.ZMQError:
                break
        sock.close(linger=0)

    # ------------------------------------------------------------- state PUB
    def _state_loop(self):
        sock = self.ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://*:{self.state_port}")
        print(f"[mock-g1] state PUB on :{self.state_port} topic={self.topic.decode()} "
              f"@ {self.state_hz} Hz")
        period = 1.0 / self.state_hz
        t0 = time.monotonic()
        while not self._stop.is_set():
            if not self.state_paused.is_set():
                t = time.monotonic() - t0
                body = (0.05 * np.sin(t + np.arange(29))).astype(np.float64)
                state = {
                    "body_q_measured": body.tolist(),
                    "left_hand_q": (0.02 * np.sin(t + np.arange(7))).tolist(),
                    "right_hand_q": (0.02 * np.cos(t + np.arange(7))).tolist(),
                    "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
                }
                sock.send(self.topic + msgpack.packb(state, use_bin_type=True))
            time.sleep(period)
        sock.close(linger=0)

    # ------------------------------------------------------------- control
    def _stdin_loop(self):
        cmds = {"pc": (self.camera_paused.set, "camera PAUSED"),
                "rc": (self.camera_paused.clear, "camera resumed"),
                "ps": (self.state_paused.set, "state PAUSED"),
                "rs": (self.state_paused.clear, "state resumed")}
        try:
            for line in sys.stdin:
                cmd = line.strip().lower()
                if cmd == "q":
                    self._stop.set()
                    return
                if cmd in cmds:
                    fn, msg = cmds[cmd]
                    fn()
                    print(f"[mock-g1] {msg}")
        except Exception:
            pass

    def run(self):
        threads = [threading.Thread(target=self._camera_loop, daemon=True),
                   threading.Thread(target=self._state_loop, daemon=True),
                   threading.Thread(target=self._stdin_loop, daemon=True)]
        for th in threads:
            th.start()
        print("[mock-g1] commands: pc/rc = pause/resume camera, ps/rs = pause/resume state, q = quit")
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        self._stop.set()
        for th in threads[:2]:
            th.join(timeout=1.0)
        self.ctx.term()


def main():
    p = argparse.ArgumentParser(description="Fake G1 camera REQ/REP + state PUB streams")
    p.add_argument("--camera-port", type=int, default=5558)
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--topic", type=str, default="g1_debug")
    p.add_argument("--frames-dir", type=str, default=None,
                   help="replay sorted images from this dir (loop); default synthetic")
    p.add_argument("--state-hz", type=float, default=50.0)
    p.add_argument("--multipart", type=int, default=1,
                   help="camera reply parts (1=RS, 4=ZedNeck emulation)")
    p.add_argument("--jpeg-quality", type=int, default=90)
    args = p.parse_args()
    MockG1Obs(camera_port=args.camera_port, state_port=args.state_port,
              topic=args.topic, frames_dir=args.frames_dir, state_hz=args.state_hz,
              multipart=args.multipart, jpeg_quality=args.jpeg_quality).run()


if __name__ == "__main__":
    main()
