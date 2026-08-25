"""Fake G1 observation streams — lets the UNMODIFIED robot clients run off-robot.

Serves the two ZMQ endpoints the clients consume, byte-compatible with the real
robot stack (wire formats read off psix_rtc_sonic_wm_client.py):

  * camera REQ/REP (default :5558): request b"get_frame" -> multipart reply whose
    part[0] is a JPEG-encoded BGR frame (RSCamera decodes exactly that; pass
    --multipart 4 to emulate the ZedNeck 4-part reply, slot 0 used).
    Frames come from --episode-dir (data.json + synchronized recorded states),
    --frames-dir (sorted images, looped), or a synthetic moving pattern. Episode
    replay starts on the first camera request and holds the last frame by default.
  * robot-state PUB (default :5557, topic "g1_debug"): topic-prefixed msgpack dict
    at --state-hz with the keys the clients read: body_q_measured(29, gentle
    sinusoid), left_hand_q(7), right_hand_q(7), base_quat_measured(4, identity).

Fault injection for the §10.1 matrix — stdin commands (or POST-free, single char + Enter):
    pc / rc   pause / resume the CAMERA (REQ times out -> client's recovery path)
    ps / rs   pause / resume the STATE stream (staleness watchdogs / frozen-frame drop)
    p / r     pause / resume recorded replay
    nN / bN   step replay forward / backward N frames; g N = goto; 0 = reset
    q         quit

Optional localhost replay API (--control-port 5559): GET /state; POST
/pause, /resume, /reset, /next?frames=N, /prev?frames=N, /goto?frame=N.

Launch:
    python3 mock_g1_obs.py --bind-host 127.0.0.1 --camera-port 5558 \
      --state-port 5557 --control-port 5559 --episode-dir <recorded-episode>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import msgpack
import numpy as np
import zmq

_IMG_EXTS = (".jpg", ".jpeg", ".png")


class MockG1Obs:
    def __init__(self, *, camera_port: int, state_port: int, topic: str = "g1_debug",
                 frames_dir: str | None = None, episode_dir: str | None = None,
                 state_hz: float = 50.0, multipart: int = 1,
                 jpeg_quality: int = 90, bind_host: str = "*",
                 replay_fps: float = 30.0, replay_mode: str = "hold",
                 start_paused: bool = False, control_host: str = "127.0.0.1",
                 control_port: int = 0):
        if frames_dir and episode_dir:
            raise ValueError("--frames-dir and --episode-dir are mutually exclusive")
        if replay_mode not in ("hold", "loop"):
            raise ValueError("replay_mode must be hold or loop")
        self.ctx = zmq.Context()
        self.camera_port = camera_port
        self.state_port = state_port
        self.topic = topic.encode("utf-8")
        self.state_hz = float(state_hz)
        self.multipart = max(1, int(multipart))
        self.jpeg_quality = int(jpeg_quality)
        self.bind_host = str(bind_host)
        self.control_host = str(control_host)
        self.control_port = max(0, int(control_port))
        self.camera_paused = threading.Event()
        self.state_paused = threading.Event()
        self._stop = threading.Event()
        self._frame_idx = 0
        self._episode_records = None
        self._frames = None
        self._replay_lock = threading.RLock()
        self._replay_fps = max(0.01, float(replay_fps))
        self._replay_mode = replay_mode
        self._replay_anchor_idx = 0
        self._replay_anchor_at = None
        self._replay_paused = bool(start_paused)
        self._control_httpd = None
        if episode_dir:
            self._episode_records, self._frames = self._load_episode(episode_dir)
        else:
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

    def _load_episode(self, episode_dir):
        """Load synchronized camera/state samples from a recorded G1 episode.

        The recorder writes one ``data.json`` row per ``color/frame_*.jpg``.
        Images and states are selected from one shared replay clock so the VLA
        sees the same 43-D observation it would have seen at that video frame.
        """
        episode_dir = os.path.abspath(episode_dir)
        data_path = os.path.join(episode_dir, "data.json")
        try:
            with open(data_path) as fh:
                records = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"[mock-g1] cannot load episode data {data_path}: {exc}")
        if not isinstance(records, list) or not records:
            raise SystemExit(f"[mock-g1] episode data must be a non-empty list: {data_path}")

        paths = []
        for i, row in enumerate(records):
            if not isinstance(row, dict):
                raise SystemExit(f"[mock-g1] episode row {i} is not an object")
            rel = row.get("image") or f"color/frame_{i:06d}.jpg"
            path = rel if os.path.isabs(rel) else os.path.join(episode_dir, rel)
            if not os.path.isfile(path):
                raise SystemExit(f"[mock-g1] episode frame missing at row {i}: {path}")
            st = row.get("states") or {}
            qpos = np.asarray(st.get("qpos", []), dtype=np.float64).reshape(-1)
            hands = np.asarray(st.get("hand_joints", []), dtype=np.float64).reshape(-1)
            quat = np.asarray(st.get("quat", []), dtype=np.float64).reshape(-1)
            if qpos.shape != (29,) or hands.shape != (14,) or quat.shape != (4,):
                raise SystemExit(
                    f"[mock-g1] bad state shapes at row {i}: "
                    f"qpos={qpos.shape} hands={hands.shape} quat={quat.shape}")
            paths.append(path)
        print(f"[mock-g1] synchronized episode: {episode_dir} "
              f"({len(records)} samples @ {self._replay_fps:g} fps, "
              f"EOF={self._replay_mode})")
        return records, paths

    def _normalize_episode_index(self, idx: int) -> int:
        n = len(self._episode_records or [])
        if n <= 0:
            return 0
        if self._replay_mode == "loop":
            return int(idx) % n
        return max(0, min(int(idx), n - 1))

    def _episode_index(self, *, start: bool = False) -> int:
        with self._replay_lock:
            now = time.monotonic()
            if start and self._replay_anchor_at is None and not self._replay_paused:
                # Do not consume the recording while the production client is
                # still doing service health checks. First camera request is t=0.
                self._replay_anchor_at = now
            idx = self._replay_anchor_idx
            if self._replay_anchor_at is not None and not self._replay_paused:
                idx += int((now - self._replay_anchor_at) * self._replay_fps)
            return self._normalize_episode_index(idx)

    def replay_pause(self) -> int:
        with self._replay_lock:
            idx = self._episode_index()
            self._replay_anchor_idx = idx
            self._replay_anchor_at = None
            self._replay_paused = True
            return idx

    def replay_resume(self) -> int:
        with self._replay_lock:
            idx = self._episode_index()
            self._replay_anchor_idx = idx
            self._replay_anchor_at = time.monotonic()
            self._replay_paused = False
            return idx

    def replay_seek(self, idx: int, *, pause: bool | None = None) -> int:
        with self._replay_lock:
            idx = self._normalize_episode_index(idx)
            if pause is not None:
                self._replay_paused = bool(pause)
            self._replay_anchor_idx = idx
            self._replay_anchor_at = None if self._replay_paused else time.monotonic()
            return idx

    def replay_step(self, delta: int) -> int:
        with self._replay_lock:
            idx = self._episode_index()
            return self.replay_seek(idx + int(delta), pause=True)

    def replay_status(self):
        if self._episode_records is None:
            return {"episode": False, "frame": self._frame_idx,
                    "paused": False, "samples": len(self._frames or [])}
        with self._replay_lock:
            idx = self._episode_index()
            return {
                "episode": True,
                "frame": idx,
                "samples": len(self._episode_records),
                "paused": self._replay_paused,
                "started": self._replay_anchor_at is not None,
                "fps": self._replay_fps,
                "mode": self._replay_mode,
                "image": os.path.basename(self._frames[idx]),
            }

    def _next_frame_bgr(self) -> np.ndarray:
        if self._episode_records is not None:
            i = self._episode_index(start=True)
            img = cv2.imread(self._frames[i], cv2.IMREAD_COLOR)
            if img is not None:
                return img
        else:
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

    def _state_sample(self, t: float):
        if self._episode_records is not None:
            i = self._episode_index()
            st = self._episode_records[i]["states"]
            qpos = np.asarray(st["qpos"], dtype=np.float64).reshape(29)
            hands = np.asarray(st["hand_joints"], dtype=np.float64).reshape(14)
            quat = np.asarray(st["quat"], dtype=np.float64).reshape(4)
            return {
                "body_q_measured": qpos.tolist(),
                "left_hand_q": hands[:7].tolist(),
                "right_hand_q": hands[7:].tolist(),
                "base_quat_measured": quat.tolist(),
                "mock_replay_frame": i,
            }
        body = (0.05 * np.sin(t + np.arange(29))).astype(np.float64)
        return {
            "body_q_measured": body.tolist(),
            "left_hand_q": (0.02 * np.sin(t + np.arange(7))).tolist(),
            "right_hand_q": (0.02 * np.cos(t + np.arange(7))).tolist(),
            "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
        }

    # ------------------------------------------------------------- camera REP
    def _camera_loop(self):
        sock = self.ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://{self.bind_host}:{self.camera_port}")
        print(f"[mock-g1] camera REP on {self.bind_host}:{self.camera_port} "
              f"(multipart={self.multipart})")
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
                sock.bind(f"tcp://{self.bind_host}:{self.camera_port}")
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
        sock.bind(f"tcp://{self.bind_host}:{self.state_port}")
        print(f"[mock-g1] state PUB on {self.bind_host}:{self.state_port} "
              f"topic={self.topic.decode()} "
              f"@ {self.state_hz} Hz")
        period = 1.0 / self.state_hz
        t0 = time.monotonic()
        while not self._stop.is_set():
            if not self.state_paused.is_set():
                t = time.monotonic() - t0
                state = self._state_sample(t)
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
                if cmd == "s":
                    print(f"[mock-g1] replay {json.dumps(self.replay_status())}")
                    continue
                if self._episode_records is not None:
                    if cmd == "p":
                        print(f"[mock-g1] replay PAUSED at {self.replay_pause()}")
                        continue
                    if cmd == "r":
                        print(f"[mock-g1] replay resumed at {self.replay_resume()}")
                        continue
                    if cmd == "0":
                        print(f"[mock-g1] replay reset to {self.replay_seek(0, pause=True)}")
                        continue
                    if cmd.startswith("n"):
                        n = int(cmd[1:].strip() or "1")
                        print(f"[mock-g1] replay stepped to {self.replay_step(n)}")
                        continue
                    if cmd.startswith("b"):
                        n = int(cmd[1:].strip() or "1")
                        print(f"[mock-g1] replay stepped to {self.replay_step(-n)}")
                        continue
                    if cmd.startswith("g "):
                        print(f"[mock-g1] replay goto {self.replay_seek(int(cmd[2:]), pause=True)}")
                        continue
                if cmd in cmds:
                    fn, msg = cmds[cmd]
                    fn()
                    print(f"[mock-g1] {msg}")
        except Exception:
            pass

    def _control_loop(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _fmt, *_args):
                return

            def reply(self, status=200, **extra):
                obj = {"ok": status < 400, **owner.replay_status(), **extra}
                body = json.dumps(obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if urlparse(self.path).path != "/state":
                    self.reply(404, error="use GET /state")
                    return
                self.reply()

            def do_POST(self):
                if owner._episode_records is None:
                    self.reply(409, error="no --episode-dir replay is active")
                    return
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                try:
                    if parsed.path == "/pause":
                        owner.replay_pause()
                    elif parsed.path == "/resume":
                        owner.replay_resume()
                    elif parsed.path == "/reset":
                        owner.replay_seek(0, pause=True)
                    elif parsed.path == "/next":
                        owner.replay_step(int(query.get("frames", ["1"])[0]))
                    elif parsed.path == "/prev":
                        owner.replay_step(-int(query.get("frames", ["1"])[0]))
                    elif parsed.path == "/goto":
                        owner.replay_seek(int(query["frame"][0]), pause=True)
                    else:
                        self.reply(404, error="unknown control path")
                        return
                except (KeyError, ValueError) as exc:
                    self.reply(400, error=str(exc))
                    return
                self.reply()

        self._control_httpd = ThreadingHTTPServer(
            (self.control_host, self.control_port), Handler)
        self._control_httpd.daemon_threads = True
        print(f"[mock-g1] replay control http://{self.control_host}:"
              f"{self.control_port}/state")
        self._control_httpd.serve_forever(poll_interval=0.2)

    def run(self):
        threads = [threading.Thread(target=self._camera_loop, daemon=True),
                   threading.Thread(target=self._state_loop, daemon=True),
                   threading.Thread(target=self._stdin_loop, daemon=True)]
        if self.control_port:
            threads.append(threading.Thread(target=self._control_loop, daemon=True))
        for th in threads:
            th.start()
        print("[mock-g1] commands: pc/rc camera, ps/rs state, "
              "p/r replay, nN/bN/g N/0 seek, s status, q quit")
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        self._stop.set()
        if self._control_httpd is not None:
            self._control_httpd.shutdown()
            self._control_httpd.server_close()
        for th in threads:
            th.join(timeout=1.0)
        self.ctx.term()


def main():
    p = argparse.ArgumentParser(description="Fake G1 camera REQ/REP + state PUB streams")
    p.add_argument("--camera-port", type=int, default=5558)
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--topic", type=str, default="g1_debug")
    p.add_argument("--bind-host", default="*",
                   help="ZMQ bind host; use 127.0.0.1 for off-robot local tests")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--frames-dir", type=str, default=None,
                        help="replay sorted images from this dir (loop); default synthetic")
    source.add_argument("--episode-dir", type=str, default=None,
                        help="recorded episode with data.json + color/; synchronized image/state replay")
    p.add_argument("--replay-fps", type=float, default=30.0)
    p.add_argument("--replay-mode", choices=["hold", "loop"], default="hold")
    p.add_argument("--start-paused", action="store_true")
    p.add_argument("--control-host", default="127.0.0.1")
    p.add_argument("--control-port", type=int, default=0,
                   help="optional localhost HTTP pause/resume/seek API; 0 disables")
    p.add_argument("--state-hz", type=float, default=50.0)
    p.add_argument("--multipart", type=int, default=1,
                   help="camera reply parts (1=RS, 4=ZedNeck emulation)")
    p.add_argument("--jpeg-quality", type=int, default=90)
    args = p.parse_args()
    MockG1Obs(camera_port=args.camera_port, state_port=args.state_port,
              topic=args.topic, frames_dir=args.frames_dir,
              episode_dir=args.episode_dir, state_hz=args.state_hz,
              multipart=args.multipart, jpeg_quality=args.jpeg_quality,
              bind_host=args.bind_host, replay_fps=args.replay_fps,
              replay_mode=args.replay_mode, start_paused=args.start_paused,
              control_host=args.control_host, control_port=args.control_port).run()


if __name__ == "__main__":
    main()
