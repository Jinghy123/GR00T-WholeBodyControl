#!/usr/bin/env python3
"""
G1 Data Collector — runs on laptop.

Data sources:
  RGB frames  : ZMQ REQ → realsense_server.py on robot  (port 5558)
  Robot state : ZMQ SUB → g1_deploy_onnx_ref output pub  (port 5557)

Keyboard controls (single key, no Enter needed):
  s  → start recording a new episode
  q  → stop recording and save
  d  → stop recording and discard
  Ctrl-C → quit

Setup:
  Robot : realsense_server.py  (change its bind port to 5558)
  Robot : g1_deploy_onnx_ref   (needs --output-type zmq or --output-type all)
  Laptop: python g1_data_server.py
"""

import datetime
import json
import os
import queue
import select
import shutil
import socket
import sys
import termios
import threading
import time
import tty

import cv2
import msgpack
import numpy as np
import zmq

from episode_writer import EpisodeWriter

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

REALSENSE_HOST = "192.168.123.164"  # Robot IP for RealSense server
REALSENSE_PORT = 5558               # Changed from 5556 to avoid collision
                                    # NOTE: also update realsense_server.py bind port to 5558

WBC_HOST  = "localhost"   # Host running g1_deploy_onnx_ref (often same machine)
WBC_PORT  = 5557          # deploy ZMQ state publisher port
WBC_TOPIC = "g1_debug"

DATA_FOLDER = os.path.expanduser("~/data")
TASK_NAME   = "demonstration"
FPS         = 30

# ──────────────────────────────────────────────────────────────────────────────
# Quaternion helpers  [w, x, y, z] convention
# ──────────────────────────────────────────────────────────────────────────────

def _quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# RealSense client  (ZMQ REQ → realsense_server.py)
# ──────────────────────────────────────────────────────────────────────────────

class RealSenseClient:
    """Fetches JPEG-encoded RGB frames from realsense_server.py on the robot."""

    def __init__(self, host=REALSENSE_HOST, port=REALSENSE_PORT):
        self._host = host
        self._port = port
        self._ctx  = zmq.Context()
        self._sock = None
        self._connect()

    def _connect(self):
        if self._sock is not None:
            self._sock.close()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)   # 1 s timeout
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{self._host}:{self._port}")

    def get_frame(self):
        """
        Request one frame. Returns BGR uint8 ndarray (H, W, 3) or None.
        realsense_server sends [rgb_jpeg, ir_jpeg, fake_depth] — we only use rgb.
        """
        try:
            self._sock.send(b"get_frame")
            parts = self._sock.recv_multipart()
            if not parts or parts[0] == b"":
                return None
            arr = np.frombuffer(parts[0], dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except zmq.Again:
            print("[RealSense] Frame timeout — reconnecting...")
            self._connect()
            return None
        except Exception as e:
            print(f"[RealSense] Error: {e}")
            return None

    def close(self):
        self._sock.close()
        self._ctx.term()

# ──────────────────────────────────────────────────────────────────────────────
# WBC state subscriber  (ZMQ SUB → g1_deploy_onnx_ref port 5557)
# ──────────────────────────────────────────────────────────────────────────────

class WBCStateReader:
    """
    Background-thread subscriber to the deploy's ZMQ state publisher.

    Provides get_state() → {"qpos": ndarray(29,), "quat": ndarray(4,)}
    where quat is relative to the first frame after the last reset_ref() call,
    matching the format recorded by the original g1_data_server.py.
    """

    def __init__(self, host=WBC_HOST, port=WBC_PORT, topic=WBC_TOPIC):
        self._topic_bytes = topic.encode()
        self._topic_len   = len(self._topic_bytes)

        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, self._topic_bytes)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt(zmq.RCVHWM, 1)       # always get latest
        self._sock.connect(f"tcp://{host}:{port}")

        self._lock     = threading.Lock()
        self._latest   = None     # {"qpos": ..., "quat": ...}  (absolute)
        self._ref_quat = None     # reference quat for relative computation
        self._stop     = threading.Event()

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[WBCState] Subscribed to {host}:{port}  topic={topic}")

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                raw     = self._sock.recv()
                payload = raw[self._topic_len:]
                data    = msgpack.unpackb(payload, raw=False)

                # measured: real robot state from IMU + encoders
                qpos = np.array(data["body_q_measured"],    dtype=np.float32)
                quat = np.array(data["base_quat_measured"], dtype=np.float32)

                # hand states (measured)
                left_hand_q = np.array(data.get("left_hand_q_measured", [0]*7), dtype=np.float32)
                right_hand_q = np.array(data.get("right_hand_q_measured", [0]*7), dtype=np.float32)

                # action: actual motor q_target sent to robot this tick
                qpos_action = np.array(data["body_q_action"], dtype=np.float32)

                # hand actions
                left_hand_action = np.array(data.get("last_left_hand_action", [0]*7), dtype=np.float32)
                right_hand_action = np.array(data.get("last_right_hand_action", [0]*7), dtype=np.float32)

                # token: encoder output (latent vector that generated this action)
                token = np.array(data.get("token_state", []), dtype=np.float32)

                with self._lock:
                    self._latest = {
                        "qpos": qpos, "quat": quat,
                        "qpos_action": qpos_action,
                        "token": token,
                        "left_hand_q": left_hand_q,
                        "right_hand_q": right_hand_q,
                        "left_hand_action": left_hand_action,
                        "right_hand_action": right_hand_action,
                    }
            except zmq.Again:
                pass
            except Exception as e:
                print(f"[WBCState] Recv error: {e}")

    def reset_ref(self):
        """Reset the quaternion reference frame (call at episode start)."""
        with self._lock:
            self._ref_quat = None

    def get_state_and_action(self):
        """
        Returns (states, actions, token) where:
          states  = {"qpos": ndarray(29,), "quat": ndarray(4,), "hand_joints": ndarray(14,)} -- measured
          actions = {"qpos": ndarray(29,), "hand_joints": ndarray(14,)}                    -- motor q_target
          token   = ndarray(N,) or empty array                                            -- encoder output
        quat in states is relative to the ref_quat set at reset_ref().
        Returns (None, None, None) if no data yet.
        """
        with self._lock:
            if self._latest is None:
                return None, None, None

            qpos_meas   = self._latest["qpos"].copy()
            q_meas      = self._latest["quat"].copy()
            qpos_action = self._latest["qpos_action"].copy()
            token       = self._latest["token"].copy()

            # hand states (measured)
            left_hand_q   = self._latest["left_hand_q"].copy()
            right_hand_q  = self._latest["right_hand_q"].copy()
            hand_q_state   = np.concatenate([left_hand_q, right_hand_q])  # [14,]

            # hand actions
            left_hand_action  = self._latest["left_hand_action"].copy()
            right_hand_action = self._latest["right_hand_action"].copy()
            hand_q_action    = np.concatenate([left_hand_action, right_hand_action])  # [14,]

            # Set ref on first call after reset
            if self._ref_quat is None:
                self._ref_quat = q_meas.copy()
                print(f"[WBCState] Reference quat set: {self._ref_quat}")

            q_rel_meas = _quat_mul(_quat_inv(self._ref_quat), q_meas)

            states  = {"qpos": qpos_meas, "quat": q_rel_meas, "hand_joints": hand_q_state}
            actions = {"qpos": qpos_action, "hand_joints": hand_q_action}
            return states, actions, token

    def close(self):
        self._stop.set()
        self._sock.close()
        self._ctx.term()

# ──────────────────────────────────────────────────────────────────────────────
# TCP CommandServer  (localhost:9999, accepts JSON from RecordingManagerThread)
# ──────────────────────────────────────────────────────────────────────────────

CMD_HOST = "localhost"
CMD_PORT = 9999

class CommandServer:
    """
    Tiny TCP server that accepts newline-delimited JSON commands from
    RecordingManagerThreadSAM (run_publisher.py).

    Protocol: {"cmd": "start", "session_id": "..."} | {"cmd": "save"} | {"cmd": "discard"}

    Each accepted connection is handled in its own thread; the server
    itself runs in a daemon thread.
    """

    def __init__(self, cmd_queue: queue.Queue, host=CMD_HOST, port=CMD_PORT):
        self._queue = cmd_queue
        self._host  = host
        self._port  = port
        self._stop  = threading.Event()
        self._srv   = None
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self._host, self._port))
        self._srv.listen(5)
        self._srv.settimeout(1.0)
        self._thread.start()
        print(f"[CommandServer] Listening on {self._host}:{self._port}")

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        buf = b""
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        cmd = msg.get("cmd", "")
                        mapping = {"start": "s", "save": "q", "discard": "d"}
                        if cmd in mapping:
                            self._queue.put(mapping[cmd])
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            conn.close()

    def close(self):
        self._stop.set()
        if self._srv:
            self._srv.close()


# ──────────────────────────────────────────────────────────────────────────────
# DataCollector — main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def _rprint(msg):
    """Print with \r\n so output looks correct while terminal is in raw mode."""
    sys.stdout.write(msg + "\r\n")
    sys.stdout.flush()

class DataCollector:
    """
    Fetches frames + state at ~30 Hz and writes them via EpisodeWriter.

    Keyboard input is handled in a background thread (raw mode, no Enter needed):
      s → start   q → save   d → discard   Ctrl-C → quit
    """

    def __init__(
        self,
        camera: RealSenseClient,
        state_reader: WBCStateReader,
        data_folder: str = DATA_FOLDER,
        task_name: str   = TASK_NAME,
        frequency: int   = FPS,
    ):
        self._camera   = camera
        self._state    = state_reader
        self._folder   = data_folder
        self._task     = task_name
        self._freq     = frequency
        self._dt       = 1.0 / frequency

        self._ep_writer  = None
        self._ep_dir     = None
        self._ep_idx     = 0
        self._phase      = "IDLE"     # "IDLE" | "RECORDING"
        self._session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self._cmd_queue = queue.Queue()
        self._stop      = threading.Event()
        self._cmd_server = CommandServer(self._cmd_queue)

    # ── episode lifecycle ──────────────────────────────────────────────────────

    def _start(self):
        if self._phase != "IDLE":
            _rprint("[Collector] Already recording. Press q to save or d to discard first.")
            return
        ep_dir = os.path.join(
            self._folder,
            f"{self._task}_{self._session_id}",
            f"episode_{self._ep_idx}",
        )
        self._ep_dir    = ep_dir
        self._ep_writer = EpisodeWriter(
            task_dir    = ep_dir,
            date        = self._session_id,
            episode_num = self._ep_idx,
            task        = self._task,
            frequency   = self._freq,
        )
        self._ep_writer.create_episode()
        self._state.reset_ref()
        self._phase = "RECORDING"
        _rprint(f"\033[92m[Collector] START episode {self._ep_idx} → {ep_dir}\033[0m")

    def _save(self):
        if self._phase != "RECORDING":
            _rprint("[Collector] Not recording.")
            return
        self._phase = "IDLE"
        self._ep_writer.save_episode()
        self._ep_writer.close()
        self._ep_writer = None
        _rprint(f"\033[92m[Collector] SAVED episode {self._ep_idx}\033[0m")
        self._ep_idx += 1

    def _discard(self):
        if self._phase != "RECORDING":
            _rprint("[Collector] Not recording.")
            return
        self._phase = "IDLE"
        ep_dir = self._ep_dir
        self._ep_writer.close()
        self._ep_writer = None
        if ep_dir and os.path.exists(ep_dir):
            shutil.rmtree(ep_dir, ignore_errors=True)
        _rprint(f"\033[91m[Collector] DISCARDED episode {self._ep_idx}\033[0m")

    # ── keyboard thread ────────────────────────────────────────────────────────

    def _keyboard_loop(self):
        fd           = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not self._stop.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch == "\x03":        # Ctrl-C
                        self._stop.set()
                        break
                    if ch in ("s", "q", "d"):
                        self._cmd_queue.put(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # ── main loop ──────────────────────────────────────────────────────────────

    def run(self):
        _rprint("[Collector] Ready.")
        _rprint("  s = start recording")
        _rprint("  q = stop and save")
        _rprint("  d = stop and discard")
        _rprint("  Ctrl-C = quit")
        _rprint("")

        self._cmd_server.start()
        kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        kb_thread.start()

        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()

                # Process any keyboard commands (executed in main thread, no lock needed)
                while not self._cmd_queue.empty():
                    ch = self._cmd_queue.get_nowait()
                    if ch == "s":
                        self._start()
                    elif ch == "q":
                        self._save()
                    elif ch == "d":
                        self._discard()

                if self._phase == "RECORDING":
                    # Fetch frame and state
                    frame = self._camera.get_frame()
                    state, action, token = self._state.get_state_and_action()

                    # Record if we have state (with or without RGB frame)
                    if state is not None:
                        colors = {"rgb": frame} if frame is not None else None
                        self._ep_writer.add_item(
                            colors=colors,
                            states=state,
                            actions=action,
                            token=token,
                        )
                    else:
                        _rprint(f"[Collector] Warning: missing state, skipping frame.")

                    # get_frame() already blocks ~33 ms, so sleep is usually near zero
                    elapsed = time.perf_counter() - t0
                    sleep_t = self._dt - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                else:
                    time.sleep(self._dt)

        finally:
            self._cmd_server.close()
            if self._phase == "RECORDING":
                _rprint("[Collector] Saving episode on exit...")
                self._save()
            _rprint("[Collector] Done.")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    camera       = RealSenseClient()
    state_reader = WBCStateReader()
    collector    = DataCollector(camera, state_reader)

    try:
        collector.run()
    finally:
        camera.close()
        state_reader.close()
