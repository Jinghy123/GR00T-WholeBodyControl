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
import logging
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

def _rprint(msg):
    """Print with \\r\\n for raw mode terminal."""
    sys.stdout.write(msg + "\r\n")
    sys.stdout.flush()

REALSENSE_HOST = "192.168.123.164"  # Robot IP for RealSense server
REALSENSE_PORT = 5558               # Changed from 5556 to avoid collision
                                    # NOTE: also update realsense_server.py bind port to 5558

WBC_HOST  = "localhost"   # Host running g1_deploy_onnx_ref (often same machine)
WBC_PORT  = 5557          # deploy ZMQ state publisher port
WBC_TOPIC = "g1_debug"

WUJI_STATE_HOST = "localhost"  # Host running wuji_hand_server.py
WUJI_STATE_PORT = 5560         # ZMQ PUB port for wuji measured state / action

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
    """Fetches JPEG-encoded frames from realsense_server.py on the robot.

    Server reply is 4-part multipart:
      [ego_rgb_jpeg, ego_stereo_jpeg, left_wrist_jpeg, right_wrist_jpeg]
    Any slot may be b"" when that camera is unavailable.
    """

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

    @staticmethod
    def _decode(jpeg_bytes):
        if not jpeg_bytes:
            return None
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def get_frames(self):
        """
        Request one frame. Returns (ego_rgb, ego_stereo, left_wrist, right_wrist)
        — each BGR uint8 ndarray (H, W, 3) or None if unavailable.
        On timeout / error, reconnects and returns (None, None, None, None).
        """
        try:
            self._sock.send(b"get_frame")
            parts = self._sock.recv_multipart()
            # Pad to 4 parts defensively.
            while len(parts) < 4:
                parts.append(b"")
            return (
                self._decode(parts[0]),
                self._decode(parts[1]),
                self._decode(parts[2]),
                self._decode(parts[3]),
            )
        except zmq.Again:
            sys.stdout.write("[RealSense] Frame timeout — reconnecting...\r\n")
            self._connect()
            return (None, None, None, None)
        except Exception as e:
            print(f"[RealSense] Error: {e}")
            return (None, None, None, None)

    def close(self):
        self._sock.close()
        self._ctx.term()

# ──────────────────────────────────────────────────────────────────────────────
# Neck-action subscriber  (ZMQ SUB → pose_publisher.py port 5559)
# ──────────────────────────────────────────────────────────────────────────────


class NeckActionReader:
    """Subscribes to pose_publisher.py's `[neck_yaw, neck_pitch]` stream.

    Same socket the G1 NeckMotor consumes — connect a second SUB so the
    recorder logs the exact value being sent to the motors. CONFLATE keeps
    only the newest sample.
    """

    def __init__(self, addr: str):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(addr)
        self._latest = None

    def get_latest(self):
        """Drain pending messages and return [yaw, pitch] in radians, or None."""
        try:
            raw = self._sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return self._latest
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._latest
        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
            self._latest = [float(msg[0]), float(msg[1])]
        return self._latest

    def close(self):
        try:
            self._sock.close(linger=0)
        except Exception:
            pass


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
# Wuji hand state subscriber  (ZMQ SUB ← wuji_hand_server.py port 5560)
# ──────────────────────────────────────────────────────────────────────────────

class WujiHandStateReader:
    """
    Background-thread subscriber to wuji_hand_server.py's state publisher.

    Provides get_wuji_state() → {"left_action": ndarray(20,), "right_action": ndarray(20,),
                                  "left_state":  ndarray(20,), "right_state":  ndarray(20,)}
    Returns None if no data has been received yet.
    """

    def __init__(self, host=WUJI_STATE_HOST, port=WUJI_STATE_PORT):
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "wuji_state")
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.connect(f"tcp://{host}:{port}")

        self._lock   = threading.Lock()
        self._latest = {
            "left_action":  np.zeros(20, dtype=np.float32),
            "right_action": np.zeros(20, dtype=np.float32),
            "left_state":   np.zeros(20, dtype=np.float32),
            "right_state":  np.zeros(20, dtype=np.float32),
        }
        self._has_data = False
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[WujiState] Subscribed to {host}:{port}")

    def _recv_loop(self):
        topic_len = len(b"wuji_state")
        while not self._stop.is_set():
            try:
                raw     = self._sock.recv()
                payload = raw[topic_len:]
                msg     = msgpack.unpackb(payload, raw=False)
                side    = str(msg.get("hand_side", "left")).lower()
                action  = np.array(msg["action"], dtype=np.float32)
                state   = np.array(msg["state"],  dtype=np.float32)
                with self._lock:
                    self._latest[f"{side}_action"] = action
                    self._latest[f"{side}_state"]  = state
                    self._has_data = True
            except zmq.Again:
                pass
            except Exception as e:
                print(f"[WujiState] Recv error: {e}")

    def get_wuji_state(self):
        """Returns dict with left/right action & state arrays (20-D each), or None."""
        with self._lock:
            if not self._has_data:
                return None
            return {k: v.copy() for k, v in self._latest.items()}

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
                            # Pass both command and session_id (for start command)
                            self._queue.put((mapping[cmd], msg.get("session_id", None)))
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
# ────────────────────────────────────────────���─────────────────────────────────

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
        hand_type: str   = "dex3",
        wuji_reader: "WujiHandStateReader | None" = None,
        neck_reader: "NeckActionReader | None" = None,
    ):
        self._camera      = camera
        self._state       = state_reader
        self._folder      = data_folder
        self._task        = task_name
        self._freq        = frequency
        self._dt          = 1.0 / frequency
        self._hand_type   = hand_type.lower()
        self._wuji_reader = wuji_reader
        self._neck_reader = neck_reader

        self._ep_writer  = None
        self._ep_dir     = None
        self._ep_idx     = 0
        self._phase      = "IDLE"     # "IDLE" | "RECORDING"
        self._session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self._cmd_queue = queue.Queue()
        self._stop      = threading.Event()
        self._cmd_server = CommandServer(self._cmd_queue)

    # ── episode lifecycle ───────────────────��──────────────────────────────────

    def _start(self, session_id=None):
        if self._phase != "IDLE":
            _rprint("[Collector] Already recording. Press q to save or d to discard first.")
            return
        # Use provided session_id, or generate only on first episode
        if session_id is not None:
            self._session_id = session_id
        elif self._session_id is None:
            # First episode: generate session_id from current timestamp
            self._session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Subsequent episodes: reuse existing self._session_id
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
                    cmd_item = self._cmd_queue.get_nowait()
                    # Handle both old format (string) and new format (tuple)
                    if isinstance(cmd_item, tuple):
                        ch, session_id = cmd_item
                    else:
                        ch, session_id = cmd_item, None
                    if ch == "s":
                        self._start(session_id=session_id)
                    elif ch == "q":
                        self._save()
                    elif ch == "d":
                        self._discard()

                if self._phase == "RECORDING":
                    # Fetch frames (ego_rgb, ego_stereo, left_wrist, right_wrist) and state
                    if self._camera is not None:
                        ego, ego_stereo, lw, rw = self._camera.get_frames()
                    else:
                        ego, ego_stereo, lw, rw = None, None, None, None
                    state, action, token = self._state.get_state_and_action()

                    # Record if we have body state OR if we're recording hand-only data
                    has_hand_data = (self._hand_type == "wuji" and self._wuji_reader is not None)

                    if state is not None or has_hand_data:
                        colors = {}
                        if ego is not None:        colors["rgb"]         = ego
                        if ego_stereo is not None: colors["stereo"]      = ego_stereo
                        if lw is not None:         colors["left_wrist"]  = lw
                        if rw is not None:         colors["right_wrist"] = rw

                        # If no body state, create minimal state/action with hand data
                        if state is None:
                            state = {"qpos": np.zeros(29, dtype=np.float32),
                                     "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}
                            action = {"qpos": np.zeros(29, dtype=np.float32)}
                            token = np.array([], dtype=np.float32)

                        if self._hand_type == "wuji" and self._wuji_reader is not None:
                            # Replace 14-D Dex3 hand_joints with 40-D Wuji hand joints
                            wuji = self._wuji_reader.get_wuji_state()
                            if wuji is not None:
                                state["hand_joints"]  = np.concatenate(
                                    [wuji["left_state"],  wuji["right_state"]]
                                )
                                action["hand_joints"] = np.concatenate(
                                    [wuji["left_action"], wuji["right_action"]]
                                )
                            else:
                                state["hand_joints"]  = np.zeros(40, dtype=np.float32)
                                action["hand_joints"] = np.zeros(40, dtype=np.float32)

                        # Neck command (yaw, pitch in radians) — same value the
                        # G1 NeckMotor is being driven by. Recorded alongside
                        # body action so replay/analysis can reproduce head pose.
                        if self._neck_reader is not None:
                            neck_cmd = self._neck_reader.get_latest()
                            action["neck"] = (
                                np.asarray(neck_cmd, dtype=np.float32)
                                if neck_cmd is not None
                                else np.zeros(2, dtype=np.float32)
                            )

                        self._ep_writer.add_item(
                            colors=colors or None,
                            states=state,
                            actions=action,
                            token=token,
                        )
                    else:
                        _rprint(f"[Collector] Warning: missing state, skipping frame.")

                    # get_frames() already blocks ~33 ms, so sleep is usually near zero
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
    import argparse

    ap = argparse.ArgumentParser(description="G1 data collector")
    ap.add_argument("--hand-type", type=str, default="dex3", choices=["dex3", "wuji"],
                    help="Hand type: 'dex3' (default) or 'wuji'")
    ap.add_argument("--wuji-state-host", type=str, default=WUJI_STATE_HOST,
                    help=f"Host running wuji_hand_server.py (default: {WUJI_STATE_HOST}; set to G1 IP when Wuji is on robot)")
    ap.add_argument("--wuji-state-port", type=int, default=WUJI_STATE_PORT,
                    help=f"ZMQ SUB port for wuji state (default: {WUJI_STATE_PORT})")
    ap.add_argument("--no-camera", action="store_true",
                    help="Disable camera (useful for hand-only recording)")
    ap.add_argument("--neck-zmq", type=str, default="",
                    help=(
                        "ZMQ SUB address of pose_publisher.py "
                        "(e.g. tcp://<desktop-ip>:5559). When set, the "
                        "[neck_yaw, neck_pitch] command is recorded into "
                        "data.json under actions['neck']."
                    ))
    args = ap.parse_args()

    camera       = None if args.no_camera else RealSenseClient()
    state_reader = WBCStateReader()
    wuji_reader  = None
    if args.hand_type == "wuji":
        wuji_reader = WujiHandStateReader(host=args.wuji_state_host, port=args.wuji_state_port)
    neck_reader  = NeckActionReader(args.neck_zmq) if args.neck_zmq else None

    collector = DataCollector(
        camera,
        state_reader,
        hand_type   = args.hand_type,
        wuji_reader = wuji_reader,
        neck_reader = neck_reader,
    )

    try:
        collector.run()
    finally:
        if camera is not None:
            camera.close()
        state_reader.close()
        if wuji_reader is not None:
            wuji_reader.close()
        if neck_reader is not None:
            neck_reader.close()
