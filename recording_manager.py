#!/usr/bin/env python3
"""
Recording Manager Thread (no-Pico variant)

Drop-in replacement for RecordingManagerThread when running without a Pico
controller (e.g. alongside run_publisher.py with a Realsense camera).

Keyboard controls (type key + Enter):
  s  →  Start a new recording episode
  q  →  Stop and save the current episode
  d  →  Stop and discard the current episode

Usage:
    from recording_manager import RecordingManagerThread

    rec = RecordingManagerThread(g1_ip="localhost")
    rec.start()
    ...
    rec.stop()
"""

import json
import socket
import threading
import time

G1_IP = "localhost"
G1_CMD_PORT = 9999


class RecordingManagerThread(threading.Thread):
    """
    Background thread that manages recording state and sends commands to G1,
    driven by keyboard input instead of a Pico controller.

    State machine:
      IDLE ──(s)──> RECORDING ──(q)──> IDLE  (saved)
                               ──(d)──> IDLE  (discarded)
    """

    def __init__(
        self,
        g1_ip=G1_IP,
        g1_cmd_port=G1_CMD_PORT,
    ):
        super().__init__(daemon=True)

        self.g1_ip = g1_ip
        self.g1_cmd_port = g1_cmd_port

        self._running = False

        self._phase = "IDLE"  # "IDLE" | "RECORDING"
        self._lock = threading.Lock()

        self._sock = None
        self._sock_lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def stop(self):
        self._running = False
        with self._sock_lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

    def get_status(self):
        with self._lock:
            return {"phase": self._phase}

    # ── Network helpers ────────────────────────────────────────────────────────

    def _connect_g1(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((self.g1_ip, self.g1_cmd_port))
            s.settimeout(None)
            with self._sock_lock:
                self._sock = s
            print(f"[RecordingManager] Connected to G1 {self.g1_ip}:{self.g1_cmd_port}")
            return True
        except Exception as e:
            print(f"[RecordingManager] G1 connect failed: {e}")
            return False

    def _send_cmd(self, cmd: str, payload: dict = None):
        msg_dict = {"cmd": cmd}
        if payload:
            msg_dict.update(payload)
        msg = (json.dumps(msg_dict) + "\n").encode()

        with self._sock_lock:
            sock = self._sock

        if sock is None:
            if not self._connect_g1():
                print(f"[RecordingManager] Could not send '{cmd}' - no connection.")
                return
            with self._sock_lock:
                sock = self._sock

        try:
            sock.sendall(msg)
            print(f"[RecordingManager] Sent: {cmd}")
        except Exception as e:
            print(f"[RecordingManager] Send error ({e}), reconnecting...")
            with self._sock_lock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
            self._connect_g1()

    # ── State machine actions ──────────────────────────────────────────────────

    def _do_start(self):
        session_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        print("\033[92m[RecordingManager] >>> START\033[0m")
        with self._lock:
            self._phase = "RECORDING"
        self._send_cmd("start", payload={"session_id": session_id})

    def _do_save(self):
        print("\033[92m[RecordingManager] >>> SAVE\033[0m")
        with self._lock:
            self._phase = "IDLE"
        self._send_cmd("save")

    def _do_discard(self):
        print("\033[91m[RecordingManager] >>> DISCARD\033[0m")
        with self._lock:
            self._phase = "IDLE"
        self._send_cmd("discard")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        print("[RecordingManager] Starting. Controls: s=start  q=save  d=discard")
        self._running = True
        self._connect_g1()

        while self._running:
            try:
                key = input().strip().lower()
            except EOFError:
                break
            except KeyboardInterrupt:
                break

            with self._lock:
                phase = self._phase

            if key == "s":
                if phase == "IDLE":
                    self._do_start()
                else:
                    print("[RecordingManager] Already recording — press q to save or d to discard first")

            elif key == "q":
                if phase == "RECORDING":
                    self._do_save()
                else:
                    print("[RecordingManager] Not currently recording")

            elif key == "d":
                if phase == "RECORDING":
                    self._do_discard()
                else:
                    print("[RecordingManager] Not currently recording")

            else:
                print(f"[RecordingManager] Unknown key '{key}' — use s / q / d")

        print("[RecordingManager] Stopped.")
