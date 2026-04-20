#!/usr/bin/env python3
"""
Wuji hand controller — runs on the machine where the Wuji hand is physically connected.

Data sources / sinks:
  Input  : ZMQ SUB  ← pico_manus_thread_server.py  (port 5559, topic "wuji_hand")
             msgpack {"left": <26D dict>, "right": <26D dict>, "timestamp": float}
  Output : ZMQ PUB  → g1_data_server.py             (port 5560, topic "wuji_state")
             msgpack {"hand_side": str, "action": [20], "state": [20], "timestamp": float}
  Hardware: wujihandpy via USB/serial

Keyboard controls (no Enter needed):
  k  → toggle follow / default (zero pose)
  p  → toggle follow / hold    (freeze last pose)
  Ctrl-C → graceful shutdown with ramp to zero

Setup:
  Laptop: python pico_manus_thread_server.py --manager --hand_type wuji
  Robot (or laptop): python wuji_hand_server.py --hand_side left  --serial_number xxx
                     python wuji_hand_server.py --hand_side right --serial_number yyy
  Laptop: python g1_data_server.py --hand-type wuji
"""

import argparse
import select
import signal
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Optional

import msgpack
import numpy as np
import zmq

# ── wuji-retargeting: look for it next to this file ─────────────────────────
_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE / "wuji-retargeting", _HERE / "wuji_retargeting"]:
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:
    from wuji_retargeting import Retargeter
    from wuji_retargeting.mediapipe import apply_mediapipe_transformations
except ImportError as _e:
    print(f"[ERROR] Cannot import wuji_retargeting: {_e}")
    print("        Install: cd wuji-retargeting && pip install -e .")
    sys.exit(1)

try:
    import wujihandpy
except ImportError:
    print("[ERROR] Missing dependency: wujihandpy")
    print("        Install with: pip install wujihandpy")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Freshness threshold: discard tracking data older than this
TRACKING_MAX_AGE_S = 0.5

# 26D → 21D (MediaPipe) joint index mapping
# MediaPipe: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# 26D input: [Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)]
_MEDIAPIPE_IDX = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]

# Joint names from wujihandpy (finger1..5, joint1..4)
_DESIRED_JOINT_NAMES = [f"finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def now_ms() -> int:
    return int(time.time() * 1000)


def hand_26d_to_mediapipe_21d(hand_dict: dict, hand_side: str) -> np.ndarray:
    """Convert 26D hand dict → (21, 3) MediaPipe-style array."""
    prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"
    joint_names_26 = [
        "Wrist", "Palm",
        "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
        "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
        "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
        "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
        "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip",
    ]
    pts26 = np.zeros((26, 3), dtype=np.float32)
    for i, name in enumerate(joint_names_26):
        key = prefix + name
        val = hand_dict.get(key)
        if val is not None:
            pts26[i] = np.asarray(val[0], dtype=np.float32)[:3]
    return pts26[_MEDIAPIPE_IDX]  # (21, 3)


def smooth_move(controller, target_qpos: np.ndarray, duration: float, steps: int):
    """Linearly interpolate from current position to target and send each step."""
    target = target_qpos.reshape(5, 4).astype(np.float32)
    try:
        cur = controller.read_joint_actual_position().reshape(5, 4).astype(np.float32)
    except Exception:
        cur = np.zeros((5, 4), dtype=np.float32)
    dt = duration / max(1, steps)
    for t in np.linspace(0, 1, steps):
        controller.set_joint_target_position(cur * (1.0 - t) + target * t)
        time.sleep(dt)

# ──────────────────────────────────────────────────────────────────────────────
# Main controller class
# ──────────────────────────────────────────────────────────────────────────────

class WujiHandServer:
    """
    ZMQ-driven Wuji hand controller.

    Subscribes to pico_manus_thread_server.py's 26D hand tracking topic,
    runs wuji-retargeting, and drives the Wuji hardware at target_fps Hz.
    Publishes measured state + action target on a ZMQ PUB socket so that
    g1_data_server.py can record them.
    """

    def __init__(
        self,
        hand_side: str = "left",
        tracking_host: str = "localhost",
        tracking_port: int = 5559,
        state_port: int = 5560,
        target_fps: int = 50,
        smooth_enabled: bool = True,
        smooth_steps: int = 5,
        serial_number: str = "",
        config_path: str = "",
        clamp_min: float = -1.5,
        clamp_max: float = 1.5,
        max_delta_per_step: float = 0.08,
    ):
        self.hand_side = hand_side.lower()
        assert self.hand_side in ("left", "right"), "hand_side must be 'left' or 'right'"
        self.control_dt = 1.0 / max(1, target_fps)
        self.target_fps = target_fps
        self.smooth_enabled = smooth_enabled
        self.smooth_steps = smooth_steps
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.max_delta_per_step = max_delta_per_step

        # Mode: "follow" | "hold" | "default"
        self._mode = "follow"
        self._mode_lock = threading.Lock()

        # ── ZMQ ──────────────────────────────────────────────────────────────
        self._ctx = zmq.Context()

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVTIMEO, 200)
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "wuji_hand")
        self._sub.connect(f"tcp://{tracking_host}:{tracking_port}")
        print(f"[WujiHand] Subscribed to tcp://{tracking_host}:{tracking_port} (topic wuji_hand)")

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://*:{state_port}")
        print(f"[WujiHand] State PUB bound to port {state_port}")

        # ── Wuji hardware ────────────────────────────────────────────────────
        print(f"[WujiHand] Initializing hardware ({self.hand_side})...")
        sn = (serial_number or "").strip()
        self.hand = wujihandpy.Hand(serial_number=sn) if sn else wujihandpy.Hand()
        self.hand.write_joint_enabled(True)
        self.controller = self.hand.realtime_controller(
            enable_upstream=True,
            filter=wujihandpy.filter.LowPass(cutoff_freq=10.0),
        )
        time.sleep(0.4)
        self.zero_pose = np.zeros_like(self.hand.read_joint_actual_position())
        print(f"[WujiHand] Hardware ready ({self.hand_side})")

        # ── wuji-retargeting ─────────────────────────────────────────────────
        if not config_path:
            # Auto-resolve config: wuji-retargeting/example/config/retarget_manus_<side>.yaml
            default_cfg = _HERE / "wuji-retargeting" / "example" / "config" / f"retarget_manus_{self.hand_side}.yaml"
            config_path = str(default_cfg)
        cfg = Path(config_path).expanduser().resolve()
        if not cfg.exists():
            raise FileNotFoundError(f"Retarget YAML not found: {cfg}")
        print(f"[WujiHand] Loading retargeter from {cfg}")
        self.retargeter = Retargeter.from_yaml(str(cfg), hand_side=self.hand_side)
        print("[WujiHand] Retargeter ready")

        # Precompute joint reordering (optimizer internal order → finger1..5 joint1..4)
        self._reorder_idx: Optional[np.ndarray] = None
        try:
            opt = getattr(self.retargeter, "optimizer", None)
            if opt is not None and hasattr(opt, "target_joint_names"):
                names = list(opt.target_joint_names)
                name2idx = {n: i for i, n in enumerate(names)}
                if all(n in name2idx for n in _DESIRED_JOINT_NAMES):
                    self._reorder_idx = np.array(
                        [name2idx[n] for n in _DESIRED_JOINT_NAMES], dtype=int
                    )
        except Exception:
            pass

        # Runtime state
        self.last_qpos = self.zero_pose.copy()
        self.running = True
        self._cleaned_up = False
        self._stop_signal: Optional[int] = None
        self._fps_start: Optional[float] = None
        self._fps_count = 0
        self._fps_interval = 100

    # ── mode helpers ──────────────────────────────────────────────────────────

    def _get_mode(self) -> str:
        with self._mode_lock:
            return self._mode

    def _set_mode(self, mode: str):
        with self._mode_lock:
            self._mode = mode
        print(f"[WujiHand] Mode → {mode}")

    # ── tracking data ─────────────────────────────────────────────────────────

    def _recv_tracking(self) -> Optional[dict]:
        """
        Try to receive one tracking message.  Returns the per-hand dict for
        self.hand_side, or None on timeout / stale data.
        """
        try:
            raw = self._sub.recv()
        except zmq.Again:
            return None
        topic_len = len(b"wuji_hand")
        payload = raw[topic_len:]
        try:
            msg = msgpack.unpackb(payload, raw=False)
        except Exception:
            return None
        age = time.time() - float(msg.get("timestamp", 0.0))
        if age > TRACKING_MAX_AGE_S:
            return None
        return msg.get(self.hand_side)

    # ── safety ────────────────────────────────────────────────────────────────

    def _apply_safety(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.clip(qpos, self.clamp_min, self.clamp_max)
        delta = qpos.reshape(-1) - self.last_qpos.reshape(-1)
        delta = np.clip(delta, -self.max_delta_per_step, self.max_delta_per_step)
        return (self.last_qpos.reshape(-1) + delta).reshape(5, 4).astype(np.float32)

    # ── hardware send ─────────────────────────────────────────────────────────

    def _send(self, target: np.ndarray):
        target = target.reshape(5, 4).astype(np.float32)
        if self.smooth_enabled:
            smooth_move(self.controller, target, self.control_dt, self.smooth_steps)
        else:
            self.controller.set_joint_target_position(target)

    # ── state publish ─────────────────────────────────────────────────────────

    def _publish_state(self, action: np.ndarray):
        try:
            state = self.controller.read_joint_actual_position().reshape(-1)
        except Exception:
            state = action.reshape(-1)
        msg = msgpack.packb({
            "hand_side": self.hand_side,
            "action": action.reshape(-1).tolist(),
            "state": state.tolist(),
            "timestamp": time.time(),
        }, use_bin_type=True)
        self._pub.send(b"wuji_state" + msg)

    # ── keyboard thread ───────────────────────────────────────────────────────

    def _keyboard_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self.running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch == "\x03":        # Ctrl-C
                        self.running = False
                        break
                    elif ch == "k":
                        # Toggle follow ↔ default
                        m = self._get_mode()
                        self._set_mode("follow" if m != "follow" else "default")
                    elif ch == "p":
                        # Toggle follow ↔ hold
                        m = self._get_mode()
                        self._set_mode("follow" if m != "follow" else "hold")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        sys.stdout.write("[WujiHand] Ready.\r\n")
        sys.stdout.write("  k = toggle follow/default\r\n")
        sys.stdout.write("  p = toggle follow/hold\r\n")
        sys.stdout.write("  Ctrl-C = quit\r\n\r\n")
        sys.stdout.flush()

        def _sig(signum, _frame):
            self._stop_signal = signum
            self.running = False

        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)

        kb = threading.Thread(target=self._keyboard_loop, daemon=True)
        kb.start()

        try:
            while self.running:
                loop_start = time.time()
                mode = self._get_mode()

                if mode in ("default", "hold"):
                    target = self.zero_pose if mode == "default" else self.last_qpos
                    if target is None:
                        target = self.zero_pose
                    self._send(target)
                    self._publish_state(target)
                    self._rate_limit(loop_start)
                    continue

                # mode == "follow"
                hand_dict = self._recv_tracking()
                if hand_dict is None:
                    self._rate_limit(loop_start)
                    continue

                try:
                    pts21 = hand_26d_to_mediapipe_21d(hand_dict, self.hand_side)
                    if not np.isfinite(pts21).all():
                        raise ValueError("non-finite in mediapipe_21d")

                    pts21_t = apply_mediapipe_transformations(pts21, hand_type=self.hand_side)
                    if not np.isfinite(pts21_t).all():
                        raise ValueError("non-finite after transformation")

                    qpos20 = np.asarray(
                        self.retargeter.retarget(pts21_t), dtype=np.float32
                    ).reshape(-1)

                    if self._reorder_idx is not None and qpos20.shape[0] >= int(self._reorder_idx.max() + 1):
                        wuji_20d = qpos20[self._reorder_idx].reshape(5, 4)
                    else:
                        wuji_20d = qpos20.reshape(5, 4)

                    wuji_20d = self._apply_safety(wuji_20d)
                    self._send(wuji_20d)
                    self._publish_state(wuji_20d)
                    self.last_qpos = wuji_20d.copy()

                    # FPS reporting
                    if self._fps_start is None:
                        self._fps_start = time.time()
                        self._fps_count = 0
                    self._fps_count += 1
                    if self._fps_count >= self._fps_interval:
                        elapsed = time.time() - self._fps_start
                        print(
                            f"[WujiHand] {self.hand_side} FPS: "
                            f"{self._fps_count / elapsed:.1f} Hz (target {self.target_fps})"
                        )
                        self._fps_start = time.time()
                        self._fps_count = 0

                except Exception as exc:
                    print(f"[WujiHand] Warning: {exc}")

                self._rate_limit(loop_start)

        finally:
            self._cleanup()

    def _rate_limit(self, loop_start: float):
        elapsed = time.time() - loop_start
        sleep_t = self.control_dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    def _cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        print("\n[WujiHand] Shutting down...")
        try:
            duration = 0.2 if self._stop_signal == signal.SIGTERM else 1.0
            steps = 10 if self._stop_signal == signal.SIGTERM else 50
            smooth_move(self.controller, self.zero_pose, duration, steps)
            print("[WujiHand] Returned to zero pose")
        except Exception:
            pass
        try:
            self.controller.close()
            self.hand.write_joint_enabled(False)
        except Exception:
            pass
        try:
            self._sub.close()
            self._pub.close()
            self._ctx.term()
        except Exception:
            pass
        print("[WujiHand] Done")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wuji hand controller: ZMQ hand tracking → retarget → hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (laptop, Wuji on laptop):
  python wuji_hand_server.py --hand_side left
  python wuji_hand_server.py --hand_side right --serial_number 337238793233

Examples (Wuji on G1, pico_manus on laptop at 192.168.1.10):
  python wuji_hand_server.py --hand_side left  --tracking_host 192.168.1.10
  python wuji_hand_server.py --hand_side right --tracking_host 192.168.1.10
""",
    )
    parser.add_argument(
        "--hand_side", type=str, default="left", choices=["left", "right"],
        help="Which hand to control (default: left)",
    )
    parser.add_argument(
        "--serial_number", type=str, default="",
        help="Wuji device serial number (leave empty to use first found)",
    )
    parser.add_argument(
        "--config", type=str, default="",
        help="Path to retarget YAML (default: wuji-retargeting/example/config/retarget_manus_<side>.yaml)",
    )
    parser.add_argument(
        "--tracking_host", type=str, default="localhost",
        help="Host running pico_manus_thread_server.py (default: localhost; set to laptop IP when Wuji is on G1)",
    )
    parser.add_argument(
        "--tracking_port", type=int, default=5559,
        help="ZMQ SUB port for hand tracking data (default: 5559)",
    )
    parser.add_argument(
        "--state_port", type=int, default=5560,
        help="ZMQ PUB port for state output (default: 5560)",
    )
    parser.add_argument(
        "--target_fps", type=int, default=50,
        help="Control loop target FPS (default: 50)",
    )
    parser.add_argument(
        "--no_smooth", action="store_true",
        help="Disable joint smoothing interpolation",
    )
    parser.add_argument(
        "--smooth_steps", type=int, default=5,
        help="Number of interpolation steps for smooth move (default: 5)",
    )
    parser.add_argument(
        "--clamp_min", type=float, default=-1.5,
        help="Minimum joint angle clamp (default: -1.5 rad)",
    )
    parser.add_argument(
        "--clamp_max", type=float, default=1.5,
        help="Maximum joint angle clamp (default: 1.5 rad)",
    )
    parser.add_argument(
        "--max_delta_per_step", type=float, default=0.08,
        help="Maximum joint angle change per control step (default: 0.08 rad)",
    )
    args = parser.parse_args()

    server = WujiHandServer(
        hand_side=args.hand_side,
        tracking_host=args.tracking_host,
        tracking_port=args.tracking_port,
        state_port=args.state_port,
        target_fps=args.target_fps,
        smooth_enabled=not args.no_smooth,
        smooth_steps=args.smooth_steps,
        serial_number=args.serial_number,
        config_path=args.config,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        max_delta_per_step=args.max_delta_per_step,
    )
    server.run()
