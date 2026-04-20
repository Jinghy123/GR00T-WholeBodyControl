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

# Freshness threshold (seconds) for tracking data.
# Default is disabled to avoid cross-machine clock skew issues.
TRACKING_MAX_AGE_S: Optional[float] = None

# 26D → 21D (MediaPipe) joint index mapping
# MediaPipe: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# 26D input: [Wrist, Palm, Thumb(4), Index(5), Middle(5), Ring(5), Pinky(5)]
_MEDIAPIPE_IDX = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]

# 26D hand joint names (aligned with humdex version)
HAND_JOINT_NAMES_26 = [
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"
]

# 26D -> 21D mapping to MediaPipe layout (aligned with humdex version)
MEDIAPIPE_MAPPING_26_TO_21 = [
    1,   # 0: Wrist -> Wrist
    2,   # 1: ThumbMetacarpal -> Thumb CMC
    3,   # 2: ThumbProximal -> Thumb MCP
    4,   # 3: ThumbDistal -> Thumb IP
    5,   # 4: ThumbTip -> Thumb Tip
    7,   # 5: IndexMetacarpal -> Index MCP
    8,   # 6: IndexProximal -> Index PIP
    9,   # 7: IndexIntermediate -> Index DIP
    10,  # 8: IndexTip -> Index Tip (skip IndexDistal)
    12,  # 9: MiddleMetacarpal -> Middle MCP
    13,  # 10: MiddleProximal -> Middle PIP
    14,  # 11: MiddleIntermediate -> Middle DIP
    15,  # 12: MiddleTip -> Middle Tip (skip MiddleDistal)
    17,  # 13: RingMetacarpal -> Ring MCP
    18,  # 14: RingProximal -> Ring PIP
    19,  # 15: RingIntermediate -> Ring DIP
    20,  # 16: RingTip -> Ring Tip (skip RingDistal)
    22,  # 17: LittleMetacarpal -> Pinky MCP
    23,  # 18: LittleProximal -> Pinky PIP
    24,  # 19: LittleIntermediate -> Pinky DIP
    25,  # 20: LittleTip -> Pinky Tip (skip LittleDistal)
]

# Joint names from wujihandpy (finger1..5, joint1..4)
_DESIRED_JOINT_NAMES = [f"finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def now_ms() -> int:
    return int(time.time() * 1000)


def hand_26d_to_mediapipe_21d(hand_dict: dict, hand_side: str) -> np.ndarray:
    """Convert 26D hand dict → (21, 3) MediaPipe-style array."""
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"

    # Build 26D position array.
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)

    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_dict:
            pos = hand_dict[key][0]  # [x, y, z]
            joint_positions_26[i] = pos
        else:
            # Fallback to zeros when a joint key is missing.
            joint_positions_26[i] = [0.0, 0.0, 0.0]

    # Remap to 21D MediaPipe order.
    mediapipe_21d = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21]

    # Use wrist as origin.
    wrist_pos = mediapipe_21d[0].copy()
    mediapipe_21d = mediapipe_21d - wrist_pos

    # Keep a dedicated scale hook for quick tuning if needed.
    scale_factor = 1.0
    mediapipe_21d[1:] = mediapipe_21d[1:] * scale_factor

    return mediapipe_21d


def smooth_move(hand, controller, target_qpos, duration=0.1, steps=10):
    """
    Smoothly interpolate from current qpos to target qpos (5x4).

    Args:
        hand: wujihandpy.Hand instance (kept for API compatibility)
        controller: realtime controller object
        target_qpos: numpy array with shape (5, 4)
        duration: interpolation duration in seconds
        steps: number of interpolation steps
    """
    target_qpos = target_qpos.reshape(5, 4)
    try:
        # cur = controller.get_joint_actual_position()
        cur = controller.read_joint_actual_position()
    except:
        cur = np.zeros((5, 4), dtype=np.float32)

    for t in np.linspace(0, 1, steps):
        q = cur * (1 - t) + target_qpos * t
        controller.set_joint_target_position(q)
        time.sleep(duration / steps)

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
        tracking_max_age_s: Optional[float] = TRACKING_MAX_AGE_S,
        enable_replay_commands: bool = False,
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
        self.tracking_max_age_s = tracking_max_age_s

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

        # Command receiver for external replay (optional, port 5561)
        # NOTE: Disabled by default to avoid blocking control loop
        # Enable with --enable-replay-commands flag if needed
        self._cmd_sock = None
        self._cmd_port = state_port + 1
        if enable_replay_commands:
            try:
                self._cmd_sock = self._ctx.socket(zmq.SUB)
                self._cmd_sock.setsockopt(zmq.RCVTIMEO, 10)  # Short timeout to avoid blocking
                self._cmd_sock.setsockopt_string(zmq.SUBSCRIBE, "wuji_replay")
                self._cmd_sock.bind(f"tcp://*:{self._cmd_port}")
                print(f"[WujiHand] Command receiver bound to port {self._cmd_port} (for replay)")
            except Exception as e:
                print(f"[WujiHand] Warning: could not bind command port {self._cmd_port}: {e}")
                self._cmd_sock = None

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
        if self.tracking_max_age_s is not None:
            age = time.time() - float(msg.get("timestamp", 0.0))
            if age > self.tracking_max_age_s:
                return None
        return msg.get(self.hand_side)

    def _recv_replay_command(self) -> Optional[np.ndarray]:
        """
        Try to receive a replay command. Returns 20D action array or None.
        Only processes commands matching self.hand_side.
        """
        if self._cmd_sock is None:
            return None
        try:
            raw = self._cmd_sock.recv()
        except zmq.Again:
            return None
        topic_len = len(b"wuji_replay")
        payload = raw[topic_len:]
        try:
            msg = msgpack.unpackb(payload, raw=False)
        except Exception:
            return None
        # Check if command is for this hand
        if msg.get("hand_side") != self.hand_side:
            return None
        action = msg.get("action")
        if action is not None:
            return np.asarray(action, dtype=np.float32).reshape(5, 4)
        return None

    # ── safety ────────────────────────────────────────────────────────────────

    def _apply_safety(self, qpos: np.ndarray) -> np.ndarray:
        """Apply clamp and per-step delta limit to model output."""
        q = np.asarray(qpos, dtype=np.float32).reshape(5, 4)
        if not np.isfinite(q).all():
            q = np.asarray(self.last_qpos if self.last_qpos is not None else self.zero_pose, dtype=np.float32).reshape(5, 4)
        q = np.clip(q, self.clamp_min, self.clamp_max)
        if self.last_qpos is not None and np.asarray(self.last_qpos).shape == q.shape:
            delta = q - self.last_qpos
            delta = np.clip(delta, -self.max_delta_per_step, self.max_delta_per_step)
            q = self.last_qpos + delta
        return q

    # ── hardware send ─────────────────────────────────────────────────────────

    def _send(self, target: np.ndarray):
        target = target.reshape(5, 4).astype(np.float32)
        if self.smooth_enabled:
            smooth_move(self.hand, self.controller, target, duration=self.control_dt, steps=self.smooth_steps)
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
                # Check for replay command first (higher priority than tracking)
                replay_action = self._recv_replay_command()
                if replay_action is not None:
                    # Directly send replay action to hardware
                    wuji_20d = self._apply_safety(replay_action)
                    self._send(wuji_20d)
                    self._publish_state(wuji_20d)
                    self.last_qpos = wuji_20d.copy()
                    self._rate_limit(loop_start)
                    continue

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
            smooth_move(self.hand, self.controller, self.zero_pose, duration=duration, steps=steps)
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
    parser.add_argument(
        "--tracking_max_age_s", type=float, default=-1.0,
        help=(
            "Discard tracking packets older than this (seconds). "
            "Set < 0 to disable (default: disabled)."
        ),
    )
    parser.add_argument(
        "--enable-replay-commands", action="store_true",
        help="Enable ZMQ command receiver for replay (may add slight latency)",
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
        tracking_max_age_s=None if args.tracking_max_age_s < 0 else args.tracking_max_age_s,
        enable_replay_commands=args.enable_replay_commands,
    )
    server.run()
