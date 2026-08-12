#!/usr/bin/env python3
"""
replay.py

Replay a recorded episode by reading data.json and publishing joint positions
to the WBC controller via ZMQ at the original recording frequency (30 Hz).

The message format mirrors WBCBridge.publish_joints() in psi-inference_rtc.py:
  - joint_pos  (1, 29) float32  ← qpos from recording
  - joint_vel  (1, 29) float32  ← zeros
  - body_quat  (1,  4) float32  ← recorded quat, sent as-is (sign-continuity corrected)
  - frame_index (1,)   int64

Usage:
    python replay.py
    python replay.py --episode-dir <path>
"""

# ── Hard-coded defaults (edit these instead of passing CLI args) ───────────────
DEFAULT_EPISODE_DIR    = "/home/hongyi/data/real/pick_place/pick_place_1/episode_45"
# DEFAULT_EPISODE_DIR = "/home/hongyi/data/real/pick_cloth/pick_cloth_4/episode_69"
DEFAULT_WBC_STATE_HOST  = "localhost"   # host where g1_deploy_onnx_ref runs
DEFAULT_WBC_STATE_PORT  = 5557          # WBC state publisher port
DEFAULT_WBC_STATE_TOPIC = "g1_debug"    # WBC state topic
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import time

import msgpack
import numpy as np
import zmq

# ── Quaternion helpers (convention: [w, x, y, z]) ─────────────────────────────

def _quat_mul(q1, q2):
    """Hamilton product q1 * q2, both [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def _quat_conj(q):
    """Conjugate (= inverse for unit quat), [w, x, y, z]."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float32)


def read_robot_quat_from_wbc(host, port, topic):
    """Read the robot's current quat from the WBC's own state publisher.

    The WBC (g1_deploy_onnx_ref) publishes base_quat_measured on port 5557.
    Using this value as q_robot_init guarantees the sign is identical to the
    WBC's internal base_quat, eliminating the double-cover mismatch that occurs
    when opening a separate hardware connection via unitree_interface.

    Returns identity [1,0,0,0] if the connection fails.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 3000)  # 3 s timeout per recv
    try:
        topic_bytes = topic.encode("utf-8")
        q = None
        for _ in range(20):  # try up to 20 frames
            try:
                msg = sock.recv()
            except zmq.Again:
                break
            payload = msg[len(topic_bytes):]
            try:
                state = msgpack.unpackb(payload, raw=False)
            except Exception:
                continue
            if "base_quat_measured" in state:
                q = np.array(state["base_quat_measured"], dtype=np.float32)
                break
        if q is None:
            raise RuntimeError("base_quat_measured not found in WBC state")
        print(f"[Replay] q_robot_init from WBC state publisher: {q}")
        return q
    except Exception as e:
        print(f"[Replay] WARNING: could not read WBC state ({e}). Using identity.")
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    finally:
        sock.close()
        ctx.term()


# ── Import pack_pose_message / build_command_message from GR00T repo ──────────
# Repo root is two levels up from this file (orchestration/<name>/replay.py).
_GROOT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)

# ── Dex3 hands ────────────────────────────────────────────────────────────────
# The recording already stores native Dex3 joints: hand_joints (14,) =
# left(7) + right(7), recorded by g1_data_server. No conversion needed.

def extract_hand_joints(frame, source="action"):
    """Return (left7, right7) Dex3 joint arrays, or None if no hand data."""
    entry = frame.get("actions" if source == "action" else "states")
    if entry is None:
        return None
    hand14 = entry.get("hand_joints")
    if hand14 is None:
        return None
    hand14 = np.asarray(hand14, dtype=np.float32)
    if hand14.shape[0] != 14:
        return None
    return hand14[:7], hand14[7:]


_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)

def _mujoco29_to_isaaclab29(joint_pos29: np.ndarray) -> np.ndarray:
    jp = np.asarray(joint_pos29, dtype=np.float32).reshape(29)
    return jp[_MUJOCO_TO_ISAACLAB_DOF].astype(np.float32).copy()

# ── ZMQ publisher (identical to WBCBridge in psi-inference_rtc.py) ────────────

class WBCBridge:
    def __init__(self, host="*", port=5556, topic="pose"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")
        self._topic = topic
        self._frame_index = 0

    def send_command(self, start=False, stop=False, planner=False):
        msg = build_command_message(start=start, stop=stop, planner=planner)
        self._socket.send(msg)
        print(f"[WBCBridge] Command: start={start} stop={stop} planner={planner}")

    def publish_joints(self, joint_pos, body_quat, left_hand=None, right_hand=None):
        pose_data = {
            "joint_pos":   joint_pos.astype(np.float32).reshape(1, 29),
            "joint_vel":   np.zeros((1, 29), dtype=np.float32),
            "body_quat":   body_quat.astype(np.float32).reshape(1, 4),
            "frame_index": np.array([self._frame_index], dtype=np.int64),
        }
        if left_hand is not None:
            pose_data["left_hand_joints"] = np.asarray(left_hand, dtype=np.float32).reshape(1, 7)
        if right_hand is not None:
            pose_data["right_hand_joints"] = np.asarray(right_hand, dtype=np.float32).reshape(1, 7)
        msg = pack_pose_message(pose_data, topic=self._topic, version=1)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close()
        self._context.term()


# ── Replay logic ──────────────────────────────────────────────────────────────

def load_episode(episode_path):
    """Load episode frames.

    `episode_path` can be either a directory containing data.json, or a direct
    path to a json file (e.g. episode_000_data.json in the Unifolm format).
    """
    if os.path.isdir(episode_path):
        json_path = os.path.join(episode_path, "data.json")
    else:
        json_path = episode_path
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"episode json not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        frames = data
        frequency = 30
    elif isinstance(data, dict) and "frames" in data:
        frames = data["frames"]
        frequency = data.get("fps", data.get("frequency", 30))
    else:
        raise ValueError(f"Unknown data format in {json_path}")

    print(f"[Replay] Loaded {len(frames)} frames at {frequency} Hz from {json_path}")
    return frames, frequency


def extract_qpos_quat(frame, source="action"):
    """Extract (qpos29_mujoco, quat_wxyz) from a frame, or None if missing.

    Format: frame["states"/"actions"]["qpos"] (29, MuJoCo order); quat (4) is
    stored only in states (relative to recording start).

    `source` selects "action" (target values) or "state" (measured values).
    """
    states = frame.get("states")
    actions = frame.get("actions")
    if source == "mixed":
        # Legs from action (measured stride is attenuated by tracking loss →
        # replaying it drags the feet); waist + arms from state (actions.qpos is
        # the motor target incl. gravity compensation → replaying it overshoots).
        if states is None or actions is None:
            return None
        qpos = np.asarray(states["qpos"], dtype=np.float32).copy()
        qpos[:12] = np.asarray(actions["qpos"], dtype=np.float32)[:12]  # mujoco order: legs = 0-11
    else:
        entry = actions if source == "action" else states
        if entry is None:
            return None
        qpos = np.asarray(entry["qpos"], dtype=np.float32)
    quat_src = states if states is not None else actions
    quat = np.asarray(quat_src["quat"], dtype=np.float32)
    return qpos, quat


def replay(episode_dir, zmq_host, zmq_pub_port, zmq_topic,
           wbc_state_host, wbc_state_port, wbc_state_topic,
           source="action", send_hands=True, dry_run=False):
    frames, frequency = load_episode(episode_dir)
    dt = 1.0 / frequency
    print(f"[Replay] Using '{source}' joint targets; sending recorded quat as-is; "
          f"hands {'on' if send_hands else 'off'}")

    bridge = WBCBridge(host="*", port=zmq_pub_port, topic=zmq_topic)

    # Give ZMQ PUB socket time to establish connections
    print("[Replay] Waiting for ZMQ connections...")
    time.sleep(1.0)

    if not dry_run:
        bridge.send_command(start=True, stop=False, planner=False)
        time.sleep(0.2)

    print(f"[Replay] Starting replay of {len(frames)} frames...")

    prev_quat = None  # for sign-continuity check
    last_qpos_isaac = None
    last_quat = None
    last_left_hand = None
    last_right_hand = None

    try:
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()

            extracted = extract_qpos_quat(frame, source=source)
            if extracted is None:
                print(f"[Replay] Frame {i}: missing {source} data, skipping")
                continue
            qpos, quat = extracted

            left_hand = right_hand = None
            if send_hands:
                # Hands always replay the commanded action: hand targets go straight
                # to Dex3 (no tracking policy in between, so no double-compensation),
                # and measured finger positions under-close when blocked by objects.
                hands = extract_hand_joints(frame, source="action")
                if hands is None:
                    hands = extract_hand_joints(frame, source="state")
                if hands is not None:
                    left_hand, right_hand = hands

            qpos_isaac = _mujoco29_to_isaaclab29(qpos)

            if qpos_isaac.shape[0] != 29:
                print(f"[Replay] Frame {i}: unexpected qpos shape {qpos_isaac.shape}, skipping")
                continue

            # Send the recorded quat as-is (no re-referencing / composition).
            quat = quat / np.linalg.norm(quat)

            # Sign-continuity fix: both q and -q represent the same rotation, but a
            # sign flip between consecutive frames would look like a 180-degree jump.
            if prev_quat is not None and np.dot(quat, prev_quat) < 0:
                quat = -quat
            prev_quat = quat.copy()

            last_qpos_isaac = qpos_isaac
            last_quat = quat
            last_left_hand = left_hand
            last_right_hand = right_hand

            if not dry_run:
                bridge.publish_joints(qpos_isaac, quat, left_hand, right_hand)

            elapsed = (i + 1) / frequency
            print(f"[Replay] Frame {i+1:5d}/{len(frames)}  t={elapsed:.2f}s", end="\r")

            # Maintain playback rate
            sleep_t = dt - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Replay] Interrupted by user.")

    print(f"\n[Replay] Done. Sent {bridge._frame_index} frames.")

    # Keep streaming the last frame so the robot stays under active control.
    # Without this, the Unitree G1's internal sport-mode watchdog detects the
    # dead data stream and autonomously re-engages, causing the robot to go limp
    # regardless of what the deploy binary does.
    if not dry_run and last_qpos_isaac is not None:
        print("[Replay] Holding last frame. Press Ctrl+C to stop.")
        try:
            while True:
                t0 = time.perf_counter()
                bridge.publish_joints(last_qpos_isaac, last_quat,
                                      last_left_hand, last_right_hand)
                sleep_t = dt - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        except KeyboardInterrupt:
            print("\n[Replay] Hold interrupted by user.")

    bridge.stop()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded episode to the WBC controller via ZMQ"
    )
    parser.add_argument(
        "--episode-dir",
        type=str,
        default=DEFAULT_EPISODE_DIR,
        help=f"Path to episode dir containing data.json, or direct json file (default: {DEFAULT_EPISODE_DIR})",
    )
    # Trade-off between sources (joint order is identical, verified):
    #   state  = measured trajectory (default) — actions.qpos is the raw motor
    #            target incl. gravity/impedance compensation and replays badly
    #   action = motor targets (arms overshoot upward ~0.3 rad at shoulders)
    #   mixed  = legs from action, waist+arms from state
    parser.add_argument(
        "--source", type=str, choices=["action", "state", "mixed"], default="state",
        help="qpos source: state (measured, default), action (motor targets), "
             "or mixed (legs from action, waist+arms from state)",
    )
    parser.add_argument(
        "--no-hands", action="store_true",
        help="Do not send the recorded Dex3 hand_joints",
    )
    parser.add_argument(
        "--zmq-host", type=str, default="*",
        help="ZMQ PUB bind host (default: * for all interfaces)",
    )
    parser.add_argument(
        "--zmq-pub-port", type=int, default=5556,
        help="ZMQ PUB port for sending pose to WBC (default: 5556)",
    )
    parser.add_argument(
        "--zmq-topic", type=str, default="pose",
        help="ZMQ topic for pose messages (default: pose)",
    )
    parser.add_argument(
        "--wbc-state-host", type=str, default=DEFAULT_WBC_STATE_HOST,
        help=f"Host of WBC state publisher (default: {DEFAULT_WBC_STATE_HOST})",
    )
    parser.add_argument(
        "--wbc-state-port", type=int, default=DEFAULT_WBC_STATE_PORT,
        help=f"Port of WBC state publisher (default: {DEFAULT_WBC_STATE_PORT})",
    )
    parser.add_argument(
        "--wbc-state-topic", type=str, default=DEFAULT_WBC_STATE_TOPIC,
        help=f"Topic of WBC state publisher (default: {DEFAULT_WBC_STATE_TOPIC})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and iterate frames without sending anything (for testing)",
    )
    args = parser.parse_args()

    replay(
        episode_dir=args.episode_dir,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_topic=args.zmq_topic,
        wbc_state_host=args.wbc_state_host,
        wbc_state_port=args.wbc_state_port,
        wbc_state_topic=args.wbc_state_topic,
        source=args.source,
        send_hands=not args.no_hands,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
