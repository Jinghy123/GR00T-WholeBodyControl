#!/usr/bin/env python3
"""
replay.py

Replay a recorded episode by reading data.json and publishing joint positions
to the WBC controller via ZMQ at the original recording frequency.

New data format:
  - qpos: states["leg_state"] + states["arm_state"]  -> 29 dims
  - quat: from states["imu"]["quaternion"] (already wxyz)

Usage:
    python replay.py
    python replay.py --episode-dir <path>
"""

# ── Hard-coded defaults ───────────────────────────────────────────────────────
DEFAULT_EPISODE_DIR = "/home/xiawei/hongyi/Unitree_Robotics/Humanoid-Teleop/teleop/data/g1_1001/Basic/Put_toys_into_box_and_lift_it_and_turn_and_put_on_the_chair/episode_41"
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import time

import numpy as np
import zmq

# ── Import pack_pose_message / build_command_message from GR00T repo ──────────
_GROOT_ROOT = os.path.expanduser(
    "~/hongyi/Unitree_Robotics/GR00T-WholeBodyControl"
)
sys.path.insert(0, _GROOT_ROOT)

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)

# ── Joint mapping: Mujoco 29 DOF -> IsaacLab 29 DOF ──────────────────────────
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [
        0, 6, 12,
        1, 7, 13,
        2, 8, 14,
        3, 9, 15,
        22, 4, 10,
        16, 23, 5,
        11, 17, 24,
        18, 25, 19,
        26, 20, 27,
        21, 28
    ],
    dtype=np.int32,
)

def _mujoco29_to_isaaclab29(joint_pos29: np.ndarray) -> np.ndarray:
    jp = np.asarray(joint_pos29, dtype=np.float32).reshape(29)
    return jp[_MUJOCO_TO_ISAACLAB_DOF].astype(np.float32).copy()


def _get_qpos29_from_states(states):
    """
    Build 29-dim joint position vector:
      leg first, then arm
    Supports both singular/plural keys.
    """
    leg = states.get("leg_states", states.get("leg_state"))
    arm = states.get("arm_states", states.get("arm_state"))

    if leg is None:
        raise KeyError("Missing leg_state/leg_states in frame states")
    if arm is None:
        raise KeyError("Missing arm_state/arm_states in frame states")

    leg = np.asarray(leg, dtype=np.float32).reshape(-1)
    arm = np.asarray(arm, dtype=np.float32).reshape(-1)

    qpos = np.concatenate([leg, arm], axis=0).astype(np.float32)

    if qpos.size != 29:
        raise ValueError(
            f"Expected 29-dim qpos, got {qpos.size} "
            f"(leg={leg.size}, arm={arm.size})"
        )
    return qpos


def _normalize_quat_wxyz(q):
    """
    Normalize quaternion in wxyz order.
    Assumes the recorded quaternion is already wxyz.
    """
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = np.linalg.norm(q)
    if n > 1e-8:
        q = q / n
    return q.astype(np.float32)


# ── ZMQ publisher ─────────────────────────────────────────────────────────────

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

    def publish_joints(self, joint_pos, body_quat, left_hand_joints=None, right_hand_joints=None):
        pose_data = {
            "joint_pos": joint_pos.astype(np.float32).reshape(1, 29),
            "joint_vel": np.zeros((1, 29), dtype=np.float32),
            "body_quat": body_quat.astype(np.float32).reshape(1, 4),
            "frame_index": np.array([self._frame_index], dtype=np.int64),
            "left_hand_joints": np.array(left_hand_joints),
            "right_hand_joints": np.array(right_hand_joints),
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=1)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close()
        self._context.term()


# ── Replay logic ──────────────────────────────────────────────────────────────

def load_episode(episode_dir):
    """Load episode from data.json (supports both old and new formats)."""
    json_path = os.path.join(episode_dir, "data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"data.json not found in {episode_dir}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        frames = data
        frequency = 30
    elif isinstance(data, dict) and "frames" in data:
        frames = data["frames"]
        frequency = data.get("frequency", 30)
    else:
        raise ValueError(f"Unknown data format in {json_path}")

    print(f"[Replay] Loaded {len(frames)} frames at {frequency} Hz from {json_path}")
    return frames, frequency


def replay(episode_dir, zmq_host, zmq_pub_port, zmq_topic, dry_run=False):
    frames, frequency = load_episode(episode_dir)
    dt = 1.0 / frequency

    bridge = WBCBridge(host=zmq_host, port=zmq_pub_port, topic=zmq_topic)

    print("[Replay] Waiting for ZMQ connections...")
    time.sleep(1.0)

    if not dry_run:
        bridge.send_command(start=True, stop=False, planner=False)
        time.sleep(0.2)

    print(f"[Replay] Starting replay of {len(frames)} frames...")

    prev_quat = None
    last_qpos = None
    last_quat = None
    hand_state = None

    try:
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()

            states = frame.get("states")
            if states is None:
                print(f"[Replay] Frame {i} has no states, skipping")
                continue

            try:
                qpos_mujoco = _get_qpos29_from_states(states)
                qpos_isaac = _mujoco29_to_isaaclab29(qpos_mujoco)
            except Exception as e:
                print(f"[Replay] Frame {i} qpos error: {e}, skipping")
                continue

            imu = states.get("imu")
            if imu is None or "quaternion" not in imu:
                print(f"[Replay] Frame {i} missing imu.quaternion, skipping")
                continue

            quat = _normalize_quat_wxyz(imu["quaternion"])

            hand_state = states.get("hand_state")

            # Sign-continuity fix
            if prev_quat is not None and np.dot(quat, prev_quat) < 0:
                quat = -quat
            prev_quat = quat.copy()

            last_qpos = qpos_isaac
            last_quat = quat

            if not dry_run:
                bridge.publish_joints(qpos_isaac, quat, left_hand_joints=hand_state[:7], right_hand_joints=hand_state[7:14])

            elapsed = (i + 1) / frequency
            print(f"[Replay] Frame {i+1:5d}/{len(frames)}  t={elapsed:.2f}s", end="\r")

            sleep_t = dt - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Replay] Interrupted by user.")

    print(f"\n[Replay] Done. Sent {bridge._frame_index} frames.")

    # Keep streaming the last frame so the robot stays under active control.
    if not dry_run and last_qpos is not None and last_quat is not None:
        print("[Replay] Holding last frame. Press Ctrl+C to stop.")
        try:
            while True:
                t0 = time.perf_counter()
                bridge.publish_joints(last_qpos, last_quat)
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
        help=f"Path to episode directory containing data.json (default: {DEFAULT_EPISODE_DIR})",
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
        "--dry-run", action="store_true",
        help="Load and iterate frames without sending anything (for testing)",
    )
    args = parser.parse_args()

    replay(
        episode_dir=args.episode_dir,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_topic=args.zmq_topic,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()