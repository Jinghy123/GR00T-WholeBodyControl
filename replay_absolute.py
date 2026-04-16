#!/usr/bin/env python3
"""
replay_absolute.py

Same as replay.py but sends q_rel DIRECTLY as body_quat (no q_robot_init composition).
This tests whether the WBC heading logic can handle absolute quats that don't
start at the robot's current orientation.

If the robot does NOT turn with this script but DOES turn with replay.py,
it proves that the q_robot_init * q_rel composition is required.
"""

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

_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)

def _mujoco29_to_isaaclab29(joint_pos29: np.ndarray) -> np.ndarray:
    jp = np.asarray(joint_pos29, dtype=np.float32).reshape(29)
    return jp[_MUJOCO_TO_ISAACLAB_DOF].astype(np.float32).copy()


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

    def publish_joints(self, joint_pos, body_quat):
        pose_data = {
            "joint_pos":   joint_pos.astype(np.float32).reshape(1, 29),
            "joint_vel":   np.zeros((1, 29), dtype=np.float32),
            "body_quat":   body_quat.astype(np.float32).reshape(1, 4),
            "frame_index": np.array([self._frame_index], dtype=np.int64),
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=1)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close()
        self._context.term()


DEFAULT_EPISODE_DIR = "/home/xiawei/data/demonstration_2026-03-12_17-20-26/episode_1"


def load_episode(episode_dir):
    json_path = os.path.join(episode_dir, "data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"data.json not found in {episode_dir}")
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = data["frames"]
    frequency = data.get("frequency", 30)
    print(f"[ReplayAbs] Loaded {len(frames)} frames at {frequency} Hz from {json_path}")
    return frames, frequency


def replay_absolute(episode_dir, dry_run=False):
    frames, frequency = load_episode(episode_dir)
    dt = 1.0 / frequency

    bridge = WBCBridge(host="*", port=5556, topic="pose")

    print("[ReplayAbs] Waiting for ZMQ connections...")
    time.sleep(1.0)

    if not dry_run:
        bridge.send_command(start=True, stop=False, planner=False)
        time.sleep(0.2)

    print(f"[ReplayAbs] Starting replay of {len(frames)} frames...")
    print("[ReplayAbs] NOTE: Sending q_rel DIRECTLY as body_quat (NO q_robot_init composition)")

    prev_quat = None
    last_qpos_isaac = None
    last_quat = None

    try:
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()

            states = frame.get("states")
            actions = frame.get("actions")
            if states is None or actions is None:
                continue

            qpos = np.array(states["qpos"], dtype=np.float32)
            q_rel = np.array(states["quat"], dtype=np.float32)  # (4,) wxyz, relative to recording start

            qpos_isaac = _mujoco29_to_isaaclab29(qpos)

            if qpos_isaac.shape[0] != 29:
                continue

            # ABSOLUTE MODE: send q_rel directly, no q_robot_init composition
            quat = q_rel.copy()

            # Sign-continuity
            if prev_quat is not None and np.dot(quat, prev_quat) < 0:
                quat = -quat
            prev_quat = quat.copy()

            last_qpos_isaac = qpos_isaac
            last_quat = quat

            if not dry_run:
                bridge.publish_joints(qpos_isaac, quat)

            elapsed = (i + 1) / frequency
            if i % 50 == 0:
                from scipy.spatial.transform import Rotation
                yaw_deg = np.degrees(Rotation.from_quat(quat[[1,2,3,0]]).as_euler('ZYX')[0])
                print(f"[ReplayAbs] Frame {i+1:5d}/{len(frames)}  t={elapsed:.2f}s  yaw={yaw_deg:.1f}°")

            sleep_t = dt - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[ReplayAbs] Interrupted by user.")

    print(f"\n[ReplayAbs] Done. Sent {bridge._frame_index} frames.")

    if not dry_run and last_qpos_isaac is not None:
        print("[ReplayAbs] Holding last frame. Press Ctrl+C to stop.")
        try:
            while True:
                t0 = time.perf_counter()
                bridge.publish_joints(last_qpos_isaac, last_quat)
                sleep_t = dt - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        except KeyboardInterrupt:
            print("\n[ReplayAbs] Hold interrupted by user.")

    bridge.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Replay with ABSOLUTE quat (no q_robot_init composition) to test heading"
    )
    parser.add_argument("--episode-dir", type=str, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    replay_absolute(episode_dir=args.episode_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
