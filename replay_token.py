#!/usr/bin/env python3
"""
replay_token.py

Replay a recorded episode by reading data.json and publishing ONLY tokens
to the WBC controller via ZMQ at the original recording frequency (30 Hz).

Uses ZMQ Protocol v4 (token-only streaming):
  - token_state (required): [N] latent vector

This is for testing/validating the token → action decode pipeline without
sending explicit joint positions.

Usage:
    python replay_token.py
    python replay_token.py --episode-dir <path>
"""

# ── Hard-coded defaults (edit these instead of passing CLI args) ───────────────
DEFAULT_EPISODE_DIR    = "/home/xiawei/data/demonstration_2026-04-14_19-46-09/episode_0"
DEFAULT_ZMQ_HOST        = "*"
DEFAULT_ZMQ_PUB_PORT    = 5556
DEFAULT_ZMQ_TOPIC      = "pose"
DEFAULT_RECORDING_FREQ = 30
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import time

import numpy as np
import zmq
import sys

# ── Import pack_pose_message from GR00T repo ──────────────────────────────────
_GROOT_ROOT = os.path.expanduser("~/hsc/GR00T-WholeBodyControl")
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)

# ── ZMQ publisher for Protocol v4 (token-only streaming) ────────────────────

class TokenPublisher:
    """ZMQ publisher for token-only streaming (Protocol v4)."""

    def __init__(self, host="*", port=5556, topic="pose"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")
        self._topic = topic
        self._frame_index = 0

    def send_command(self, start=False, stop=False, planner=False):
        msg = build_command_message(start=start, stop=stop, planner=planner)
        self._socket.send(msg)
        print(f"[TokenPublisher] Command: start={start} stop={stop} planner={planner}")

    def publish_token(self, token, left_hand=None, right_hand=None):
        """
        Publish token-only message (Protocol v4).

        Args:
            token: np.ndarray of shape (N,) - latent vector from encoder
            left_hand: np.ndarray of shape (7,) - left hand 7-DOF joint positions (optional)
            right_hand: np.ndarray of shape (7,) - right hand 7-DOF joint positions (optional)
        """
        pose_data = {
            "token_state": token.astype(np.float32).reshape(1, -1),
        }
        if left_hand is not None:
            pose_data["left_hand_joints"] = np.array(left_hand, dtype=np.float32).reshape(1, -1)
        if right_hand is not None:
            pose_data["right_hand_joints"] = np.array(right_hand, dtype=np.float32).reshape(1, -1)

        msg = pack_pose_message(pose_data, topic=self._topic, version=4)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close()
        self._context.term()


# ── Load episode (new format: list of dict) ─────────────────────────────────

def load_episode(episode_dir):
    """Load episode from data.json (new format: list of dict)."""
    json_path = os.path.join(episode_dir, "data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"data.json not found in {episode_dir}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        frames = data
        frequency = DEFAULT_RECORDING_FREQ
    elif isinstance(data, dict) and "frames" in data:
        frames = data["frames"]
        frequency = data.get("frequency", DEFAULT_RECORDING_FREQ)
    else:
        raise ValueError(f"Unknown data format in {json_path}")

    print(f"[ReplayToken] Loaded {len(frames)} frames at {frequency} Hz from {json_path}")
    return frames, frequency


# ── Replay logic ──────────────────────────────────────────────────────────────

def replay(episode_dir, zmq_host, zmq_pub_port, zmq_topic,
           frequency, dry_run=False):
    frames, _ = load_episode(episode_dir)
    dt = 1.0 / frequency

    publisher = TokenPublisher(host=zmq_host, port=zmq_pub_port, topic=zmq_topic)

    print("[ReplayToken] Waiting for ZMQ connections...")
    time.sleep(1.0)

    if not dry_run:
        publisher.send_command(start=True, stop=False, planner=True)
        time.sleep(0.2)

    print(f"[ReplayToken] Starting token replay of {len(frames)} frames...")

    last_token = None
    last_left_hand = None
    last_right_hand = None

    try:
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()

            actions = frame.get("actions")
            if actions is None:
                print(f"[ReplayToken] Frame {i} has no actions, skipping")
                continue

            token = actions.get("token", [])
            if token is None or len(token) == 0:
                print(f"[ReplayToken] Frame {i} has no token, skipping")
                continue

            token = np.array(token, dtype=np.float32)

            hand_joints = actions.get("hand_joints", None)
            left_hand = right_hand = None
            if hand_joints is not None and len(hand_joints) == 14:
                hand_joints = np.array(hand_joints, dtype=np.float32)
                left_hand = hand_joints[:7]
                right_hand = hand_joints[7:]

            if not dry_run:
                publisher.publish_token(token, left_hand=left_hand, right_hand=right_hand)

            last_token = token
            last_left_hand = left_hand
            last_right_hand = right_hand

            elapsed = (i + 1) / frequency
            hand_info = ""
            if left_hand is not None and right_hand is not None:
                hand_info = f" L={np.linalg.norm(left_hand):.2f} R={np.linalg.norm(right_hand):.2f}"
            print(f"[ReplayToken] Frame {i+1:5d}/{len(frames)}  t={elapsed:.2f}s  token_dim={token.shape[0]}{hand_info}", end="\r")

            sleep_t = dt - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[ReplayToken] Interrupted by user.")
        publisher.stop()
        return

    print(f"\n[ReplayToken] Done. Sent {publisher._frame_index} frames.")

    if not dry_run and last_token is not None:
        print("[ReplayToken] Holding last token. Press Ctrl+C to stop.")
        try:
            while True:
                t0 = time.perf_counter()
                publisher.publish_token(last_token, left_hand=last_left_hand, right_hand=last_right_hand)
                sleep_t = dt - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        except KeyboardInterrupt:
            print("\n[ReplayToken] Hold interrupted by user.")

    publisher.stop()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded episode via token-only streaming (Protocol v4)"
    )
    parser.add_argument(
        "--episode-dir",
        type=str,
        default=DEFAULT_EPISODE_DIR,
        help=f"Path to episode directory containing data.json (default: {DEFAULT_EPISODE_DIR})",
    )
    parser.add_argument(
        "--zmq-host", type=str, default=DEFAULT_ZMQ_HOST,
        help="ZMQ PUB bind host (default: *)",
    )
    parser.add_argument(
        "--zmq-pub-port", type=int, default=DEFAULT_ZMQ_PUB_PORT,
        help="ZMQ PUB port (default: 5556)",
    )
    parser.add_argument(
        "--zmq-topic", type=str, default=DEFAULT_ZMQ_TOPIC,
        help="ZMQ topic (default: pose)",
    )
    parser.add_argument(
        "--frequency", type=int, default=DEFAULT_RECORDING_FREQ,
        help="Playback frequency in Hz (default: 30)",
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
        frequency=args.frequency,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
