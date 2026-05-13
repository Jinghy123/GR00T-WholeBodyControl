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
# DEFAULT_EPISODE_DIR    = "/home/xiawei/hongyi/Unitree_Robotics/Humanoid-Teleop/teleop/data/g1_1001/Basic/Pick_bottle_and_turn_and_pour_into_cup/episode_16"
# DEFAULT_EPISODE_DIR    = "/home/xiawei/hongyi/Unitree_Robotics/Humanoid-Teleop/teleop/data/g1_1001/Basic/Spray_the_bowl_and_wipe_it_and_stack_it_up/episode_1"
# DEFAULT_EPISODE_DIR    = "/home/xiawei/data/unfold_a_tablet_cover_h1/episode_0"
DEFAULT_EPISODE_DIR    = "/home/xiawei/data/press_a_big_button_h1/episode_0"
DEFAULT_ZMQ_HOST        = "*"
DEFAULT_ZMQ_PUB_PORT    = 5556
DEFAULT_ZMQ_TOPIC      = "pose"
DEFAULT_RECORDING_FREQ = 30
# ── Neck-motor replay (matches pose_publisher.py wire format) ──────────────────
DEFAULT_ENABLE_NECK    = False
DEFAULT_NECK_PUB_HOST  = "*"
DEFAULT_NECK_PUB_PORT  = 5570
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


class NeckPublisher:
    """ZMQ PUB of `[neck_yaw, neck_pitch]` JSON for the G1 NeckMotor.

    Wire format matches pose_publisher.py exactly (SNDHWM=1, LINGER=0,
    payload = json.dumps([float(yaw), float(pitch)]).encode()), so
    realsense_server.py's --pose-zmq subscriber consumes this stream
    without any change. NOTE: stop pose_publisher.py before running
    replay or the bind on port 5570 will collide.
    """

    def __init__(self, host=DEFAULT_NECK_PUB_HOST, port=DEFAULT_NECK_PUB_PORT):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        bind_addr = f"tcp://{host}:{port}"
        try:
            self._sock.bind(bind_addr)
        except zmq.ZMQError as e:
            self._sock.close(linger=0)
            self._ctx.term()
            raise RuntimeError(
                f"NeckPublisher bind failed on {bind_addr}: {e}. "
                f"Is pose_publisher.py still running? `pkill -f pose_publisher.py`"
            ) from e
        print(f"[NeckPublisher] PUB bound to {bind_addr}")
        self._last = None

    def publish(self, yaw, pitch):
        msg = json.dumps([float(yaw), float(pitch)]).encode("utf-8")
        self._sock.send(msg)
        self._last = (float(yaw), float(pitch))

    def publish_last(self):
        """Re-publish the last sent value (used while holding the final token)."""
        if self._last is not None:
            self.publish(*self._last)

    def stop(self):
        self._sock.close(linger=0)
        self._ctx.term()


def _extract_neck(frame):
    """Return (yaw_rad, pitch_rad) from frame.actions['neck'], or None if absent."""
    actions = frame.get("actions") or {}
    neck = actions.get("neck")
    if neck is None:
        return None
    try:
        if len(neck) >= 2:
            return float(neck[0]), float(neck[1])
    except (TypeError, ValueError):
        pass
    return None


# ── Load episode (new format: list of dict) ─────────────────────────────────

def load_episode(episode_dir):
    """Load episode from data_sonic.json (new format: list of dict)."""
    json_path = os.path.join(episode_dir, "data_sonic.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"data_sonic.json not found in {episode_dir}")

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
           frequency, dry_run=False,
           enable_neck=DEFAULT_ENABLE_NECK,
           neck_pub_host=DEFAULT_NECK_PUB_HOST,
           neck_pub_port=DEFAULT_NECK_PUB_PORT):
    frames, _ = load_episode(episode_dir)
    dt = 1.0 / frequency

    publisher = TokenPublisher(host=zmq_host, port=zmq_pub_port, topic=zmq_topic)

    neck_publisher = None
    if enable_neck and not dry_run:
        n_with_neck = sum(1 for f in frames if _extract_neck(f) is not None)
        if n_with_neck == 0:
            print("[ReplayToken] --enable-neck set but no frame has actions['neck']; skipping neck replay.")
        else:
            print(f"[ReplayToken] Neck replay enabled: {n_with_neck}/{len(frames)} frames have neck data.")
            neck_publisher = NeckPublisher(host=neck_pub_host, port=neck_pub_port)

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

            if neck_publisher is not None:
                neck = _extract_neck(frame)
                if neck is not None:
                    neck_publisher.publish(*neck)
                else:
                    neck_publisher.publish_last()  # hold last value if this frame is missing it

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
        if neck_publisher is not None:
            neck_publisher.stop()
        return

    print(f"\n[ReplayToken] Done. Sent {publisher._frame_index} frames.")

    if not dry_run and last_token is not None:
        print("[ReplayToken] Holding last token. Press Ctrl+C to stop.")
        try:
            while True:
                t0 = time.perf_counter()
                publisher.publish_token(last_token, left_hand=last_left_hand, right_hand=last_right_hand)
                if neck_publisher is not None:
                    neck_publisher.publish_last()
                sleep_t = dt - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        except KeyboardInterrupt:
            print("\n[ReplayToken] Hold interrupted by user.")

    publisher.stop()
    if neck_publisher is not None:
        neck_publisher.stop()


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
    parser.add_argument(
        "--enable-neck", action="store_true", default=DEFAULT_ENABLE_NECK,
        help="Replay actions['neck'] to the G1 NeckMotor over ZMQ PUB. "
             "Stop pico_manus_thread_server.py (or pose_publisher.py if used) "
             "before enabling — both bind port 5570.",
    )
    parser.add_argument(
        "--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST,
        help=f"Neck PUB bind host (default: {DEFAULT_NECK_PUB_HOST})",
    )
    parser.add_argument(
        "--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT,
        help=f"Neck PUB port (default: {DEFAULT_NECK_PUB_PORT})",
    )
    args = parser.parse_args()

    replay(
        episode_dir=args.episode_dir,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_topic=args.zmq_topic,
        frequency=args.frequency,
        dry_run=args.dry_run,
        enable_neck=args.enable_neck,
        neck_pub_host=args.neck_pub_host,
        neck_pub_port=args.neck_pub_port,
    )


if __name__ == "__main__":
    main()
