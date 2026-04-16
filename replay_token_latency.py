#!/usr/bin/env python3
"""
replay_token_latency.py

Same as replay_token.py but with configurable chunk size and latency frames,
to simulate different inference latencies.

Pattern per cycle:
  1. Send CHUNK_SIZE tokens at 30 Hz  (EXECUTING)
  2. Run encoder on current robot state → freeze token, send LATENCY_FRAMES times (WAITING)
  3. Repeat

Usage:
    python replay_token_latency.py
    python replay_token_latency.py --latency-frames 16   # ~533ms latency
    python replay_token_latency.py --chunk-size 16 --latency-frames 4
"""

# ── Hard-coded defaults ────────────────────────────────────────────────────────
DEFAULT_EPISODE_DIR    = "/home/xiawei/data/pick_bottle_and_turn_and_pour_into_cup/episode_35"
DEFAULT_ZMQ_HOST       = "*"
DEFAULT_ZMQ_PUB_PORT   = 5556
DEFAULT_ZMQ_TOPIC      = "pose"
DEFAULT_RECORDING_FREQ = 30
DEFAULT_CHUNK_SIZE     = 24
DEFAULT_LATENCY_FRAMES = 30    # frames @ 30Hz to hold after each chunk
# ── WBC state subscriber defaults ─────────────────────────────────────────────
DEFAULT_WBC_HOST       = "localhost"
DEFAULT_WBC_PORT       = 5557
DEFAULT_WBC_TOPIC      = "g1_debug"
# ── Encoder model ──────────────────────────────────────────────────────────────
ENCODER_MODEL = "gear_sonic_deploy/policy/release/model_encoder.onnx"
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import threading
import time

import msgpack
import numpy as np
import zmq
import sys

_GROOT_ROOT = os.path.expanduser("~/hsc/GR00T-WholeBodyControl")
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)
from encoder_client import EncoderClient

# Joint order conversion: WBC publishes in Mujoco order, encoder expects IsaacLab order
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)

def _mujoco29_to_isaaclab29(qpos: np.ndarray) -> np.ndarray:
    return np.asarray(qpos, dtype=np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB_DOF].copy()


class WBCStateReader:
    """Lightweight background subscriber to WBC state publisher."""

    def __init__(self, host=DEFAULT_WBC_HOST, port=DEFAULT_WBC_PORT, topic=DEFAULT_WBC_TOPIC):
        self._topic_bytes = topic.encode()
        self._topic_len   = len(self._topic_bytes)

        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, self._topic_bytes)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.connect(f"tcp://{host}:{port}")

        self._lock   = threading.Lock()
        self._latest = None
        self._stop   = threading.Event()

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[WBCState] Subscribed to {host}:{port} topic={topic}")

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                raw     = self._sock.recv()
                payload = raw[self._topic_len:]
                data    = msgpack.unpackb(payload, raw=False)
                with self._lock:
                    self._latest = {
                        "qpos":      np.array(data["body_q_measured"],  dtype=np.float32),
                        "base_quat": np.array(data["base_quat_measured"], dtype=np.float32),
                    }
            except zmq.Again:
                pass
            except Exception as e:
                print(f"[WBCState] Recv error: {e}")

    def get_state(self):
        with self._lock:
            if self._latest is None:
                return None
            return {k: v.copy() for k, v in self._latest.items()}

    def close(self):
        self._stop.set()
        self._sock.close()
        self._ctx.term()


class TokenPublisher:
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


def load_episode(episode_dir):
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
    print(f"[ReplayLatency] Loaded {len(frames)} frames at {frequency} Hz from {json_path}")
    return frames, frequency


def _extract_tokens(frames):
    tokens = []
    for frame in frames:
        actions = frame.get("actions")
        if actions is None:
            continue
        token = actions.get("token", [])
        if token is None or len(token) == 0:
            continue
        token = np.array(token, dtype=np.float32)
        hand_joints = actions.get("hand_joints", None)
        left_hand = right_hand = None
        if hand_joints is not None and len(hand_joints) == 14:
            hand_joints = np.array(hand_joints, dtype=np.float32)
            left_hand = hand_joints[:7]
            right_hand = hand_joints[7:]
        tokens.append((token, left_hand, right_hand))
    return tokens


def _send_one(publisher, token, left_hand, right_hand, dt, dry_run):
    t0 = time.perf_counter()
    if not dry_run:
        publisher.publish_token(token, left_hand=left_hand, right_hand=right_hand)
    sleep_t = dt - (time.perf_counter() - t0)
    if sleep_t > 0:
        time.sleep(sleep_t)


def _compute_freeze_token(encoder, state_reader, last_token, last_left, last_right):
    """
    Read current robot state and run encoder to produce a freeze token.
    Falls back to last_token if state is unavailable.

    Returns (freeze_token, last_left, last_right) — hand joints unchanged.
    """
    state = state_reader.get_state() if state_reader is not None else None
    if state is not None:
        qpos      = _mujoco29_to_isaaclab29(state["qpos"])  # (29,) reordered to IsaacLab
        base_quat = state["base_quat"]                      # (4,) wxyz

        joint_pos = np.tile(qpos,      (10, 1)).astype(np.float32)  # (10, 29)
        joint_vel = np.zeros((10, 29), dtype=np.float32)
        body_quat = np.tile(base_quat, (10, 1)).astype(np.float32)  # (10, 4)

        freeze_token = encoder.encode(joint_pos, joint_vel, body_quat)  # (64,)
        print(f"\n[ReplayLatency] Encoder freeze token computed from robot state.")
    else:
        freeze_token = last_token
        print(f"\n[ReplayLatency] No robot state available — repeating last token.")
    return freeze_token, last_left, last_right


def replay(episode_dir, zmq_host, zmq_pub_port, zmq_topic,
           frequency, chunk_size, latency_frames, dry_run=False,
           wbc_host=DEFAULT_WBC_HOST, wbc_port=DEFAULT_WBC_PORT,
           wbc_topic=DEFAULT_WBC_TOPIC):
    frames, _ = load_episode(episode_dir)
    dt = 1.0 / frequency
    tokens = _extract_tokens(frames)
    total = len(tokens)

    publisher = TokenPublisher(host=zmq_host, port=zmq_pub_port, topic=zmq_topic)

    # Start WBC state reader and encoder (skip in dry-run)
    state_reader = None
    encoder      = None
    if not dry_run:
        state_reader = WBCStateReader(host=wbc_host, port=wbc_port, topic=wbc_topic)
        encoder_path = os.path.join(_GROOT_ROOT, ENCODER_MODEL)
        encoder = EncoderClient(encoder_path, mode=0)

    print("[ReplayLatency] Waiting for ZMQ connections...")
    time.sleep(1.0)

    if not dry_run:
        publisher.send_command(start=True, stop=False, planner=True)
        time.sleep(0.2)

    n_chunks = (total + chunk_size - 1) // chunk_size
    latency_ms = latency_frames * (1000.0 / frequency)
    print(f"[ReplayLatency] {total} tokens → {n_chunks} chunks")
    print(f"[ReplayLatency] chunk_size={chunk_size}  latency_frames={latency_frames} ({latency_ms:.0f} ms)")

    try:
        for chunk_idx in range(n_chunks):
            chunk = tokens[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]

            # ── EXECUTING: send chunk ─────────────────────────────────────
            for j, (token, left_hand, right_hand) in enumerate(chunk):
                frame_no = chunk_idx * chunk_size + j + 1
                print(f"[ReplayLatency] chunk {chunk_idx+1}/{n_chunks}  frame {frame_no:4d}/{total}  [EXEC  ]", end="\r")
                _send_one(publisher, token, left_hand, right_hand, dt, dry_run)

            # ── WAITING: encoder freeze token for latency_frames ticks ────
            last_token, last_left, last_right = chunk[-1]
            if not dry_run:
                freeze_token, freeze_left, freeze_right = _compute_freeze_token(
                    encoder, state_reader, last_token, last_left, last_right
                )
            else:
                freeze_token, freeze_left, freeze_right = last_token, last_left, last_right

            for h in range(latency_frames):
                print(f"[ReplayLatency] chunk {chunk_idx+1}/{n_chunks}  hold   {h+1:4d}/{latency_frames}  [WAITING]", end="\r")
                _send_one(publisher, freeze_token, freeze_left, freeze_right, dt, dry_run)

    except KeyboardInterrupt:
        print("\n[ReplayLatency] Interrupted by user.")
        publisher.stop()
        if state_reader:
            state_reader.close()
        return

    print(f"\n[ReplayLatency] Done. Sent {publisher._frame_index} frames.")

    last_token, last_left, last_right = tokens[-1]
    print("[ReplayLatency] Holding last token. Press Ctrl+C to stop.")
    try:
        while True:
            _send_one(publisher, last_token, last_left, last_right, dt, dry_run)
    except KeyboardInterrupt:
        print("\n[ReplayLatency] Hold interrupted by user.")

    publisher.stop()
    if state_reader:
        state_reader.close()


def main():
    parser = argparse.ArgumentParser(
        description="Replay tokens with configurable chunk size and simulated inference latency"
    )
    parser.add_argument("--episode-dir", type=str, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--zmq-host", type=str, default=DEFAULT_ZMQ_HOST)
    parser.add_argument("--zmq-pub-port", type=int, default=DEFAULT_ZMQ_PUB_PORT)
    parser.add_argument("--zmq-topic", type=str, default=DEFAULT_ZMQ_TOPIC)
    parser.add_argument("--frequency", type=int, default=DEFAULT_RECORDING_FREQ,
                        help="Playback frequency in Hz (default: 30)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Tokens per inference chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--latency-frames", type=int, default=DEFAULT_LATENCY_FRAMES,
                        help=f"Hold frames between chunks to simulate inference latency "
                             f"(default: {DEFAULT_LATENCY_FRAMES} = "
                             f"{DEFAULT_LATENCY_FRAMES*1000//DEFAULT_RECORDING_FREQ}ms)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse frames without sending (skips WBC/encoder)")
    parser.add_argument("--wbc-host", type=str, default=DEFAULT_WBC_HOST,
                        help=f"WBC state publisher host (default: {DEFAULT_WBC_HOST})")
    parser.add_argument("--wbc-port", type=int, default=DEFAULT_WBC_PORT,
                        help=f"WBC state publisher port (default: {DEFAULT_WBC_PORT})")
    parser.add_argument("--wbc-topic", type=str, default=DEFAULT_WBC_TOPIC,
                        help=f"WBC state topic (default: {DEFAULT_WBC_TOPIC})")
    args = parser.parse_args()

    replay(
        episode_dir=args.episode_dir,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_topic=args.zmq_topic,
        frequency=args.frequency,
        chunk_size=args.chunk_size,
        latency_frames=args.latency_frames,
        dry_run=args.dry_run,
        wbc_host=args.wbc_host,
        wbc_port=args.wbc_port,
        wbc_topic=args.wbc_topic,
    )


if __name__ == "__main__":
    main()
