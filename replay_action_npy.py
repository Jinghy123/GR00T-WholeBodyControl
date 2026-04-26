#!/usr/bin/env python3
"""
replay_action_npy.py

Replay a raw action.npy file (shape [T, D] float32, each row is a token)
by publishing tokens to the WBC controller via ZMQ at 30 Hz.

Uses ZMQ Protocol v4 (token-only streaming). No hand joints are sent.

Usage:
    python replay_action_npy.py
    python replay_action_npy.py --npy-path <path>
"""

# ── Hard-coded defaults ──────────────────────────────────────────────────────
DEFAULT_NPY_PATH       = "/home/xiawei/data/actions_pour_water.npy"
DEFAULT_TOKEN_DIM      = 64   # layout: hand_joints(14) + token(64) = 78
DEFAULT_HAND_DIM       = 14
DEFAULT_ZMQ_HOST       = "*"

# FSQ configuration (from g1_sonic_client.py)
FSQ_MIN  = -0.625
FSQ_MAX  = 0.625
FSQ_STEP = 0.0625  # = 1/16
DEFAULT_ZMQ_PUB_PORT   = 5556
DEFAULT_ZMQ_TOPIC      = "pose"
DEFAULT_RECORDING_FREQ = 30
DEFAULT_REPEAT         = 1  # send each action 30 times (≈1s each at 30Hz)
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys
import time

import numpy as np
import zmq

_GROOT_ROOT = os.path.expanduser("~/hsc/GR00T-WholeBodyControl")
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


def fsq_quantize(x, fsq_min=FSQ_MIN, fsq_max=FSQ_MAX, fsq_step=FSQ_STEP):
    clipped = np.clip(x, fsq_min, fsq_max)
    quantized = np.round(clipped / fsq_step) * fsq_step
    return np.clip(quantized, fsq_min, fsq_max)


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


def load_tokens(npy_path, token_dim):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"npy not found: {npy_path}")
    arr = np.load(npy_path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array [T, D], got shape {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    print(f"[ReplayActionNpy] Loaded npy shape={arr.shape} from {npy_path}")
    expected = DEFAULT_HAND_DIM + token_dim
    if arr.shape[1] != expected:
        raise ValueError(f"npy cols={arr.shape[1]} != expected {expected} (hand {DEFAULT_HAND_DIM} + token {token_dim})")
    hands = arr[:, :DEFAULT_HAND_DIM]
    tokens_raw = arr[:, DEFAULT_HAND_DIM:]
    tokens = fsq_quantize(tokens_raw).astype(np.float32)
    print(f"[ReplayActionNpy] Layout: hand_joints(14) + token({token_dim})")
    print(f"[ReplayActionNpy] Token FSQ-quantized: raw_range=[{tokens_raw.min():.4f},{tokens_raw.max():.4f}] "
          f"-> qtz_range=[{tokens.min():.4f},{tokens.max():.4f}]")
    return tokens, hands


def build_phases(tokens, hands, mode, repeat):
    """Return list of (label, token(N,), left_hand|None, right_hand|None, repeat)."""
    def split_hand(row):
        if hands is None:
            return None, None
        return row[:7], row[7:]

    if mode == "normal":
        phases = []
        for i, tok in enumerate(tokens):
            lh, rh = split_hand(hands[i]) if hands is not None else (None, None)
            phases.append((f"action_{i+1}", tok, lh, rh, repeat))
        return phases

    if mode == "sweep":
        tok0 = tokens[0]
        lh0, rh0 = split_hand(hands[0]) if hands is not None else (None, None)
        phases = [("orig", tok0, lh0, rh0, repeat)]
        for d in range(tok0.shape[0]):
            perturbed = tok0.copy()
            perturbed[d] = perturbed[d] - FSQ_STEP
            perturbed = fsq_quantize(perturbed).astype(np.float32)
            phases.append((f"dim{d:02d}-1", perturbed, lh0, rh0, repeat))
        return phases

    raise ValueError(f"unknown mode: {mode}")


def replay(npy_path, zmq_host, zmq_pub_port, zmq_topic, frequency, token_dim, repeat, mode, dry_run=False):
    tokens, hands = load_tokens(npy_path, token_dim)
    dt = 1.0 / frequency

    phases = build_phases(tokens, hands, mode, repeat)
    total_sends = sum(p[4] for p in phases)

    publisher = TokenPublisher(host=zmq_host, port=zmq_pub_port, topic=zmq_topic)

    print("[ReplayActionNpy] Waiting for ZMQ connections...")
    time.sleep(1.0)

    if not dry_run:
        publisher.send_command(start=True, stop=False, planner=True)
        time.sleep(0.2)

    print(f"[ReplayActionNpy] Mode={mode}  phases={len(phases)}  total_sends={total_sends} at {frequency} Hz")

    last_token = None
    last_left_hand = None
    last_right_hand = None
    send_idx = 0
    try:
        for p_idx, (label, token, left_hand, right_hand, reps) in enumerate(phases):
            print(f"\n[ReplayActionNpy] Phase {p_idx+1}/{len(phases)}: {label}  "
                  f"token range=[{token.min():.4f},{token.max():.4f}]  x{reps}")

            for r in range(reps):
                t0 = time.perf_counter()

                if not dry_run:
                    publisher.publish_token(token, left_hand=left_hand, right_hand=right_hand)

                send_idx += 1
                elapsed = send_idx / frequency
                hand_info = ""
                if left_hand is not None and right_hand is not None:
                    hand_info = f" L={np.linalg.norm(left_hand):.2f} R={np.linalg.norm(right_hand):.2f}"
                print(f"[ReplayActionNpy] {label} rep {r+1:2d}/{reps}  send {send_idx}/{total_sends}  t={elapsed:.2f}s{hand_info}", end="\r")

                sleep_t = dt - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)

            last_token = token
            last_left_hand = left_hand
            last_right_hand = right_hand

    except KeyboardInterrupt:
        print("\n[ReplayActionNpy] Interrupted by user.")
        publisher.stop()
        return

    print(f"\n[ReplayActionNpy] Done. Sent {publisher._frame_index} frames.")

    if not dry_run and last_token is not None:
        print("[ReplayActionNpy] Holding last token. Press Ctrl+C to stop.")
        try:
            while True:
                t0 = time.perf_counter()
                publisher.publish_token(last_token, left_hand=last_left_hand, right_hand=last_right_hand)
                sleep_t = dt - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        except KeyboardInterrupt:
            print("\n[ReplayActionNpy] Hold interrupted by user.")

    publisher.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Replay a raw action.npy file via token-only streaming (Protocol v4)"
    )
    parser.add_argument("--npy-path", type=str, default=DEFAULT_NPY_PATH)
    parser.add_argument("--zmq-host", type=str, default=DEFAULT_ZMQ_HOST)
    parser.add_argument("--zmq-pub-port", type=int, default=DEFAULT_ZMQ_PUB_PORT)
    parser.add_argument("--zmq-topic", type=str, default=DEFAULT_ZMQ_TOPIC)
    parser.add_argument("--frequency", type=int, default=DEFAULT_RECORDING_FREQ)
    parser.add_argument("--token-dim", type=int, default=DEFAULT_TOKEN_DIM)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help="Send each action this many times (default: 30)")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "sweep"],
                        help="normal: replay all actions. sweep: use only first token and run 4 phases "
                             "[orig, +1 FSQ step, orig, -1 FSQ step], each x --repeat times")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    replay(
        npy_path=args.npy_path,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_topic=args.zmq_topic,
        frequency=args.frequency,
        token_dim=args.token_dim,
        repeat=args.repeat,
        mode=args.mode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
