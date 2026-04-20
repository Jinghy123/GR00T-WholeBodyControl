#!/usr/bin/env python3
"""
replay_wuji_hand.py

Replay recorded Wuji hand data (left hand only) from a saved episode.

Reads data.json from an episode directory and publishes left hand actions
to wuji_hand_server.py via ZMQ at the original recording frequency.

Usage:
    # Terminal 1: Start wuji_hand_server.py with replay enabled
    cd /path/to/GR00T-WholeBodyControl
    python wuji_hand_server.py --hand_side left --no_smooth --enable-replay-commands

    # Terminal 2: Run replay
    python replay_wuji_hand.py --episode-dir <path>

Requirements:
    - wuji_hand_server.py must be running with --enable-replay-commands flag
    - Episode directory must contain data.json with Wuji hand data

Workflow:
    1. Record an episode using g1_data_server.py (press 's' to start, 'q' to save)
    2. Find the episode directory (e.g., ~/data/demonstration_2025-04-20_12-00-00/episode_0)
    3. Stop wuji_hand_server.py and restart with --enable-replay-commands
    4. Run replay: python replay_wuji_hand.py --episode-dir <episode_dir>
    5. Watch the Wuji hand replicate the recorded motions
"""

import argparse
import json
import os
import sys
import time

import msgpack
import numpy as np
import zmq


# Default configuration
DEFAULT_EPISODE_DIR = ""
DEFAULT_WUJI_HOST = "localhost"
DEFAULT_WUJI_PORT = 5561  # wuji_hand_server.py command receiver port (state_port + 1)


def load_episode_data(episode_dir):
    """Load data.json from episode directory."""
    data_file = os.path.join(episode_dir, "data.json")
    if not os.path.exists(data_file):
        print(f"[ERROR] data.json not found in {episode_dir}")
        sys.exit(1)

    with open(data_file, "r") as f:
        data = json.load(f)

    print(f"[Replay] Loaded {len(data)} frames from {data_file}")
    return data


def extract_wuji_actions(data, hand_side="left"):
    """Extract Wuji hand actions from episode data.

    Args:
        data: List of frames from data.json
        hand_side: "left" or "right"

    Returns:
        List of (timestamp, action_20d) tuples
    """
    actions = []
    hand_idx = 0 if hand_side == "left" else 1

    for frame in data:
        try:
            # Note: key is 'actions' (plural), not 'action'
            action = frame.get("actions", {})
            hand_joints = action.get("hand_joints", None)

            if hand_joints is None:
                print(f"[WARNING] Frame missing hand_joints, skipping")
                continue

            hand_joints = np.array(hand_joints, dtype=np.float32)

            # Wuji format: [left_20, right_20] = 40 total
            if len(hand_joints) == 40:
                wuji_20d = hand_joints[hand_idx * 20:(hand_idx + 1) * 20]
            elif len(hand_joints) == 20:
                # Only one hand recorded
                wuji_20d = hand_joints
            else:
                print(f"[WARNING] Unexpected hand_joints shape {len(hand_joints)}, skipping")
                continue

            # Get timestamp if available
            timestamp = frame.get("timestamp", time.time())

            actions.append((timestamp, wuji_20d))

        except Exception as e:
            print(f"[WARNING] Error processing frame: {e}")
            continue

    print(f"[Replay] Extracted {len(actions)} valid {hand_side} hand actions")
    return actions


def replay_wuji_actions(actions, wuji_host, wuji_port, hand_side, fps=30):
    """Replay Wuji hand actions via ZMQ to wuji_hand_server.py.

    Note: This requires wuji_hand_server.py to be modified to accept external
    commands via ZMQ. Currently it only has state publishing.
    """
    print(f"[Replay] Starting replay at {fps} Hz...")
    print(f"[Replay] Press Ctrl-C to stop")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{wuji_port}")

    # Give subscriber time to connect
    time.sleep(1)

    try:
        frame_interval = 1.0 / fps

        for i, (timestamp, action_20d) in enumerate(actions):
            # Send command
            msg = msgpack.packb({
                "hand_side": hand_side,
                "action": action_20d.tolist(),
                "timestamp": time.time(),
            }, use_bin_type=True)

            sock.send(b"wuji_replay" + msg)

            if i % 30 == 0:  # Print every second
                print(f"[Replay] Frame {i}/{len(actions)}: action = {action_20d[:5]}...")

            # Maintain frame rate
            time.sleep(frame_interval)

        print(f"[Replay] Completed {len(actions)} frames")

    except KeyboardInterrupt:
        print(f"\n[Replay] Stopped by user at frame {i}/{len(actions)}")

    finally:
        sock.close()
        ctx.term()


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded Wuji hand data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Replay left hand from episode
    python replay_wuji_hand.py --episode-dir ~/data/demonstration_2025-01-01_12-00-00/episode_0

    # Replay right hand
    python replay_wuji_hand.py --episode-dir <path> --hand-side right
        """
    )

    parser.add_argument(
        "--episode-dir",
        type=str,
        required=True,
        help="Path to episode directory containing data.json"
    )
    parser.add_argument(
        "--hand-side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Hand to replay (default: left)"
    )
    parser.add_argument(
        "--wuji-host",
        type=str,
        default=DEFAULT_WUJI_HOST,
        help=f"Wuji hand server host (default: {DEFAULT_WUJI_HOST})"
    )
    parser.add_argument(
        "--wuji-port",
        type=int,
        default=DEFAULT_WUJI_PORT,
        help=f"Wuji hand server command port (default: {DEFAULT_WUJI_PORT})"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Replay frame rate (default: 30)"
    )

    args = parser.parse_args()

    # Load episode data
    data = load_episode_data(args.episode_dir)

    # Extract Wuji actions
    actions = extract_wuji_actions(data, args.hand_side)

    if not actions:
        print("[ERROR] No valid Wuji actions found in episode data")
        sys.exit(1)

    # Replay
    print(f"[Replay] Replaying {args.hand_side} hand ({len(actions)} frames)")
    print(f"[Replay] Episode: {args.episode_dir}")
    print(f"[Replay] Target: {args.wuji_host}:{args.wuji_port}")

    replay_wuji_actions(
        actions,
        args.wuji_host,
        args.wuji_port,
        args.hand_side,
        args.fps
    )


if __name__ == "__main__":
    main()
