#!/usr/bin/env python3
"""
replay_wuji_hand_direct.py

Direct Wuji hand replay - controls hardware directly without wuji_hand_server.py.

Usage:
    python replay_wuji_hand_direct.py --episode-dir <path>
    python replay_wuji_hand_direct.py --episode-dir <path> --hand-side left
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import wujihandpy
except ImportError:
    print("[ERROR] wujihandpy not installed. Install with: pip install wujihandpy")
    sys.exit(1)


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
    """Extract Wuji hand actions from episode data."""
    actions = []
    hand_idx = 0 if hand_side == "left" else 1

    for frame in data:
        try:
            # Note: key is 'actions' (plural), not 'action'
            action = frame.get("actions", {})
            hand_joints = action.get("hand_joints", None)

            if hand_joints is None:
                continue

            hand_joints = np.array(hand_joints, dtype=np.float32)

            # Wuji format: [left_20, right_20] = 40 total
            if len(hand_joints) == 40:
                wuji_20d = hand_joints[hand_idx * 20:(hand_idx + 1) * 20]
            elif len(hand_joints) == 20:
                wuji_20d = hand_joints
            else:
                print(f"[WARNING] Unexpected hand_joints shape {len(hand_joints)}, skipping")
                continue

            actions.append(wuji_20d)

        except Exception as e:
            print(f"[WARNING] Error processing frame: {e}")
            continue

    print(f"[Replay] Extracted {len(actions)} valid {hand_side} hand actions")
    return actions


def replay_wuji_actions(actions, serial_number="", fps=30):
    """Replay Wuji hand actions by directly controlling hardware."""
    print(f"[Replay] Initializing Wuji hand...")

    try:
        if serial_number:
            hand = wujihandpy.Hand(serial_number=serial_number)
        else:
            hand = wujihandpy.Hand()

        hand.write_joint_enabled(True)
        controller = hand.realtime_controller(
            enable_upstream=True,
            filter=wujihandpy.filter.LowPass(cutoff_freq=10.0),
        )
        print(f"[Replay] Wuji hand ready")

    except Exception as e:
        print(f"[ERROR] Failed to initialize Wuji hand: {e}")
        sys.exit(1)

    print(f"[Replay] Starting replay at {fps} Hz...")
    print(f"[Replay] Press Ctrl-C to stop")

    frame_interval = 1.0 / fps

    try:
        for i, action_20d in enumerate(actions):
            # Reshape to (5, 4) for wujihandpy
            qpos = action_20d.reshape(5, 4).astype(np.float32)

            # Send to hardware
            controller.set_joint_target_position(qpos)

            if i % 30 == 0:  # Print every second
                print(f"[Replay] Frame {i}/{len(actions)}: action = {qpos[0]}...")

            # Maintain frame rate
            time.sleep(frame_interval)

        print(f"[Replay] Completed {len(actions)} frames")

    except KeyboardInterrupt:
        print(f"\n[Replay] Stopped by user at frame {i}/{len(actions)}")

    finally:
        try:
            # Return to zero pose
            zero_pose = np.zeros((5, 4), dtype=np.float32)
            for _ in range(50):
                controller.set_joint_target_position(zero_pose)
                time.sleep(0.02)
            print("[Replay] Returned to zero pose")
        except Exception:
            pass

        try:
            controller.close()
            hand.write_joint_enabled(False)
            print("[Replay] Hand disabled")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Replay Wuji hand data (direct hardware control)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Replay left hand from episode
    python replay_wuji_hand_direct.py --episode-dir ~/data/demonstration_2025-01-01_12-00-00/episode_0

    # Replay right hand with specific serial number
    python replay_wuji_hand_direct.py --episode-dir <path> --hand-side right --serial-number 337238793233
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
        "--serial-number",
        type=str,
        default="",
        help="Wuji device serial number (leave empty to use first found)"
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

    replay_wuji_actions(
        actions,
        serial_number=args.serial_number,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
