#!/usr/bin/env python3
"""
Apply the initial pose from data.json to the robot.
Uses the encoder to convert qpos to token, matching g1_sonic_client.py behavior.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import zmq

_GROOT_ROOT = os.path.expanduser("/mnt/data/weiduo/heng/GR00T-WholeBodyControl")
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

# ZMQ configuration
DEFAULT_ZMQ_HOST = "*"
DEFAULT_ZMQ_PORT = 5556
DEFAULT_ZMQ_TOPIC = "pose"

# Neck publisher configuration
DEFAULT_NECK_PUB_HOST = "*"
DEFAULT_NECK_PUB_PORT = 5570

# Encoder model path
ENCODER_MODEL = os.path.join(_GROOT_ROOT, "gear_sonic_deploy/policy/release/model_encoder.onnx")

HAND_DIM = 14
NECK_DIM = 2
TOKEN_DIM = 64


def load_initial_pose(data_path):
    """Load the first frame's state from data.json."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    if not data:
        raise ValueError("data.json is empty")

    first_frame = data[0]
    state = first_frame['states']
    action = first_frame['actions']

    # Use STATES for qpos/hand (actual robot position), ACTION for neck (command format)
    # states.neck is motor feedback, actions.neck is the command format needed
    initial_pose = {
        'qpos': np.array(state['qpos'], dtype=np.float32),  # (29,) - actual position
        'quat': np.array(state['quat'], dtype=np.float32),  # (4,) wxyz - only in state
        'hand_joints': np.array(state['hand_joints'], dtype=np.float32),  # (14,) - actual position
        'neck': np.array(action['neck'], dtype=np.float32),  # (2,) - command format!
    }

    return initial_pose


def apply_initial_pose(pose, zmq_host, zmq_port, zmq_topic, neck_pub_host, neck_pub_port):
    """Publish the initial pose to the robot."""

    # Initialize encoder
    print("[Encoder] Loading model...")
    encoder = EncoderClient(ENCODER_MODEL, mode=0)
    print("[Encoder] Model loaded")

    # Create ZMQ context
    ctx = zmq.Context()

    # Token publisher (body + hand joints)
    token_pub = ctx.socket(zmq.PUB)
    token_pub.bind(f"tcp://{zmq_host}:{zmq_port}")
    print(f"[TokenPub] PUB bound to tcp://{zmq_host}:{zmq_port}")

    # Neck publisher
    neck_pub = ctx.socket(zmq.PUB)
    neck_pub.setsockopt(zmq.SNDHWM, 1)
    neck_pub.setsockopt(zmq.LINGER, 0)
    neck_pub.bind(f"tcp://{neck_pub_host}:{neck_pub_port}")
    print(f"[NeckPub] PUB bound to tcp://{neck_pub_host}:{neck_pub_port}")

    time.sleep(0.2)  # Wait for connections to establish

    # Send start command first
    start_cmd = build_command_message(start=True, stop=False, planner=True)
    token_pub.send(start_cmd)
    print(f"[TokenPub] Sent start command")

    time.sleep(0.1)

    # Encode the initial pose to get token
    # Convert qpos from Mujoco order to IsaacLab order (encoder expects IsaacLab)
    qpos_isaaclab = _mujoco29_to_isaaclab29(pose['qpos'])
    base_quat = pose['quat']  # (4,) wxyz

    # Prepare encoder inputs (matching g1_sonic_client.py _publish_loop)
    joint_pos = np.tile(qpos_isaaclab, (10, 1)).astype(np.float32)  # (10, 29)
    joint_vel = np.zeros((10, 29), dtype=np.float32)
    body_quat = np.tile(base_quat, (10, 1)).astype(np.float32)  # (10, 4)

    # Encode to get token
    token = encoder.encode(joint_pos, joint_vel, body_quat)  # (64,)
    print(f"[Encoder] Generated token from qpos, range=[{token.min():.4f},{token.max():.4f}]")

    # Build action: hand(14) + neck(2) + token(64) = 80 (include-neck mode)
    action = np.concatenate([
        pose['hand_joints'],  # (14,)
        pose['neck'],          # (2,)
        token,                 # (64,)
    ])

    # Publish the action (matching g1_sonic_client.py publish_token)
    pose_data = {
        "token_state": action[HAND_DIM + NECK_DIM:HAND_DIM + NECK_DIM + TOKEN_DIM].reshape(1, -1),
        "left_hand_joints": action[:7].reshape(1, 7),
        "right_hand_joints": action[7:14].reshape(1, 7),
        "body_quat_w": np.asarray(base_quat, dtype=np.float32).reshape(1, 4),
    }
    msg = pack_pose_message(pose_data, topic=zmq_topic, version=4)
    token_pub.send(msg)
    print(f"[TokenPub] Sent action with body_quat_w={base_quat}")

    # Publish neck angle (realsense_server.py will apply NECK_YAW_SIGN and NECK_PITCH_SIGN)
    neck_msg = json.dumps([float(pose['neck'][0]), float(pose['neck'][1])]).encode("utf-8")
    neck_pub.send(neck_msg)
    print(f"[NeckPub] Sent neck: yaw={pose['neck'][0]:.4f}, pitch={pose['neck'][1]:.4f}")

    print("\n[Done] Initial pose sent. The robot should now move to the starting position.")

    # Send a few more times to ensure delivery
    for _ in range(5):
        token_pub.send(msg)
        neck_pub.send(neck_msg)
        time.sleep(0.01)

    # Cleanup
    token_pub.close()
    neck_pub.close()
    ctx.term()
    print("[Exit] Done.")


def main():
    parser = argparse.ArgumentParser(description="Apply initial pose from data.json")
    parser.add_argument("--data", type=str,
                       default="/mnt/data/weiduo/heng/GR00T-WholeBodyControl/data.json",
                       help="Path to data.json file")
    parser.add_argument("--zmq-host", type=str, default=DEFAULT_ZMQ_HOST,
                       help="ZMQ publisher bind host (default: *)")
    parser.add_argument("--zmq-port", type=int, default=DEFAULT_ZMQ_PORT,
                       help="ZMQ publisher port (default: 5556)")
    parser.add_argument("--zmq-topic", type=str, default=DEFAULT_ZMQ_TOPIC,
                       help="ZMQ topic (default: pose)")
    parser.add_argument("--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST,
                       help=f"Neck PUB bind host (default: {DEFAULT_NECK_PUB_HOST})")
    parser.add_argument("--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT,
                       help=f"Neck PUB port (default: {DEFAULT_NECK_PUB_PORT})")

    args = parser.parse_args()

    # Load initial pose
    print(f"[Load] Reading initial pose from {args.data}")
    pose = load_initial_pose(args.data)

    print(f"[Pose] qpos[0:5]: {pose['qpos'][:5]}")
    print(f"[Pose] quat (wxyz): {pose['quat']}")
    print(f"[Pose] hand_joints: {pose['hand_joints']}")
    print(f"[Pose] neck (yaw, pitch) from actions: {pose['neck']}")

    # Apply pose
    apply_initial_pose(
        pose,
        args.zmq_host,
        args.zmq_port,
        args.zmq_topic,
        args.neck_pub_host,
        args.neck_pub_port,
    )


if __name__ == "__main__":
    main()
