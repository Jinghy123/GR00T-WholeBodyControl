"""Desktop-side neck-angle publisher for the G1 neck-motor loop.

Runs on the machine with the XRoboToolKit daemon, the body trackers, and the
`general_motion_retargeting` package. Pulls SMPL-X body data from
`XRobotStreamer`, extracts neck yaw/pitch via `human_head_to_robot_neck`, and
publishes the 2-vector `[neck_yaw, neck_pitch]` (radians, robot convention)
as JSON over a ZMQ PUB socket.

Going through SMPL-X instead of the raw headset pose decouples neck rotation
from torso lean: the spine joints absorb body lean, leaving the neck angles
unchanged when the operator walks or leans without rotating the head.

On the G1, point `realsense_server.py --pose-zmq tcp://<desktop-ip>:5559`
at this publisher.

Prereqs (desktop):
    conda activate gmr
    # see TOPOLOGY2_QUICKSTART.md for the gmr install steps.

Usage:
    python pose_publisher.py --bind tcp://0.0.0.0:5559 --hz 50 \
        --neck-retarget-scale 1.5
"""
from __future__ import annotations

import argparse
import json
import signal
import time

import zmq


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Publish SMPL-X-derived neck angles over ZMQ PUB for the remote "
            "realsense_server.py neck-motor loop."
        )
    )
    p.add_argument(
        "--bind", default="tcp://0.0.0.0:5559",
        help="ZMQ PUB bind address (default tcp://0.0.0.0:5559)",
    )
    p.add_argument(
        "--hz", type=int, default=50,
        help="Publish rate in Hz (default 50, matches NECK_CONTROL_HZ)",
    )
    p.add_argument(
        "--neck-retarget-scale", type=float, default=1.5,
        help=(
            "Multiplier applied to (yaw, pitch) before publishing — matches "
            "the `--neck_retarget_scale` knob from the past pipeline."
        ),
    )
    args = p.parse_args()

    from general_motion_retargeting import XRobotStreamer, human_head_to_robot_neck

    streamer = XRobotStreamer()
    print("[pose_publisher] XRobotStreamer initialized.")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)   # keep only the newest frame
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(args.bind)
    print(f"[pose_publisher] PUB bound to {args.bind}")

    running = [True]

    def _stop(sig, _frame):
        print(f"\n[pose_publisher] signal {sig}, stopping.")
        running[0] = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    dt = 1.0 / args.hz
    scale = args.neck_retarget_scale
    n_no_data_warned = False
    n_valid = 0
    next_tick = time.time()

    try:
        while running[0]:
            next_tick += dt
            smplx_data, *_rest = streamer.get_current_frame()
            if smplx_data is not None:
                neck_yaw, neck_pitch = human_head_to_robot_neck(smplx_data)
                neck_yaw  *= scale
                neck_pitch *= scale
                sock.send(
                    json.dumps([float(neck_yaw), float(neck_pitch)]).encode("utf-8")
                )
                n_valid += 1
                n_no_data_warned = False
            elif not n_no_data_warned:
                print(
                    "[pose_publisher] smplx_data is None — body trackers not "
                    "streaming yet (check XRoboToolKit daemon + headset/trackers)."
                )
                n_no_data_warned = True

            if n_valid and n_valid % (args.hz * 2) == 0:
                print(
                    f"[pose_publisher] published {n_valid} neck samples @ ~{args.hz} Hz"
                )

            sleep_s = next_tick - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.time()
    finally:
        sock.close(linger=0)
        ctx.term()
        print("[pose_publisher] stopped cleanly.")


if __name__ == "__main__":
    main()
