"""Desktop-side head-pose publisher for the G1 neck-motor loop.

Runs on whichever machine has the XRoboToolkit PC Service daemon and the
Python binding installed (typically a x86_64 desktop). Reads the Pico
headset pose via `xrobotoolkit_sdk` and publishes the raw 7-vector
`[x, y, z, qx, qy, qz, qw]` as JSON over a ZMQ PUB socket.

On the G1, point `realsense_server.py --pose-zmq tcp://<desktop-ip>:5559`
at this publisher to bypass running the XRoboToolKit daemon on aarch64.

Prereqs (same env as the daemon + SDK):
    export PYTHONPATH=".../XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$PYTHONPATH"
    export LD_LIBRARY_PATH=".../XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/lib:$LD_LIBRARY_PATH"
    pip install pyzmq

Usage:
    python pose_publisher.py --bind tcp://0.0.0.0:5559 --hz 50
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
            "Publish Pico headset pose over ZMQ PUB for the remote "
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
    args = p.parse_args()

    import xrobotoolkit_sdk as xrt
    xrt.init()
    print("[pose_publisher] XRoboToolkit SDK initialized.")

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
    n_zero_warned = False
    n_valid = 0
    next_tick = time.time()

    try:
        while running[0]:
            next_tick += dt
            pose = xrt.get_headset_pose()
            if pose is not None and len(pose) >= 7:
                qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
                if qx * qx + qy * qy + qz * qz + qw * qw > 1e-6:
                    sock.send(
                        json.dumps([float(v) for v in pose]).encode("utf-8")
                    )
                    n_valid += 1
                    n_zero_warned = False
                elif not n_zero_warned:
                    print(
                        "[pose_publisher] quat is zero — Pico not streaming "
                        "(check service daemon + XRoboToolkit app on headset)."
                    )
                    n_zero_warned = True

            if n_valid and n_valid % (args.hz * 2) == 0:
                print(
                    f"[pose_publisher] published {n_valid} poses @ ~{args.hz} Hz"
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
