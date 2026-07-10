#!/usr/bin/env python3
"""
Reset the G1 neck to its initial position (yaw=0, pitch=0 by default).

Publishes `[yaw, pitch]` JSON on the neck PUB port (default 5570), the same
wire format as pose_publisher.py / NeckPublisher in g1_sonic_client.py, so the
NeckMotor SUB inside realsense_server.py / realsense_native_server.py consumes
it without any change. The on-robot NeckMotor applies its own EMA smoothing
(NECK_SMOOTH_ALPHA), so sending the raw target is safe.

NOTE: stop pose_publisher.py / pico_manus_thread_server.py /
g1_sonic_client_rtc.py before running this, or the bind on port 5570 will
collide.

Usage:
    python reset_neck.py                 # reset to (0, 0)
    python reset_neck.py --yaw 0.2 --pitch -0.1
"""

import argparse
import json
import time

import zmq

DEFAULT_NECK_PUB_HOST = "*"
DEFAULT_NECK_PUB_PORT = 5570
DEFAULT_NECK_STATE_ZMQ = "tcp://192.168.123.164:5560"


def read_neck_state(ctx, addr, wait_s=1.0):
    """Best-effort read of the current neck present position from 5560."""
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(addr)
    latest = None
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            raw = sock.recv(flags=zmq.NOBLOCK)
            msg = json.loads(raw.decode("utf-8"))
            if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                latest = [float(msg[0]), float(msg[1])]
        except zmq.Again:
            time.sleep(0.02)
        except (ValueError, UnicodeDecodeError):
            pass
    sock.close(linger=0)
    return latest


def main():
    parser = argparse.ArgumentParser(description="Reset the G1 neck position")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="Target neck yaw in rad (default: 0.0)")
    parser.add_argument("--pitch", type=float, default=0.0,
                        help="Target neck pitch in rad (default: 0.0)")
    parser.add_argument("--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST,
                        help=f"Neck PUB bind host (default: {DEFAULT_NECK_PUB_HOST})")
    parser.add_argument("--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT,
                        help=f"Neck PUB port (default: {DEFAULT_NECK_PUB_PORT})")
    parser.add_argument("--neck-state-zmq", type=str, default=DEFAULT_NECK_STATE_ZMQ,
                        help="Neck present-position SUB address, used only for "
                             f"before/after printout (default: {DEFAULT_NECK_STATE_ZMQ})")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="How long to keep publishing the target, in seconds "
                             "(covers PUB/SUB slow-joiner + NeckMotor EMA settle; default: 3.0)")
    args = parser.parse_args()

    ctx = zmq.Context()

    state = read_neck_state(ctx, args.neck_state_zmq, wait_s=1.0)
    if state is not None:
        print(f"[NeckState] current: yaw={state[0]:+.4f}, pitch={state[1]:+.4f}")
    else:
        print("[NeckState] no state received (camera server not running?), sending anyway")

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.setsockopt(zmq.LINGER, 0)
    bind_addr = f"tcp://{args.neck_pub_host}:{args.neck_pub_port}"
    try:
        pub.bind(bind_addr)
    except zmq.ZMQError as e:
        pub.close(linger=0)
        ctx.term()
        raise SystemExit(
            f"[NeckPub] bind failed on {bind_addr}: {e}\n"
            "Is pose_publisher.py / pico_manus_thread_server.py / "
            "g1_sonic_client_rtc.py still running? Stop it first."
        )
    print(f"[NeckPub] PUB bound to {bind_addr}")

    msg = json.dumps([float(args.yaw), float(args.pitch)]).encode("utf-8")
    print(f"[NeckPub] Sending target: yaw={args.yaw:+.4f}, pitch={args.pitch:+.4f} "
          f"for {args.duration:.1f}s")
    # The NeckMotor SUB uses CONFLATE (keeps only the newest sample), so
    # re-sending at 100 Hz is safe and rides out the PUB/SUB slow-joiner delay.
    end = time.time() + args.duration
    while time.time() < end:
        pub.send(msg)
        time.sleep(0.01)

    state = read_neck_state(ctx, args.neck_state_zmq, wait_s=1.0)
    if state is not None:
        print(f"[NeckState] after: yaw={state[0]:+.4f}, pitch={state[1]:+.4f}")

    pub.close(linger=0)
    ctx.term()
    print("[Exit] Done.")


if __name__ == "__main__":
    main()
