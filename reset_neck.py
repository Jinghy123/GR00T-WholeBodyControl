#!/usr/bin/env python3
"""
Reset the G1 neck to its initial position (yaw=0, pitch=0 by default).

Publishes `[yaw, pitch]` JSON on the neck PUB port (default 5570), the same
wire format as pose_publisher.py / NeckPublisher in g1_sonic_client.py, so the
NeckMotor SUB inside realsense_server.py / realsense_native_server.py consumes
it without any change.

The Dynamixels run with I-gain 0, so a plain zero target leaves a 3-5 deg
steady-state error (the servo stalls against gear friction before reaching the
goal tick). This script therefore closes the loop over the network: it reads
the present position from the neck state PUB (port 5560) and biases the
published target opposite to the measured error until the head actually sits
at the requested pose. Without a state stream it falls back to open-loop
publishing for --timeout seconds.

NOTE: stop pose_publisher.py / pico_manus_thread_server.py /
g1_sonic_client.py before running this, or the bind on port 5570 will collide.

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

# Must match NECK_YAW_SIGN / NECK_PITCH_SIGN in realsense_server.py: the server
# multiplies received targets by these before writing the motors, while the
# 5560 state stream is raw motor-frame radians.
YAW_SIGN = 1.0
PITCH_SIGN = -1.0

PUBLISH_HZ = 50.0
ADJUST_EVERY_S = 0.4    # bias update period (leave time for the EMA + motion)
ADJUST_GAIN = 0.7       # fraction of the measured error folded into the bias
BIAS_LIMIT_RAD = 0.35   # safety clamp on the compensation bias
TOL_RAD = 0.015         # ~0.86 deg: close enough to call it zero
SETTLE_COUNT = 3        # consecutive in-tolerance adjustments to finish
# Static friction holds the head anywhere within ~0.06 rad of the goal, so a
# proportional step near zero breaks free and slips past the tolerance (limit
# cycle). Inside NEAR_BAND creep in tiny steps instead, so the slip after
# breakaway stays bounded.
NEAR_BAND_RAD = 0.06
NEAR_STEP_MAX_RAD = 0.012


class NeckStateSub:
    """CONFLATE SUB of the `[yaw, pitch]` present-position stream (port 5560)."""

    def __init__(self, ctx, addr):
        self._sock = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(addr)
        self._latest = None

    def poll(self):
        try:
            raw = self._sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return self._latest
        try:
            msg = json.loads(raw.decode("utf-8"))
            if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                self._latest = [float(msg[0]), float(msg[1])]
        except (ValueError, UnicodeDecodeError):
            pass
        return self._latest

    def wait(self, wait_s):
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.poll() is not None and time.time() > deadline - wait_s + 0.3:
                # Keep polling a bit so CONFLATE settles on a fresh sample.
                return self._latest
            time.sleep(0.02)
        return self._latest

    def close(self):
        self._sock.close(linger=0)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


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
                        help="Neck present-position SUB address used for the "
                             f"closed loop (default: {DEFAULT_NECK_STATE_ZMQ})")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Give up after this many seconds (default: 15.0). "
                             "Also the open-loop publish duration when no state "
                             "stream is available.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Deprecated alias for --timeout.")
    args = parser.parse_args()
    if args.duration is not None:
        args.timeout = args.duration

    ctx = zmq.Context()

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
            "g1_sonic_client.py still running? Stop it first."
        )
    print(f"[NeckPub] PUB bound to {bind_addr}")

    state = NeckStateSub(ctx, args.neck_state_zmq)
    present = state.wait(1.0)
    if present is not None:
        print(f"[NeckState] current: yaw={present[0]:+.4f}, pitch={present[1]:+.4f}")
    else:
        print("[NeckState] no state stream on 5560 — open-loop fallback "
              f"(publishing raw target for {args.timeout:.1f}s)")

    # Desired pose in motor frame, for comparing against the 5560 stream.
    desired_motor = [args.yaw * YAW_SIGN, args.pitch * PITCH_SIGN]
    bias = [0.0, 0.0]  # additive compensation, in the published (client) frame
    signs = [YAW_SIGN, PITCH_SIGN]
    targets = [args.yaw, args.pitch]

    print(f"[NeckPub] target: yaw={args.yaw:+.4f}, pitch={args.pitch:+.4f} "
          f"(closed-loop, tol={TOL_RAD:.3f} rad, timeout={args.timeout:.1f}s)")

    dt = 1.0 / PUBLISH_HZ
    deadline = time.time() + args.timeout
    next_adjust = time.time() + ADJUST_EVERY_S
    settled = 0
    converged = False
    while time.time() < deadline:
        msg = json.dumps([targets[0] + bias[0], targets[1] + bias[1]]).encode("utf-8")
        pub.send(msg)
        present = state.poll()

        if present is not None and time.time() >= next_adjust:
            next_adjust = time.time() + ADJUST_EVERY_S
            err = [present[i] - desired_motor[i] for i in range(2)]
            if max(abs(err[0]), abs(err[1])) < TOL_RAD:
                settled += 1
                if settled >= SETTLE_COUNT:
                    converged = True
                    break
            else:
                settled = 0
                # Fold the motor-frame error back into the published frame.
                for i in range(2):
                    step = ADJUST_GAIN * err[i]
                    if abs(err[i]) < NEAR_BAND_RAD:
                        step = clamp(step, -NEAR_STEP_MAX_RAD, NEAR_STEP_MAX_RAD)
                    bias[i] = clamp(bias[i] - step * signs[i],
                                    -BIAS_LIMIT_RAD, BIAS_LIMIT_RAD)
                print(f"[NeckPub] err yaw={err[0]:+.4f} pitch={err[1]:+.4f} "
                      f"-> bias yaw={bias[0]:+.4f} pitch={bias[1]:+.4f}")
        time.sleep(dt)

    # Hold the final (biased) command briefly so the server's EMA settles on
    # it; the server then keeps writing that goal after we exit.
    hold_end = time.time() + 0.5
    while time.time() < hold_end:
        pub.send(json.dumps([targets[0] + bias[0], targets[1] + bias[1]]).encode("utf-8"))
        time.sleep(dt)

    present = state.poll()
    if present is not None:
        print(f"[NeckState] final: yaw={present[0]:+.4f}, pitch={present[1]:+.4f}")
    if converged:
        print("[Exit] Converged.")
    elif present is not None:
        print("[Exit] Timeout before convergence — check that the neck can move freely.")
    else:
        print("[Exit] Done (open-loop).")

    state.close()
    pub.close(linger=0)
    ctx.term()


if __name__ == "__main__":
    main()
