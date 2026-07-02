#!/usr/bin/env python3
"""
replay_sonic.py

Replay a recording made by record_sonic.py into the (sim) WBC deploy.

It takes the role of g1_sonic_client: it BINDS the token PUB on 5556 and the
neck PUB on 5570, sends the start command, then republishes every recorded
tick at its original timing. Because the recording stores each 30 Hz tick
(including the frozen tokens published during the WAITING phase), the pauses
between action chunks are reproduced exactly.

Start the sim WBC deploy first (it connects to 5556/5570 as a subscriber),
make sure no real g1_sonic_client is running (it would collide on the binds),
then:

    python apply_initial_pose.py        # optional, same as the live run
    python replay_sonic.py --in recordings/run1.pkl
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

_GROOT_ROOT = os.path.expanduser("/mnt/data/weiduo/heng/GR00T-WholeBodyControl")
sys.path.insert(0, _GROOT_ROOT)

from g1_sonic_client import (
    TokenPublisher,
    NeckPublisher,
    HAND_DIM,
    NECK_DIM,
    DEFAULT_ZMQ_HOST,
    DEFAULT_ZMQ_PORT,
    DEFAULT_ZMQ_TOPIC,
    DEFAULT_NECK_PUB_HOST,
    DEFAULT_NECK_PUB_PORT,
)


def main():
    p = argparse.ArgumentParser(description="Replay recorded states+actions into the sim WBC.")
    p.add_argument("--in", dest="in_path", type=str, required=True,
                   help="Recording pickle from record_sonic.py")
    p.add_argument("--zmq-host", type=str, default=DEFAULT_ZMQ_HOST)
    p.add_argument("--zmq-port", type=int, default=DEFAULT_ZMQ_PORT)
    p.add_argument("--zmq-topic", type=str, default=DEFAULT_ZMQ_TOPIC)
    p.add_argument("--neck-pub-host", type=str, default=DEFAULT_NECK_PUB_HOST)
    p.add_argument("--neck-pub-port", type=int, default=DEFAULT_NECK_PUB_PORT)
    p.add_argument("--rate", type=float, default=1.0,
                   help="Playback speed multiplier (default 1.0 = real time)")
    p.add_argument("--warmup", type=float, default=0.5,
                   help="Seconds to wait after bind for subscribers to join (default 0.5)")
    p.add_argument("--loop", action="store_true", help="Replay in a loop until Ctrl+C")
    args = p.parse_args()

    with open(args.in_path, "rb") as f:
        data = pickle.load(f)

    include_neck = data["include_neck"]
    ticks = data["ticks"]
    t = ticks["t"]
    actions = ticks["action"]
    n = len(t)
    if n == 0:
        print("[Replay] Recording has no ticks.")
        return

    dur = t[-1] - t[0]
    print(f"[Replay] Loaded {n} ticks ({dur:.1f}s), include_neck={include_neck}, "
          f"action_dim={actions.shape[1]}")

    token_pub = TokenPublisher(host=args.zmq_host, port=args.zmq_port,
                               topic=args.zmq_topic, include_neck=include_neck)
    neck_pub = (NeckPublisher(host=args.neck_pub_host, port=args.neck_pub_port)
                if include_neck else None)

    # Let subscribers (sim WBC) finish connecting before the start command.
    time.sleep(args.warmup)

    try:
        while True:
            token_pub.send_command(start=True, stop=False, planner=True)
            print("[Replay] start sent, streaming ticks...")

            t0 = t[0]
            wall0 = time.perf_counter()
            for i in range(n):
                action = actions[i]
                token_pub.publish_token(action)
                if include_neck:
                    neck_pub.publish(action[HAND_DIM], action[HAND_DIM + 1])

                # Sleep until this tick's recorded offset (scaled by rate).
                target = wall0 + (t[i] - t0) / args.rate
                sleep_t = target - time.perf_counter()
                if sleep_t > 0:
                    time.sleep(sleep_t)

                if (i + 1) % data["freq"] == 0:
                    print(f"[Replay] {i + 1}/{n} ticks")

            if not args.loop:
                break
            print("[Replay] loop: restarting...")

    except KeyboardInterrupt:
        print("\n[Replay] Stopped by user.")
    finally:
        token_pub.send_command(start=False, stop=True, planner=False)
        time.sleep(0.2)
        token_pub.stop()
        if neck_pub is not None:
            neck_pub.stop()
        print("[Replay] Done.")


if __name__ == "__main__":
    main()
