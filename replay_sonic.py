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

The start command is sent with planner=0, which puts the deploy's ZMQManager in
STREAMED_MOTION mode. That is the only mode in which streamed tokens reach the
policy: with planner=1 the WBC receives the token messages and drops them, and
the robot simply stands through the whole replay. Use --planner to restore the
old planner=1 behaviour.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))
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
    p.add_argument("--planner", action="store_true",
                   help="Send the start command with planner=1 (WBC ZMQManager stays in "
                        "PLANNER mode). Tokens are ignored in that mode, so the robot just "
                        "stands; default is planner=0 (STREAMED_MOTION), which plays them.")
    p.add_argument("--no-stop", dest="send_stop", action="store_false",
                   help="Do not send the stop command when the replay ends. Stop terminates "
                        "WBC control and the deploy exits, so in sim the robot goes limp and "
                        "collapses the moment the episode finishes; with --no-stop the WBC "
                        "keeps holding the final pose.")
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
            token_pub.send_command(start=True, stop=False, planner=args.planner)
            mode = "PLANNER (tokens ignored)" if args.planner else "STREAMED_MOTION"
            print(f"[Replay] start sent ({mode}), streaming ticks...")

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
        if args.send_stop:
            token_pub.send_command(start=False, stop=True, planner=False)
        else:
            print("[Replay] --no-stop: leaving WBC control running")
        time.sleep(0.2)
        token_pub.stop()
        if neck_pub is not None:
            neck_pub.stop()
        print("[Replay] Done.")


if __name__ == "__main__":
    main()
