#!/usr/bin/env python3
"""
record_sonic.py

Passive recorder for `g1_sonic_client.py --action-only --include-neck`.

Run this in a second terminal *while* the policy client is driving the robot.
It subscribes (read-only, no port binding) to every stream the client
publishes/consumes, and dumps a per-tick log of the published action together
with the aligned robot state. The pauses between action chunks are captured
naturally: during the WAITING phase the client keeps publishing the frozen
token at 30 Hz, so every frozen tick is recorded with its own timestamp and
the gap is reproduced exactly on replay.

Streams subscribed:
  - token PUB  (tcp://localhost:5556, topics "pose"/"command")  <- TokenPublisher
  - neck PUB   (tcp://localhost:5570, JSON [yaw, pitch])         <- NeckPublisher
  - WBC state  (tcp://localhost:5557, topic "g1_debug")          <- WBCStateReader
  - neck state (tcp://192.168.123.164:5560, JSON [yaw, pitch])   <- NeckStateReader

Usage:
    python record_sonic.py --out recordings/run1.pkl
    # ... let the client run, then Ctrl+C here to stop & save.
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import zmq

_GROOT_ROOT = os.path.expanduser("/mnt/data/weiduo/heng/GR00T-WholeBodyControl")
sys.path.insert(0, _GROOT_ROOT)

from ours.clients.g1_sonic_client import (
    WBCStateReader,
    NeckStateReader,
    HAND_DIM,
    NECK_DIM,
    TOKEN_DIM,
    FREQ_POLICY,
    WBC_HOST,
    WBC_PORT,
    WBC_TOPIC,
    DEFAULT_NECK_STATE_ZMQ,
)

# ── Wire-format constants (match zmq_planner_sender.py) ─────────────────────
HEADER_SIZE = 1280
_DTYPE = {
    "f32": "<f4",
    "f64": "<f8",
    "i32": "<i4",
    "i64": "<i8",
    "u8": "u1",
    "bool": "?",
}
_KNOWN_TOPICS = (b"command", b"planner", b"pose")


def parse_zmq_message(raw: bytes):
    """Decode a [topic][1280-byte JSON header][binary] message.

    Returns (topic_str, fields_dict). fields_dict maps field name -> np.ndarray.
    """
    topic = next((t for t in _KNOWN_TOPICS if raw.startswith(t)), None)
    if topic is None:
        raise ValueError(f"Unknown topic prefix in message of {len(raw)} bytes")
    tl = len(topic)
    header_json = raw[tl:tl + HEADER_SIZE].split(b"\x00", 1)[0]
    header = json.loads(header_json.decode("utf-8"))
    payload = raw[tl + HEADER_SIZE:]

    out = {}
    off = 0
    for f in header["fields"]:
        np_dt = _DTYPE[f["dtype"]]
        n = int(np.prod(f["shape"])) if f["shape"] else 1
        nbytes = n * np.dtype(np_dt).itemsize
        arr = np.frombuffer(payload[off:off + nbytes], dtype=np_dt).reshape(f["shape"])
        out[f["name"]] = arr.copy()
        off += nbytes
    return topic.decode("utf-8"), out


class Recorder:
    def __init__(self, args):
        self._include_neck = args.include_neck
        self._action_dim = HAND_DIM + (NECK_DIM if args.include_neck else 0) + TOKEN_DIM

        # Token stream SUB (passive connect to the client's bound PUB).
        self._ctx = zmq.Context()
        self._token_sock = self._ctx.socket(zmq.SUB)
        self._token_sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._token_sock.setsockopt(zmq.RCVHWM, 0)  # never drop
        self._token_sock.connect(args.token_zmq)
        print(f"[Recorder] token SUB connected to {args.token_zmq}")

        # Neck action stream (the [yaw, pitch] the client publishes on 5570).
        # CONFLATE keeps only the freshest value so we can pair it with the
        # pose tick that arrives on the token socket.
        if args.include_neck:
            self._neck_action = NeckStateReader(args.neck_zmq)
        else:
            self._neck_action = None

        # Robot state readers (same classes the client uses).
        self._state = WBCStateReader(host=args.wbc_host, port=args.wbc_port, topic=args.wbc_topic)
        if args.include_neck:
            self._neck_state = NeckStateReader(args.neck_state_zmq)
        else:
            self._neck_state = None

        # Storage
        self._t = []
        self._action = []
        self._qpos = []
        self._lhand = []
        self._rhand = []
        self._quat = []
        self._neck_obs = []
        self._is_repeat = []
        self._commands = []
        self._prev_action = None

    def _reconstruct_action(self, fields: dict) -> np.ndarray:
        left = np.asarray(fields["left_hand_joints"], dtype=np.float32).reshape(-1)   # (7,)
        right = np.asarray(fields["right_hand_joints"], dtype=np.float32).reshape(-1)  # (7,)
        token = np.asarray(fields["token_state"], dtype=np.float32).reshape(-1)        # (64,)
        if self._include_neck:
            neck = self._neck_action.get_latest()
            neck = (np.asarray(neck, dtype=np.float32).reshape(NECK_DIM)
                    if neck is not None else np.zeros(NECK_DIM, dtype=np.float32))
            return np.concatenate([left, right, neck, token]).astype(np.float32)
        return np.concatenate([left, right, token]).astype(np.float32)

    def run(self):
        print("[Recorder] Recording... press Ctrl+C to stop and save.")
        poller = zmq.Poller()
        poller.register(self._token_sock, zmq.POLLIN)
        n_pose = 0
        try:
            while True:
                socks = dict(poller.poll(timeout=200))
                if self._token_sock not in socks:
                    continue
                raw = self._token_sock.recv()
                t = time.perf_counter()
                topic, fields = parse_zmq_message(raw)

                if topic == "command":
                    cmd = {
                        "t": t,
                        "start": int(fields["start"][0]),
                        "stop": int(fields["stop"][0]),
                        "planner": int(fields["planner"][0]),
                    }
                    self._commands.append(cmd)
                    print(f"[Recorder] command @ {t:.3f}: {cmd}")
                    continue

                if topic != "pose":
                    continue

                action = self._reconstruct_action(fields)

                # Aligned robot state at this tick.
                st = self._state.get_state()
                if st is not None:
                    qpos = st["qpos"]
                    lhand = st["left_hand_q"]
                    rhand = st["right_hand_q"]
                    quat = st["base_quat"]
                else:
                    qpos = np.full(29, np.nan, np.float32)
                    lhand = np.full(7, np.nan, np.float32)
                    rhand = np.full(7, np.nan, np.float32)
                    quat = np.full(4, np.nan, np.float32)

                if self._include_neck:
                    ns = self._neck_state.get_latest()
                    neck_obs = (np.asarray(ns, dtype=np.float32).reshape(NECK_DIM)
                                if ns is not None else np.full(NECK_DIM, np.nan, np.float32))
                else:
                    neck_obs = np.full(NECK_DIM, np.nan, np.float32)

                is_repeat = (self._prev_action is not None
                             and np.allclose(action, self._prev_action, atol=1e-8))
                self._prev_action = action

                self._t.append(t)
                self._action.append(action)
                self._qpos.append(qpos)
                self._lhand.append(lhand)
                self._rhand.append(rhand)
                self._quat.append(quat)
                self._neck_obs.append(neck_obs)
                self._is_repeat.append(is_repeat)

                n_pose += 1
                if n_pose % FREQ_POLICY == 0:
                    waited = sum(self._is_repeat[-FREQ_POLICY:])
                    print(f"[Recorder] {n_pose} ticks recorded "
                          f"(last 1s: {FREQ_POLICY - waited} fresh / {waited} frozen)")
        except KeyboardInterrupt:
            print("\n[Recorder] Stopped by user.")

    def save(self, path: str):
        if not self._t:
            print("[Recorder] No ticks recorded, nothing to save.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "include_neck": self._include_neck,
            "action_dim": self._action_dim,
            "freq": FREQ_POLICY,
            "ticks": {
                "t": np.asarray(self._t, dtype=np.float64),
                "action": np.stack(self._action).astype(np.float32),
                "qpos": np.stack(self._qpos).astype(np.float32),
                "left_hand_q": np.stack(self._lhand).astype(np.float32),
                "right_hand_q": np.stack(self._rhand).astype(np.float32),
                "base_quat": np.stack(self._quat).astype(np.float32),
                "neck_state": np.stack(self._neck_obs).astype(np.float32),
                "is_repeat": np.asarray(self._is_repeat, dtype=bool),
            },
            "commands": self._commands,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        dur = self._t[-1] - self._t[0]
        print(f"[Recorder] Saved {len(self._t)} ticks ({dur:.1f}s) to {path}")

    def close(self):
        self._token_sock.close(linger=0)
        self._state.close()
        if self._neck_action is not None:
            self._neck_action.close()
        if self._neck_state is not None:
            self._neck_state.close()
        self._ctx.term()


def main():
    p = argparse.ArgumentParser(description="Record states+actions from a running g1_sonic_client.")
    p.add_argument("--out", type=str, default="recordings/sonic_record.pkl",
                   help="Output pickle path (default: recordings/sonic_record.pkl)")
    p.add_argument("--token-zmq", type=str, default="tcp://localhost:5556",
                   help="Token PUB to subscribe (default: tcp://localhost:5556)")
    p.add_argument("--neck-zmq", type=str, default="tcp://localhost:5570",
                   help="Neck action PUB to subscribe (default: tcp://localhost:5570)")
    p.add_argument("--wbc-host", type=str, default=WBC_HOST)
    p.add_argument("--wbc-port", type=int, default=WBC_PORT)
    p.add_argument("--wbc-topic", type=str, default=WBC_TOPIC)
    p.add_argument("--neck-state-zmq", type=str, default=DEFAULT_NECK_STATE_ZMQ)
    p.add_argument("--include-neck", dest="include_neck", action="store_true", default=True,
                   help="Record neck (80-dim action). Default on, matches --include-neck client.")
    p.add_argument("--no-include-neck", dest="include_neck", action="store_false",
                   help="78-dim hand+token only, no neck I/O.")
    args = p.parse_args()

    print(f"[Recorder] include_neck={args.include_neck}, action_dim="
          f"{HAND_DIM + (NECK_DIM if args.include_neck else 0) + TOKEN_DIM}")

    rec = Recorder(args)
    try:
        rec.run()
    finally:
        rec.save(args.out)
        rec.close()


if __name__ == "__main__":
    main()
