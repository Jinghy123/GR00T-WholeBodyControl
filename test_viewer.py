"""
Live ZED-only viewer for realsense_server.py.

Subscribes to the server's dedicated viewer PUB stream (2-part multipart:
[ego_jpeg, stereo_jpeg]) on port 5559. This is SEPARATE from the REP
recording socket (5558), so running this viewer never slows down the
recorder — PUB drops frames for a slow subscriber instead of backpressuring.
Press 'q' or ESC to quit. Use --duration N to auto-exit after N seconds.

Usage:
  python test_viewer.py                                      # connect to 127.0.0.1:5559
  python test_viewer.py --server 192.168.123.164             # robot IP
  python test_viewer.py --show-stereo                        # add stereo window
  python test_viewer.py --duration 15                        # auto-close after 15s
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import zmq


def _decode(jpeg_bytes):
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _make_sock(ctx, addr):
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.RCVTIMEO, 3000)
    s.setsockopt(zmq.RCVHWM, 2)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(addr)
    return s


def run(args):
    addr = f"tcp://{args.server}:{args.port}"
    ctx = zmq.Context()
    sock = _make_sock(ctx, addr)
    print(f"[viewer] connected to {addr}")

    # DISABLED: D405 wrists — only ZED windows are created.
    windows = ["ZED Ego"]
    if args.show_stereo:
        windows.insert(1, "ZED Stereo L|R")
    for w in windows:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    deadline = time.time() + args.duration if args.duration > 0 else None
    frames = 0
    t0 = time.time()
    last_report = t0

    try:
        while True:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                print("[viewer] no frames from server, reconnecting...")
                sock.close()
                sock = _make_sock(ctx, addr)
                continue

            while len(parts) < 2:
                parts.append(b"")
            ego, stereo = (_decode(p) for p in parts[:2])

            if ego is not None:
                cv2.imshow("ZED Ego", ego)
            if args.show_stereo and stereo is not None:
                cv2.imshow("ZED Stereo L|R", stereo)

            frames += 1
            now = time.time()
            if now - last_report > 2.0:
                fps = frames / (now - t0)
                got = [n for n, x in zip(["ego", "stereo"],
                                          [ego, stereo]) if x is not None]
                print(f"[viewer] {frames} frames, {fps:.1f} fps, streams={got}")
                last_report = now

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:   # q or ESC
                break
            if deadline is not None and now >= deadline:
                print(f"[viewer] --duration {args.duration}s elapsed, exiting")
                break
    finally:
        cv2.destroyAllWindows()
        sock.close()
        ctx.term()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5559)
    p.add_argument("--show-stereo", action="store_true",
                   help="also open a window for the ZED stereo L|R view")
    p.add_argument("--duration", type=float, default=0,
                   help="auto-exit after N seconds (0 = run until 'q'/ESC)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
