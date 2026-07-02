"""
Viewer for the robot-side ~/realsense_server.py (REP on 192.168.123.164:5556).

The server replies to each request with a 3-part multipart message:
  Part 0 - RGB   JPEG   (640x480 BGR)
  Part 1 - IR    JPEG   (left|right hstacked, 1280x480)
  Part 2 - Depth raw    (z16, 640x480 uint16)

Sends a request, decodes the reply, shows the windows. Press 'q'/ESC to quit.

Usage:
  python realsense_viewer.py                          # 192.168.123.164:5556
  python realsense_viewer.py --server 192.168.123.164 --port 5556
  python realsense_viewer.py --show-ir --show-depth
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import zmq


def _decode_jpeg(buf):
    if not buf:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _decode_depth(buf, w=640, h=480):
    if not buf or len(buf) != w * h * 2:
        return None
    depth = np.frombuffer(buf, dtype=np.uint16).reshape(h, w)
    vis = cv2.convertScaleAbs(depth, alpha=0.03)  # ~ up to 8.5m -> 0..255
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def run(args):
    addr = f"tcp://{args.server}:{args.port}"
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    sock.setsockopt(zmq.SNDTIMEO, 3000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(addr)
    print(f"[viewer] connected to {addr}")

    frames = 0
    t0 = time.time()
    last_report = t0

    try:
        while True:
            try:
                sock.send(b"get")
                parts = sock.recv_multipart()
            except zmq.Again:
                print("[viewer] timeout, reconnecting...")
                sock.close()
                sock = ctx.socket(zmq.REQ)
                sock.setsockopt(zmq.RCVTIMEO, 3000)
                sock.setsockopt(zmq.SNDTIMEO, 3000)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(addr)
                continue

            if len(parts) < 3:
                print("[viewer] server has no frame yet...")
                time.sleep(0.1)
                continue

            rgb = _decode_jpeg(parts[0])
            if rgb is not None:
                cv2.imshow("RealSense RGB", rgb)
            if args.show_ir:
                ir = _decode_jpeg(parts[1])
                if ir is not None:
                    cv2.imshow("RealSense IR L|R", ir)
            if args.show_depth:
                depth = _decode_depth(parts[2])
                if depth is not None:
                    cv2.imshow("RealSense Depth", depth)

            frames += 1
            now = time.time()
            if now - last_report > 2.0:
                fps = frames / (now - t0)
                print(f"[viewer] {frames} frames, {fps:.1f} fps")
                last_report = now

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                break
    finally:
        cv2.destroyAllWindows()
        sock.close()
        ctx.term()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="192.168.123.164")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--show-ir", action="store_true", help="also show IR L|R window")
    p.add_argument("--show-depth", action="store_true", help="also show colorized depth")
    run(p.parse_args())


if __name__ == "__main__":
    main()
