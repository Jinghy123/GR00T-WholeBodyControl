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


def _load_overlay_frame(path, swap_channels=False):
    """First frame of a video file, for blending over the live RGB view.

    swap_channels flips B<->R so the overlay renders in false color and stands
    out against the similarly-colored live camera view.
    """
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"[viewer] could not read first frame from {path}")
    if swap_channels:
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    print(f"[viewer] overlay: first frame of {path} ({frame.shape[1]}x{frame.shape[0]}"
          f"{', channels swapped' if swap_channels else ''})")
    return frame


def _show(name, img, scale):
    """imshow with an optional display-only scale (frame data untouched)."""
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LINEAR)
    cv2.imshow(name, img)


def _decode_depth(buf, w=640, h=480):
    if not buf or len(buf) != w * h * 2:
        return None
    depth = np.frombuffer(buf, dtype=np.uint16).reshape(h, w)
    vis = cv2.convertScaleAbs(depth, alpha=0.03)  # ~ up to 8.5m -> 0..255
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def run(args):
    overlay = (
        _load_overlay_frame(args.overlay, swap_channels=args.overlay_swap_channels)
        if args.overlay else None
    )

    ctx = zmq.Context()

    if args.sub:
        # SUB mode: watch the server's viewer PUB (5559) without stealing frames
        # from the REP recording/inference socket. Use this while recording.
        port = args.port if args.port != 5556 else 5559
        addr = f"tcp://{args.server}:{port}"
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.setsockopt(zmq.CONFLATE, 0)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(addr)
        print(f"[viewer] SUB connected to {addr}")
    else:
        # REQ mode: talk to the REP socket directly (do NOT use this while
        # recording — it steals every other frame from the recorder).
        addr = f"tcp://{args.server}:{args.port}"
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.setsockopt(zmq.SNDTIMEO, 3000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(addr)
        print(f"[viewer] REQ connected to {addr}")

    frames = 0
    t0 = time.time()
    last_report = t0

    try:
        while True:
            try:
                if not args.sub:
                    sock.send(b"get")
                parts = sock.recv_multipart()
            except zmq.Again:
                print("[viewer] timeout...")
                if args.sub:
                    continue
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
                if overlay is not None:
                    if overlay.shape[:2] != rgb.shape[:2]:
                        overlay = cv2.resize(overlay, (rgb.shape[1], rgb.shape[0]))
                    rgb = cv2.addWeighted(
                        rgb, 1.0 - args.overlay_alpha, overlay, args.overlay_alpha, 0
                    )
                _show("RealSense RGB", rgb, args.scale)
            if args.show_ir:
                ir = _decode_jpeg(parts[1])
                if ir is not None:
                    _show("RealSense IR L|R", ir, args.scale)
            if args.show_depth:
                depth = _decode_depth(parts[2])
                if depth is not None:
                    _show("RealSense Depth", depth, args.scale)

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
    p.add_argument("--sub", action="store_true",
                   help="SUB to the viewer PUB (default port 5559) instead of "
                        "REQ to the REP socket. Use this while recording so the "
                        "viewer never steals frames from the recorder.")
    p.add_argument("--show-ir", action="store_true", help="also show IR L|R window")
    p.add_argument("--show-depth", action="store_true", help="also show colorized depth")
    p.add_argument("--overlay", default="",
                   help="video file whose FIRST frame is alpha-blended over the "
                        "live RGB window (e.g. to line the camera up with a "
                        "recorded episode)")
    p.add_argument("--overlay-alpha", type=float, default=0.5,
                   help="overlay opacity, 0..1 (default 0.5)")
    p.add_argument("--overlay-swap-channels", action="store_true",
                   help="swap the overlay's B<->R channels (false color) so it "
                        "contrasts with the live camera view")
    p.add_argument("--scale", type=float, default=1.0,
                   help="display-only window scale factor (default 1.0); the "
                        "frame data itself is never resized")
    run(p.parse_args())


if __name__ == "__main__":
    main()
