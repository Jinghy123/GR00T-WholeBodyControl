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
import collections
import time

import cv2
import numpy as np
import zmq


def _decode(jpeg_bytes):
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _load_overlay_frame(path, swap_channels=False):
    """First frame of a video file, for blending over the live ego view.

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

    overlay = (
        _load_overlay_frame(args.overlay, swap_channels=args.overlay_swap_channels)
        if args.overlay else None
    )

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
    # Sliding window of recent frame timestamps for an instantaneous FPS.
    recv_times: collections.deque = collections.deque()
    WINDOW_S = 2.0

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
            ego = _decode(parts[0])
            stereo = _decode(parts[1]) if args.show_stereo else None

            if ego is not None:
                if overlay is not None:
                    if overlay.shape[:2] != ego.shape[:2]:
                        overlay = cv2.resize(overlay, (ego.shape[1], ego.shape[0]))
                    ego = cv2.addWeighted(
                        ego, 1.0 - args.overlay_alpha, overlay, args.overlay_alpha, 0
                    )
                cv2.imshow("ZED Ego", ego)
            if args.show_stereo and stereo is not None:
                cv2.imshow("ZED Stereo L|R", stereo)

            frames += 1
            now = time.time()
            recv_times.append(now)
            while recv_times and now - recv_times[0] > WINDOW_S:
                recv_times.popleft()
            if now - last_report > 2.0:
                span = now - recv_times[0] if len(recv_times) > 1 else 0.0
                fps = (len(recv_times) - 1) / span if span > 0 else 0.0
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
    p.add_argument("--overlay", default="",
                   help="video file whose first frame is blended over the live "
                        "ego view (e.g. to line the camera up with an episode)")
    p.add_argument("--overlay-alpha", type=float, default=0.5,
                   help="overlay opacity, 0..1 (default 0.5)")
    p.add_argument("--overlay-swap-channels", action="store_true",
                   help="swap the overlay's B<->R channels (false color) so it "
                        "stands out against the live view")
    run(p.parse_args())


if __name__ == "__main__":
    main()
