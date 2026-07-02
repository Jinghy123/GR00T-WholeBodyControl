"""
Native RealSense head-camera server (robot-side) — no ZED, no neck, no Pico.

A standalone alternative to realsense_server.py (which drives the neck-mounted
ZED). This one drives the robot's *native* RealSense head camera directly via
pyrealsense2 and serves it over ZMQ REP, matching the 3-part contract that the
desktop clients already speak in their non-`--include-neck` path:

  - g1_sonic_client.py / psix_rtc_sonic_client.py  → RSCamera (reads part 0 RGB)
  - realsense_viewer.py                            → RGB / IR L|R / Depth

ZMQ REP bind: tcp://0.0.0.0:5558  (matches RSCamera's default port)
Each request receives a 3-part multipart reply (b"" if a stream is unavailable):

  Part 0 — RGB   JPEG   (640x480 BGR)
  Part 1 — IR    JPEG   (left|right hstacked, 1280x480; b"" if --no-ir)
  Part 2 — Depth raw    (z16, 640x480 uint16 little-endian; b"" if --no-depth)

Usage (on the robot):
    python realsense_native_server.py --zmq-bind tcp://0.0.0.0:5558

    python realsense_native_server.py --list-devices     # enumerate + exit
    python realsense_native_server.py --no-ir --no-depth  # RGB only (lightest)

Environment overrides: RS_ZMQ_BIND, RS_FPS, RS_WIDTH, RS_HEIGHT,
                       RS_JPEG_QUALITY, RS_SERIAL
"""

from __future__ import annotations

import argparse
import io
import os
import signal
import threading
import time
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

# ──────────────────────────────────────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────────────────────────────────────

latest_rgb_bytes: Optional[bytes] = None
latest_ir_bytes: Optional[bytes] = None
latest_depth_bytes: Optional[bytes] = None
frame_seq = 0
frame_cond = threading.Condition()

ZMQ_BIND_DEFAULT = os.environ.get("RS_ZMQ_BIND", "tcp://0.0.0.0:5558")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _jpeg_encode(img: np.ndarray, quality: int) -> bytes:
    """Encode a BGR or single-channel image to JPEG bytes (OpenCV, Pillow fallback)."""
    q = max(1, min(100, int(quality)))
    try:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            return buf.tobytes()
    except (cv2.error, TypeError, ValueError):
        pass
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "JPEG encode failed in OpenCV and Pillow is not installed"
        ) from e
    if img.ndim == 2:
        im = Image.fromarray(img, mode="L")
    else:
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        im = Image.fromarray(rgb, mode="RGB")
    bio = io.BytesIO()
    im.save(bio, format="JPEG", quality=q)
    return bio.getvalue()


def list_realsense_devices() -> None:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("[RealSense] No devices connected.")
        return
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"[{i}] {name}  serial={serial}")


# ──────────────────────────────────────────────────────────────────────────────
# Native RealSense capture
# ──────────────────────────────────────────────────────────────────────────────


def capture_thread(cfg: argparse.Namespace) -> None:
    global latest_rgb_bytes, latest_ir_bytes, latest_depth_bytes, frame_seq

    pipeline = rs.pipeline()
    config = rs.config()
    if cfg.serial:
        config.enable_device(cfg.serial)
    config.enable_stream(
        rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps
    )
    if cfg.enable_depth:
        config.enable_stream(
            rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps
        )
    if cfg.enable_ir:
        # infrared 1 = left, infrared 2 = right (y8)
        config.enable_stream(
            rs.stream.infrared, 1, cfg.width, cfg.height, rs.format.y8, cfg.fps
        )
        config.enable_stream(
            rs.stream.infrared, 2, cfg.width, cfg.height, rs.format.y8, cfg.fps
        )

    pipeline.start(config)
    print(
        f"[RealSense] Started: RGB {cfg.width}x{cfg.height}@{cfg.fps} "
        f"(depth={cfg.enable_depth}, ir={cfg.enable_ir})"
    )

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except Exception as e:
                print(f"[RealSense] wait_for_frames error: {e}")
                time.sleep(0.01)
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.ascontiguousarray(
                np.asarray(color_frame.get_data(), dtype=np.uint8)
            )
            if color_image.ndim != 3 or color_image.size == 0:
                continue
            enc_rgb = _jpeg_encode(color_image, cfg.jpeg_quality)

            enc_ir = None
            if cfg.enable_ir:
                irl = frames.get_infrared_frame(1)
                irr = frames.get_infrared_frame(2)
                if irl and irr:
                    left = np.asarray(irl.get_data(), dtype=np.uint8)
                    right = np.asarray(irr.get_data(), dtype=np.uint8)
                    ir_lr = np.ascontiguousarray(np.hstack((left, right)))
                    enc_ir = _jpeg_encode(ir_lr, cfg.jpeg_quality)

            depth_raw = None
            if cfg.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth = np.asarray(depth_frame.get_data(), dtype=np.uint16)
                    depth_raw = np.ascontiguousarray(depth).tobytes()

            with frame_cond:
                latest_rgb_bytes = enc_rgb
                latest_ir_bytes = enc_ir
                latest_depth_bytes = depth_raw
                frame_seq += 1
                frame_cond.notify_all()
    finally:
        pipeline.stop()


# ──────────────────────────────────────────────────────────────────────────────
# ZMQ REP server
# ──────────────────────────────────────────────────────────────────────────────


def start_server(cfg: argparse.Namespace) -> None:
    threading.Thread(target=capture_thread, args=(cfg,), daemon=True).start()

    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.bind(cfg.zmq_bind)
    print(f"[ZMQ] REP bound to {cfg.zmq_bind}, waiting for requests...")

    def _force_exit(sig, _frame):
        print(f"\n[Server] Signal {sig} received - shutting down.")
        os._exit(0)

    signal.signal(signal.SIGINT, _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    last_sent_seq = 0
    try:
        while True:
            _ = sock.recv()

            with frame_cond:
                while frame_seq == last_sent_seq:
                    frame_cond.wait(timeout=0.1)
                rgb = latest_rgb_bytes
                ir = latest_ir_bytes
                depth = latest_depth_bytes
                last_sent_seq = frame_seq

            sock.send_multipart([
                rgb if rgb is not None else b"",
                ir if ir is not None else b"",
                depth if depth is not None else b"",
            ])
    finally:
        sock.close()
        context.term()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Native RealSense head-camera server (no ZED, no neck, no Pico)."
    )
    p.add_argument("--zmq-bind", default=ZMQ_BIND_DEFAULT,
                   help="ZMQ REP bind address (default tcp://0.0.0.0:5558)")
    p.add_argument("--fps", type=int, default=int(os.environ.get("RS_FPS", "30")))
    p.add_argument("--width", type=int, default=int(os.environ.get("RS_WIDTH", "640")))
    p.add_argument("--height", type=int, default=int(os.environ.get("RS_HEIGHT", "480")))
    p.add_argument("--jpeg-quality", type=int,
                   default=int(os.environ.get("RS_JPEG_QUALITY", "80")))
    p.add_argument("--serial", default=os.environ.get("RS_SERIAL", ""),
                   help="RealSense device serial (empty = first device found)")
    p.add_argument("--no-ir", dest="enable_ir", action="store_false",
                   help="Do not stream/serve the IR L|R pair")
    p.add_argument("--no-depth", dest="enable_depth", action="store_false",
                   help="Do not stream/serve the depth map")
    p.add_argument("--list-devices", action="store_true",
                   help="List connected RealSense devices and exit")
    p.set_defaults(enable_ir=True, enable_depth=True)
    return p.parse_args()


def main() -> None:
    ns = _parse_args()
    if ns.list_devices:
        list_realsense_devices()
        return
    start_server(ns)


if __name__ == "__main__":
    main()
