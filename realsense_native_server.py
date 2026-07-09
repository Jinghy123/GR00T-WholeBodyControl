"""
Native RealSense head-camera server (robot-side) — no ZED. Optional 2-DOF neck
motor control (--enable-neck-motor), same as realsense_server.py: SUB neck
angles from pose_publisher / pico_manus_thread_server on --pose-zmq, drive the
Dynamixels, PUB present position on --neck-state-pub (default tcp://*:5560).

A standalone alternative to realsense_server.py (which drives the neck-mounted
ZED). This one drives the robot's *native* RealSense head camera directly via
pyrealsense2 and serves it to THREE independent consumers at the same time so
recording, a live viewer and the Pico headset never steal frames from each
other:

  1. ZMQ REP  (default tcp://0.0.0.0:5558) — the recording / inference client
     (RSCamera / RealSenseClient). Strict req-reply, full frame rate. This is
     the only socket the desktop clients speak, so they stay unchanged.
  2. ZMQ PUB  (default tcp://0.0.0.0:5559) — the live viewer stream. Any number
     of viewers SUB here without back-pressuring the recorder (PUB drops frames
     for a slow subscriber). Use `realsense_viewer.py --sub`.
  3. Pico H.264 (--enable-pico) — GStreamer x264 over TCP to the Pico headset,
     so the teleoperator sees a first-person view while recording.

Each REP reply and each PUB message is a 3-part multipart (b"" if a stream is
off):
  Part 0 — RGB   JPEG   (640x480 BGR)
  Part 1 — IR    JPEG   (left|right hstacked, 1280x480; b"" if --no-ir)
  Part 2 — Depth raw    (z16, 640x480 uint16 little-endian; b"" if --no-depth)

Usage (on the robot, in the `ruohai` env — system librealsense / v4l2 backend):
    # record + viewer + pico all at once, RGB(+depth) at 30 fps on USB2:
    python realsense_native_server.py --no-ir \
        --enable-pico --pico-ip 192.168.0.241

    python realsense_native_server.py --list-devices     # enumerate + exit

USB2 note: this camera is on a USB2 port. RGB or RGB+depth run at 30 fps, but
turning IR on (both infrared streams) drops it to ~15-22 fps. Keep `--no-ir`
unless the camera is moved to a USB3 port.

Environment overrides: RS_ZMQ_BIND, RS_VIEWER_PUB, RS_FPS, RS_WIDTH, RS_HEIGHT,
                       RS_JPEG_QUALITY, RS_SERIAL, RS_EXPOSURE,
                       PICO_IP, VIDEO_PORT, VR_WIDTH, VR_HEIGHT
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import signal
import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

# GStreamer is loaded lazily only when --enable-pico is set.
gi = None
Gst = None

# ──────────────────────────────────────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────────────────────────────────────

latest_rgb_bytes: Optional[bytes] = None
latest_ir_bytes: Optional[bytes] = None
latest_depth_bytes: Optional[bytes] = None
latest_rgb_frame: Optional[np.ndarray] = None   # raw BGR ndarray, for Pico
latest_ir_frame: Optional[np.ndarray] = None    # raw L|R gray ndarray, for Pico
frame_seq = 0
frame_cond = threading.Condition()

ZMQ_BIND_DEFAULT = os.environ.get("RS_ZMQ_BIND", "tcp://0.0.0.0:5558")
VIEWER_PUB_DEFAULT = os.environ.get("RS_VIEWER_PUB", "tcp://0.0.0.0:5559")
PICO_IP = os.environ.get("PICO_IP", "192.168.0.241")
VIDEO_PORT = int(os.environ.get("VIDEO_PORT", "12345"))
VR_WIDTH = int(os.environ.get("VR_WIDTH", "0"))    # 0 = native frame size
VR_HEIGHT = int(os.environ.get("VR_HEIGHT", "0"))
FPS = 30


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


def _tune_color_sensor(profile, cfg: argparse.Namespace) -> None:
    """Keep the RGB (UVC) sensor at a constant frame rate.

    The D435i colour sensor defaults to `auto_exposure_priority=1`, which lets
    the driver *drop the frame rate* (e.g. 30->15) to lengthen exposure in dim
    light. Force it to 0 so we always get the requested `--fps`. Optionally pin
    a fixed exposure/gain via --exposure to remove auto-exposure entirely.
    """
    try:
        color_sensor = profile.get_device().first_color_sensor()
    except Exception as e:
        print(f"[RealSense] no color sensor to tune: {e}")
        return

    if color_sensor.supports(rs.option.auto_exposure_priority):
        try:
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
            print("[RealSense] auto_exposure_priority=0 (constant fps)")
        except Exception as e:
            print(f"[RealSense] could not clear auto_exposure_priority: {e}")

    if cfg.exposure > 0:
        try:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, float(cfg.exposure))
            print(f"[RealSense] fixed exposure={cfg.exposure}")
        except Exception as e:
            print(f"[RealSense] could not set fixed exposure: {e}")


def _tune_depth_sensor(profile, cfg: argparse.Namespace) -> None:
    """Disable the IR dot-pattern projector when the IR pair is meant for human
    viewing (--pico-source ir) — otherwise the whole view is covered in
    structured-light speckles. Costs depth quality, so only done on demand."""
    if not cfg.disable_emitter:
        return
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            print("[RealSense] IR emitter disabled (clean stereo view, weaker depth)")
    except Exception as e:
        print(f"[RealSense] could not disable emitter: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Native RealSense capture
# ──────────────────────────────────────────────────────────────────────────────


def capture_thread(cfg: argparse.Namespace) -> None:
    global latest_rgb_bytes, latest_ir_bytes, latest_depth_bytes
    global latest_rgb_frame, latest_ir_frame, frame_seq

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

    profile = pipeline.start(config)
    _tune_color_sensor(profile, cfg)
    _tune_depth_sensor(profile, cfg)
    print(
        f"[RealSense] Started: RGB {cfg.width}x{cfg.height}@{cfg.fps} "
        f"(depth={cfg.enable_depth}, ir={cfg.enable_ir})"
    )

    fps_t0 = time.time()
    fps_n = 0
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
            ir_lr = None
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
                latest_rgb_frame = color_image
                latest_ir_frame = ir_lr
                frame_seq += 1
                frame_cond.notify_all()

            fps_n += 1
            if time.time() - fps_t0 >= 2.0:
                print(f"[RealSense] capture {fps_n / (time.time() - fps_t0):.1f} fps")
                fps_t0 = time.time()
                fps_n = 0
    finally:
        pipeline.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Viewer PUB — broadcasts frames to any number of viewers, off the REP path
# ──────────────────────────────────────────────────────────────────────────────


def viewer_pub_thread(cfg: argparse.Namespace) -> None:
    """PUB the latest [rgb, ir, depth] so viewers never contend with the REP
    recording socket. A slow viewer just drops frames (SNDHWM), it can't slow
    down recording."""
    if not cfg.viewer_pub:
        return
    try:
        sock = zmq.Context.instance().socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(cfg.viewer_pub)
        print(f"[ZMQ] Viewer PUB bound: {cfg.viewer_pub}")
    except Exception as e:
        print(f"[ZMQ] Viewer PUB bind failed: {e}")
        return

    last_seq = 0
    while True:
        with frame_cond:
            while frame_seq == last_seq:
                frame_cond.wait(timeout=0.1)
            rgb = latest_rgb_bytes
            ir = latest_ir_bytes
            depth = latest_depth_bytes
            last_seq = frame_seq
        if rgb is None:
            continue
        try:
            sock.send_multipart(
                [rgb, ir if ir is not None else b"", depth if depth is not None else b""],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Pico Video Streamer (GStreamer H.264 + TCP) — streams the RGB frame
# ──────────────────────────────────────────────────────────────────────────────


class PicoVideoStreamer:
    """Streams the native RGB BGR frame to the Pico headset via GStreamer x264.

    Ported from realsense_server.py's ZED streamer; feeds `latest_rgb_frame`
    (mono RGB) instead of the ZED stereo frame.
    """

    def __init__(
        self,
        pico_ip: str = PICO_IP,
        port: int = VIDEO_PORT,
        width: int = VR_WIDTH,
        height: int = VR_HEIGHT,
        fps: int = FPS,
        source: str = "rgb",
    ):
        self.pico_ip = pico_ip
        self.port = port
        self.source = source  # "rgb" (mono color) or "ir" (L|R stereo pair)
        self.vr_width = width
        self.vr_height = height
        self.fps = fps
        self._running = False
        self._connected = False
        self._sock: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()
        self._pipeline = None
        self._appsrc = None
        self._frame_id = 0
        self._pipeline_started = False
        self._send_q: collections.deque = collections.deque(maxlen=3)
        self._send_event = threading.Event()

    def start(self) -> None:
        if Gst is None:
            raise RuntimeError("GStreamer (gi) not loaded; cannot start Pico streamer")
        Gst.init([])
        self._running = True
        self._pipeline_started = False
        threading.Thread(target=self._connection_loop, daemon=True).start()
        threading.Thread(target=self._push_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()

    def _connection_loop(self) -> None:
        while self._running:
            if not self._connected:
                old = self._sock
                self._sock = None
                if old:
                    try:
                        old.close()
                    except Exception:
                        pass
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    s.settimeout(2)
                    s.connect((self.pico_ip, self.port))
                    s.settimeout(None)
                    with self._sock_lock:
                        self._sock = s
                        self._connected = True
                    print(f"[PicoStreamer] Connected to Pico {self.pico_ip}:{self.port}")
                except Exception:
                    pass
            time.sleep(2)

    def _start_pipeline(self, width: int, height: int) -> None:
        self.vr_width = width
        self.vr_height = height
        pipe_str = (
            f"appsrc name=src is-live=True format=time do-timestamp=True ! "
            f"video/x-raw,format=BGR,width={width},height={height},"
            f"framerate={self.fps}/1 ! "
            f"videoconvert ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast "
            f"bitrate=4000 vbv-buf-capacity=500 key-int-max=3 threads=2 ! "
            f"video/x-h264,profile=baseline ! "
            f"h264parse config-interval=-1 ! "
            f"video/x-h264,stream-format=byte-stream,alignment=au ! "
            f"appsink name=sink emit-signals=True sync=False"
        )
        self._pipeline = Gst.parse_launch(pipe_str)
        self._appsrc = self._pipeline.get_by_name("src")
        appsink = self._pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_encoded_frame)
        self._pipeline.set_state(Gst.State.PLAYING)
        print(f"[PicoStreamer] GStreamer pipeline: {width}x{height}@{self.fps}")

    def _on_encoded_frame(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            data = bytes(info.data)
            buf.unmap(info)
            self._send_q.append(data)
            self._send_event.set()
        return Gst.FlowReturn.OK

    def _send_loop(self) -> None:
        while self._running:
            self._send_event.wait(timeout=0.1)
            self._send_event.clear()
            while self._send_q:
                data = self._send_q.popleft()
                with self._sock_lock:
                    if not self._connected or self._sock is None:
                        continue
                    try:
                        header = struct.pack(">I", len(data))
                        self._sock.sendall(header + data)
                    except Exception:
                        self._connected = False
                        print("[PicoStreamer] Connection lost. Retrying...")

    def _push_loop(self) -> None:
        last_seq = 0
        while self._running:
            frame = None
            with frame_cond:
                while frame_seq == last_seq and self._running:
                    frame_cond.wait(timeout=0.1)
                src = latest_ir_frame if self.source == "ir" else latest_rgb_frame
                if src is not None:
                    frame = src
                    last_seq = frame_seq

            if frame is None:
                continue

            if frame.ndim == 2:  # IR L|R is grayscale; x264 pipeline expects BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            h, w = frame.shape[:2]
            if not self._pipeline_started:
                out_w = self.vr_width if self.vr_width > 0 else w
                out_h = self.vr_height if self.vr_height > 0 else h
                self._start_pipeline(out_w, out_h)
                self._pipeline_started = True

            if w != self.vr_width or h != self.vr_height:
                frame = cv2.resize(frame, (self.vr_width, self.vr_height))

            frame = np.ascontiguousarray(frame)
            gst_buf = Gst.Buffer.new_wrapped(frame.tobytes())
            gst_buf.pts = self._frame_id * (Gst.SECOND // self.fps)
            gst_buf.duration = Gst.SECOND // self.fps
            self._appsrc.emit("push-buffer", gst_buf)
            self._frame_id += 1

    def stop(self) -> None:
        self._running = False
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)


# ──────────────────────────────────────────────────────────────────────────────
# Neck motor driver — same calibration/behavior as realsense_server.py
# ──────────────────────────────────────────────────────────────────────────────

NECK_PORT             = "/dev/ttyUSB0"
NECK_BAUD             = 2_000_000
NECK_YAW_ID           = 0
NECK_PITCH_ID         = 1
NECK_YAW_ZERO_TICK    = 1897
NECK_PITCH_ZERO_TICK  = 3275
NECK_YAW_LIMIT_DEG    = 60.0
NECK_PITCH_LIMIT_DEG  = 45.0
NECK_SMOOTH_ALPHA     = 0.3
NECK_CONTROL_HZ       = 50
NECK_YAW_SIGN         = 1
NECK_PITCH_SIGN       = -1

ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132
ADDR_HW_ERROR_STATUS  = 70
TICKS_PER_REV         = 4096


class NeckMotor:
    """SUB [neck_yaw, neck_pitch] over ZMQ → drive 2 Dynamixels; PUB present position."""

    def __init__(
        self,
        pose_zmq_addr: str = "",
        state_pub_addr: str = "",
    ):
        self.port_path = NECK_PORT
        self.baud = NECK_BAUD
        self.yaw_id = NECK_YAW_ID
        self.pitch_id = NECK_PITCH_ID
        self.yaw_zero_tick = NECK_YAW_ZERO_TICK
        self.pitch_zero_tick = NECK_PITCH_ZERO_TICK
        self.yaw_limit_rad = math.radians(NECK_YAW_LIMIT_DEG)
        self.pitch_limit_rad = math.radians(NECK_PITCH_LIMIT_DEG)
        self.smooth_alpha = NECK_SMOOTH_ALPHA
        self.control_hz = NECK_CONTROL_HZ
        self.yaw_sign = NECK_YAW_SIGN
        self.pitch_sign = NECK_PITCH_SIGN
        self.pose_zmq_addr = pose_zmq_addr
        self.state_pub_addr = state_pub_addr

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._port = None
        self._packet = None
        self._pose_sub = None
        self._state_pub = None
        self._zmq_ctx = None

    @staticmethod
    def _rad_to_tick(rad: float, zero_tick: int) -> int:
        return zero_tick + int(rad * TICKS_PER_REV / (2 * math.pi))

    @staticmethod
    def _tick_to_rad(tick: int, zero_tick: int) -> float:
        return (tick - zero_tick) * (2 * math.pi) / TICKS_PER_REV

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _check_and_reboot(self, motor_id: int) -> bool:
        hw, _, err = self._packet.read1ByteTxRx(
            self._port, motor_id, ADDR_HW_ERROR_STATUS
        )
        if err == 128 or hw != 0:
            print(f"[Neck] ID {motor_id} hw_error=0b{hw:08b} -> rebooting")
            self._packet.reboot(self._port, motor_id)
            time.sleep(0.5)
            self._packet.write1ByteTxRx(
                self._port, motor_id, ADDR_TORQUE_ENABLE, 1
            )
            return True
        return False

    def start(self) -> bool:
        try:
            from dynamixel_sdk import PacketHandler, PortHandler
        except ImportError as e:
            print(f"[Neck] Import failed: {e}. pip install dynamixel-sdk.")
            return False

        if not self.pose_zmq_addr:
            print("[Neck] --pose-zmq is required with --enable-neck-motor.")
            return False

        self._zmq_ctx = zmq.Context.instance()
        self._pose_sub = self._zmq_ctx.socket(zmq.SUB)
        self._pose_sub.setsockopt(zmq.CONFLATE, 1)
        self._pose_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._pose_sub.setsockopt(zmq.RCVTIMEO, 100)
        try:
            self._pose_sub.connect(self.pose_zmq_addr)
            print(f"[Neck] ZMQ SUB neck-angle source: {self.pose_zmq_addr}")
        except Exception as e:
            print(f"[Neck] ZMQ connect failed: {e}")
            return False

        self._port = PortHandler(self.port_path)
        self._packet = PacketHandler(2.0)
        if not self._port.openPort() or not self._port.setBaudRate(self.baud):
            print(f"[Neck] Could not open {self.port_path} @ {self.baud}")
            return False

        for i in (self.yaw_id, self.pitch_id):
            _, res, _ = self._packet.ping(self._port, i)
            if res != 0:
                print(f"[Neck] Ping ID {i} failed (res={res})")
                self._port.closePort()
                return False
            self._check_and_reboot(i)
            self._packet.write1ByteTxRx(self._port, i, ADDR_TORQUE_ENABLE, 1)

        self._packet.write4ByteTxRx(
            self._port, self.yaw_id, ADDR_GOAL_POSITION,
            self._rad_to_tick(0.0, self.yaw_zero_tick),
        )
        self._packet.write4ByteTxRx(
            self._port, self.pitch_id, ADDR_GOAL_POSITION,
            self._rad_to_tick(0.0, self.pitch_zero_tick),
        )
        time.sleep(0.3)

        if self.state_pub_addr:
            self._state_pub = self._zmq_ctx.socket(zmq.PUB)
            self._state_pub.setsockopt(zmq.SNDHWM, 1)
            self._state_pub.setsockopt(zmq.LINGER, 0)
            try:
                self._state_pub.bind(self.state_pub_addr)
                print(f"[Neck] State PUB bound: {self.state_pub_addr}")
            except Exception as e:
                print(f"[Neck] State PUB bind failed: {e}")
                self._state_pub.close(linger=0)
                self._state_pub = None

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        print(
            f"[Neck] Started: {self.port_path}@{self.baud} "
            f"IDs {self.yaw_id}/{self.pitch_id} "
            f"limits yaw±{NECK_YAW_LIMIT_DEG:.0f}° pitch±{NECK_PITCH_LIMIT_DEG:.0f}° "
            f"@ {self.control_hz}Hz"
        )
        return True

    def _read_present_state(self):
        if self._packet is None or self._port is None:
            return None, None
        try:
            yaw_tick, _, yerr = self._packet.read4ByteTxRx(
                self._port, self.yaw_id, ADDR_PRESENT_POSITION
            )
            pitch_tick, _, perr = self._packet.read4ByteTxRx(
                self._port, self.pitch_id, ADDR_PRESENT_POSITION
            )
            if yerr != 0 or perr != 0:
                return None, None
            return (
                self._tick_to_rad(yaw_tick, self.yaw_zero_tick),
                self._tick_to_rad(pitch_tick, self.pitch_zero_tick),
            )
        except Exception:
            return None, None

    def _read_neck_angles(self):
        if self._pose_sub is None:
            return None
        try:
            raw = self._pose_sub.recv(flags=0)
        except zmq.Again:
            return None
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(msg, (list, tuple)) or len(msg) < 2:
            return None
        return float(msg[0]), float(msg[1])

    def _loop(self) -> None:
        dt = 1.0 / self.control_hz
        yaw_cmd = 0.0
        pitch_cmd = 0.0
        last_hw_check = 0.0
        last_status_print = 0.0
        last_warn_print = 0.0
        pose_valid = False
        next_tick = time.time()

        while self._running:
            next_tick += dt

            yaw_target = yaw_cmd
            pitch_target = pitch_cmd
            pose_ok = False
            try:
                angles = self._read_neck_angles()
                if angles is not None:
                    yaw_target = angles[0] * self.yaw_sign
                    pitch_target = angles[1] * self.pitch_sign
                    pose_ok = True
            except Exception as e:
                now = time.time()
                if now - last_warn_print > 2.0:
                    print(f"[Neck] neck-angle read error: {e}")
                    last_warn_print = now

            if not pose_ok:
                now = time.time()
                if pose_valid or now - last_warn_print > 5.0:
                    print("[Neck] no neck data yet (pose publisher running?)")
                    last_warn_print = now
                pose_valid = False
                yaw_target = yaw_cmd
                pitch_target = pitch_cmd
            else:
                pose_valid = True

            yaw_target = self._clamp(
                yaw_target, -self.yaw_limit_rad, self.yaw_limit_rad
            )
            pitch_target = self._clamp(
                pitch_target, -self.pitch_limit_rad, self.pitch_limit_rad
            )

            yaw_cmd += self.smooth_alpha * (yaw_target - yaw_cmd)
            pitch_cmd += self.smooth_alpha * (pitch_target - pitch_cmd)

            yaw_tick = self._rad_to_tick(yaw_cmd, self.yaw_zero_tick)
            pitch_tick = self._rad_to_tick(pitch_cmd, self.pitch_zero_tick)
            try:
                self._packet.write4ByteTxRx(
                    self._port, self.yaw_id, ADDR_GOAL_POSITION, yaw_tick
                )
                self._packet.write4ByteTxRx(
                    self._port, self.pitch_id, ADDR_GOAL_POSITION, pitch_tick
                )
            except Exception as e:
                print(f"[Neck] write error: {e}")

            if self._state_pub is not None:
                yaw_meas, pitch_meas = self._read_present_state()
                if yaw_meas is not None:
                    try:
                        self._state_pub.send_string(
                            json.dumps([yaw_meas, pitch_meas])
                        )
                    except Exception:
                        pass

            now = time.time()

            if now - last_hw_check > 2.0:
                for i in (self.yaw_id, self.pitch_id):
                    try:
                        self._check_and_reboot(i)
                    except Exception as e:
                        print(f"[Neck] hw check error: {e}")
                last_hw_check = now

            if now - last_status_print > 0.5:
                print(
                    f"[Neck] yaw {math.degrees(yaw_cmd):+6.1f}° "
                    f"(tick {yaw_tick:4d})  pitch "
                    f"{math.degrees(pitch_cmd):+6.1f}° (tick {pitch_tick:4d})"
                )
                last_status_print = now

            sleep_s = next_tick - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.time()

    def stop(self) -> None:
        if not self._running and self._port is None:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            if self._port is not None and self._packet is not None:
                self._packet.write4ByteTxRx(
                    self._port, self.yaw_id, ADDR_GOAL_POSITION,
                    self._rad_to_tick(0.0, self.yaw_zero_tick),
                )
                self._packet.write4ByteTxRx(
                    self._port, self.pitch_id, ADDR_GOAL_POSITION,
                    self._rad_to_tick(0.0, self.pitch_zero_tick),
                )
                time.sleep(0.5)
                for i in (self.yaw_id, self.pitch_id):
                    self._packet.write1ByteTxRx(
                        self._port, i, ADDR_TORQUE_ENABLE, 0
                    )
                self._port.closePort()
                print("[Neck] Shutdown: zeroed, torque off, port closed.")
        except Exception as e:
            print(f"[Neck] Shutdown error: {e}")
        finally:
            self._port = None
            self._packet = None
            if self._pose_sub is not None:
                try:
                    self._pose_sub.close(linger=0)
                except Exception:
                    pass
                self._pose_sub = None
            if self._state_pub is not None:
                try:
                    self._state_pub.close(linger=0)
                except Exception:
                    pass
                self._state_pub = None


# ──────────────────────────────────────────────────────────────────────────────
# ZMQ REP server
# ──────────────────────────────────────────────────────────────────────────────


def start_server(cfg: argparse.Namespace) -> None:
    global gi, Gst

    threading.Thread(target=capture_thread, args=(cfg,), daemon=True).start()
    threading.Thread(target=viewer_pub_thread, args=(cfg,), daemon=True).start()

    pico_streamer: Optional[PicoVideoStreamer] = None
    if cfg.enable_pico:
        try:
            import gi as _gi
            _gi.require_version("Gst", "1.0")
            from gi.repository import Gst as _Gst
            gi = _gi
            Gst = _Gst
        except ImportError as e:
            print(f"--enable-pico set but GStreamer/PyGObject not available: {e}")
            cfg.enable_pico = False

    if cfg.enable_pico and Gst is not None:
        pico_streamer = PicoVideoStreamer(
            pico_ip=cfg.pico_ip, port=cfg.pico_port, source=cfg.pico_source
        )
        pico_streamer.start()
        print(
            f"[PicoStreamer] Started {cfg.pico_source.upper()} streaming to "
            f"Pico {cfg.pico_ip}:{cfg.pico_port}"
        )

    neck_motor: Optional[NeckMotor] = None
    if cfg.enable_neck_motor:
        neck_motor = NeckMotor(
            pose_zmq_addr=cfg.pose_zmq,
            state_pub_addr=cfg.neck_state_pub,
        )
        if not neck_motor.start():
            print("[Neck] Failed to start; continuing without motor control.")
            neck_motor = None

    context = zmq.Context.instance()
    sock = context.socket(zmq.REP)
    sock.bind(cfg.zmq_bind)
    print(f"[ZMQ] REP bound to {cfg.zmq_bind}, waiting for requests...")

    def _force_exit(sig, _frame):
        print(f"\n[Server] Signal {sig} received - shutting down.")
        if neck_motor is not None:
            neck_motor.stop()
        if pico_streamer is not None:
            pico_streamer.stop()
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
        if pico_streamer is not None:
            pico_streamer.stop()
        if neck_motor is not None:
            neck_motor.stop()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Native RealSense head-camera server (no ZED, no neck). "
                    "REP recording + PUB viewer + optional Pico H.264."
    )
    p.add_argument("--zmq-bind", default=ZMQ_BIND_DEFAULT,
                   help="ZMQ REP bind for recording/inference (default tcp://0.0.0.0:5558)")
    p.add_argument("--viewer-pub", default=VIEWER_PUB_DEFAULT,
                   help="ZMQ PUB bind for the live viewer stream (default "
                        "tcp://0.0.0.0:5559). Empty string disables it.")
    p.add_argument("--fps", type=int, default=int(os.environ.get("RS_FPS", "30")))
    p.add_argument("--width", type=int, default=int(os.environ.get("RS_WIDTH", "640")))
    p.add_argument("--height", type=int, default=int(os.environ.get("RS_HEIGHT", "480")))
    p.add_argument("--jpeg-quality", type=int,
                   default=int(os.environ.get("RS_JPEG_QUALITY", "80")))
    p.add_argument("--exposure", type=int,
                   default=int(os.environ.get("RS_EXPOSURE", "0")),
                   help="Fixed RGB exposure (us). 0 = keep auto-exposure but "
                        "force constant fps via auto_exposure_priority=0.")
    p.add_argument("--serial", default=os.environ.get("RS_SERIAL", ""),
                   help="RealSense device serial (empty = first device found)")
    p.add_argument("--no-ir", dest="enable_ir", action="store_false",
                   help="Do not stream/serve the IR L|R pair (needed for 30fps on USB2)")
    p.add_argument("--no-depth", dest="enable_depth", action="store_false",
                   help="Do not stream/serve the depth map")
    p.add_argument("--enable-neck-motor", action="store_true",
                   help="Drive the 2-DOF neck (Dynamixel IDs 0=yaw, 1=pitch) from "
                        "the [yaw, pitch] stream on --pose-zmq")
    p.add_argument("--pose-zmq", default=os.environ.get("POSE_ZMQ", ""),
                   help="ZMQ SUB address of the neck-angle publisher, e.g. "
                        "tcp://<desktop-ip>:5570. Required with --enable-neck-motor.")
    p.add_argument("--neck-state-pub",
                   default=os.environ.get("NECK_STATE_PUB", "tcp://*:5560"),
                   help="ZMQ PUB bind for neck present-position [yaw_rad, pitch_rad] "
                        "(default tcp://*:5560; empty string disables)")
    p.add_argument("--enable-pico", action="store_true",
                   help="Stream video to the Pico headset over H.264/TCP")
    p.add_argument("--pico-ip", default=PICO_IP)
    p.add_argument("--pico-port", type=int, default=VIDEO_PORT)
    p.add_argument("--pico-source", choices=("rgb", "ir"), default="rgb",
                   help="What to stream to the Pico: 'rgb' = mono color frame; "
                        "'ir' = the L|R infrared pair side by side (true stereo "
                        "view, grayscale). 'ir' requires IR enabled (no --no-ir) "
                        "and auto-disables the IR dot projector.")
    p.add_argument("--list-devices", action="store_true",
                   help="List connected RealSense devices and exit")
    p.set_defaults(enable_ir=True, enable_depth=True)
    return p.parse_args()


def main() -> None:
    ns = _parse_args()
    if ns.list_devices:
        list_realsense_devices()
        return
    # IR-to-Pico needs the IR pair captured, and the dot projector off so the
    # operator isn't looking at a speckle field.
    ns.disable_emitter = ns.enable_pico and ns.pico_source == "ir"
    if ns.pico_source == "ir" and ns.enable_pico and not ns.enable_ir:
        raise SystemExit("--pico-source ir requires IR capture (remove --no-ir)")
    start_server(ns)


if __name__ == "__main__":
    main()
