"""
Unified camera server (robot-side):
  - ZED egocentric camera  (left RGB + stereo L|R)
  - RealSense D405 wrist cameras (left, right)

ZMQ REP bind: tcp://192.168.123.164:5558
Each request receives a 4-part multipart reply (all JPEG bytes, b"" if unavailable):

  Part 0 — Ego left RGB JPEG       (ZED left)
  Part 1 — Ego stereo JPEG         (ZED left|right, for VR / archival)
  Part 2 — Left wrist RGB JPEG     (D405 serial 218622276113)
  Part 3 — Right wrist RGB JPEG    (D405 serial 218622276849)

Optional: H.264 TCP stream of ZED stereo to Pico VR (--enable-pico).

Environment overrides: ZMQ_BIND, ENABLE_PICO, PICO_IP, VIDEO_PORT,
                      ZED_RESOLUTION, ZED_FPS, VR_WIDTH, VR_HEIGHT
"""

from __future__ import annotations

import argparse
import collections
import io
import math
import os
import signal
import socket
import struct
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
import pyzed.sl as sl
import pyrealsense2 as rs
import zmq

gi = None
Gst = None

# ──────────────────────────────────────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────────────────────────────────────

latest_ego_rgb_bytes: Optional[bytes] = None
latest_ego_stereo_bytes: Optional[bytes] = None
latest_ego_stereo_frame: Optional[np.ndarray] = None
latest_left_wrist_bytes: Optional[bytes] = None
latest_right_wrist_bytes: Optional[bytes] = None
frame_seq = 0
frame_cond = threading.Condition()

# Defaults (overridden by CLI / env)
ZMQ_BIND_DEFAULT = os.environ.get("ZMQ_BIND", "tcp://192.168.123.164:5558")
PICO_IP = os.environ.get("PICO_IP", "192.168.0.128")
VIDEO_PORT = int(os.environ.get("VIDEO_PORT", "12345"))
VR_WIDTH = int(os.environ.get("VR_WIDTH", "0"))   # 0 = use native stereo size
VR_HEIGHT = int(os.environ.get("VR_HEIGHT", "0"))
FPS = 30

RESOLUTION_MAP = {
    "auto": sl.RESOLUTION.AUTO,
    "vga": sl.RESOLUTION.VGA,
    "hd720": sl.RESOLUTION.HD720,
    "hd1080": sl.RESOLUTION.HD1080,
    "hd1200": sl.RESOLUTION.HD1200,
    "hd2k": sl.RESOLUTION.HD2K,
}

# Neck-motor configuration — edit these in place to recalibrate.
NECK_PORT             = "/dev/ttyUSB0"
NECK_BAUD             = 2_000_000
NECK_YAW_ID           = 0
NECK_PITCH_ID         = 1
NECK_YAW_ZERO_TICK    = 1897
NECK_PITCH_ZERO_TICK  = 2900
NECK_YAW_LIMIT_DEG    = 60.0
NECK_PITCH_LIMIT_DEG  = 45.0
NECK_SMOOTH_ALPHA     = 0.3    # 0 = frozen, 1 = no smoothing
NECK_CONTROL_HZ       = 50
# Headset quaternion → Euler. Calibrate signs/order by watching real motor vs. head.
NECK_EULER_ORDER      = "yxz"   # scipy convention; returns (yaw, pitch, roll)
NECK_YAW_SIGN         = -1
NECK_PITCH_SIGN       = -1

ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_HW_ERROR_STATUS  = 70
TICKS_PER_REV         = 4096


def _bgra_to_bgr(bgra: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgra[:, :, :3])


def _jpeg_encode_bgr(bgr: np.ndarray, quality: int) -> bytes:
    q = max(1, min(100, int(quality)))
    try:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
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
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    im = Image.fromarray(rgb, mode="RGB")
    bio = io.BytesIO()
    im.save(bio, format="JPEG", quality=q)
    return bio.getvalue()


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


# ──────────────────────────────────────────────────────────────────────────────
# ZED egocentric capture
# ──────────────────────────────────────────────────────────────────────────────


def zed_capture_thread(cfg: Any) -> None:
    global latest_ego_rgb_bytes, latest_ego_stereo_bytes
    global latest_ego_stereo_frame, frame_seq

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = cfg.resolution
    init.camera_fps = cfg.fps
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.sdk_verbose = cfg.sdk_verbose

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[ZED] Open failed: {repr(err)}. Ego capture thread exits.")
        return

    left_mat = sl.Mat()
    right_mat = sl.Mat()
    runtime = sl.RuntimeParameters()

    print(f"[ZED] Started: resolution={cfg.resolution_name} fps={cfg.fps}")

    while True:
        try:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(left_mat, sl.VIEW.LEFT, sl.MEM.CPU)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT, sl.MEM.CPU)

            left_rgba = np.ascontiguousarray(
                np.asarray(left_mat.get_data(), dtype=np.uint8)
            )
            right_rgba = np.ascontiguousarray(
                np.asarray(right_mat.get_data(), dtype=np.uint8)
            )
            if left_rgba.ndim != 3 or right_rgba.ndim != 3:
                continue
            if left_rgba.shape[2] != 4 or right_rgba.shape[2] != 4:
                continue

            left_bgr = _bgra_to_bgr(left_rgba)
            right_bgr = _bgra_to_bgr(right_rgba)
            stereo_bgr = np.ascontiguousarray(np.hstack((left_bgr, right_bgr)))

            enc_rgb = _jpeg_encode_bgr(left_bgr, cfg.jpeg_rgb_quality)
            enc_stereo = _jpeg_encode_bgr(stereo_bgr, cfg.jpeg_stereo_quality)

            with frame_cond:
                latest_ego_rgb_bytes = enc_rgb
                latest_ego_stereo_bytes = enc_stereo
                latest_ego_stereo_frame = stereo_bgr.copy()
                frame_seq += 1
                frame_cond.notify_all()

        except Exception as e:
            print(f"[ZED] Capture error: {e}")
            time.sleep(0.01)


# ──────────────────────────────────────────────────────────────────────────────
# RealSense D405 wrist capture
# ──────────────────────────────────────────────────────────────────────────────


def wrist_capture_thread(serial: str, side: str) -> None:
    global latest_left_wrist_bytes, latest_right_wrist_bytes

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
        print(f"[Wrist-{side}] Started (serial={serial}): RGB active.")
    except Exception as e:
        print(f"[Wrist-{side}] Failed to start (serial={serial}): {e}")
        return

    while True:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.ascontiguousarray(
                np.asarray(color_frame.get_data(), dtype=np.uint8)
            )
            if color_image.ndim != 3 or color_image.size == 0:
                continue
            encoded_rgb = _jpeg_encode_bgr(color_image, 80)

            with frame_cond:
                if side == "left":
                    latest_left_wrist_bytes = encoded_rgb
                else:
                    latest_right_wrist_bytes = encoded_rgb

        except Exception as e:
            print(f"[Wrist-{side}] Capture error: {e}")
            time.sleep(0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Pico Video Streamer (GStreamer H.264 + TCP) — streams ZED stereo
# ──────────────────────────────────────────────────────────────────────────────


class PicoVideoStreamer:
    """Streams ego stereo BGR frames to Pico via GStreamer x264.

    3 threads:
      _push_loop        — waits for fresh ZED stereo frames, feeds appsrc
      _on_encoded_frame — GStreamer callback, queues H.264 NALUs (never blocks)
      _send_loop        — pops from queue, does TCP sendall (isolated from pipeline)
      _connection_loop  — reconnects TCP on failure
    """

    def __init__(
        self,
        pico_ip: str = PICO_IP,
        port: int = VIDEO_PORT,
        width: int = VR_WIDTH,
        height: int = VR_HEIGHT,
        fps: int = FPS,
    ):
        self.pico_ip = pico_ip
        self.port = port
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
        self._send_q: collections.deque = collections.deque(maxlen=3)
        self._send_event = threading.Event()

    def start(self) -> None:
        global Gst
        if Gst is None:
            raise RuntimeError("GStreamer (gi) not loaded; cannot start Pico streamer")
        # PyGObject >=3.50 rejects None here; pass an empty argv list.
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
                    print(
                        f"\033[92m[PicoStreamer] Connected to Pico "
                        f"{self.pico_ip}:{self.port}\033[0m"
                    )
                except Exception:
                    pass
            time.sleep(2)

    def _start_pipeline(self, width: int, height: int) -> None:
        self.vr_width = width
        self.vr_height = height
        pipe_str = (
            f"appsrc name=src is-live=True format=time "
            f"do-timestamp=True ! "
            f"video/x-raw,format=BGR,"
            f"width={width},height={height},"
            f"framerate={self.fps}/1 ! "
            f"videoconvert ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast "
            f"bitrate=4000 vbv-buf-capacity=500 "
            f"key-int-max=3 threads=2 ! "
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
        print(
            f"[PicoStreamer] GStreamer pipeline: {width}x{height}@{self.fps} "
            f"bitrate=4000k preset=ultrafast keyint=3"
        )

    def _on_encoded_frame(self, sink) -> Any:
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
                        print("\033[91m[PicoStreamer] Connection lost. Retrying...\033[0m")

    def _push_loop(self) -> None:
        last_seq = 0
        while self._running:
            frame = None
            with frame_cond:
                while frame_seq == last_seq and self._running:
                    frame_cond.wait(timeout=0.1)
                if latest_ego_stereo_frame is not None:
                    frame = latest_ego_stereo_frame
                    last_seq = frame_seq

            if frame is None:
                continue

            h, w = frame.shape[:2]

            if not self._pipeline_started:
                out_w = self.vr_width if self.vr_width > 0 else w
                out_h = self.vr_height if self.vr_height > 0 else h
                self._start_pipeline(out_w, out_h)
                self._pipeline_started = True

            if w != self.vr_width or h != self.vr_height:
                frame = cv2.resize(frame, (self.vr_width, self.vr_height))

            stereo = np.ascontiguousarray(frame)
            gst_buf = Gst.Buffer.new_wrapped(stereo.tobytes())
            gst_buf.pts = self._frame_id * (Gst.SECOND // self.fps)
            gst_buf.duration = Gst.SECOND // self.fps
            self._appsrc.emit("push-buffer", gst_buf)
            self._frame_id += 1

    def stop(self) -> None:
        self._running = False
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)


# ──────────────────────────────────────────────────────────────────────────────
# Neck motor driver (Pico head pose → 2x Dynamixel)
# ──────────────────────────────────────────────────────────────────────────────


class NeckMotor:
    """Reads Pico headset pose via xrobotoolkit_sdk and drives 2 Dynamixels.

    Thread model:
      _loop — CONTROL_HZ: poll xrt.get_headset_pose() → yaw/pitch → clamp →
              smooth → rad_to_tick → Dynamixel write. Periodic HW-error
              check + auto-reboot.

    Shutdown via stop(): zero-pose write, sleep 0.5s, torque off, close port.
    """

    def __init__(
        self,
        port: str = NECK_PORT,
        baud: int = NECK_BAUD,
        yaw_id: int = NECK_YAW_ID,
        pitch_id: int = NECK_PITCH_ID,
        yaw_zero_tick: int = NECK_YAW_ZERO_TICK,
        pitch_zero_tick: int = NECK_PITCH_ZERO_TICK,
        yaw_limit_rad: float = math.radians(NECK_YAW_LIMIT_DEG),
        pitch_limit_rad: float = math.radians(NECK_PITCH_LIMIT_DEG),
        smooth_alpha: float = NECK_SMOOTH_ALPHA,
        control_hz: int = NECK_CONTROL_HZ,
        euler_order: str = NECK_EULER_ORDER,
        yaw_sign: int = NECK_YAW_SIGN,
        pitch_sign: int = NECK_PITCH_SIGN,
    ):
        self.port_path = port
        self.baud = baud
        self.yaw_id = yaw_id
        self.pitch_id = pitch_id
        self.yaw_zero_tick = yaw_zero_tick
        self.pitch_zero_tick = pitch_zero_tick
        self.yaw_limit_rad = yaw_limit_rad
        self.pitch_limit_rad = pitch_limit_rad
        self.smooth_alpha = smooth_alpha
        self.control_hz = control_hz
        self.euler_order = euler_order
        self.yaw_sign = yaw_sign
        self.pitch_sign = pitch_sign

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._port = None
        self._packet = None
        self._xrt = None
        self._R = None

    @staticmethod
    def _rad_to_tick(rad: float, zero_tick: int) -> int:
        return zero_tick + int(rad * TICKS_PER_REV / (2 * math.pi))

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
        """Open SDK + serial port, enable torque, spawn loop thread.

        Returns True on success, False on failure (caller should continue
        without motor control).
        """
        try:
            import xrobotoolkit_sdk as xrt
            from dynamixel_sdk import PacketHandler, PortHandler
            from scipy.spatial.transform import Rotation as R
        except ImportError as e:
            print(
                f"\033[91m[Neck] Import failed: {e}. "
                f"pip install dynamixel-sdk scipy and install xrobotoolkit_sdk.\033[0m"
            )
            return False

        self._xrt = xrt
        self._R = R

        try:
            xrt.init()
            print("[Neck] XRoboToolkit SDK initialized.")
        except Exception as e:
            print(f"\033[91m[Neck] xrt.init() failed: {e}\033[0m")
            return False

        self._port = PortHandler(self.port_path)
        self._packet = PacketHandler(2.0)
        if not self._port.openPort() or not self._port.setBaudRate(self.baud):
            print(
                f"\033[91m[Neck] Could not open {self.port_path} @ {self.baud}\033[0m"
            )
            return False

        for i in (self.yaw_id, self.pitch_id):
            _, res, _ = self._packet.ping(self._port, i)
            if res != 0:
                print(f"\033[91m[Neck] Ping ID {i} failed (res={res})\033[0m")
                self._port.closePort()
                return False
            self._check_and_reboot(i)
            self._packet.write1ByteTxRx(
                self._port, i, ADDR_TORQUE_ENABLE, 1
            )

        # Start at zero pose
        self._packet.write4ByteTxRx(
            self._port, self.yaw_id, ADDR_GOAL_POSITION,
            self._rad_to_tick(0.0, self.yaw_zero_tick),
        )
        self._packet.write4ByteTxRx(
            self._port, self.pitch_id, ADDR_GOAL_POSITION,
            self._rad_to_tick(0.0, self.pitch_zero_tick),
        )
        time.sleep(0.3)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        print(
            f"\033[92m[Neck] Started: {self.port_path}@{self.baud} "
            f"IDs {self.yaw_id}/{self.pitch_id} "
            f"limits yaw±{math.degrees(self.yaw_limit_rad):.0f}° "
            f"pitch±{math.degrees(self.pitch_limit_rad):.0f}° "
            f"@ {self.control_hz}Hz\033[0m"
        )
        return True

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

            # Read headset pose
            yaw_target = yaw_cmd
            pitch_target = pitch_cmd
            pose_ok = False
            try:
                pose = self._xrt.get_headset_pose()
                if pose is not None and len(pose) >= 7:
                    qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
                    # SDK returns zero-filled quat when no live data has
                    # arrived yet from the Pico; skip until stream is up.
                    if (qx * qx + qy * qy + qz * qz + qw * qw) > 1e-6:
                        euler = self._R.from_quat(
                            [qx, qy, qz, qw]
                        ).as_euler(self.euler_order)
                        yaw_target = float(euler[0]) * self.yaw_sign
                        pitch_target = float(euler[1]) * self.pitch_sign
                        pose_ok = True
            except Exception as e:
                now = time.time()
                if now - last_warn_print > 2.0:
                    print(f"[Neck] headset read error: {e}")
                    last_warn_print = now

            if not pose_ok:
                now = time.time()
                if pose_valid or now - last_warn_print > 5.0:
                    print(
                        "[Neck] no headset data yet "
                        "(Pico not streaming / service down?)"
                    )
                    last_warn_print = now
                pose_valid = False
                # Hold current cmd — don't snap to zero.
                yaw_target = yaw_cmd
                pitch_target = pitch_cmd
            else:
                pose_valid = True

            # Clamp
            yaw_target = self._clamp(
                yaw_target, -self.yaw_limit_rad, self.yaw_limit_rad
            )
            pitch_target = self._clamp(
                pitch_target, -self.pitch_limit_rad, self.pitch_limit_rad
            )

            # Low-pass smoothing
            yaw_cmd += self.smooth_alpha * (yaw_target - yaw_cmd)
            pitch_cmd += self.smooth_alpha * (pitch_target - pitch_cmd)

            # Write motors
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

            now = time.time()

            # Periodic hw-error check every 2s
            if now - last_hw_check > 2.0:
                for i in (self.yaw_id, self.pitch_id):
                    try:
                        self._check_and_reboot(i)
                    except Exception as e:
                        print(f"[Neck] hw check error: {e}")
                last_hw_check = now

            # Status line every 0.5s
            if now - last_status_print > 0.5:
                print(
                    f"[Neck] yaw {math.degrees(yaw_cmd):+6.1f}° "
                    f"(tick {yaw_tick:4d})  pitch "
                    f"{math.degrees(pitch_cmd):+6.1f}° (tick {pitch_tick:4d})"
                )
                last_status_print = now

            # Sleep
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


# ──────────────────────────────────────────────────────────────────────────────
# RealSense device listing
# ──────────────────────────────────────────────────────────────────────────────


def list_realsense_devices() -> None:
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices detected.")
        return
    print(f"Found {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        print(f"  [{i}] {name}  serial={serial}  firmware={fw}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI + server entry
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified camera server: ZED ego + RealSense D405 wrist (ZMQ + optional Pico)"
    )

    # ZED
    p.add_argument(
        "--resolution",
        default=os.environ.get("ZED_RESOLUTION", "vga"),
        choices=sorted(RESOLUTION_MAP.keys()),
        help="ZED resolution preset",
    )
    p.add_argument(
        "--fps", type=int,
        default=int(os.environ.get("ZED_FPS", "30")),
        help="ZED camera FPS",
    )
    p.add_argument(
        "--sdk-verbose", type=int,
        default=int(os.environ.get("ZED_SDK_VERBOSE", "0")),
    )
    p.add_argument(
        "--jpeg-rgb-quality", type=int,
        default=int(os.environ.get("ZED_JPEG_RGB_QUALITY", "80")),
    )
    p.add_argument(
        "--jpeg-stereo-quality", type=int,
        default=int(os.environ.get("ZED_JPEG_STEREO_QUALITY", "60")),
    )

    # Wrist (D405 serials documented in past_streaming_pipeline)
    p.add_argument(
        "--left-wrist-serial", default="218622276113",
        help="Serial of left wrist D405 (empty string to disable)",
    )
    p.add_argument(
        "--right-wrist-serial", default="218622276849",
        help="Serial of right wrist D405 (empty string to disable)",
    )
    p.add_argument(
        "--zed-only", action="store_true",
        default=_env_bool("ZED_ONLY", False),
        help="Use ZED only; disable both RealSense wrist cameras",
    )

    # Neck motor
    p.add_argument(
        "--enable-neck-motor", action="store_true",
        default=_env_bool("NECK_MOTOR", False),
        help="Drive the 2-DOF neck (Dynamixel IDs 0=yaw, 1=pitch) from Pico headset pose",
    )

    # Network
    p.add_argument(
        "--zmq-bind",
        default=ZMQ_BIND_DEFAULT,
        help="ZMQ REP bind address (default tcp://192.168.123.164:5558)",
    )

    # Pico
    p.set_defaults(enable_pico=_env_bool("ENABLE_PICO", False))
    p.add_argument(
        "--enable-pico", action="store_true", dest="enable_pico",
        help="Enable Pico GStreamer TCP stream of ZED stereo",
    )
    p.add_argument(
        "--no-enable-pico", action="store_false", dest="enable_pico",
        help="Disable Pico stream",
    )
    p.add_argument("--pico-ip", default=PICO_IP)
    p.add_argument("--pico-port", type=int, default=VIDEO_PORT)

    # Utility
    p.add_argument(
        "--list-devices", action="store_true",
        help="List connected RealSense devices and exit",
    )
    return p.parse_args()


def _build_config(ns: argparse.Namespace) -> Any:
    class C:
        pass

    c = C()
    c.resolution = RESOLUTION_MAP[ns.resolution]
    c.resolution_name = ns.resolution
    c.fps = ns.fps
    c.sdk_verbose = ns.sdk_verbose
    c.jpeg_rgb_quality = ns.jpeg_rgb_quality
    c.jpeg_stereo_quality = ns.jpeg_stereo_quality
    c.zmq_bind = ns.zmq_bind
    c.enable_pico = ns.enable_pico
    c.pico_ip = ns.pico_ip
    c.pico_port = ns.pico_port
    c.left_wrist_serial = "" if ns.zed_only else ns.left_wrist_serial
    c.right_wrist_serial = "" if ns.zed_only else ns.right_wrist_serial
    c.zed_only = ns.zed_only
    c.enable_neck_motor = ns.enable_neck_motor
    return c


def start_server(cfg: Any) -> None:
    global Gst, gi

    if cfg.zed_only:
        print("[Server] Mode: ZED only (wrist cameras disabled).")
    else:
        print("[Server] Mode: ZED + 2x RealSense wrist cameras.")
        list_realsense_devices()

    threading.Thread(
        target=zed_capture_thread, args=(cfg,), daemon=True
    ).start()

    if cfg.left_wrist_serial:
        threading.Thread(
            target=wrist_capture_thread,
            args=(cfg.left_wrist_serial, "left"),
            daemon=True,
        ).start()
    else:
        print("[Wrist-left] Disabled (no serial provided).")

    if cfg.right_wrist_serial:
        threading.Thread(
            target=wrist_capture_thread,
            args=(cfg.right_wrist_serial, "right"),
            daemon=True,
        ).start()
    else:
        print("[Wrist-right] Disabled (no serial provided).")

    pico_streamer: Optional[PicoVideoStreamer] = None
    if cfg.enable_pico:
        try:
            import gi as _gi
            _gi.require_version("Gst", "1.0")
            from gi.repository import Gst as _Gst
            gi = _gi
            Gst = _Gst
        except ImportError as e:
            print(f"ENABLE_PICO set but GStreamer/PyGObject not available: {e}")
            cfg.enable_pico = False

    if cfg.enable_pico and Gst is not None:
        pico_streamer = PicoVideoStreamer(
            pico_ip=cfg.pico_ip, port=cfg.pico_port
        )
        pico_streamer.start()
        print("\033[92m[PicoStreamer] Started ego stereo streaming to Pico\033[0m")

    neck_motor: Optional[NeckMotor] = None
    if cfg.enable_neck_motor:
        neck_motor = NeckMotor()
        if not neck_motor.start():
            print("\033[91m[Neck] Failed to start; continuing without motor control.\033[0m")
            neck_motor = None

    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.bind(cfg.zmq_bind)
    print(f"[ZMQ] REP bound to {cfg.zmq_bind}, waiting for requests...")

    def _force_exit(sig, _frame):
        print(f"\n\033[91m[Server] Signal {sig} received – shutting down.\033[0m")
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
                ego_rgb = latest_ego_rgb_bytes
                ego_stereo = latest_ego_stereo_bytes
                lw = latest_left_wrist_bytes
                rw = latest_right_wrist_bytes
                last_sent_seq = frame_seq

            if ego_rgb is None:
                sock.send_multipart([b"", b"", b"", b""])
            else:
                sock.send_multipart([
                    ego_rgb,
                    ego_stereo if ego_stereo is not None else b"",
                    lw if lw is not None else b"",
                    rw if rw is not None else b"",
                ])
    finally:
        sock.close()
        context.term()
        if pico_streamer is not None:
            pico_streamer.stop()
        if neck_motor is not None:
            neck_motor.stop()


def main() -> None:
    ns = _parse_args()
    if ns.list_devices:
        list_realsense_devices()
        return
    cfg = _build_config(ns)
    start_server(cfg)


if __name__ == "__main__":
    main()
