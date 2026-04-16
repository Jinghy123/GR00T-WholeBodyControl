import datetime
import signal
import socket
import struct
import threading
import time
import cv2
import gi
import numpy as np
import pyrealsense2 as rs
import zmq

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# Shared variables
latest_rgb_bytes = None
latest_ir_bytes = None
latest_rgb_frame = None  # For ZMQ clients
latest_ir_stereo_frame = None  # For Pico streaming (IR left+right stacked)
# Pre-generate a fake depth buffer (zeros) to maintain pipeline compatibility
# 640x480 uint16 = 614,400 bytes
FAKE_DEPTH_BYTES = np.zeros((480, 640), dtype=np.uint16).tobytes()
frame_lock = threading.Lock()

# Pico streaming configuration
PICO_IP = "192.168.0.128"
VIDEO_PORT = 12345
VR_WIDTH = 1920
VR_HEIGHT = 1080
FPS = 30

def frame_capture_thread():
    global latest_rgb_bytes, latest_ir_bytes, latest_rgb_frame, latest_ir_stereo_frame

    pipeline = rs.pipeline()
    config = rs.config()

    # Enable ONLY RGB and IR (to fit USB 2.0 bandwidth)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    try:
        pipeline.start(config)
        print("RealSense: RGB + IR active. Depth is MOCKED (zeros) for USB 2.0 compatibility.")
    except Exception as e:
        print(f"Failed to start RealSense: {e}")
        return
    while True:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            ir_left_frame = frames.get_infrared_frame(1)
            ir_right_frame = frames.get_infrared_frame(2)

            if not (color_frame and ir_left_frame and ir_right_frame):
                continue

            # Process Color
            color_image = np.asanyarray(color_frame.get_data())
            # Use JPEG quality 80 for USB 2.0 stability
            _, encoded_rgb = cv2.imencode(".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Process IR
            ir_l = np.asanyarray(ir_left_frame.get_data())
            ir_r = np.asanyarray(ir_right_frame.get_data())
            # Convert to BGR to match your original processing logic
            ir_l_bgr = cv2.cvtColor(ir_l, cv2.COLOR_GRAY2BGR)
            ir_r_bgr = cv2.cvtColor(ir_r, cv2.COLOR_GRAY2BGR)
            ir_combined = np.hstack((ir_l_bgr, ir_r_bgr))
            _, encoded_ir = cv2.imencode(".jpg", ir_combined, [cv2.IMWRITE_JPEG_QUALITY, 60])

            with frame_lock:
                latest_rgb_bytes = encoded_rgb.tobytes()
                latest_ir_bytes = encoded_ir.tobytes()
                latest_rgb_frame = color_image.copy()  # For ZMQ clients
                latest_ir_stereo_frame = ir_combined.copy()  # For Pico streaming (stereo IR)

        except Exception as e:
            print(f"Capture error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Pico Video Streamer (based on teleop g1_data_server.py)
# ──────────────────────────────────────────────────────────────────────────────

class PicoVideoStreamer:
    """Streams IR stereo frames to Pico headset via GStreamer H.264."""

    def __init__(self, pico_ip=PICO_IP, port=VIDEO_PORT,
                 width=VR_WIDTH, height=VR_HEIGHT, fps=FPS):
        self.pico_ip   = pico_ip
        self.port      = port
        self.vr_width  = width
        self.vr_height = height
        self.fps       = fps

        self._running      = False
        self._connected    = False
        self._sock         = None
        self._pipeline     = None
        self._appsrc       = None
        self._frame_id     = 0

    def start(self):
        Gst.init(None)
        self._running = True
        self._start_pipeline()
        threading.Thread(target=self._connection_loop, daemon=True).start()
        threading.Thread(target=self._push_loop, daemon=True).start()

    def _connection_loop(self):
        """Continuously attempts (re)connect the TCP socket to Pico."""
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
                    s.settimeout(2)
                    s.connect((self.pico_ip, self.port))
                    s.settimeout(None)
                    self._sock = s
                    self._connected = True
                    print(f"\033[92m[PicoStreamer] Connected to Pico {self.pico_ip}:{self.port}\033[0m")
                except Exception:
                    pass
            time.sleep(2)

    def _start_pipeline(self):
        """Build and start the GStreamer encode pipeline."""
        pipe_str = (
            f"appsrc name=src is-live=True format=time ! "
            f"video/x-raw,format=BGR,"
            f"width={self.vr_width},height={self.vr_height},"
            f"framerate={self.fps}/1 ! "
            f"videoconvert ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast key-int-max=15 ! "
            f"video/x-h264,profile=baseline ! "
            f"h264parse config-interval=-1 ! "
            f"video/x-h264,stream-format=byte-stream,alignment=au ! "
            f"appsink name=sink emit-signals=True sync=False"
        )
        self._pipeline = Gst.parse_launch(pipe_str)
        self._appsrc   = self._pipeline.get_by_name("src")
        appsink        = self._pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_encoded_frame)
        self._pipeline.set_state(Gst.State.PLAYING)

    def _on_encoded_frame(self, sink):
        """Forward each encoded H.264 NAL unit to the Pico socket."""
        sample = sink.emit("pull-sample")
        buf    = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            if self._connected:
                sock = self._sock
                if sock is not None:
                    try:
                        header = struct.pack(">I", len(info.data))
                        sock.sendall(header + info.data)
                    except Exception:
                        self._connected = False
                        print("\033[91m[PicoStreamer] Connection lost. Retrying...\033[0m")
            buf.unmap(info)
        return Gst.FlowReturn.OK

    def _push_loop(self):
        """Continuously push IR stereo frames into the GStreamer pipeline."""
        dt = 1.0 / self.fps
        while self._running:
            t0 = time.time()

            if self._appsrc is not None:
                frame = None
                with frame_lock:
                    if latest_ir_stereo_frame is not None:
                        frame = latest_ir_stereo_frame.copy()

                if frame is not None:
                    # Resize IR stereo frame to VR resolution
                    # frame is already 1280x480 (640+640, 480), resize to VR_WIDTH x VR_HEIGHT
                    resized = cv2.resize(frame, (self.vr_width, self.vr_height))
                    stereo = np.ascontiguousarray(resized)

                    gst_buf = Gst.Buffer.new_wrapped(stereo.tobytes())
                    gst_buf.pts      = self._frame_id * (Gst.SECOND // self.fps)
                    gst_buf.duration = Gst.SECOND // self.fps
                    self._appsrc.emit("push-buffer", gst_buf)
                    self._frame_id += 1

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def stop(self):
        self._running = False
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)


# ──────────────────────────────────────────────────────────────────────────────
# Original ZMQ Server (5558 port)
# ──────────────────────────────────────────────────────────────────────────────

def start_server():
    # Start camera capture thread
    threading.Thread(target=frame_capture_thread, daemon=True).start()

    # Start Pico video streaming thread
    pico_streamer = PicoVideoStreamer()
    pico_streamer.start()
    print("\033[92m[PicoStreamer] Started IR stereo video streaming to Pico\033[0m")

    # Start ZMQ server for psi-inference (5558 port)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://192.168.123.164:5558")
    print("ZMQ server started on port 5558, waiting for client requests...")

    # Register signal handler for clean exit
    def _force_exit(sig, _frame):
        print(f"\n\033[91m[Server] Signal {sig} received – forcing exit.\033[0m")
        import os
        os._exit(0)

    signal.signal(signal.SIGINT,  _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    try:
        while True:
            cur = time.time()
            request = socket.recv()
            print(f"req time: {time.time() - cur}")
            cur = time.time()
            with frame_lock:
                rgb = latest_rgb_bytes
                ir = latest_ir_bytes

            if rgb is None or ir is None:
                socket.send(b"")
            else:
                # Send RGB, IR, and the FAKE depth zeros
                socket.send_multipart([rgb, ir, FAKE_DEPTH_BYTES])
                print(f"send time: {time.time() - cur}")
    finally:
        socket.close()
        context.term()
        pico_streamer.stop()

if __name__ == "__main__":
    start_server()