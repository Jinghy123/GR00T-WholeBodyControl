# realsense_server_with_neck.py

Native RealSense head camera (the working `pyrealsense2` path from
`realsense_native_server.py`) + the 2-DOF Dynamixel neck (ported from
`realsense_server.py`). No ZED, no pyzed import.

## What it serves

| Channel | Address (default) | Content |
|---|---|---|
| ZMQ REP | `tcp://0.0.0.0:5558` | 3-part reply: RGB JPEG / IR L\|R JPEG / depth z16 raw (`b""` if disabled). Same contract as `realsense_native_server.py`, so existing recording clients work unchanged. |
| ZMQ PUB (viewer) | `tcp://0.0.0.0:5559` | Same 3 parts, drop-on-slow. Use `realsense_viewer.py --sub`. |
| Pico H.264/TCP | `--pico-ip:12345` | RGB or IR stereo pair, x264 zerolatency. |
| ZMQ PUB (neck state) | `tcp://*:5560` | `[yaw_rad, pitch_rad]` present position at 50 Hz, for the data recorder. Only when `--enable-neck-motor`. |

## Prerequisites (on the robot)

1. Free the RealSense from the vendor process (it holds the camera and
   `pipeline.start` will fail with "Device or resource busy" otherwise):

   ```bash
   sudo killall -9 videohub_pc4
   ```

2. Give the neck serial port permissions (only needed with `--enable-neck-motor`):

   ```bash
   sudo chmod 777 /dev/ttyUSB0
   ```

3. Environment: use `ruohai` (the env where `realsense_native_server.py`
   already works — it has the system librealsense / v4l2 backend).
   The neck additionally needs `dynamixel-sdk`; if the server prints
   `[Neck] Import failed`, install it once:

   ```bash
   conda activate ruohai
   pip install dynamixel-sdk
   ```

   Note: the old `LD_PRELOAD=/lib/aarch64-linux-gnu/libffi.so.7` was a
   workaround for the `sonic` env (ZED/GStreamer conflict). The `ruohai`
   command has been run without it; only add it if you see a libffi import
   error.

4. On the desktop, `pose_publisher.py` must be running and publishing
   `[neck_yaw, neck_pitch]` on port 5570 — otherwise the neck holds its
   current pose and prints "no neck data yet" (it will start following as
   soon as data arrives, no restart needed).

## Launch

Full pipeline (camera + Pico IR stereo view + neck), matching the flags the
two old servers were run with:

```bash
sudo killall -9 videohub_pc4
sudo chmod 777 /dev/ttyUSB0
conda activate ruohai
cd ~/GR00T-WholeBodyControl
python weiduo_server/realsense_server_with_neck.py --no-depth \
    --enable-pico --pico-ip 192.168.0.241 --pico-source ir \
    --enable-neck-motor --pose-zmq tcp://192.168.123.222:5570
```

Camera only (no neck, behaves exactly like `realsense_native_server.py`):

```bash
python weiduo_server/realsense_server_with_neck.py --no-depth \
    --enable-pico --pico-ip 192.168.0.241 --pico-source ir
```

List cameras and exit:

```bash
python weiduo_server/realsense_server_with_neck.py --list-devices
```

## Flag notes

- `--pico-ip`: the ZED server used `192.168.0.242`, the native server used
  `192.168.0.241`. Default here is `241`; pass whichever the Pico currently
  has.
- `--pico-source ir` streams the true stereo IR pair to the headset and
  auto-disables the IR dot projector (clean view, weaker depth). It requires
  IR capture, so do NOT combine with `--no-ir`.
- USB2 limitation (from the native server): RGB or RGB+depth run at 30 fps,
  but enabling both IR streams drops capture to ~15-22 fps. If the Pico view
  is not needed, run with `--no-ir` for full 30 fps recording.
- `--pose-zmq` is mandatory with `--enable-neck-motor`; the server exits
  immediately with a clear message if it is missing.
- `--neck-state-pub ''` disables the 5560 neck-state stream if something else
  already binds that port.
- Neck calibration constants (zero ticks 1897/3275, limits 60/45 deg, signs,
  smoothing 0.3, 50 Hz) are module-level constants at the top of the script,
  copied verbatim from `realsense_server.py`. Edit in place to recalibrate.

## Shutdown behavior

Ctrl-C (SIGINT) or SIGTERM: the neck is commanded back to zero pose, torque
is disabled, the serial port is closed, then the process exits. If the neck
fails to start (port missing, ping failure, no dynamixel-sdk), the server
prints a red error and keeps running camera-only rather than dying.

## Ports summary

- 5558 REP — recording/inference frames
- 5559 PUB — viewer frames
- 5560 PUB — neck present position (neck enabled only)
- 5570 (remote, desktop) — pose_publisher.py neck-angle source
- 12345 TCP (remote, Pico) — H.264 video
