# Pico VR + Trackers → ZED + Neck Motor Teleop

The neck driver is fed from **SMPL-X-derived neck angles**, not the raw
headset pose. The operator wears the Pico headset plus body trackers; a
desktop process retargets the body data to a 2-vector `[neck_yaw, neck_pitch]`
and publishes it to the G1 over ZMQ. Going through SMPL-X decouples neck
rotation from torso lean — when the operator walks or leans without rotating
their head, the neck angles stay still and the motors don't move.

```
  Pico VR headset + body trackers
        │
        │  ─── (tracking + body pose — XRoboToolKit protocol) ───┐
        │                                                        │
        │  ◄──────── H.264 stereo (TCP :12345) ──────────────────────────┐
        │                                                        │       │
                                                                 ▼       │
┌──────────────────────────────────────┐               ┌───────────────────────────┐
│      Desktop PC (x86_64)             │               │   G1 onboard PC           │
│                                      │               │                           │
│  XRoboToolKit service (.deb)         │               │  realsense_server.py      │
│  pose_publisher.py    ── ZMQ PUB ───►│──────────────►│    --pose-zmq tcp://…:5559│
│  (gmr env: XRobotStreamer +          │  :5559 JSON   │                           │
│   human_head_to_robot_neck)          │  [yaw, pitch] │  ├─ ZED Mini              │
│                                      │               │  └─ U2D2 + 2 Dynamixels   │
│  test_viewer.py  (optional)  ◄── REP─┤◄──────────────┤                           │
│                                 :5558│               │                           │
└──────────────────────────────────────┘               └───────────────────────────┘
```

Responsibilities:

- **Desktop PC** runs the **XRoboToolKit PC Service daemon**, the Python
  binding (`xrobotoolkit_sdk`), the `general_motion_retargeting` package
  (`gmr` env), and [pose_publisher.py](pose_publisher.py). It pulls SMPL-X
  body data via `XRobotStreamer`, calls `human_head_to_robot_neck` to extract
  neck yaw/pitch, and publishes the two angles over ZMQ PUB.
- **G1 onboard PC** owns the ZED, the U2D2 (`/dev/ttyUSB0`), and the 2
  Dynamixels. It does **not** need the XRoboToolKit binding or the gmr
  package — it only subscribes to the published angles.
- **Pico** runs the XRoboToolkit Client app and points at the **desktop's** IP.
- **Body trackers** (Pico Swift / equivalent) stream into the same
  XRoboToolKit daemon so SMPL-X body data is available.

---

## Hardware checklist

Connected to the **G1 onboard PC**:
- ZED Mini camera on USB3.
- U2D2 on `/dev/ttyUSB0`, two Dynamixels daisy-chained:
  - Yaw motor — ID `0`, zero tick `1897`, limit ±60°.
  - Pitch motor — ID `1`, zero tick `2900`, limit ±45°.
- (Optional) 2x RealSense D405 wrist cameras — disabled via `--zed-only`.

Worn by the operator:
- Pico VR headset.
- Body trackers paired with the XRoboToolKit Client app.

Network: Pico, G1, and Desktop all on the same LAN, able to ping each other.

---

## One-time setup

For the focused step-by-step, see
[TOPOLOGY2_QUICKSTART.md](TOPOLOGY2_QUICKSTART.md). The summary:

### Desktop PC

1. **XRoboToolKit Python binding (pre-built x86_64 `.so`)** — set
   `PYTHONPATH` and `LD_LIBRARY_PATH` per
   [TOPOLOGY2_QUICKSTART.md](TOPOLOGY2_QUICKSTART.md#desktop) step 2.
2. **XRoboToolKit PC Service daemon** — install the amd64 `.deb`
   (`XRoboToolkit_PC_Service_…_amd64.deb`).
3. **`gmr` env** for `general_motion_retargeting`. The `XRobotStreamer` and
   `human_head_to_robot_neck` symbols come from this package — see
   [TOPOLOGY2_QUICKSTART.md](TOPOLOGY2_QUICKSTART.md#desktop) step 4 for
   install commands.
4. `pip install pyzmq` in the same env, `sudo ufw disable`.

### G1 onboard PC

```bash
conda activate sonic
pip install dynamixel-sdk pyzmq numpy opencv-python scipy
sudo ufw disable
sudo usermod -aG dialout $USER     # log out/in for it to apply
```

No XRoboToolKit binding, no daemon, no gmr.

### Motor calibration (in-file constants)

All knobs live at the top of
[realsense_server.py](realsense_server.py):

```python
NECK_PORT             = "/dev/ttyUSB0"
NECK_BAUD             = 2_000_000
NECK_YAW_ID           = 0
NECK_PITCH_ID         = 1
NECK_YAW_ZERO_TICK    = 1897
NECK_PITCH_ZERO_TICK  = 2900
NECK_YAW_LIMIT_DEG    = 60.0
NECK_PITCH_LIMIT_DEG  = 45.0
NECK_SMOOTH_ALPHA     = 0.3
NECK_CONTROL_HZ       = 50
NECK_YAW_SIGN         = +1     # flip if motor moves opposite to head
NECK_PITCH_SIGN       = +1     # flip if motor moves opposite to head
```

Restart the server after any edit — no rebuild needed.

The publisher exposes `--neck-retarget-scale` (default `1.5`) — bump it up or
down on the desktop to scale total neck travel without recompiling.

### Pico headset

- Install the **XRoboToolkit Client** app.
- Set its Server IP to the **desktop's** LAN IP.
- Enable head tracking and pair the body trackers.

---

## Running the pipeline

**Desktop terminal 1 — XRoboToolKit service daemon (leave running):**
```bash
sudo bash /opt/apps/roboticsservice/runService.sh
```
Verify with `pgrep -fa RoboticsServiceProcess` — should print a PID.

**Pico headset:** launch the XRoboToolkit Client app, confirm it shows
"Connected" to the desktop, start tracking. Make sure body trackers are
streaming.

**Desktop terminal 2 — pose publisher (leave running):**
```bash
conda activate gmr
cd $GR00T_ROOT
python pose_publisher.py \
    --bind tcp://0.0.0.0:5559 \
    --hz 50 \
    --neck-retarget-scale 1.5
```
Expected logs: `XRobotStreamer initialized.`, `PUB bound to tcp://0.0.0.0:5559`,
then periodic `published N neck samples @ ~50 Hz`. If it warns
`smplx_data is None`, the body trackers aren't streaming yet (see troubleshooting).

**G1 terminal — launch the camera + motor server:**
```bash
conda activate sonic
cd ~/Desktop/GR00T-WholeBodyControl     # adjust path on the robot

python realsense_server.py \
    --zed-only \
    --zmq-bind tcp://0.0.0.0:5558 \
    --enable-pico --pico-ip <PICO_IP> \
    --enable-neck-motor \
    --pose-zmq tcp://<DESKTOP_IP>:5559
```

Expected lines in order:
- `[ZED] Started: resolution=vga fps=30`
- `[PicoStreamer] Connected to Pico <PICO_IP>:12345`
- `[Neck] ZMQ SUB neck-angle source: tcp://<DESKTOP_IP>:5559`
- `[Neck] Started: /dev/ttyUSB0@2000000 IDs 0/1 ...`
- Periodic `[Neck] yaw ... pitch ...` that track your head.

**Desktop terminal 3 — optional live ZED viewer:**
```bash
python test_viewer.py --server <G1_IP> --port 5558 --show-stereo
```

Ctrl-C on either machine stops cleanly. Motors zero and release torque on
the G1 side.

---

## Command-line flags reference

| Flag | Default | Purpose |
|------|---------|---------|
| `--zed-only` | off (env `ZED_ONLY`) | Disable both RealSense wrist cameras. |
| `--zmq-bind <addr>` | `tcp://192.168.123.164:5558` | ZMQ REP bind. Use `tcp://0.0.0.0:5558` to bind all interfaces. |
| `--enable-pico` | off (env `ENABLE_PICO`) | Stream ZED stereo as H.264 to the Pico. |
| `--pico-ip <ip>` | `192.168.0.128` | Pico headset IP. |
| `--pico-port <port>` | `12345` | Pico TCP video port. |
| `--enable-neck-motor` | off (env `NECK_MOTOR`) | Drive the 2-DOF neck. Requires `--pose-zmq`. |
| `--pose-zmq <addr>` | "" (env `POSE_ZMQ`) | **Required with `--enable-neck-motor`.** ZMQ SUB address of `pose_publisher.py`. Wire format: JSON `[neck_yaw, neck_pitch]` (radians, robot convention). Example: `tcp://192.168.0.50:5559`. |
| `--resolution <preset>` | `vga` | `vga`, `hd720`, `hd1080`, `hd1200`, `hd2k`, `auto`. |
| `--fps <n>` | `30` | ZED FPS. |
| `--list-devices` | — | Print attached RealSense devices and exit. |

Publisher flags (`pose_publisher.py`):

| Flag | Default | Purpose |
|------|---------|---------|
| `--bind <addr>` | `tcp://0.0.0.0:5559` | ZMQ PUB bind. |
| `--hz <n>` | `50` | Publish rate (matches `NECK_CONTROL_HZ`). |
| `--neck-retarget-scale <f>` | `1.5` | Multiplier on `(yaw, pitch)` before publishing. |

---

## Troubleshooting

**`Cannot assign requested address (addr='tcp://192.168.123.164:5558')`**
The PC has no interface with that IP. Override with `--zmq-bind tcp://0.0.0.0:5558`.

**`[Neck] --pose-zmq is required`**
The neck driver no longer has a local-SDK fallback. You must run
`pose_publisher.py` somewhere reachable and pass
`--pose-zmq tcp://<publisher-host>:5559` to the server.

**`[Neck] Import failed: No module named 'dynamixel_sdk'`**
`pip install dynamixel-sdk` in the G1's env.

**Publisher: `smplx_data is None — body trackers not streaming yet`**
The XRoboToolKit daemon is up but no SMPL-X stream is arriving. Check:
1. `pgrep -fa RoboticsServiceProcess` prints a PID.
2. XRoboToolkit Client app on the Pico is running, connected to the desktop's
   IP, with body tracking enabled.
3. Body trackers are paired and powered.
4. `sudo ufw status` — disable if active.

**Publisher: `ImportError: No module named 'general_motion_retargeting'`**
The `gmr` env isn't active. `conda activate gmr`, or follow
[TOPOLOGY2_QUICKSTART.md](TOPOLOGY2_QUICKSTART.md#desktop) step 4 to install.

**G1: `[Neck] no neck data yet`**
The G1 connected to the publisher but no valid samples are arriving. Check
the publisher terminal — if its counter isn't increasing, fix the desktop
side first. Otherwise verify the G1 can reach the desktop:
`ping <desktop-ip>` from the G1.

**Motor moves the wrong direction**
Flip `NECK_YAW_SIGN` or `NECK_PITCH_SIGN` between `+1` and `-1` in
[realsense_server.py](realsense_server.py). Restart.

**Total neck travel is too small / too large**
Scale `--neck-retarget-scale` on the publisher (no G1 restart needed).

**Motor stops responding mid-session**
Dynamixel overload trip. Code auto-reboots every 2 s on detected `hw_error`;
watch for `[Neck] ID <n> hw_error=... -> rebooting`. If recurring, lower
`NECK_SMOOTH_ALPHA` (more damping) or tighten limits.

**`/dev/ttyUSB0` fails to open**
Another process has it. `sudo fuser -v /dev/ttyUSB0` lists holders. Kill any
stale `realsense_server.py` or legacy `neck_redis_driver.py`.

**`zmq.error.ZMQError: Address already in use` on the desktop**
An old `pose_publisher.py` is still running: `pkill -f pose_publisher.py`.

**Neck moves when leaning torso (the bug this pipeline was built to fix)**
That's a regression from a misconfigured publisher. Double-check that
`pose_publisher.py` is using `XRobotStreamer` + `human_head_to_robot_neck`,
not the raw `xrobotoolkit_sdk.get_headset_pose()` — the latter is what the
old pipeline did and it's exactly what conflates body lean with head pitch.
