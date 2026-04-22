# Pico VR → ZED + Neck Motor Teleop

End-to-end VR teleop pipeline with two machines:

```
  Pico VR headset
        │
        │  ───────── (tracking pose — XRoboToolKit protocol) ────────┐
        │                                                            │
        │  ◄──────── H.264 stereo (TCP :12345) ────────┐             │
        │                                              │             │
                                                       │             ▼
┌──────────────────────────┐                 ┌───────────────────────────┐
│      Desktop PC          │                 │   G1 onboard PC           │
│                          │                 │                           │
│  test_viewer.py          │◄── ZMQ (:5558) ─┤  realsense_server.py      │
│  (or teleop client)      │   REQ/REP JPEG  │                           │
│                          │                 │  ├─ ZED Mini (USB3)       │
│                          │                 │  ├─ U2D2 + 2 Dynamixels   │
│                          │                 │  │   (yaw ID 0, pitch 1)  │
│                          │                 │  └─ XRoboToolKit service  │
└──────────────────────────┘                 └───────────────────────────┘
```

Responsibilities:
- **G1 onboard PC** runs [realsense_server.py](realsense_server.py). It owns the
  ZED, the U2D2 (`/dev/ttyUSB0`), and the XRoboToolKit service that receives
  head-pose from the Pico. It serves ZED frames over ZMQ, streams ZED stereo as
  H.264 to the Pico, and drives the 2-DOF neck directly from the Pico head
  pose.
- **Desktop PC** runs the consumer side: either [test_viewer.py](test_viewer.py)
  for a live debug view of the ZED feed, or any downstream teleop client
  that talks to the G1's ZMQ endpoint.
- **Pico VR** headset runs the XRoboToolKit Client app (streams tracking to G1)
  and receives the H.264 video for VR display.

---

## Hardware checklist

Connected to the **G1 onboard PC**:
- ZED Mini camera on USB3.
- U2D2 on `/dev/ttyUSB0`, two Dynamixels daisy-chained:
  - Yaw motor — ID `0`, zero tick `1897`, limit ±60°.
  - Pitch motor — ID `1`, zero tick `2900`, limit ±45°.
- (Optional) 2x RealSense D405 wrist cameras — disabled via `--zed-only`.

Network: Pico, G1, and Desktop all on the same LAN, able to ping each other.

---

## One-time setup

### A. G1 onboard PC setup

#### A.1 — Python env

Use the existing `sonic` env (Python 3.10). Add the two packages required by
the neck driver:

```bash
conda activate sonic
pip install dynamixel-sdk scipy
```

#### A.2 — XRoboToolKit Python SDK

A pre-built Python 3.10 `.so` is shipped for **x86_64** under
`external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/`. On
**aarch64** (Jetson Orin G1), the `.so` is built on demand by `install_pico.sh`.

Put the module on the import path (add to `~/.bashrc` on the G1):

```bash
export GR00T_ROOT="$HOME/Desktop/GR00T-WholeBodyControl"      # adjust path
export PYTHONPATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$PYTHONPATH"
export LD_LIBRARY_PATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/lib:$LD_LIBRARY_PATH"
```

Verify:
```bash
python -c "import xrobotoolkit_sdk as xrt; xrt.init(); print('ok')"
```

> `pip install -e external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/`
> triggers a CMake rebuild that fails without cmake/pybind11 in the env.
> The PYTHONPATH approach reuses the pre-built `.so` and avoids the build.

#### A.3 — XRoboToolKit PC Service daemon

Daemon that receives head-pose from the Pico app and publishes it to
`xrobotoolkit_sdk`.

```bash
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
```

#### A.4 — Firewall

Open the service ports, or simplest:
```bash
sudo ufw disable
```

#### A.5 — USB port permissions

Add your user to `dialout` so `/dev/ttyUSB0` opens without `sudo`:
```bash
sudo usermod -aG dialout $USER
# log out / log in for the group to apply
```

#### A.6 — Motor calibration (in-file constants)

All tuning knobs are module-level constants at the top of
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
NECK_EULER_ORDER      = "yxz"
NECK_YAW_SIGN         = -1    # flip if motor moves opposite to head
NECK_PITCH_SIGN       = +1    # flip if motor moves opposite to head
```

After any edit, restart the server — no rebuild needed.

### B. Desktop PC setup

The desktop only consumes the ZMQ feed — it does **not** need the SDK, the
daemon, or the motor libraries.

```bash
conda activate sonic
pip install pyzmq opencv-python numpy    # likely already present
```

If you're running downstream teleop clients (e.g. files under
`gear_sonic/scripts/`), follow their own install instructions — they are
unaffected by this pipeline.

### C. Pico headset setup

- Install the **XRoboToolkit Client** app on the headset. This is different
  from any VR video viewer app.
- In the app, set the Server IP / Host IP to the **G1 onboard PC's LAN IP**
  (`ip -4 addr show | grep inet` on the G1).
- Make sure head tracking is enabled in the app.

---

## Running the pipeline

### On the G1

Open three terminals on the G1 (or use `tmux`).

**G1 terminal 1 — XRoboToolKit service daemon** (leave running):
```bash
sudo ufw disable     # skip if already off
sudo bash /opt/apps/roboticsservice/runService.sh
```
> Invoke with `bash`, not `sh` — the script uses `${BASH_SOURCE[0]}` which
> `dash` rejects ("Bad substitution").

Verify in any terminal: `pgrep -fa RoboticsServiceProcess` should print a PID.

**On the Pico headset:** launch the XRoboToolkit Client, confirm it shows
"Connected" with the G1's IP, and start tracking.

**G1 terminal 2 — pose sanity check** (only needed the first time or when
debugging):
```bash
conda activate sonic
cd $GR00T_ROOT
python external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/examples/run_binding_continuous.py
```
Move your head. `Headset Pose` quaternion values must change. If they stay
`0.0, 0.0, 0.0, 0.0`, the Pico → daemon link is not alive — fix before
launching the server.

**G1 terminal 3 — launch the server:**
```bash
conda activate sonic
cd $GR00T_ROOT

python realsense_server.py \
    --zed-only \
    --zmq-bind tcp://0.0.0.0:5558 \
    --enable-pico --pico-ip <PICO_IP> \
    --enable-neck-motor
```

Expected log lines:
- `[ZED] Started: resolution=vga fps=30`
- `[PicoStreamer] Started ego stereo streaming to Pico`
- `[PicoStreamer] Connected to Pico <PICO_IP>:12345`
- `[Neck] XRoboToolkit SDK initialized.`
- `[Neck] Started: /dev/ttyUSB0@2000000 IDs 0/1 limits yaw±60° pitch±45° @ 50Hz`
- Periodic `[Neck] yaw ... pitch ...` with values that track your head.

**Ctrl-C** for clean shutdown: motors return to zero, torque released,
USB port closed.

### On the desktop PC

Point the viewer at the G1's LAN IP:

```bash
conda activate sonic
cd <repo path on desktop>

python test_viewer.py --server <G1_IP> --port 5558 --show-stereo
```

Press `q` or `ESC` to quit. `--show-stereo` opens a second window with the
ZED stereo pair; omit it for a lighter viewer.

### Pico headset

Once the server is running and `[PicoStreamer] Connected to Pico ...` appears,
the VR video viewer on the headset will show the ZED stereo feed. Moving your
head rotates the physical neck on the robot.

---

## Command-line flags reference

| Flag | Default | Purpose |
|------|---------|---------|
| `--zed-only` | off (env `ZED_ONLY`) | Disable both RealSense wrist cameras. |
| `--zmq-bind <addr>` | `tcp://192.168.123.164:5558` | ZMQ REP bind. Use `tcp://0.0.0.0:5558` to bind all interfaces. |
| `--enable-pico` | off (env `ENABLE_PICO`) | Stream ZED stereo as H.264 to the Pico. |
| `--pico-ip <ip>` | `192.168.0.128` | Pico headset IP. |
| `--pico-port <port>` | `12345` | Pico TCP video port. |
| `--enable-neck-motor` | off (env `NECK_MOTOR`) | Drive the 2-DOF neck from Pico head pose. |
| `--resolution <preset>` | `vga` | `vga`, `hd720`, `hd1080`, `hd1200`, `hd2k`, `auto`. |
| `--fps <n>` | `30` | ZED FPS. |
| `--list-devices` | — | Print attached RealSense devices and exit. |

Common combinations (on the G1):
```bash
# ZED view only (no motor, no Pico) — useful for testing viewer alone
python realsense_server.py --zed-only --zmq-bind tcp://0.0.0.0:5558

# Full VR teleop
python realsense_server.py --zed-only --zmq-bind tcp://0.0.0.0:5558 \
    --enable-pico --pico-ip <PICO_IP> --enable-neck-motor
```

---

## Troubleshooting

**`Cannot assign requested address (addr='tcp://192.168.123.164:5558')`**
The PC has no interface with that IP. Override with `--zmq-bind tcp://0.0.0.0:5558`.

**`[Neck] Import failed: No module named 'xrobotoolkit_sdk'`**
Run the PYTHONPATH/LD_LIBRARY_PATH exports from A.2 before launching the
server, or bake them into `~/.bashrc`.

**`[Neck] headset read error: Found zero norm quaternions` / `[Neck] no headset data yet`**
`xrt.get_headset_pose()` is returning zeros. SDK is up but no live pose is
arriving. Check, in order:
1. Service daemon running on G1: `pgrep -fa RoboticsServiceProcess`.
2. XRoboToolkit Client app on the Pico is launched, connected, pointed at the
   G1's IP.
3. Firewall: `sudo ufw status` — disable if active.
4. G1 and Pico on the same subnet.

The server holds last motor command until valid pose data returns.

**`/opt/apps/roboticsservice/runService.sh: 1: Bad substitution`**
`sudo bash /opt/apps/roboticsservice/runService.sh` (not `sh`).

**Viewer fps stuck around 24 Hz at HD720**
Combined cost of viewer `cv2.waitKey(1)` (~15 ms on Linux) + JPEG decode +
server-side JPEG encode exceeds a 33 ms frame budget. Options:
- `--resolution vga` on the server (cuts encode cost).
- Drop `--show-stereo` on the viewer.
The Pico H.264 stream is independent of the viewer loop and unaffected.

**Motor moves the wrong direction**
Flip `NECK_YAW_SIGN` or `NECK_PITCH_SIGN` in
[realsense_server.py](realsense_server.py) between `+1` and `-1`. Restart.

**Yaw and pitch are swapped**
Change `NECK_EULER_ORDER`. Common values: `"yxz"` (default), `"zyx"`,
`"zxy"`. Recheck signs afterward.

**Motor stops responding mid-session**
Dynamixel overload trip. Code auto-reboots every 2 s on detected `hw_error`;
watch for `[Neck] ID <n> hw_error=... -> rebooting`. If recurring, damp the
input: lower `NECK_SMOOTH_ALPHA` (more smoothing) or tighten limits.

**`/dev/ttyUSB0` fails to open**
Another process has it. `sudo fuser -v /dev/ttyUSB0` lists holders. Kill any
stale `realsense_server.py` or legacy `neck_redis_driver.py`.
