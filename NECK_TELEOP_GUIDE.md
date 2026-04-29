# Pico VR + Trackers → ZED + Neck Motor Teleop

The neck driver is fed from **SMPL-X-derived neck angles**, not the raw
headset pose. The operator wears the Pico headset plus body trackers; the
desktop retargets the body data to a 2-vector `[neck_yaw, neck_pitch]` and
publishes it to the G1 over ZMQ. Going through SMPL-X decouples neck
rotation from torso lean — when the operator walks or leans without
rotating their head, the neck angles stay near zero and the motors hold.

## Topology

`realsense_server.py` runs on the **G1 Orin** (Ethernet, fixed IP
`192.168.123.164`). Everything else — XRoboToolKit daemon, Python binding,
SMPL-X publisher, data collector — runs on the **desktop**.

```
  Pico VR headset + body trackers
        │
        │  ─── (tracking + body pose — XRoboToolKit protocol) ───┐
        │                                                        │
        │  ◄──────── H.264 stereo (TCP :12345) ──────────────────────────┐
        │                                                        │       │
                                                                 ▼       │
┌──────────────────────────────────────┐               ┌───────────────────────────┐
│      Desktop PC (x86_64)             │               │   G1 Orin  192.168.123.164│
│                                      │               │                           │
│  XRoboToolKit service (.deb)         │               │  realsense_server.py      │
│  pose_publisher.py     ── PUB :5559 ─┤──────────────►│  --pose-zmq tcp://desktop │
│  (gmr shim: XRobotStreamer +         │  JSON         │                  :5559    │
│   human_head_to_robot_neck)          │  [yaw,pitch]  │  ├─ ZED Mini              │
│                                      │               │  └─ U2D2 + 2 Dynamixels   │
│  g1_data_server.py     ◄── REP :5558 ┤◄──────────────┤    (yaw=ID0, pitch=ID1)   │
│  (records data.json)   ◄── PUB :5560 ┤◄──────────────┤    (state PUB :5560)      │
└──────────────────────────────────────┘               └───────────────────────────┘
```

Responsibilities:

- **Desktop PC** runs the XRoboToolKit PC Service daemon, the
  `xrobotoolkit_sdk` binding, the in-tree `general_motion_retargeting`
  shim, [pose_publisher.py](pose_publisher.py), and
  [g1_data_server.py](g1_data_server.py).
- **G1 Orin** owns the ZED Mini, the U2D2 (`/dev/ttyUSB0`), and the two
  Dynamixels. It only runs [realsense_server.py](realsense_server.py) — no
  XRoboToolKit install, no gmr install, no daemon.
- **Pico** runs the XRoboToolkit Client app and points its Server IP at
  the **desktop's** LAN IP. Body trackers (Pico Swift / equivalent) pair
  with the same daemon so SMPL-X body data is available.

---

## Hardware checklist

Connected to the **G1 Orin**:
- ZED Mini camera on USB3.
- U2D2 on `/dev/ttyUSB0`, two Dynamixels daisy-chained:
  - Yaw motor — ID `0`, zero tick `1897`, limit ±60°.
  - Pitch motor — ID `1`, zero tick `2900`, limit ±45°.
- (Optional) 2x RealSense D405 wrist cameras — disabled via `--zed-only`.

Worn by the operator:
- Pico VR headset.
- Body trackers paired with the XRoboToolKit Client app.

Network: G1 reachable from the desktop at `192.168.123.164` over
Ethernet. Pico on the same LAN as the desktop, able to reach it.

---

## One-time setup

### Desktop

#### 1. Python env
```bash
micromamba create -n sonic python=3.10 -y
micromamba activate sonic
pip install pyzmq numpy scipy opencv-python
```

#### 2. `general_motion_retargeting` shim (in-tree)

The pipeline uses two helpers — `XRobotStreamer` and
`human_head_to_robot_neck` — that live in this repo at
[external_dependencies/gmr_shim/general_motion_retargeting/](external_dependencies/gmr_shim/general_motion_retargeting/).
There is no upstream package to install; just point Python at the shim:

```bash
cd /path/to/GR00T-WholeBodyControl
export GR00T_ROOT="$PWD"
export PYTHONPATH="$GR00T_ROOT/external_dependencies/gmr_shim:$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$PYTHONPATH"
export LD_LIBRARY_PATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$LD_LIBRARY_PATH"
```

Add the three exports to `~/.bashrc` so they persist across sessions.

#### 3. XRoboToolKit PC Service daemon
```bash
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
sudo ufw disable
```

#### 4. Verify the imports
```bash
python -c "import xrobotoolkit_sdk as xrt; xrt.init(); print('xrt ok')"
python -c "from general_motion_retargeting import XRobotStreamer, human_head_to_robot_neck; print('gmr ok')"
```
Both must print `ok`.

### G1 Orin

#### 1. Python env
```bash
micromamba create -n sonic python=3.10 -y
micromamba activate sonic
pip install numpy opencv-python pyzmq dynamixel-sdk scipy
```
No XRoboToolKit binding, no daemon, no gmr.

#### 2. ZED SDK (JetPack 6.x / CUDA 12.6)
The installer is bundled in the repo for the G1's L4T 36.5 image:
[ZED_SDK_Tegra_L4T36.5_v5.2.3.zstd.run](ZED_SDK_Tegra_L4T36.5_v5.2.3.zstd.run).

```bash
cd $GR00T_ROOT
chmod +x ZED_SDK_Tegra_L4T36.5_v5.2.3.zstd.run
./ZED_SDK_Tegra_L4T36.5_v5.2.3.zstd.run        # follow the interactive prompts
cd /usr/local/zed
python get_python_api.py
```

If your JetPack / CUDA version differs from L4T 36.5, fetch the matching
build instead from [stereolabs.com/developers](https://www.stereolabs.com/developers/release).

#### 3. Host / permissions
```bash
sudo ufw disable
sudo usermod -aG dialout $USER     # for /dev/ttyUSB0; log out/in to apply
```

#### 4. Sanity-check the motor port
```bash
python -c "
from dynamixel_sdk import PortHandler, PacketHandler
p = PortHandler('/dev/ttyUSB0'); p.openPort(); p.setBaudRate(2_000_000)
pk = PacketHandler(2.0)
for i in (0, 1):
    _, res, _ = pk.ping(p, i)
    print(f'ID {i}: {\"ok\" if res == 0 else \"FAIL\"}')
p.closePort()
"
```
Both IDs should print `ok`.

### Pico headset

- Install the **XRoboToolkit Client** app.
- Set Server IP to the **desktop's** LAN IP.
- Enable head tracking and pair the body trackers.

SMPL-X retargeting requires the body trackers — without them
`XRobotStreamer` returns `smplx_data = None` and the publisher logs a
warning instead of publishing samples.

---

## Motor calibration

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
NECK_SMOOTH_ALPHA     = 0.3        # lower = more damping
NECK_CONTROL_HZ       = 50
NECK_YAW_SIGN         = +1         # flip to -1 if motor moves opposite to head
NECK_PITCH_SIGN       = +1         # flip to -1 if motor moves opposite to head
```

Restart the server after any edit — no rebuild needed.

The publisher exposes `--neck-retarget-scale` (default `1.5`) — bump up
or down on the desktop to scale total neck travel without recompiling.

---

## Runtime playbook

Three terminals on the desktop, one ssh'd into the G1.

### Desktop terminal 1 — XRoboToolKit daemon (leave running)
```bash
sudo bash /opt/apps/roboticsservice/runService.sh
```
Verify in any other terminal: `pgrep -fa RoboticsServiceProcess` prints a
PID.

### Pico headset
Put on the headset, launch XRoboToolkit Client, confirm "Connected" to
the desktop's IP, start tracking. Make sure body trackers are paired and
streaming.

### Desktop terminal 2 — pose publisher (leave running)
```bash
micromamba activate sonic
cd $GR00T_ROOT
python pose_publisher.py \
    --bind tcp://0.0.0.0:5559 \
    --hz 50 \
    --neck-retarget-scale 1.5
```
Expected logs:
- `XRobotStreamer initialized.`
- `PUB bound to tcp://0.0.0.0:5559`
- Periodic `published N neck samples @ ~50 Hz` (as you move your head).

If it warns `smplx_data is None`, the body trackers aren't streaming yet
(see [Troubleshooting](#troubleshooting)).

### G1 terminal — camera + neck server

Open the motor port for this session (the `dialout` group fix from setup
sometimes doesn't survive an SSH login on JetPack — easiest workaround):
```bash
sudo chmod 777 /dev/ttyUSB0
```

Then launch:
```bash
micromamba activate sonic
cd ~/Desktop/GR00T-WholeBodyControl     # adjust path on the robot

python realsense_server.py \
    --zed-only \
    --zmq-bind tcp://0.0.0.0:5558 \
    --enable-pico --pico-ip <PICO_IP> \
    --enable-neck-motor \
    --pose-zmq tcp://<DESKTOP_IP>:5559
```

Expected lines, in order:
- `[ZED] Started: resolution=vga fps=30`
- `[PicoStreamer] Connected to Pico <PICO_IP>:12345`
- `[Neck] ZMQ SUB neck-angle source: tcp://<DESKTOP_IP>:5559`
- `[Neck] State PUB bound: tcp://*:5560`
- `[Neck] Started: /dev/ttyUSB0@2000000 IDs 0/1 ...`
- Periodic `[Neck] yaw ... pitch ...` with the **tick value changing**
  (if the tick stays at the zero tick, motor writes are off — see
  troubleshooting).

The motor present-position is published on `:5560` so `g1_data_server.py`
can record it as `states['neck']` (see
[Recording episodes](#recording-episodes-g1_data_serverpy) below). Pass
`--neck-state-pub ""` to disable.

**Lean test** (the actual reason this pipeline exists): hold your head
still relative to your torso, then lean forward / back. The
`[Neck] yaw / pitch` values should stay near zero — body lean is
absorbed by the spine joints in SMPL-X, leaving the neck angles
invariant.

Ctrl-C on the G1 zeroes the motors, releases torque, and closes the USB.

### Desktop terminal 3 — optional live ZED viewer
```bash
python test_viewer.py --server 192.168.123.164 --port 5558 --show-stereo
```
`q` or `ESC` to quit.

---

## CLI flags reference

`realsense_server.py`:

| Flag | Default | Purpose |
|------|---------|---------|
| `--zed-only` | off (env `ZED_ONLY`) | Disable both RealSense wrist cameras. |
| `--zmq-bind <addr>` | `tcp://192.168.123.164:5558` | ZMQ REP bind. Use `tcp://0.0.0.0:5558` to bind all interfaces. |
| `--enable-pico` | off (env `ENABLE_PICO`) | Stream ZED stereo as H.264 to the Pico. |
| `--pico-ip <ip>` | `192.168.0.128` | Pico headset IP. |
| `--pico-port <port>` | `12345` | Pico TCP video port. |
| `--enable-neck-motor` | off (env `NECK_MOTOR`) | Drive the 2-DOF neck. Requires `--pose-zmq`. |
| `--pose-zmq <addr>` | "" (env `POSE_ZMQ`) | **Required with `--enable-neck-motor`.** ZMQ SUB address of `pose_publisher.py`. Wire format: JSON `[neck_yaw, neck_pitch]` (radians). Example: `tcp://<desktop-ip>:5559`. |
| `--neck-state-pub <addr>` | `tcp://*:5560` (env `NECK_STATE_PUB`) | ZMQ PUB bind for the neck motor present-position stream `[yaw_rad, pitch_rad]`. Read from the Dynamixels every control tick. Empty string disables. |
| `--resolution <preset>` | `vga` | `vga`, `hd720`, `hd1080`, `hd1200`, `hd2k`, `auto`. |
| `--fps <n>` | `30` | ZED FPS. |
| `--list-devices` | — | Print attached RealSense devices and exit. |

`pose_publisher.py`:

| Flag | Default | Purpose |
|------|---------|---------|
| `--bind <addr>` | `tcp://0.0.0.0:5559` | ZMQ PUB bind. |
| `--hz <n>` | `50` | Publish rate (matches `NECK_CONTROL_HZ`). |
| `--neck-retarget-scale <f>` | `1.5` | Multiplier on `(yaw, pitch)` before publishing. |

---

## Recording episodes (`g1_data_server.py`)

To record both the neck **command** (publisher → motors) and the neck
**state** (motor present-position) into `data.json`, run the data
collector with both ZMQ subscriptions:

```bash
python g1_data_server.py \
    --neck-zmq       tcp://localhost:5559 \              # action label  (publisher on this machine)
    --neck-state-zmq tcp://192.168.123.164:5560          # proprioception (G1 Ethernet IP)
```

For each frame `data.json` then contains:

- `actions['neck']` = `[yaw_cmd, pitch_cmd]` (radians, post-smoothing)
- `states['neck']`  = `[yaw_meas, pitch_meas]` (radians, motor present-position)

Both keys are 2-vectors. Either flag is independent — pass only
`--neck-zmq` to record commands only, or only `--neck-state-zmq` to
record state only.

The signs/zero-ticks/`NECK_SMOOTH_ALPHA` baked into `realsense_server.py`
at record time define the convention; if you flip them later, prior
recordings need offline correction.

`WBC_HOST` is hardcoded to `localhost` in `g1_data_server.py` — with the
recorder on the desktop and the WBC on the robot you must edit it to
`192.168.123.164` (it is not env-overridable yet).

Output folder structure:
```
~/data/demonstration_<session-id>/episode_<N>/
```
where `<session-id>` is the timestamp set on the first `s` press, reused
for subsequent episodes. Each `episode_<N>` directory contains
`data.json` plus per-stream image folders.

Keyboard controls in the `g1_data_server.py` terminal:
- `s` — start a new episode
- `q` — stop and save
- `d` — stop and discard
- `Ctrl-C` — quit

---

## Troubleshooting

**`Cannot assign requested address (addr='tcp://192.168.123.164:5558')`**
The G1 has no interface with that IP. Override with
`--zmq-bind tcp://0.0.0.0:5558`.

**`[Neck] --pose-zmq is required`**
You launched `realsense_server.py --enable-neck-motor` without
`--pose-zmq`. The local-SDK fallback was removed; the publisher is the
only supported source.

**`[Neck] Import failed: No module named 'dynamixel_sdk'`**
`pip install dynamixel-sdk` in the G1's env.

**`[Neck] Could not open /dev/ttyUSB0`**
- USB cable plugged in? `ls /dev/ttyUSB*`.
- Permissions: `sudo chmod 777 /dev/ttyUSB0` (the `dialout` group fix
  from setup sometimes doesn't survive an SSH login).
- User in `dialout`? `groups $USER` — if missing, `sudo usermod -aG
  dialout $USER` and log out/in (long-term fix).
- Another process has it: `sudo fuser -v /dev/ttyUSB0`.

**Publisher: `smplx_data is None — body trackers not streaming yet`**
The XRoboToolKit daemon is up but no SMPL-X stream is arriving. Check:
1. `pgrep -fa RoboticsServiceProcess` prints a PID.
2. XRoboToolkit Client app on the Pico is running, connected to the
   desktop's IP, with body tracking enabled.
3. Body trackers are paired and powered.
4. `sudo ufw status` — disable if active.

**Publisher: `ImportError: No module named 'general_motion_retargeting'`**
The PYTHONPATH export pointing at `external_dependencies/gmr_shim`
isn't set in this shell. Re-source `~/.bashrc` or re-run the exports
from setup step 2.

**G1: `[Neck] no neck data yet`**
The G1 connected to the publisher but no valid samples are arriving.
Check the publisher terminal — if its counter isn't increasing, fix the
desktop side first. Otherwise verify the G1 can reach the desktop:
`ping <desktop-ip>` from the G1.

**Motor moves the wrong direction**
Flip `NECK_YAW_SIGN` or `NECK_PITCH_SIGN` between `+1` and `-1` in
[realsense_server.py](realsense_server.py). Restart.

**Total neck travel is too small / too large**
Scale `--neck-retarget-scale` on the publisher (no G1 restart needed).

**Motor stops responding mid-session (Dynamixel overload trip)**
Code auto-reboots every 2 s on detected `hw_error`; watch for
`[Neck] ID <n> hw_error=... -> rebooting`. If recurring, lower
`NECK_SMOOTH_ALPHA` (more damping) or tighten limits.

**`/dev/ttyUSB0` fails to open**
Another process has it. `sudo fuser -v /dev/ttyUSB0` lists holders.
Kill any stale `realsense_server.py`.

**`zmq.error.ZMQError: Address already in use` on the desktop**
An old `pose_publisher.py` is still running:
`pkill -f pose_publisher.py`.

**`[Neck] State PUB bind failed: Address already in use`**
A previous `realsense_server.py` is still bound to `:5560`. Either kill
it (`pkill -f realsense_server.py`), use a different port via
`--neck-state-pub tcp://*:5561`, or pass `--neck-state-pub ""` to
disable state publishing this run.

**`states['neck']` is missing from `data.json`**
You didn't pass `--neck-state-zmq` to `g1_data_server.py`, or the
address is wrong. The state publisher is `realsense_server.py` on the
robot, so from the desktop use `tcp://192.168.123.164:5560`.

**`[Neck] yaw / pitch` values change but the `tick` stays at the zero tick**
Motor writes are disabled in the source. The current branch keeps them
enabled; if you've patched them off (e.g. for a passive bring-up), the
ticks won't follow `yaw_cmd` / `pitch_cmd`. Re-enable the
`write4ByteTxRx` calls in the neck loop.

**Desktop crash during teleop**
The G1 `NeckMotor._loop` holds the last commanded position when the
ZMQ stream goes silent — the motor freezes at the last neck angle
instead of snapping back to zero. Ctrl-C the G1 server to zero-return
cleanly.

**Neck moves when leaning torso (the bug this pipeline was built to fix)**
That's a regression from a misconfigured publisher. Confirm
`pose_publisher.py` logs `XRobotStreamer initialized.` (not
`XRoboToolkit SDK initialized.`) and that it's importing
`XRobotStreamer` + `human_head_to_robot_neck` from
`general_motion_retargeting`, not falling back to the raw
`xrt.get_headset_pose()`.
