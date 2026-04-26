# Topology 2 Quickstart — Desktop + G1 Orin

Focused setup for the specific case:

- **Desktop PC** (x86_64) — runs everything XRoboToolKit-related:
  the PC Service daemon, the Python binding, the `gmr` env with
  `general_motion_retargeting`, and `pose_publisher.py`. The publisher pulls
  SMPL-X body data from `XRobotStreamer`, calls `human_head_to_robot_neck`
  to extract neck yaw/pitch, and publishes `[neck_yaw, neck_pitch]`.
- **G1 onboard PC** (aarch64 / JetPack 6 / CUDA 12.6) — minimal deps only:
  `realsense_server.py` subscribes to the published angles over ZMQ. No
  XRoboToolKit install of any kind on the robot, no `gmr`.
- **Pico VR headset + body trackers** — point the XRoboToolkit Client app at
  the **desktop's** LAN IP. SMPL-X requires the body trackers; without them
  the publisher will warn `smplx_data is None`.

```
  Pico VR headset + body trackers
        │
        │  ─── tracking + body pose (XRoboToolKit) ────┐
        │                                              │
        │  ◄── H.264 stereo (TCP :12345) ──────────────────────┐
        │                                              │       │
                                                       ▼       │
┌──────────────────────────────────────┐      ┌───────────────────────────┐
│      Desktop PC (x86_64)             │      │   G1 Orin (aarch64)       │
│                              ── PUB ─┤      │  realsense_server.py      │
│  XRoboToolKit service (.deb)│ :5559  │─────►│    --pose-zmq tcp://…:5559│
│  gmr env + XRobotStreamer    │ JSON  │      │                           │
│  pose_publisher.py        [yaw,pitch]│      │  ├─ ZED Mini              │
│                                      │      │  └─ U2D2 + 2 Dynamixels   │
│  test_viewer.py  (optional)  ◄─ REP ─┤◄─────┤                           │
│                                 :5558│      │                           │
└──────────────────────────────────────┘      └───────────────────────────┘
```

---

## One-time setup

### Desktop

#### 1. `gmr` Python env (Python 3.10 recommended)

`general_motion_retargeting` ships its own conda env. Clone the package's
repo (the same one used by the past project — the `conda activate gmr` line
at [past_streaming_pipeline/pico_contol.py:2](past_streaming_pipeline/pico_contol.py#L2)
implies its repo URL is the one already on file for this team) and install
in develop mode:

```bash
git clone <gmr-repo-url> ~/general_motion_retargeting
cd ~/general_motion_retargeting
conda env create -f environment.yml      # creates `gmr` env
conda activate gmr
pip install -e .
pip install pyzmq                         # used by pose_publisher.py
```

Verify:
```bash
python -c "from general_motion_retargeting import XRobotStreamer, human_head_to_robot_neck; print('ok')"
```

If you don't yet have the `gmr` repo URL, ask whoever set up the past
pipeline — `gear_sonic/scripts/pico_gmr_thread_server.py:46` in this repo is
another in-tree consumer of the same package and confirms the import path.

#### 2. XRoboToolKit Python binding (pre-built)
```bash
cd /path/to/GR00T-WholeBodyControl
export GR00T_ROOT="$PWD"
export PYTHONPATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$PYTHONPATH"
export LD_LIBRARY_PATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/lib:$LD_LIBRARY_PATH"
```
Add the three exports to `~/.bashrc` so they persist. Note: these still
matter because `XRobotStreamer` reads from the same daemon as
`xrobotoolkit_sdk`.

Verify (inside the `gmr` env):
```bash
python -c "import xrobotoolkit_sdk as xrt; xrt.init(); print('ok')"
```

#### 3. PC Service daemon
```bash
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
sudo ufw disable
```

### G1 Orin

#### 1. Python 3.10 env
```bash
micromamba create -n sonic python=3.10 -y
micromamba activate sonic
```

#### 2. Python deps (minimal)
```bash
pip install numpy opencv-python pyzmq dynamixel-sdk scipy
```

That's it — no XRoboToolKit binding, no daemon, no pybind CMake build.

#### 3. ZED SDK (JetPack 6.x / CUDA 12.6)
Download the Jetson version matching your JetPack from
[stereolabs.com/developers](https://www.stereolabs.com/developers/release).
After installation:
```bash
cd /usr/local/zed
python get_python_api.py
```

#### 4. Host / permissions
```bash
sudo ufw disable
sudo usermod -aG dialout $USER      # for /dev/ttyUSB0; log out/in to apply
```

#### 5. Sanity-check the motor port (wearing no one's head yet)
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

### Pico headset + body trackers

- Install the **XRoboToolkit Client** app on the headset.
- Set its Server IP to the **desktop's** LAN IP.
- Pair the body trackers (Pico Swift / equivalent) and start tracking.

SMPL-X requires the body trackers — without them, `XRobotStreamer` returns
`smplx_data = None` and the publisher logs a warning instead of publishing
samples.

---

## Runtime playbook

Every teleop session follows this order. Three terminals on the desktop, one
on the G1.

### Desktop terminal 1 — service daemon (leave running)
```bash
sudo bash /opt/apps/roboticsservice/runService.sh
```
Verify in any terminal: `pgrep -fa RoboticsServiceProcess` prints a PID.

### Pico headset
Put on the headset, launch XRoboToolkit Client, confirm "Connected" to the
desktop's IP, start tracking.

### Desktop terminal 2 — pose publisher (leave running)
```bash
conda activate gmr
cd $GR00T_ROOT
python pose_publisher.py \
    --bind tcp://0.0.0.0:5559 \
    --hz 50 \
    --neck-retarget-scale 1.5
```
Look for:
- `XRobotStreamer initialized.`
- `PUB bound to tcp://0.0.0.0:5559`
- Periodic `published N neck samples @ ~50 Hz` (as you move your head).

If it logs `smplx_data is None — body trackers not streaming yet`, the
daemon is up but the body trackers aren't producing data — confirm the Pico
app shows tracking active and the trackers are paired/powered, then re-check.

### G1 terminal — launch the camera + motor server
```bash
micromamba activate sonic
cd ~/Desktop/GR00T-WholeBodyControl    # adjust path on the robot

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
- Periodic `[Neck] yaw ... pitch ...` that tracks your head.

Lean test (the actual reason this pipeline exists): hold your head still
relative to your torso, then lean forward / back. The `[Neck] yaw / pitch`
values should stay near zero — body lean is absorbed by the spine joints in
SMPL-X, leaving the neck angles invariant.

Ctrl-C to stop — motors zero, torque releases, USB closes.

### Desktop terminal 3 — optional live ZED viewer
```bash
python test_viewer.py --server <G1_IP> --port 5558 --show-stereo
```
`q` or `ESC` to quit.

---

## Motor calibration

G1-side knobs at the top of [realsense_server.py](realsense_server.py):

```python
NECK_YAW_ZERO_TICK    = 1897
NECK_PITCH_ZERO_TICK  = 2900
NECK_YAW_LIMIT_DEG    = 60.0
NECK_PITCH_LIMIT_DEG  = 45.0
NECK_SMOOTH_ALPHA     = 0.3     # lower = more damping, smoother
NECK_YAW_SIGN         = +1      # flip to -1 if yaw goes the wrong way
NECK_PITCH_SIGN       = +1      # flip to -1 if pitch goes the wrong way
```
Edit in place, restart the server — no rebuild needed.

Desktop-side knob: `--neck-retarget-scale` on `pose_publisher.py` (default
`1.5`) scales total neck travel without rebuilding anything.

---

## Troubleshooting

**`No module named 'general_motion_retargeting'` on the desktop**
The `gmr` env isn't active. `conda activate gmr`, or re-run Desktop step 1
to install it.

**`No module named 'xrobotoolkit_sdk'` on the desktop**
The PYTHONPATH/LD_LIBRARY_PATH exports in Desktop step 2 aren't set in the
current shell. Re-source `~/.bashrc` or re-run the exports.

**`pose_publisher.py` logs `smplx_data is None — body trackers not streaming yet`**
- `pgrep -fa RoboticsServiceProcess` should print a PID; if empty, restart
  with `sudo bash /opt/apps/roboticsservice/runService.sh`.
- Confirm the Pico app is launched, connected to the desktop's IP, with
  tracking enabled.
- Confirm the body trackers are paired and powered.
- `sudo ufw status` on the desktop — disable if active.

**G1 log: `[Neck] --pose-zmq is required`**
You launched `realsense_server.py --enable-neck-motor` without
`--pose-zmq tcp://<desktop-ip>:5559`. The local-SDK fallback no longer
exists; the publisher is the only supported source.

**G1 log: `[Neck] no neck data yet`**
The G1 reached the publisher but no valid samples arrived. Check the
publisher terminal — if its counter isn't increasing, fix the desktop side
first. Otherwise check that the G1 can reach the desktop:
`ping <desktop-ip>` from the G1.

**G1 log: `Cannot assign requested address (addr='tcp://192.168.123.164:5558')`**
This is the default bind IP hardcoded for the old robot network. Override:
`--zmq-bind tcp://0.0.0.0:5558`.

**G1 log: `[Neck] Could not open /dev/ttyUSB0`**
- Motor USB cable plugged in? `ls /dev/ttyUSB*`.
- User in `dialout`? `groups $USER` — if missing, `sudo usermod -aG dialout
  $USER` and log out/in.
- Another process has it: `sudo fuser -v /dev/ttyUSB0`.

**Motor moves the wrong way**
Flip `NECK_YAW_SIGN` or `NECK_PITCH_SIGN` between `+1` and `-1`. Restart.

**Total neck travel too small / too large**
Adjust `--neck-retarget-scale` on the publisher (no G1 restart needed).

**Dynamixel overload trip mid-session** (motor stops accepting commands)
The code auto-reboots on detected `hw_error` every 2 s — look for
`[Neck] ID <n> hw_error=... -> rebooting`. If recurring, lower
`NECK_SMOOTH_ALPHA` (more damping) or tighten limits.

**Desktop crash during teleop**
The G1 `NeckMotor._loop` holds the last commanded position when the ZMQ
stream goes silent — the motor freezes at the last neck angle instead of
snapping back to zero. Ctrl-C the G1 server to zero-return cleanly.

**Neck still moves when leaning torso (the bug we're trying to fix)**
That should not happen with this pipeline — `human_head_to_robot_neck`
extracts neck rotation relative to the spine. If you see it, the publisher
is probably running the old version that read `xrt.get_headset_pose()`
directly. Pull the latest `pose_publisher.py` and confirm its log says
`XRobotStreamer initialized.`, not `XRoboToolkit SDK initialized.`.
