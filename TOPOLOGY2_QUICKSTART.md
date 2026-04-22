# Topology 2 Quickstart — Desktop + G1 Orin

Focused setup for the specific case:

- **Desktop PC** (x86_64) — runs everything XRoboToolKit-related:
  the PC Service daemon, the Python binding, and `pose_publisher.py`.
- **G1 onboard PC** (aarch64 / JetPack 6 / CUDA 12.6) — minimal deps only:
  `realsense_server.py` subscribes to head pose over ZMQ. No XRoboToolKit
  install of any kind on the robot.
- **Pico VR headset** points its XRoboToolkit Client app at the **desktop's**
  LAN IP.

For the full two-topology reference, see
[NECK_TELEOP_README.md](NECK_TELEOP_README.md). This doc only covers
Topology 2.

```
  Pico VR headset
        │
        │  ─── tracking pose (XRoboToolKit) ────┐
        │                                       │
        │  ◄── H.264 stereo (TCP :12345) ─────────────────┐
        │                                       │         │
                                                ▼         │
┌──────────────────────────────┐        ┌───────────────────────────┐
│      Desktop PC (x86_64)     │        │   G1 Orin (aarch64)       │
│                              │        │                           │
│  XRoboToolKit service (.deb) │        │  realsense_server.py      │
│  pose_publisher.py           │── PUB ─┤    --pose-zmq tcp://…:5559│
│                              │ :5559  │                           │
│  test_viewer.py  (optional)  │◄─ REP ─┤  ├─ ZED Mini              │
│                              │ :5558  │  └─ U2D2 + 2 Dynamixels   │
└──────────────────────────────┘        └───────────────────────────┘
```

---

## One-time setup

### Desktop

#### 1. Python env
```bash
conda activate sonic          # or create a new Python 3.10 env
pip install pyzmq
```

#### 2. XRoboToolKit Python binding (pre-built)
```bash
cd /path/to/GR00T-WholeBodyControl
export GR00T_ROOT="$PWD"
export PYTHONPATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$PYTHONPATH"
export LD_LIBRARY_PATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/lib:$LD_LIBRARY_PATH"
```
Add the three exports to `~/.bashrc` so they persist.

Verify:
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

### Pico headset

- Install the **XRoboToolkit Client** app on the headset.
- Set its Server IP to the **desktop's** LAN IP.
- Start head tracking in the app.

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
conda activate sonic
cd $GR00T_ROOT
python pose_publisher.py --bind tcp://0.0.0.0:5559 --hz 50
```
Look for:
- `XRoboToolkit SDK initialized.`
- `PUB bound to tcp://0.0.0.0:5559`
- Periodic `published N poses @ ~50 Hz` (as you move your head).

If it logs `quat is zero — Pico not streaming`, the daemon ↔ Pico link is
broken — fix the Pico app or daemon before continuing.

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
- `[Neck] ZMQ SUB pose source: tcp://<DESKTOP_IP>:5559`
- `[Neck] Started: /dev/ttyUSB0@2000000 IDs 0/1 ...`
- Periodic `[Neck] yaw ... pitch ...` that tracks your head.

Ctrl-C to stop — motors zero, torque releases, USB closes.

### Desktop terminal 3 — optional live ZED viewer
```bash
python test_viewer.py --server <G1_IP> --port 5558 --show-stereo
```
`q` or `ESC` to quit.

---

## Motor calibration

All knobs live at the top of
[realsense_server.py](realsense_server.py) on the G1:

```python
NECK_YAW_ZERO_TICK    = 1897
NECK_PITCH_ZERO_TICK  = 2900
NECK_YAW_LIMIT_DEG    = 60.0
NECK_PITCH_LIMIT_DEG  = 45.0
NECK_SMOOTH_ALPHA     = 0.3     # lower = more damping, smoother
NECK_EULER_ORDER      = "yxz"
NECK_YAW_SIGN         = -1      # flip if yaw goes the wrong way
NECK_PITCH_SIGN       = +1      # flip if pitch goes the wrong way
```
Edit in place, restart the server — no rebuild needed.

---

## Troubleshooting

**`No module named 'xrobotoolkit_sdk'` on the desktop**
The PYTHONPATH/LD_LIBRARY_PATH exports in Desktop step 2 aren't set in the
current shell. Re-source `~/.bashrc` or re-run the exports.

**`pose_publisher.py` logs `quat is zero — Pico not streaming`**
- `pgrep -fa RoboticsServiceProcess` should print a PID; if empty, restart
  with `sudo bash /opt/apps/roboticsservice/runService.sh`.
- Confirm the Pico app is launched, connected to the desktop's IP, with
  tracking enabled.
- `sudo ufw status` on the desktop — disable if active.

**G1 log: `[Neck] no headset data yet`**
The G1 reached the publisher but no valid poses arrived. Check the publisher
terminal — if its counter isn't increasing, fix the desktop side first.
Otherwise check that the G1 can reach the desktop:
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

**Yaw and pitch are swapped**
Change `NECK_EULER_ORDER` to `"zyx"` or `"zxy"`. Recheck signs.

**Dynamixel overload trip mid-session** (motor stops accepting commands)
The code auto-reboots on detected `hw_error` every 2 s — look for
`[Neck] ID <n> hw_error=... -> rebooting`. If recurring, lower
`NECK_SMOOTH_ALPHA` (more damping) or tighten limits.

**Desktop crash during teleop**
The G1 `NeckMotor._loop` holds the last commanded position when the ZMQ
stream goes silent — the motor freezes at the last head pose instead of
snapping back to zero. Ctrl-C the G1 server to zero-return cleanly.
