# Pico VR → ZED + Neck Motor Teleop

Two deployment topologies are supported. Pick the one that matches your
hardware — both use the same `realsense_server.py` binary, differentiated by
the `--pose-zmq` flag.

### Topology 1 — all-on-G1 (requires aarch64 XRoboToolKit daemon)

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

### Topology 2 — daemon on desktop, pose over ZMQ (recommended for Orin)

Use this when the G1 is aarch64 and you don't want to build the
XRoboToolKit daemon for Orin. The daemon + binding run on the desktop; a
small `pose_publisher.py` forwards headset pose to the G1 over ZMQ.

```
  Pico VR headset
        │
        │  ─── (tracking pose — XRoboToolKit protocol) ───┐
        │                                                 │
        │  ◄──────── H.264 stereo (TCP :12345) ─────────────────────┐
        │                                                 │         │
                                                          ▼         │
┌──────────────────────────────┐                 ┌───────────────────────────┐
│      Desktop PC              │                 │   G1 onboard PC           │
│                              │                 │                           │
│  XRoboToolKit service (.deb) │                 │  realsense_server.py      │
│  pose_publisher.py           │── ZMQ PUB ─────►│    --pose-zmq tcp://…:5559│
│                              │   (:5559 JSON)  │                           │
│  test_viewer.py              │◄── ZMQ (:5558) ─┤  ├─ ZED Mini              │
│                              │   REQ/REP JPEG  │  └─ U2D2 + 2 Dynamixels   │
└──────────────────────────────┘                 └───────────────────────────┘
```

Responsibilities (Topology 1):
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

Responsibilities (Topology 2 — recommended for Orin):
- **Desktop PC** additionally runs the **XRoboToolKit PC Service daemon**
  (amd64 `.deb`) and [pose_publisher.py](pose_publisher.py), which publishes
  the Pico headset pose over ZMQ.
- **G1 onboard PC** does **NOT** need the daemon or the Python binding. It
  runs `realsense_server.py --pose-zmq tcp://<desktop-ip>:5559`, reads the
  published pose instead of calling the SDK locally, and drives the motors.
- The **Pico** app points its tracking target at the **desktop's** IP (not
  the G1's) in this topology.

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

> **Note for Topology 2 (recommended for Orin):** On the G1 you only need
> A.1, A.4, A.5, A.6. **Skip A.2 and A.3** — XRoboToolKit is installed on the
> desktop instead (see section B.2). In Topology 1, do all of A.1 through A.6
> on the G1.

#### A.1 — Python env

Use the existing `sonic` env (Python 3.10). Add the two packages required by
the neck driver:

```bash
conda activate sonic
pip install dynamixel-sdk scipy pyzmq
```

#### A.2 — XRoboToolKit Python SDK (Topology 1 only — skip for Topology 2)

Before following the steps, check your architecture:
```bash
uname -m         # x86_64 → follow A.2.x86; aarch64 → follow A.2.aarch64
```

##### A.2.x86 — x86_64 machines

A pre-built Python 3.10 `.so` is shipped at
`external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so`.
Put it on the import path (add to `~/.bashrc`):

```bash
export GR00T_ROOT="$HOME/Desktop/GR00T-WholeBodyControl"      # adjust path
export PYTHONPATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$PYTHONPATH"
export LD_LIBRARY_PATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/lib:$LD_LIBRARY_PATH"
```

> Skipping `pip install -e …/` deliberately — it triggers a CMake rebuild that
> fails without cmake/pybind11 in the env. The PYTHONPATH approach reuses the
> pre-built `.so`.

##### A.2.aarch64 — Jetson Orin (JetPack 6.x / CUDA 12.6)

There is no pre-built Python `.so` for aarch64. You must build it once. The
aarch64 native library (`libPXREARobotSDK.so`) is already committed under
`.../lib/aarch64/`, so only the Python binding needs to compile.

```bash
conda activate sonic          # Python 3.10
cd $GR00T_ROOT
pip install cmake pybind11 setuptools
export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
pip install --no-build-isolation -e \
    external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/
```

(Or run `bash install_scripts/install_pico.sh` which does the same steps and
also rebuilds `libPXREARobotSDK.so` from source if the aarch64 copy is
missing — see
[install_pico.sh:70-94](install_scripts/install_pico.sh#L70-L94).)

Then set `LD_LIBRARY_PATH` so the binding finds the aarch64 native lib:
```bash
export GR00T_ROOT="$HOME/Desktop/GR00T-WholeBodyControl"      # adjust path
export LD_LIBRARY_PATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/lib/aarch64:$LD_LIBRARY_PATH"
```

(`pip install -e` makes `xrobotoolkit_sdk` importable without PYTHONPATH
surgery.)

##### Verify (either architecture)

```bash
python -c "import xrobotoolkit_sdk as xrt; xrt.init(); print('ok')"
```

#### A.3 — XRoboToolKit PC Service daemon (Topology 1 only — skip for Topology 2)

Daemon that receives head-pose from the Pico app and publishes it to
`xrobotoolkit_sdk`.

##### A.3.x86 — x86_64

```bash
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
```

##### A.3.aarch64 — Jetson Orin

**The shipped `.deb` is amd64-only and will NOT install on Orin.** You need an
aarch64 build of `RoboticsServiceProcess`. Two options:

1. **Ask XR-Robotics for an aarch64 `.deb`** (same version as the amd64 one).
2. **Build from source** using the `orin` branch of the upstream repo
   (which `install_pico.sh` already clones at the binding-build step):
   ```bash
   cd /tmp
   git clone -b orin https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
   cd XRoboToolkit-PC-Service
   # Follow the build instructions in that repo's README.
   # The resulting binary ecosystem (RoboticsServiceProcess, setting.ini,
   # support libs) must be installed to /opt/apps/roboticsservice/ so the
   # rest of this README works unchanged.
   ```

Without a running service daemon, `xrt.init()` still succeeds but
`get_headset_pose()` always returns zeros (what we debugged in the x86 setup).
Verify the install with:
```bash
pgrep -fa RoboticsServiceProcess   # should print a PID after launch
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

Two variants depending on topology.

#### B.1 — Viewer-only desktop (Topology 1)

The desktop only consumes the ZMQ feed — no SDK, no daemon, no motor libs.
```bash
conda activate sonic
pip install pyzmq opencv-python numpy    # likely already present
```

#### B.2 — Pose-publisher desktop (Topology 2, recommended for Orin)

The desktop runs the XRoboToolKit PC Service daemon and forwards the Pico
headset pose to the G1. Follow A.2.x86 and A.3.x86 above on the desktop (the
amd64 `.deb` installs cleanly here), and also:

```bash
conda activate sonic
pip install pyzmq
sudo ufw disable          # or open port 5559
```

The publisher itself is shipped as [pose_publisher.py](pose_publisher.py);
no extra setup.

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

Pick the subsection matching your topology.

### Topology 2 (daemon on desktop, Orin G1) — recommended

**Desktop terminal 1 — XRoboToolKit service daemon:**
```bash
sudo ufw disable
sudo bash /opt/apps/roboticsservice/runService.sh
```

**Pico headset:** launch XRoboToolkit Client, point its Server IP at the
**desktop's** IP, start tracking.

**Desktop terminal 2 — pose publisher:**
```bash
conda activate sonic
cd $GR00T_ROOT
export PYTHONPATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$PYTHONPATH"
export LD_LIBRARY_PATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/lib:$LD_LIBRARY_PATH"

python pose_publisher.py --bind tcp://0.0.0.0:5559 --hz 50
```
Expected logs: `XRoboToolkit SDK initialized.`, `PUB bound to tcp://0.0.0.0:5559`, then periodic `published N poses @ ~50 Hz`. If it warns `quat is zero`, the Pico→daemon link is not live (see troubleshooting).

**Desktop terminal 3 — optional live ZED viewer** (after launching G1 below):
```bash
python test_viewer.py --server <G1_IP> --port 5558 --show-stereo
```

**G1 terminal — launch the server:**
```bash
conda activate sonic
cd ~/Desktop/GR00T-WholeBodyControl      # adjust path on the robot

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
- Periodic `[Neck] yaw ... pitch ...` that track your head movement.

Ctrl-C on either machine stops cleanly. Motors zero and release torque on
the G1 side.

### Topology 1 (daemon on G1) — only if you have an aarch64 daemon

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
| `--pose-zmq <addr>` | "" (env `POSE_ZMQ`) | Read headset pose from a remote `pose_publisher.py` over ZMQ SUB instead of calling `xrobotoolkit_sdk` in-process. Use on Orin G1 when the daemon runs on the desktop. Example: `tcp://192.168.0.50:5559`. |
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

**(Topology 2) `[Neck] no headset data yet` on the G1**
The G1 is connected to the publisher but no valid poses are arriving.
Check on the desktop:
1. `pgrep -fa RoboticsServiceProcess` prints a PID.
2. `pose_publisher.py` log shows `published N poses @ ~50 Hz` (not
   `quat is zero`).
3. Firewall on the desktop isn't blocking 5559 (`sudo ufw status`).
4. G1 can reach the desktop: `ping <desktop-ip>` from the G1.

**(Topology 2) `zmq.error.ZMQError: Address already in use` on the desktop**
An old `pose_publisher.py` is still running: `pkill -f pose_publisher.py`.
