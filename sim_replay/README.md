# Replaying episodes in the MuJoCo sim

Streams a recorded episode's action tokens into the WBC deploy running against
the MuJoCo simulator, so a demonstration can be re-executed without hardware.

```sh
sim_replay/replay_lerobot.sh .data/g1_sonic_lerobot_0810_merged_val 0   # LeRobot episode
sim_replay/replay_in_mujoco.sh recordings/run1.pkl                      # record_sonic pickle

# first 100 frames only (~3 s at 30 Hz), for a quick smoke test
sim_replay/replay_lerobot.sh .data/g1_sonic_lerobot_0810_merged_val 0 --max-allowed-frames 100
```

`--max-allowed-frames N` stops the parquet read once N rows are in hand, so a
long episode is neither fully extracted nor fully replayed. The npz, pickle and
video are suffixed `_f<N>`, so a truncated run never overwrites the full one.

Both take ~90 s: ~40 s of WBC startup (TensorRT engine build on first run is
slower), ~14 s to stand the robot up and settle, then the episode at its
recorded rate. Each of the silent startup stages shows a progress bar on the
terminal; the episode itself reports its own tick count. Logs go to
`.data/g1_sim_replay/` (`sim.log`, `deploy.log`, `replay.log`, `start_control.log`);
override with `LOG_DIR`. When the episode ends the WBC and the
sim are shut down, closing the MuJoCo viewer; pass `KEEP_SIM=1` to leave them
running (stop them later with `pkill -f g1_deploy_onnx_ref`).

The viewer window is screen-recorded (ffmpeg x11grab, 30 fps h264) for the
length of the episode - `replay_lerobot.sh` writes `<dataset>_ep<N>.mp4` and
`replay_in_mujoco.sh` writes `<episode>.mp4`, both next to the logs. `VIDEO=`
picks another path, `NO_VIDEO=1` turns it off. Being a screen grab it records
whatever is on top of the viewer, so leave the window unobstructed; needs
`ffmpeg`, `xwininfo` and an X11 `DISPLAY`, and quietly skips recording if any of
them is missing.

## Files

| file | role |
|---|---|
| `replay_lerobot.sh` | one LeRobot episode: extract -> repack -> replay |
| `replay_in_mujoco.sh` | brings up sim + WBC + replay in the right order |
| `run_sim_loop_dropctl.py` | `run_sim_loop.py` plus flag-file elastic-band control and a 1 Hz pelvis trace |
| `run_wbc_deploy.sh` | launches `g1_deploy_onnx_ref` for sim (see host caveats below) |
| `start_control.py` | sends the start command alone, so the policy holds the robot before it is set down |
| `lerobot_extract.py` | LeRobot parquet -> npz (needs a python with pyarrow) |
| `lerobot_to_replay.py` | npz -> `record_sonic.py`-style replay pickle |
| `capture_state.py` | records measured joint state from `:5557`, for checking tracking |

## Things that are easy to get wrong

* **Streamed-motion mode.** `ZMQManager` only forwards tokens to the policy in
  `STREAMED_MOTION`, selected by `planner=0` in the start command. With
  `planner=1` the WBC receives every token and drops it, and the robot stands
  motionless for the whole episode - it looks like a working replay of a very
  boring demonstration. `replay_sonic.py` now defaults to `planner=0`.
* **`--input-type zmq_manager`, not `zmq`.** The plain `zmq` endpoint subscribes
  to the pose topic only, so it never sees the start command and control never
  begins.
* **Order of standing vs. dropping.** Release the elastic band before control
  starts and a limp robot falls; release it mid-episode and the transient throws
  the robot. `replay_in_mujoco.sh` starts control, lowers the band, settles, then
  streams.
* **The stop command.** `replay_sonic.py` sends stop when the episode ends, which
  terminates WBC control and exits the deploy, so the robot collapses right after
  the last tick. `--no-stop` keeps it standing.

## Host-specific paths

`run_wbc_deploy.sh` launches the prebuilt binary through a nix glibc loader: it
needs GLIBC >= 2.38 (Ubuntu 22.04 has 2.35) and `libcudart.so.13` (system CUDA
here is 11.8/12.8). Override `GLIBC_ROOT` / `CUDA13_LIB` / `TensorRT_ROOT`, or
call the binary directly if your host resolves these normally.

`replay_lerobot.sh` shells out to `PYARROW_PY` for the parquet read because
neither `.venv_sim` nor `.venv_teleop` has pyarrow.

## Verifying a replay actually moved

```sh
.venv_teleop/bin/python sim_replay/capture_state.py 30 /tmp/measured.npz   # during a replay
```

Compare the per-joint std against the episode's own `qpos`: a working replay of
`g1_sonic_lerobot_0810_merged_val` episode 0 gives ~9.8 deg mean / 24.9 deg max
against the dataset's 8.4 / 23.2. Near-zero std means the tokens are not being
applied. `sim.log`'s `[trace]` lines give pelvis height and yaw, which separate a
real fall from a controller that never started.
