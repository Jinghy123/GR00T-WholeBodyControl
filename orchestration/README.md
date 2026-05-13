# Headless online token recording

End-to-end automation for recording closed-loop FSQ tokens from
`walk_then_replay.py` on a headless server. No display, no manual key presses.

Two files, both new — nothing under `gear_sonic/`, `gear_sonic_deploy/`, or
`walk_then_replay.py` was modified.

## Files

- `sim_runner_headless.py` — wraps `gear_sonic/scripts/run_sim_loop.py`:
  - forces `enable_onscreen=False` so no MuJoCo viewer is spawned
  - starts the sim as a thread and, after `--auto-drop-after-secs N`, sets
    `sim.elastic_band.enable = False` — programmatic equivalent of pressing `9`
  - prints `SIM_READY` once the sim thread is ticking
- `orchestrator.py` — main driver:
  1. spawns `sim_runner_headless.py`, waits for `SIM_READY`
  2. spawns `bash deploy.sh sim`, pipes `Y\n` past the `[Y/n]` prompt
  3. watches deploy stdout for `Init Done` (full init complete — output
     handlers, robot config, etc. all up)
  4. writes `]` to deploy stdin -> `start_control = true`
  5. waits for the sim's auto-drop + a settle window
  6. writes `\n` to deploy stdin -> `toggle_zmq_mode = true` (streaming on)
  7. runs `walk_then_replay.py <episode_dir> --record-tokens <out>` blocking
  8. SIGTERM (then SIGKILL) every child process

## Usage

```bash
python orchestration/orchestrator.py /path/to/episode_15 \
    --out /path/to/episode_15/data_sonic.json \
    --auto-drop-after-secs 10 \
    --settle-secs 3
```

Defaults: `--out=<episode_dir>/data_sonic.json`, drop 10 s, settle 3 s,
`--deploy-mode sim`.

## Notes / caveats

- `--deploy-ready-timeout` is a *max wait*, not a fixed sleep — we exit as
  soon as deploy prints the `Press ENTER to toggle ...` banner. Typical first
  run on a warm build: 20–30s. Bump it only if you really do hit the cap.
- Each child runs in its own process group (`os.setsid`) so cleanup kills
  the whole tree, including the binary spawned by `just run`.
- For parallel workers, give each one its own DDS `DOMAIN_ID` and ZMQ ports
  via `sim_runner_headless.py` config; the current orchestrator does not yet
  thread those through.
