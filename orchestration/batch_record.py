"""Batch headless token recording across multiple episodes.

Shares one sim + one deploy across N episodes — TRT engine is loaded once,
saving ~30s per episode vs single-shot orchestrator.

Flow:
  ┌─ first time only ─┐         ┌──── per episode ────────────────┐
  start sim                     │ (if not first):                 │
  start deploy                  │   send Enter -> loaded motion    │
  wait deploy ready             │   wait reset_settle_secs         │
  send ] -> start_control       │     (robot self-stabilizes under │
  SIGUSR1 (=press 9)            │      loaded reference motion)    │
  wait settle_secs (stand up)   │   send Enter -> ZMQ streaming    │
  send Enter -> streaming       │   wait pre_episode_secs          │
                                │ run walk_then_replay --record-tokens
                                │ wait for it to finish            │
                                └──────────────────────────────────┘
  cleanup (SIGTERM sim+deploy)

The Enter-twice cycle (in --input-type zmq) toggles between ZMQ pose stream
and the deploy's loaded reference motion (the `--default-motion` from
deploy.sh, e.g. neutral_kick — a balanced standing motion). Cycling to
loaded-motion between episodes lets the robot recover to a clean standing
pose on its own — no re-hanging the band needed.

Usage:
  python orchestration/batch_record.py \\
      /path/to/episode_15 /path/to/episode_16 \\
      [--out-name data_sonic.json]              \\
      [--settle-secs 5]                         \\
      [--reset-settle-secs 2]                   \\
      [--pre-episode-secs 1.5]                  \\
      [--deploy-mode sim]
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = REPO_ROOT / "gear_sonic_deploy"
DEPLOY_SH = DEPLOY_DIR / "deploy.sh"
SIM_RUNNER = REPO_ROOT / "orchestration" / "sim_runner_headless.py"
WALK_THEN_REPLAY = REPO_ROOT / "walk_then_replay.py"

SIM_READY_MARKER = "SIM_READY"
DEPLOY_READY_MARKER = "Init Done"


def _log(tag: str, msg: str) -> None:
    print(f"[batch] {tag}: {msg}", flush=True)


class _PipePump:
    """Pumps a subprocess pipe to our stdout and fires events on markers."""

    def __init__(self, name: str, pipe: IO[str]):
        self.name = name
        self.pipe = pipe
        self.lock = threading.Lock()
        self._markers: dict[str, threading.Event] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def watch(self, marker: str) -> threading.Event:
        ev = threading.Event()
        with self.lock:
            self._markers[marker] = ev
        return ev

    def _run(self) -> None:
        try:
            for line in iter(self.pipe.readline, ""):
                sys.stdout.write(f"[{self.name}] {line}")
                sys.stdout.flush()
                with self.lock:
                    for m, ev in self._markers.items():
                        if not ev.is_set() and m in line:
                            ev.set()
        except Exception as e:  # noqa: BLE001
            _log(self.name, f"pump err: {e!r}")
        finally:
            try:
                self.pipe.close()
            except Exception:
                pass


def _terminate(proc: Optional[subprocess.Popen], name: str, grace: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    _log(name, "SIGTERM")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=grace)
        _log(name, f"exited rc={proc.returncode}")
    except subprocess.TimeoutExpired:
        _log(name, "SIGKILL")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+", help="Episode dirs (each must contain data.json)")
    ap.add_argument("--out-name", type=str, default="data_sonic.json",
                    help="Output filename inside each episode dir.")
    ap.add_argument("--settle-secs", type=float, default=5.0,
                    help="After initial drop, wait this long for robot to stand up before first Enter.")
    ap.add_argument("--reset-settle-secs", type=float, default=2.0,
                    help="Between episodes: after Enter -> loaded motion, wait this long for the robot"
                         " to self-stabilize under the reference motion before re-enabling streaming.")
    ap.add_argument("--pre-episode-secs", type=float, default=1.5,
                    help="After streaming re-enabled, wait this long before launching walk_then_replay.")
    ap.add_argument("--deploy-mode", type=str, default="sim")
    ap.add_argument("--viewer", action="store_true",
                    help="Show mujoco viewer (debug only; default headless).")
    ap.add_argument("--sim-ready-timeout", type=float, default=120.0)
    ap.add_argument("--deploy-ready-timeout", type=float, default=120.0)
    args = ap.parse_args()

    episodes = [Path(p).resolve() for p in args.episodes]
    for ep in episodes:
        if not (ep / "data.json").exists():
            _log("init", f"no data.json in {ep}")
            return 1

    sim_proc: Optional[subprocess.Popen] = None
    deploy_proc: Optional[subprocess.Popen] = None
    sim_pump: Optional[_PipePump] = None
    deploy_pump: Optional[_PipePump] = None

    try:
        # ---- start sim ----
        _log("setup", f"starting sim (viewer={args.viewer})")
        sim_cmd = [sys.executable, "-u", str(SIM_RUNNER)]
        if args.viewer:
            sim_cmd.append("--viewer")
        sim_proc = subprocess.Popen(
            sim_cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, preexec_fn=os.setsid,
        )
        sim_pump = _PipePump("sim", sim_proc.stdout)  # type: ignore[arg-type]
        sim_ready = sim_pump.watch(SIM_READY_MARKER)
        if not sim_ready.wait(args.sim_ready_timeout):
            _log("setup", "timeout waiting for SIM_READY")
            return 1
        _log("setup", "sim ready")

        # ---- start deploy ----
        _log("setup", f"starting deploy.sh --input-type zmq {args.deploy_mode}")
        deploy_proc = subprocess.Popen(
            ["bash", str(DEPLOY_SH), "--input-type", "zmq", args.deploy_mode],
            cwd=str(DEPLOY_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, preexec_fn=os.setsid,
        )
        deploy_pump = _PipePump("deploy", deploy_proc.stdout)  # type: ignore[arg-type]
        deploy_ready = deploy_pump.watch(DEPLOY_READY_MARKER)

        assert deploy_proc.stdin is not None
        deploy_proc.stdin.write("Y\n")
        deploy_proc.stdin.flush()

        if not deploy_ready.wait(args.deploy_ready_timeout):
            _log("setup", "timeout waiting for deploy Init Done")
            return 1
        _log("setup", "deploy ready")

        # ---- first-time setup: ] -> SIGUSR1 (press 9) -> wait stand -> Enter ----
        time.sleep(0.5)
        _log("setup", "sending ']' (start_control)")
        deploy_proc.stdin.write("]")
        deploy_proc.stdin.flush()

        _log("setup", "SIGUSR1 -> sim (press 9: drop band)")
        os.killpg(sim_proc.pid, signal.SIGUSR1)

        _log("setup", f"waiting {args.settle_secs}s for robot to stand")
        time.sleep(args.settle_secs)

        _log("setup", "sending Enter (toggle ZMQ streaming ON)")
        deploy_proc.stdin.write("\n")
        deploy_proc.stdin.flush()
        time.sleep(1.0)

        # ---- per-episode loop ----
        results: List[tuple[str, int, bool]] = []
        for idx, ep in enumerate(episodes):
            _log("episode", f"[{idx+1}/{len(episodes)}] {ep.name}")

            if idx > 0:
                _log("reset", "Enter -> loaded reference motion")
                deploy_proc.stdin.write("\n")
                deploy_proc.stdin.flush()
                _log("reset", f"settling {args.reset_settle_secs}s")
                time.sleep(args.reset_settle_secs)
                _log("reset", "Enter -> ZMQ streaming ON")
                deploy_proc.stdin.write("\n")
                deploy_proc.stdin.flush()
                time.sleep(args.pre_episode_secs)

            out_path = ep / args.out_name
            _log("episode", f"running walk_then_replay -> {out_path}")
            walk_proc = subprocess.Popen(
                [sys.executable, "-u", str(WALK_THEN_REPLAY),
                 str(ep), "--record-tokens", str(out_path)],
                cwd=str(REPO_ROOT),
                preexec_fn=os.setsid,
            )
            try:
                walk_rc = walk_proc.wait()
            except KeyboardInterrupt:
                _terminate(walk_proc, "walk_then_replay", 2.0)
                raise
            ok = walk_rc == 0 and out_path.exists()
            _log("episode", f"{ep.name} rc={walk_rc} out_exists={out_path.exists()}")
            results.append((str(ep), walk_rc, ok))

        # ---- summary ----
        print()
        _log("summary", f"{sum(1 for _,_,ok in results if ok)}/{len(results)} succeeded")
        for ep, rc, ok in results:
            mark = "OK " if ok else "FAIL"
            print(f"  [{mark}] rc={rc}  {ep}")
        return 0 if all(ok for _, _, ok in results) else 2

    except KeyboardInterrupt:
        _log("error", "interrupted by user")
        return 130
    except Exception as e:  # noqa: BLE001
        _log("error", f"{type(e).__name__}: {e}")
        return 1
    finally:
        _terminate(deploy_proc, "deploy", grace=5.0)
        _terminate(sim_proc, "sim", grace=5.0)


if __name__ == "__main__":
    sys.exit(main())
