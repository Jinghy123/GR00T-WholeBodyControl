"""End-to-end headless orchestrator for token recording.

Pipeline:
  1. start sim_runner_headless.py  -> wait for "SIM_READY"
  2. start deploy.sh sim           -> feed "Y\n" (skip prompt), watch for
                                      "Press ENTER to toggle" (= main loop up)
  3. write "]"  to deploy stdin    -> start_control
  4. wait auto_drop_after_secs + settle -> robot is on ground and stable
  5. write "\n" to deploy stdin    -> toggle_zmq_mode (streaming)
  6. run walk_then_replay.py --record-tokens <episode>  (blocking)
  7. SIGTERM deploy & sim, wait, SIGKILL on timeout

Usage:
  python orchestration/orchestrator.py /path/to/episode_dir \
      [--out /tmp/data_sonic.json] [--auto-drop-after-secs 10] \
      [--settle-secs 3.0] [--deploy-mode sim]

Run on a headless server. No display required.
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
    print(f"[orchestrator] {tag}: {msg}", flush=True)


def _stream_to_stdout(name: str, pipe: IO[str], buf: List[str], marker: Optional[str],
                       marker_event: Optional[threading.Event]) -> None:
    """Read pipe line-by-line, mirror to our stdout, optionally set an event when marker seen."""
    try:
        for line in iter(pipe.readline, ""):
            sys.stdout.write(f"[{name}] {line}")
            sys.stdout.flush()
            buf.append(line)
            if marker is not None and marker_event is not None and not marker_event.is_set():
                if marker in line:
                    marker_event.set()
    except Exception as e:  # noqa: BLE001
        _log(name, f"stdout pump error: {e!r}")
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _wait_for(event: threading.Event, name: str, timeout: float) -> None:
    if not event.wait(timeout):
        raise TimeoutError(f"timeout waiting for {name} (>{timeout}s)")


def _terminate(proc: Optional[subprocess.Popen], name: str, grace: float = 5.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        _log(name, f"already exited rc={proc.returncode}")
        return
    _log(name, "SIGTERM")
    try:
        # Send to the whole process group so children (e.g. just->deploy binary) also die.
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
            _log(name, "still alive after SIGKILL — giving up")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir", type=str, help="Path to episode dir (contains data.json)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output data_sonic.json path. Default: <episode_dir>/data_sonic.json")
    ap.add_argument("--auto-drop-after-secs", type=float, default=10.0,
                    help="Seconds after sim start before disabling elastic_band.")
    ap.add_argument("--settle-secs", type=float, default=3.0,
                    help="Extra seconds after drop before sending Enter to deploy.")
    ap.add_argument("--deploy-mode", type=str, default="sim",
                    help="Argument passed to deploy.sh (sim/real/<iface>). Default: sim.")
    ap.add_argument("--sim-ready-timeout", type=float, default=120.0,
                    help="Max wait (not a sleep). Returns as soon as SIM_READY appears.")
    ap.add_argument("--deploy-ready-timeout", type=float, default=120.0,
                    help="Max wait (not a sleep). Returns as soon as the deploy banner appears. "
                         "Typical first run: 20-30s. Bump if `just build` takes longer.")
    args = ap.parse_args()

    episode_dir = Path(args.episode_dir).resolve()
    if not (episode_dir / "data.json").exists():
        _log("init", f"no data.json in {episode_dir}")
        return 1
    out_path = Path(args.out) if args.out else (episode_dir / "data_sonic.json")

    sim_proc: Optional[subprocess.Popen] = None
    deploy_proc: Optional[subprocess.Popen] = None
    walk_proc: Optional[subprocess.Popen] = None

    sim_buf: List[str] = []
    deploy_buf: List[str] = []
    sim_ready = threading.Event()
    deploy_ready = threading.Event()

    try:
        # ---------------- Stage 1: sim ----------------
        _log("stage1", f"starting headless sim ({SIM_RUNNER.name}) auto_drop={args.auto_drop_after_secs}s")
        sim_cmd = [
            sys.executable, "-u", str(SIM_RUNNER),
            "--auto-drop-after-secs", str(args.auto_drop_after_secs),
        ]
        sim_proc = subprocess.Popen(
            sim_cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            preexec_fn=os.setsid,
        )
        threading.Thread(
            target=_stream_to_stdout,
            args=("sim", sim_proc.stdout, sim_buf, SIM_READY_MARKER, sim_ready),
            daemon=True,
        ).start()
        _wait_for(sim_ready, "sim SIM_READY", args.sim_ready_timeout)
        _log("stage1", "sim ready")
        sim_started_at = time.monotonic()

        # ---------------- Stage 2: deploy ----------------
        _log("stage2", f"starting deploy.sh {args.deploy_mode}")
        deploy_proc = subprocess.Popen(
            ["bash", str(DEPLOY_SH), args.deploy_mode],
            cwd=str(DEPLOY_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            preexec_fn=os.setsid,
        )
        threading.Thread(
            target=_stream_to_stdout,
            args=("deploy", deploy_proc.stdout, deploy_buf, DEPLOY_READY_MARKER, deploy_ready),
            daemon=True,
        ).start()

        # Pass the [Y/n] confirmation. Bash's `read` will pick this up.
        assert deploy_proc.stdin is not None
        deploy_proc.stdin.write("Y\n")
        deploy_proc.stdin.flush()
        _log("stage2", "sent 'Y' for deploy confirmation; waiting for main loop banner...")

        _wait_for(deploy_ready, "deploy main-loop banner", args.deploy_ready_timeout)
        _log("stage2", "deploy main loop is up")

        # ---------------- Stage 3: press ] ----------------
        time.sleep(0.5)  # tiny pause so the binary's stdin handler is definitely armed
        _log("stage3", "sending ']' (start_control)")
        deploy_proc.stdin.write("]")
        deploy_proc.stdin.flush()

        # ---------------- Stage 4: wait for drop + settle ----------------
        # auto-drop fires at sim_started_at + auto_drop_after_secs (from sim's
        # perspective). It started slightly before deploy. Compute remaining.
        drop_at = sim_started_at + args.auto_drop_after_secs
        remaining = drop_at - time.monotonic()
        if remaining > 0:
            _log("stage4", f"waiting {remaining:.1f}s for auto-drop")
            time.sleep(remaining)
        _log("stage4", f"settling for {args.settle_secs}s")
        time.sleep(args.settle_secs)

        # ---------------- Stage 5: press Enter ----------------
        _log("stage5", "sending Enter (toggle_zmq_mode -> streaming)")
        deploy_proc.stdin.write("\n")
        deploy_proc.stdin.flush()
        time.sleep(1.0)

        # ---------------- Stage 6: walk_then_replay with token recording ----------------
        _log("stage6", f"running walk_then_replay {episode_dir} -> {out_path}")
        walk_cmd = [
            sys.executable, "-u", str(WALK_THEN_REPLAY),
            str(episode_dir),
            "--record-tokens", str(out_path),
        ]
        walk_proc = subprocess.Popen(
            walk_cmd,
            cwd=str(REPO_ROOT),
            stdout=None, stderr=None,  # inherit; let user see live
            preexec_fn=os.setsid,
        )
        walk_rc = walk_proc.wait()
        _log("stage6", f"walk_then_replay rc={walk_rc}")

        return 0 if walk_rc == 0 and out_path.exists() else max(1, walk_rc)

    except Exception as e:  # noqa: BLE001
        _log("error", f"{type(e).__name__}: {e}")
        return 1
    finally:
        _terminate(walk_proc, "walk_then_replay", grace=2.0)
        _terminate(deploy_proc, "deploy", grace=5.0)
        _terminate(sim_proc, "sim", grace=5.0)


if __name__ == "__main__":
    sys.exit(main())
