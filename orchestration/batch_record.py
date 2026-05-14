"""Batch headless token recording across multiple episodes.

Shares one sim + one deploy across N episodes — TRT engine is loaded once,
saving ~30s per episode vs single-shot orchestrator.

Input auto-detection (each positional arg):
  * directory containing data.json           → single episode
  * directory containing episode_* subdirs   → task (all its episodes)
  * directory containing task subdirs        → category (all tasks × all episodes);
                                               writes conversion_status.txt here

On first walk_then_replay failure: writes FAILED + STOPPED to the category
status file, tears down sim/deploy, exits non-zero.

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
  python orchestration/batch_record.py /hfm/data/HE_RAW/Articulated
  python orchestration/batch_record.py /path/to/task_folder
  python orchestration/batch_record.py /path/to/episode_15 /path/to/episode_16
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


# ── input auto-detection ─────────────────────────────────────────────────────

def _looks_like_task(p: Path) -> bool:
    """Task = dir that contains at least one episode_* subdir with data.json."""
    if not p.is_dir():
        return False
    for sub in p.iterdir():
        if sub.is_dir() and sub.name.startswith("episode") and (sub / "data.json").is_file():
            return True
    return False


def _list_episodes(task_dir: Path) -> List[Path]:
    eps = [p for p in task_dir.iterdir()
           if p.is_dir() and p.name.startswith("episode") and (p / "data.json").is_file()]
    return sorted(eps, key=lambda p: p.name)


def _list_tasks(category_dir: Path) -> List[Path]:
    return sorted([p for p in category_dir.iterdir() if _looks_like_task(p)],
                  key=lambda p: p.name)


def classify(path: Path) -> str:
    """Returns 'episode' | 'task' | 'category'."""
    if (path / "data.json").is_file():
        return "episode"
    if _looks_like_task(path):
        return "task"
    return "category"


class StatusWriter:
    """Append-only, auto-flushed status log. Disabled (no-op) when path is None."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self.fh = open(path, "a") if path is not None else None
        if self.fh is not None:
            self.fh.write("\n")

    def _ts(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def write(self, line: str) -> None:
        msg = f"[{self._ts()}] {line}\n"
        if self.fh is not None:
            self.fh.write(msg)
            self.fh.flush()

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


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
    ap.add_argument("paths", nargs="+",
                    help="Category dir, task dir, or single episode dir (auto-detect)")
    ap.add_argument("--out-name", type=str, default="data_sonic.json",
                    help="Output filename inside each episode dir.")
    ap.add_argument("--status-name", type=str, default="conversion_status.txt",
                    help="Status filename written inside each category dir.")
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

    # Expand each path into a "group" = (root, kind, status_path, [(task_name, [eps])]).
    paths = [Path(p).resolve() for p in args.paths]
    for p in paths:
        if not p.is_dir():
            _log("init", f"not a directory: {p}")
            return 1

    groups: List[dict] = []
    total_eps = 0
    for p in paths:
        kind = classify(p)
        if kind == "episode":
            tasks = [(p.parent.name, [p])]
            status_path = None
        elif kind == "task":
            tasks = [(p.name, _list_episodes(p))]
            status_path = None
        else:  # category
            tasks = [(t.name, _list_episodes(t)) for t in _list_tasks(p)]
            status_path = p / args.status_name
        n_eps = sum(len(eps) for _, eps in tasks)
        total_eps += n_eps
        _log("init", f"{p} → kind={kind} ({len(tasks)} tasks, {n_eps} episodes)")
        groups.append({"root": p, "kind": kind, "status_path": status_path,
                       "tasks": tasks})

    if total_eps == 0:
        _log("init", "no episodes to process")
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

        # ---- per-group loop (stop-on-first-failure) ----
        global_idx = 0
        for g in groups:
            status = StatusWriter(g["status_path"])
            if g["kind"] == "category":
                n_eps_in_g = sum(len(eps) for _, eps in g["tasks"])
                status.write(f"STARTED — {g['root'].name} "
                             f"({len(g['tasks'])} tasks, {n_eps_in_g} episodes)")

            stopped = False
            failed_ref: Optional[str] = None
            for task_name, eps in g["tasks"]:
                if not eps:
                    status.write(f"SKIP   {task_name} (no episodes with data.json)")
                    continue
                t_task = time.perf_counter()
                for ep in eps:
                    if global_idx > 0:
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
                    _log("episode", f"[{global_idx+1}/{total_eps}] "
                                    f"{task_name}/{ep.name} → {out_path}")
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
                    _log("episode", f"{task_name}/{ep.name} rc={walk_rc} "
                                    f"out_exists={out_path.exists()}")
                    global_idx += 1
                    if not ok:
                        failed_ref = f"{task_name}/{ep.name}"
                        status.write(f"FAILED {failed_ref} "
                                     f"(rc={walk_rc}, out_exists={out_path.exists()})")
                        status.write("STOPPED — first failure")
                        stopped = True
                        break
                if stopped:
                    break
                status.write(f"DONE   {task_name} ({len(eps)} episodes, "
                             f"{time.perf_counter()-t_task:.1f}s)")

            if not stopped and g["kind"] == "category":
                status.write(f"FINISHED — {g['root'].name}")
            status.close()
            if stopped:
                _log("error", f"stopped at {failed_ref}")
                return 2

        return 0

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
