"""Offline token recording for stationary manipulation episodes.

For datasets where the robot never walks/turns (all phase A in walk_then_replay),
token computation has no sim/WBC/planner dependency — it's a pure offline
transform of data.json.

Input auto-detection (each positional arg):
  * directory containing data.json           → single episode
  * directory containing episode_* subdirs   → task (all its episodes)
  * directory containing task subdirs        → category (all tasks × all episodes);
                                               writes conversion_status.txt here

Robot type per episode is auto-detected from frame 0's `robot_type` key:
  * present → G1 path: reuses walk_then_replay.load_full_episode + write_sonic_json.
  * absent  → H1 path:
      - legs  : forced to G1_DEFAULT_ANGLES_MUJOCO[:15]
      - arms  : H1 actions.sol_q (14d, mujoco arm order)
      - quat  : identity
      - hands : action.left_angles / right_angles via the sonic 1.7-x mapping
                → 6d per side → pad each with a 0 → 14d (left_7 + right_7)
      - states.hand_joints: H1 states.hand_state (12d right-first) → swap and
                            pad → 14d

On first failure: writes the error to conversion_status.txt and exits with
non-zero (stops at the first broken episode).

Usage:
  python orchestration/offline_record.py /hfm/data/HE_RAW/Articulated
  python orchestration/offline_record.py /path/to/single/episode_0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from encoder_client import EncoderClient  # noqa: E402
from walk_forward_token import (  # noqa: E402
    ENCODER_MODEL, G1_DEFAULT_ANGLES_MUJOCO, _MUJOCO_TO_ISAACLAB_DOF,
    fsq_quantize, resample_30hz_to_50hz,
)
from walk_then_replay import (  # noqa: E402
    ENCODER_MODEL_V1_1, build_data_window, load_full_episode, write_sonic_json,
)


G1_STABLE_LEG_MUJOCO = G1_DEFAULT_ANGLES_MUJOCO[:15].copy()


def _log(msg: str) -> None:
    print(f"[offline] {msg}", flush=True)


def _load_frames(episode_dir: Path):
    with open(episode_dir / "data.json") as f:
        d = json.load(f)
    return d if isinstance(d, list) else d.get("frames", d)


def is_g1_episode(episode_dir: Path) -> bool:
    return "robot_type" in _load_frames(episode_dir)[0]


# ── H1 hand conversion (1.7-x mapping from sonic_client) ─────────────────────

def _h1_hand_action_14(left_angles_12, right_angles_12) -> list[float]:
    """H1 left/right_angles (each 12d) → 14d sonic hand action (left_7, right_7).
    Each side: 6d from the 1.7-x mapping, pad a trailing 0 to make 7."""
    def side6(q):
        out = [1.7 - q[i] for i in (4, 6, 2, 0)]
        out.append(1.2 - q[8])
        out.append(0.5 - q[9])
        return out
    return side6(left_angles_12) + [0.0] + side6(right_angles_12) + [0.0]


def _h1_hand_state_14(hand_state_12) -> list[float]:
    """H1 states.hand_state is 12d, ordered (right_6, left_6). Swap to
    (left_6, right_6) and pad each 6-block with a trailing 0 → 14d."""
    h = list(hand_state_12)
    return h[6:12] + [0.0] + h[:6] + [0.0]


# ── H1 loader (encoder input) ────────────────────────────────────────────────

def load_h1_episode(episode_dir: Path):
    """Returns (qpos_isaac_30, imu_quat_30) for encoder input. Legs forced to
    G1 stable standing; arms from actions.sol_q (14d, mujoco arm order); quat
    forced to identity."""
    frames = _load_frames(episode_dir)
    n = len(frames)
    qpos_mj = np.zeros((n, 29), dtype=np.float32)
    qpos_mj[:, :15] = G1_STABLE_LEG_MUJOCO
    for i, fr in enumerate(frames):
        sol_q = (fr.get("actions") or {}).get("sol_q")
        if sol_q is None or len(sol_q) < 14:
            raise ValueError(f"frame {i}: missing/short actions.sol_q")
        qpos_mj[i, 15:] = np.asarray(sol_q, dtype=np.float32).reshape(-1)[:14]
    qpos_isaac = qpos_mj[:, _MUJOCO_TO_ISAACLAB_DOF].astype(np.float32)
    imu_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    return qpos_isaac, imu_quat


# ── H1 sonic writer ──────────────────────────────────────────────────────────

def write_sonic_json_h1(episode_dir: Path, out_path: Path, tokens_30) -> None:
    """Per-frame sonic JSON:
      states.qpos        = G1_STABLE_LEG (15) + states.arm_state (14) = 29
      states.quat        = identity [1,0,0,0]
      states.hand_joints = H1 hand_state (12, right-first) → 14 (left, right)
      actions.hand_joints = H1 left/right_angles via 1.7-x → 14
      actions.token       = computed 64-d FSQ token
    """
    frames = _load_frames(episode_dir)
    n = len(frames)
    if len(tokens_30) != n:
        raise ValueError(f"tokens has {len(tokens_30)} but data has {n} frames")

    stable_leg = G1_STABLE_LEG_MUJOCO.tolist()
    out = []
    for i, fr in enumerate(frames):
        s = fr["states"]
        a = fr.get("actions") or {}
        arm = np.asarray(s.get("arm_state", s.get("arm_states", [])),
                         dtype=np.float32).reshape(-1)[:14].tolist()
        if len(arm) < 14:
            raise ValueError(f"frame {i}: arm_state has <14 entries")
        hs = s.get("hand_state")
        if hs is None or len(hs) < 12:
            raise ValueError(f"frame {i}: missing/short states.hand_state")
        la, ra = a.get("left_angles"), a.get("right_angles")
        if la is None or ra is None or len(la) < 12 or len(ra) < 12:
            raise ValueError(f"frame {i}: missing/short actions.left/right_angles")

        frame_out = {
            "states": {
                "qpos":        stable_leg + arm,
                "quat":        [1.0, 0.0, 0.0, 0.0],
                "hand_joints": _h1_hand_state_14(hs),
            },
            "actions": {
                "hand_joints": _h1_hand_action_14(la, ra),
                "token":       tokens_30[i].tolist(),
            },
            "image": fr.get("image", ""),
        }
        t = fr.get("time")
        if isinstance(t, (int, float)):
            frame_out["timestamp"] = int(t * 1e9)
        out.append(frame_out)

    with open(out_path, "w") as f:
        json.dump(out, f)
    _log(f"wrote {n} frames → {out_path}")


# ── Top-level encode (G1 / H1 routing) ───────────────────────────────────────

def encode_episode(episode_dir: Path, encoder: EncoderClient, out_path: Path) -> None:
    if is_g1_episode(episode_dir):
        # _cmd (torso_vx/vy + target_yaw) is unused here — offline episodes are
        # all phase A (no walking), so motion-command detection never matters.
        # Just absorb the 6th return value and drop it.
        qpos_30, _h, imu_q_30, _v, sol_dim, _cmd = load_full_episode(str(episode_dir))
        tag = f"G1 sol_dim={sol_dim}"
        writer = lambda toks: write_sonic_json(str(episode_dir), str(out_path), toks)
    else:
        qpos_30, imu_q_30 = load_h1_episode(episode_dir)
        tag = "H1 (stable-leg + sol_q arms + 1.7-x hands)"
        writer = lambda toks: write_sonic_json_h1(episode_dir, out_path, toks)

    n_30 = len(qpos_30)
    qpos_50 = resample_30hz_to_50hz(qpos_30)
    quat_50 = resample_30hz_to_50hz(imu_q_30)
    quat_50 /= np.linalg.norm(quat_50, axis=1, keepdims=True).clip(1e-8)
    n_50 = len(qpos_50)
    _log(f"{episode_dir.name}: {tag} n_30={n_30} n_50={n_50}")

    tokens_50 = np.zeros((n_50, 64), dtype=np.float32)
    t0 = time.perf_counter()
    for pub_frame in range(n_50):
        jp, jv, bq = build_data_window(qpos_50, quat_50, pub_frame)
        tok = fsq_quantize(encoder.encode(joint_pos=jp, joint_vel=jv, body_quat=bq))
        tokens_50[pub_frame] = tok.astype(np.float32)

    idx_30_to_50 = np.round(np.arange(n_30) * 5.0 / 3.0).astype(int).clip(0, n_50 - 1)
    writer(tokens_50[idx_30_to_50])
    _log(f"{episode_dir.name}: done ({time.perf_counter()-t0:.1f}s, {n_50} frames)")


# ── Input walking (episode / task / category auto-detect) ───────────────────

def _looks_like_task(p: Path) -> bool:
    """Task = dir that contains at least one episode_* subdir with data.json."""
    if not p.is_dir():
        return False
    for sub in p.iterdir():
        if sub.is_dir() and sub.name.startswith("episode") and (sub / "data.json").is_file():
            return True
    return False


def _list_episodes(task_dir: Path) -> list[Path]:
    eps = [p for p in task_dir.iterdir()
           if p.is_dir() and p.name.startswith("episode") and (p / "data.json").is_file()]
    return sorted(eps, key=lambda p: p.name)


def _list_tasks(category_dir: Path) -> list[Path]:
    return sorted([p for p in category_dir.iterdir() if _looks_like_task(p)],
                  key=lambda p: p.name)


def classify(path: Path) -> str:
    """Returns 'episode' | 'task' | 'category'."""
    if (path / "data.json").is_file():
        return "episode"
    if _looks_like_task(path):
        return "task"
    return "category"


# ── Status file (live progress / errors, per category) ───────────────────────

class StatusWriter:
    """Append-only, auto-flushed status log. Disabled for single-episode runs."""

    def __init__(self, path: Path | None):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="Category dir, task dir, or single episode dir (auto-detect)")
    ap.add_argument("--out-name", type=str, default=None,
                    help="Output filename inside each episode dir "
                         "(default: data_sonic.json, or data_sonic_v1_1.json with --v1.1).")
    ap.add_argument("--v1.1", "--v1_1", dest="v1_1", action="store_true",
                    help="Encode with the sonic v1.1 checkpoint.")
    ap.add_argument("--status-name", type=str, default="conversion_status.txt")
    args = ap.parse_args()

    if args.out_name is None:
        args.out_name = "data_sonic_v1_1.json" if args.v1_1 else "data_sonic.json"

    paths = [Path(p).resolve() for p in args.paths]
    for p in paths:
        if not p.is_dir():
            _log(f"not a directory: {p}")
            return 1

    if args.v1_1:
        _log(f"loading encoder {ENCODER_MODEL_V1_1} (v1.1)")
        encoder = EncoderClient(ENCODER_MODEL_V1_1, mode=0, version="v1_1")
    else:
        _log(f"loading encoder {ENCODER_MODEL}")
        encoder = EncoderClient(ENCODER_MODEL, mode=0)

    overall_ok = True
    for p in paths:
        kind = classify(p)
        _log(f"input {p} → kind={kind}")

        if kind == "episode":
            status = StatusWriter(None)
            tasks = [(p.parent.name, [p])]
        elif kind == "task":
            status = StatusWriter(None)
            tasks = [(p.name, _list_episodes(p))]
        else:  # category
            status_path = p / args.status_name
            status = StatusWriter(status_path)
            tlist = _list_tasks(p)
            tasks = [(t.name, _list_episodes(t)) for t in tlist]
            n_total_eps = sum(len(eps) for _, eps in tasks)
            status.write(f"STARTED — {p.name} ({len(tasks)} tasks, {n_total_eps} episodes)")
            _log(f"category {p.name}: {len(tasks)} tasks, {n_total_eps} episodes; "
                 f"status → {status_path}")

        stopped = False
        for task_name, eps in tasks:
            if not eps:
                status.write(f"SKIP   {task_name} (no episodes with data.json)")
                continue
            t_task = time.perf_counter()
            for ep in eps:
                try:
                    encode_episode(ep, encoder, ep / args.out_name)
                except Exception as e:  # noqa: BLE001
                    status.write(f"FAILED {task_name}/{ep.name}: {type(e).__name__}: {e}")
                    status.write("STOPPED — first failure")
                    _log(f"STOPPED — {task_name}/{ep.name} failed: {type(e).__name__}: {e}")
                    overall_ok = False
                    stopped = True
                    break
            if stopped:
                break
            status.write(f"DONE   {task_name} ({len(eps)} episodes, "
                         f"{time.perf_counter()-t_task:.1f}s)")

        if not stopped and kind == "category":
            status.write(f"FINISHED — {p.name}")
        status.close()
        if stopped:
            return 2

    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
