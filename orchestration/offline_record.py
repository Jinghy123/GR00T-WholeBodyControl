"""Offline token recording for stationary manipulation episodes.

For datasets where the robot never walks/turns (all phase A in walk_then_replay),
token computation has no sim/WBC/planner dependency — it's a pure offline
transform of data.json.

Robot type is auto-detected from frame 0's `robot_type` key:
  * present → G1 path: reuses walk_then_replay.load_full_episode + write_sonic_json.
  * absent  → H1 path:
      - legs   : forced to G1_DEFAULT_ANGLES_MUJOCO[:15]
      - arms   : H1 actions.sol_q (14d, mujoco arm order)
      - quat   : identity
      - hands  : action.left_angles / right_angles via the sonic 1.7-x mapping
                 → 6d per side → pad each with a 0 → 14d (left_7 + right_7)
      - states.hand_joints: H1 states.hand_state (12d right-first) → swap and
                            pad → 14d

Usage:
  python orchestration/offline_record.py /path/to/ep1 /path/to/ep2 [...]
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
    build_data_window, load_full_episode, write_sonic_json,
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
        qpos_30, _h, imu_q_30, _v, sol_dim = load_full_episode(str(episode_dir))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+", help="Episode dirs (each must contain data.json)")
    ap.add_argument("--out-name", type=str, default="data_sonic.json")
    args = ap.parse_args()

    episodes = [Path(p).resolve() for p in args.episodes]
    for ep in episodes:
        if not (ep / "data.json").exists():
            _log(f"no data.json in {ep}")
            return 1

    _log(f"loading encoder {ENCODER_MODEL}")
    encoder = EncoderClient(ENCODER_MODEL, mode=0)

    results = []
    for idx, ep in enumerate(episodes):
        _log(f"[{idx+1}/{len(episodes)}] {ep}")
        try:
            encode_episode(ep, encoder, ep / args.out_name)
            results.append((str(ep), True))
        except Exception as e:  # noqa: BLE001
            _log(f"FAIL {type(e).__name__}: {e}")
            results.append((str(ep), False))

    print()
    _log(f"{sum(1 for _,ok in results if ok)}/{len(results)} succeeded")
    for ep, ok in results:
        print(f"  [{'OK ' if ok else 'FAIL'}] {ep}")
    return 0 if all(ok for _, ok in results) else 2


if __name__ == "__main__":
    sys.exit(main())
