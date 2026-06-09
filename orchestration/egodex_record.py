"""Offline sonic-token recording for EgoDex-retargeted G1 episodes.

Source is an NPZ produced by the egodex_retargeting repo
(`scripts/retarget/g1_dex31.py`): G1 torso/arms + Dex3.1 hands. The upper body
(arms + hands) is real retargeted motion; the lower body is forced to a stable
standing pose — exactly mirroring the H1 path in offline_record.py.

Per episode:
  * legs  : forced to G1_DEFAULT_ANGLES_MUJOCO[:15]  (stable standing)
  * arms  : NPZ g1_qpos arm columns, reordered (by joint name) to the 14-d
            mujoco arm order
  * quat  : identity [1,0,0,0]
  * hands : NPZ left_hand_qpos(7) + right_hand_qpos(7), reordered (by joint
            name) to the 14-d deploy order (left_7, right_7). NOTE the right
            hand uses index-before-middle, unlike the left hand.

Token computation is identical to offline_record / walk_then_replay: build the
29-d mujoco qpos (legs + arms), reorder to isaaclab, resample 30->50Hz, run the
encoder window + FSQ quantize, then map the 50Hz tokens back to 30Hz frames.
Hands and quat are pass-through — they never enter the encoder, so the stable
legs + identity quat fully determine the token (arms drive all the motion).

Output is the standard G1 sonic JSON (states.qpos/quat/hand_joints,
actions.hand_joints/token), one file per source NPZ.

Usage:
  python orchestration/egodex_record.py outputs/result.npz
  python orchestration/egodex_record.py outputs/result.npz --out-path foo_sonic.json
  python orchestration/egodex_record.py /dir/of/npzs        # all *.npz in dir
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from encoder_client import EncoderClient  # noqa: E402
from walk_forward_token import (  # noqa: E402
    ENCODER_MODEL, G1_DEFAULT_ANGLES_MUJOCO, _MUJOCO_TO_ISAACLAB_DOF,
    fsq_quantize, resample_30hz_to_50hz,
)
from walk_then_replay import build_data_window  # noqa: E402


G1_STABLE_LEG_MUJOCO = G1_DEFAULT_ANGLES_MUJOCO[:15].copy()

# Waist joints inside the 15-d mujoco leg block (Unitree G1 29-dof order):
#   idx 12 = waist_yaw, idx 13 = waist_roll, idx 14 = waist_pitch.
WAIST_YAW_IDX, WAIST_ROLL_IDX, WAIST_PITCH_IDX = 12, 13, 14

# 14-d mujoco arm order — matches G1_DEFAULT_ANGLES_MUJOCO[15:29] (left 7, right 7).
MUJOCO_ARM_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# 14-d deploy hand order (left_7, right_7). The RIGHT hand is index-before-middle,
# unlike the left hand — this asymmetry is the deploy convention, so we reorder
# the NPZ hand columns by name rather than concatenating raw.
DEPLOY_HAND_JOINTS = (
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
)


def _log(msg: str) -> None:
    print(f"[egodex] {msg}", flush=True)


# ── NPZ loader ───────────────────────────────────────────────────────────────

def load_egodex_npz(npz_path: Path):
    """Returns (arms_30 [n,14] mujoco arm order, hands_30 [n,14] deploy order,
    T_root [n,4,4] torso world pose).

    Arms/hands are pulled out of the NPZ by joint name, so column order in the
    NPZ does not matter — only the joint names must be present.
    """
    d = np.load(npz_path, allow_pickle=True)

    # --- arms: from g1_qpos, indexed by g1_joint_order ---
    g1_order = [str(x) for x in d["g1_joint_order"]]
    g1_idx = {n: i for i, n in enumerate(g1_order)}
    g1_qpos = np.asarray(d["g1_qpos"], dtype=np.float32)
    n = g1_qpos.shape[0]

    arms_30 = np.zeros((n, 14), dtype=np.float32)
    for k, jn in enumerate(MUJOCO_ARM_JOINTS):
        if jn not in g1_idx:
            raise ValueError(f"{npz_path.name}: g1_joint_order missing arm joint '{jn}'")
        arms_30[:, k] = g1_qpos[:, g1_idx[jn]]

    # --- hands: build a name -> per-frame-column lookup spanning both hands ---
    hand_cols: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        qpos = np.asarray(d[f"{side}_hand_qpos"], dtype=np.float32).reshape(n, -1)
        order = [str(x) for x in d[f"{side}_hand_joint_order"]]
        if qpos.shape[1] != len(order):
            raise ValueError(
                f"{npz_path.name}: {side}_hand_qpos has {qpos.shape[1]} cols "
                f"but {len(order)} joint names")
        for j, jn in enumerate(order):
            hand_cols[jn] = qpos[:, j]

    hands_30 = np.zeros((n, 14), dtype=np.float32)
    for k, jn in enumerate(DEPLOY_HAND_JOINTS):
        if jn not in hand_cols:
            raise ValueError(f"{npz_path.name}: hand joint '{jn}' not found in NPZ")
        hands_30[:, k] = hand_cols[jn]

    T_root = np.asarray(d["g1_T_w_root"], dtype=np.float32).reshape(n, 4, 4)
    return arms_30, hands_30, T_root


# ── waist from torso ─────────────────────────────────────────────────────────

def build_legs_30(T_root: np.ndarray, *, waist_from_torso: bool, baseline: str,
                  yaw_sign: float, roll_sign: float,
                  pitch_from_torso: bool, pitch_sign: float) -> np.ndarray:
    """Build the per-frame 15-d mujoco leg block.

    Legs are always the stable standing pose. When waist_from_torso is set, the
    waist yaw/roll come from the retargeted torso orientation; waist pitch is
    forced to 0 unless pitch_from_torso is also set, in which case it uses the
    torso pitch too. Without waist_from_torso the waist stays at the stable 0.

    The torso world rotation carries a constant frame-alignment offset (EgoDex
    world != robot world), so absolute yaw/roll/pitch are meaningless for the
    waist. We therefore decompose the rotation *relative to a baseline* (default
    the first frame, consistent with the identity base quat) as intrinsic ZXY =
    (yaw, roll, pitch). Note that the relative pitch is typically small — the
    bulk of the torso pitch lives in the constant offset that the baseline
    removes.
    """
    n = len(T_root)
    legs = np.tile(G1_STABLE_LEG_MUJOCO, (n, 1)).astype(np.float32)
    if not waist_from_torso:
        return legs

    Rm = R.from_matrix(T_root[:, :3, :3])
    if baseline == "first":
        R_ref = Rm[0]
    elif baseline == "mean":
        R_ref = Rm.mean()
    elif baseline == "none":
        R_ref = R.identity()
    else:
        raise ValueError(f"unknown waist baseline '{baseline}'")
    rel = (R_ref.inv() * Rm).as_euler("ZXY")  # (n, 3) = (yaw, roll, pitch)

    legs[:, WAIST_YAW_IDX] = (yaw_sign * rel[:, 0]).astype(np.float32)
    legs[:, WAIST_ROLL_IDX] = (roll_sign * rel[:, 1]).astype(np.float32)
    legs[:, WAIST_PITCH_IDX] = (pitch_sign * rel[:, 2]).astype(np.float32) if pitch_from_torso else 0.0
    return legs


def build_encoder_inputs(legs_30: np.ndarray, arms_30: np.ndarray):
    """29-d isaaclab qpos (legs+waist + arms) + identity imu quat, for the encoder."""
    n = len(arms_30)
    qpos_mj = np.zeros((n, 29), dtype=np.float32)
    qpos_mj[:, :15] = legs_30
    qpos_mj[:, 15:] = arms_30
    qpos_isaac = qpos_mj[:, _MUJOCO_TO_ISAACLAB_DOF].astype(np.float32)
    imu_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    return qpos_isaac, imu_quat


# ── sonic writer ───────────────────────────────────────────────────────────────

def write_sonic_json_egodex(out_path: Path, legs_30, arms_30, hands_30, tokens_30, fps: float) -> None:
    """Per-frame sonic JSON (G1 format):
        states.qpos        = legs+waist (15) + arms (14) = 29  (mujoco order)
        states.quat        = identity [1,0,0,0]
        states.hand_joints = deploy-ordered hands (14)
        actions.hand_joints = same hands (14)
        actions.token       = computed 64-d FSQ token
        image               = "" (no image in NPZ)
        timestamp           = frame_index / fps in ns
    """
    n = len(tokens_30)
    if not (len(legs_30) == len(arms_30) == len(hands_30) == n):
        raise ValueError(
            f"length mismatch: legs={len(legs_30)} arms={len(arms_30)} "
            f"hands={len(hands_30)} tokens={n}")

    out = []
    for i in range(n):
        hands = hands_30[i].tolist()
        frame_out = {
            "states": {
                "qpos":        legs_30[i].tolist() + arms_30[i].tolist(),
                "quat":        [1.0, 0.0, 0.0, 0.0],
                "hand_joints": hands,
            },
            "actions": {
                "hand_joints": hands,
                "token":       tokens_30[i].tolist(),
            },
            "image": "",
            "timestamp": int(i / fps * 1e9),
        }
        out.append(frame_out)

    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(out, f)
    tmp_path.replace(out_path)
    _log(f"wrote {n} frames -> {out_path}")


# ── encode one NPZ ───────────────────────────────────────────────────────────

def encode_npz(npz_path: Path, encoder: EncoderClient, out_path: Path, fps: float,
               *, waist_from_torso: bool, waist_baseline: str,
               waist_yaw_sign: float, waist_roll_sign: float,
               waist_pitch_from_torso: bool, waist_pitch_sign: float) -> None:
    arms_30, hands_30, T_root = load_egodex_npz(npz_path)
    legs_30 = build_legs_30(
        T_root, waist_from_torso=waist_from_torso, baseline=waist_baseline,
        yaw_sign=waist_yaw_sign, roll_sign=waist_roll_sign,
        pitch_from_torso=waist_pitch_from_torso, pitch_sign=waist_pitch_sign,
    )
    qpos_30, imu_q_30 = build_encoder_inputs(legs_30, arms_30)

    n_30 = len(qpos_30)
    qpos_50 = resample_30hz_to_50hz(qpos_30)
    quat_50 = resample_30hz_to_50hz(imu_q_30)
    quat_50 /= np.linalg.norm(quat_50, axis=1, keepdims=True).clip(1e-8)
    n_50 = len(qpos_50)
    if waist_from_torso:
        wy, wr, wp = (legs_30[:, WAIST_YAW_IDX], legs_30[:, WAIST_ROLL_IDX],
                      legs_30[:, WAIST_PITCH_IDX])
        pitch_tag = (f"pitch[{np.degrees(wp.min()):.1f},{np.degrees(wp.max()):.1f}]deg"
                     if waist_pitch_from_torso else "pitch=0")
        waist_tag = (f"waist yaw[{np.degrees(wy.min()):.1f},{np.degrees(wy.max()):.1f}]deg "
                     f"roll[{np.degrees(wr.min()):.1f},{np.degrees(wr.max()):.1f}]deg {pitch_tag}")
    else:
        waist_tag = "waist=stable"
    _log(f"{npz_path.name}: stable-leg + egodex arms + dex3.1 hands  "
         f"n_30={n_30} n_50={n_50}  {waist_tag}")

    tokens_50 = np.zeros((n_50, 64), dtype=np.float32)
    t0 = time.perf_counter()
    for pub_frame in range(n_50):
        jp, jv, bq = build_data_window(qpos_50, quat_50, pub_frame)
        tok = fsq_quantize(encoder.encode(joint_pos=jp, joint_vel=jv, body_quat=bq))
        tokens_50[pub_frame] = tok.astype(np.float32)

    idx_30_to_50 = np.round(np.arange(n_30) * 5.0 / 3.0).astype(int).clip(0, n_50 - 1)
    write_sonic_json_egodex(out_path, legs_30, arms_30, hands_30, tokens_50[idx_30_to_50], fps)
    _log(f"{npz_path.name}: done ({time.perf_counter()-t0:.1f}s, {n_50} frames)")


# ── input walking (npz file / dir auto-detect) ───────────────────────────────

def _collect_npzs(path: Path) -> list[tuple[Path, Path]]:
    """Returns a list of (npz_path, default_out_path) for a file or directory."""
    if path.is_file() and path.suffix == ".npz":
        return [(path, path.with_name(f"{path.stem}_sonic.json"))]
    if path.is_dir():
        npzs = sorted(path.glob("*.npz"))
        return [(p, p.with_name(f"{p.stem}_sonic.json")) for p in npzs]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="NPZ file(s) or directory(ies) of *.npz")
    ap.add_argument("--out-path", type=str, default=None,
                    help="Explicit output path (only valid for a single NPZ input)")
    ap.add_argument("--fps", type=float, default=30.0, help="Source frame rate (EgoDex = 30)")
    ap.add_argument("--waist-from-torso", action=argparse.BooleanOptionalAction, default=True,
                    help="Drive waist yaw/roll from the torso orientation (pitch forced 0). "
                         "Use --no-waist-from-torso to keep the waist at stable 0.")
    ap.add_argument("--waist-baseline", choices=("first", "mean", "none"), default="first",
                    help="Reference subtracted before decomposing torso rotation into waist "
                         "yaw/roll. 'first' = relative to frame 0 (recommended); 'none' = "
                         "absolute (carries the EgoDex frame offset — usually wrong).")
    ap.add_argument("--waist-yaw-sign", type=float, default=1.0, help="Flip to -1 if waist yaw is mirrored")
    ap.add_argument("--waist-roll-sign", type=float, default=1.0, help="Flip to -1 if waist roll is mirrored")
    ap.add_argument("--waist-pitch-from-torso", action=argparse.BooleanOptionalAction, default=False,
                    help="Also drive waist pitch from the torso pitch (default: pitch forced 0). "
                         "Relative pitch is usually small — the constant offset is baseline-removed.")
    ap.add_argument("--waist-pitch-sign", type=float, default=1.0, help="Flip to -1 if waist pitch is mirrored")
    args = ap.parse_args()

    jobs: list[tuple[Path, Path]] = []
    for p in (Path(x).resolve() for x in args.paths):
        found = _collect_npzs(p)
        if not found:
            _log(f"no NPZ found at: {p}")
            return 1
        jobs.extend(found)

    if args.out_path is not None:
        if len(jobs) != 1:
            _log("--out-path is only valid when the input resolves to a single NPZ")
            return 1
        jobs = [(jobs[0][0], Path(args.out_path).resolve())]

    _log(f"loading encoder {ENCODER_MODEL}")
    encoder = EncoderClient(ENCODER_MODEL, mode=0)

    ok = True
    for npz_path, out_path in jobs:
        try:
            encode_npz(
                npz_path, encoder, out_path, args.fps,
                waist_from_torso=args.waist_from_torso,
                waist_baseline=args.waist_baseline,
                waist_yaw_sign=args.waist_yaw_sign,
                waist_roll_sign=args.waist_roll_sign,
                waist_pitch_from_torso=args.waist_pitch_from_torso,
                waist_pitch_sign=args.waist_pitch_sign,
            )
        except Exception as e:  # noqa: BLE001
            _log(f"FAILED {npz_path}: {type(e).__name__}: {e}")
            ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
