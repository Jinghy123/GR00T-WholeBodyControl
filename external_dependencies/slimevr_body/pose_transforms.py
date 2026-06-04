"""Coordinate transforms for the SlimeVR → GMR retarget path.

Vendored verbatim from the humdex repo (deploy_real/pose_csv_loader.py) so the
SlimeVR teleop server is self-contained and matches the behavior validated in
humdex. Two entry points are used by slimevr_manus_thread_server.py:

    apply_bvh_like_coordinate_transform(frame, pos_unit="m", apply_rotation=True)
        — rotate a {joint: [pos, quat_wxyz]} frame into the BVH/GMR convention.
    gmr_rename_and_footmod(frame, fmt="nokov")
        — apply the BVH loader's rename aliases + FootMod construction so the
          frame is consumable by GeneralMotionRetargeting(src_human="bvh_<fmt>").
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

# BVH → GMR fixed axis correction. Positions: p' = ROT_M @ p. Rotations (wxyz):
# q' = ROT_Q * q (global mode). ROT_M is the same Unity/Y-up → Z-up matrix used
# elsewhere in the pipeline.
BVH_GMR_ROT_M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
BVH_GMR_ROT_Q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in wxyz order."""
    aw, ax, ay, az = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bw, bx, by, bz = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def apply_bvh_like_coordinate_transform(
    frame: Dict[str, Any],
    *,
    pos_unit: str = "m",
    apply_rotation: bool = True,
    rot_mode: str = "global",
    rot_tweak: str = "",
    rot_tweak_order: str = "post",
    apply_pos_rotation: bool = True,
    apply_quat_rotation: bool = True,
) -> Dict[str, Any]:
    """
    Convert a pose frame to the same coordinate convention as our BVH loader output.

    - **pos_unit**: unit of positions ('m'|'cm'|'mm')
    - **apply_rotation**: whether to apply BVH_GMR_ROT_M / BVH_GMR_ROT_Q

    Returns a NEW dict, does not mutate input.
    """
    unit = str(pos_unit).lower().strip()
    if unit == "m":
        s = 1.0
    elif unit == "cm":
        s = 0.01
    elif unit == "mm":
        s = 0.001
    else:
        raise ValueError(f"Invalid pos_unit: {pos_unit} (expected 'm'|'cm'|'mm')")

    out: Dict[str, Any] = {}
    rot_m = BVH_GMR_ROT_M
    rot_q = BVH_GMR_ROT_Q
    mode = str(rot_mode).lower().strip()
    if mode not in ["global", "basis"]:
        raise ValueError(f"Invalid rot_mode: {rot_mode} (expected global|basis)")
    tweak = str(rot_tweak).lower().strip()
    order = str(rot_tweak_order).lower().strip()
    if apply_rotation and tweak and tweak != "none":
        if tweak in ["rx180", "x180"]:
            extra_m = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            extra_q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        elif tweak in ["ry180", "y180"]:
            extra_m = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
            extra_q = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        elif tweak in ["rz180", "z180"]:
            extra_m = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
            extra_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            raise ValueError(f"Invalid rot_tweak: {rot_tweak} (expected none|rx180|ry180|rz180)")
        if order not in ["pre", "post"]:
            raise ValueError(f"Invalid rot_tweak_order: {rot_tweak_order} (expected pre|post)")
        if order == "pre":
            rot_m = rot_m @ extra_m
            rot_q = quat_mul_wxyz(rot_q, extra_q)
        else:
            rot_m = extra_m @ rot_m
            rot_q = quat_mul_wxyz(extra_q, rot_q)
    for name, v in frame.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        pos = np.asarray(v[0], dtype=np.float32).reshape(3) * float(s)
        quat = np.asarray(v[1], dtype=np.float32).reshape(4)
        if apply_rotation:
            if bool(apply_pos_rotation):
                pos = pos @ rot_m.T
            if bool(apply_quat_rotation):
                if mode == "global":
                    quat = quat_mul_wxyz(rot_q, quat)
                else:
                    quat = quat_mul_wxyz(quat_mul_wxyz(rot_q, quat), quat_conj_wxyz(rot_q))
        out[name] = [pos, quat]
    return out


def gmr_rename_and_footmod(frame: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    """
    Apply the same rename aliases and FootMod construction used by our BVH loader,
    but on a frame already containing global pos/quat.
    """
    out = dict(frame)

    rename = {
        "LeftUpperLeg": "LeftUpLeg",
        "RightUpperLeg": "RightUpLeg",
        "LeftLowerLeg": "LeftLeg",
        "RightLowerLeg": "RightLeg",
        "LeftUpperArm": "LeftArm",
        "RightUpperArm": "RightArm",
        "LeftLowerArm": "LeftForeArm",
        "RightLowerArm": "RightForeArm",
    }
    for src, dst in rename.items():
        if src in out and dst not in out:
            out[dst] = out[src]

    # Toe / ToeBase aliases
    if "LeftToe" in out and "LeftToeBase" not in out:
        out["LeftToeBase"] = out["LeftToe"]
    if "RightToe" in out and "RightToeBase" not in out:
        out["RightToeBase"] = out["RightToe"]
    if "LeftToeBase" in out and "LeftToe" not in out:
        out["LeftToe"] = out["LeftToeBase"]
    if "RightToeBase" in out and "RightToe" not in out:
        out["RightToe"] = out["RightToeBase"]

    # FootMod selection: position from Foot, orientation from Toe or ToeBase
    if fmt == "lafan1":
        left_toe = "LeftToe" if "LeftToe" in out else "LeftToeBase"
        right_toe = "RightToe" if "RightToe" in out else "RightToeBase"
    elif fmt == "nokov":
        left_toe = "LeftToeBase" if "LeftToeBase" in out else "LeftToe"
        right_toe = "RightToeBase" if "RightToeBase" in out else "RightToe"
    else:
        raise ValueError(f"Invalid format: {fmt}")

    if "LeftFoot" in out and left_toe in out:
        out["LeftFootMod"] = [out["LeftFoot"][0], out[left_toe][1]]
    if "RightFoot" in out and right_toe in out:
        out["RightFootMod"] = [out["RightFoot"][0], out[right_toe][1]]

    return out
