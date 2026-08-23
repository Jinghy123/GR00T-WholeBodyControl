#!/usr/bin/env python3
"""G1 upper-body FK for eepose columns — mirrors psi/scripts/data/raw_he_to_psi0.py.

Semantics (identical to the psi converter, by design):
  - Fixed-base URDF (pelvis at world origin), q starts at pin.neutral — waist
    and legs stay at zero. The output is "upper-body pose in a torso-upright,
    pelvis-origin frame": squatting / turning / waist lean do not affect it.
  - Only arm (7 per side) and hand (7 per side) joints are filled.
  - Wrist frames get a +5cm x offset; fingertip frames get the constant tip
    offsets from OFFSETS_G1.
  - Rotations are returned as rpy via the same matrix_to_rpy convention.

Input action layout: 28 = [left_hand7, right_hand7, left_arm7, right_arm7],
joint order matching LEFT/RIGHT_HAND_JOINTS_G1 and LEFT/RIGHT_ARM_JOINTS_G1
(same as the G1_HAND_NAMES / G1_ARM_NAMES columns in the psix dataset).
"""

from __future__ import annotations

import os

import numpy as np
import pinocchio as pin

# Repo-local URDF copy (assets/README.md) — meshes stripped, so the model must
# be loaded kinematics-only via pin.buildModelFromUrdf (verified identical FK
# to RobotWrapper.BuildFromURDF on the original psi asset).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_G1_URDF = os.path.join(_REPO_ROOT, "assets", "g1", "g1_body29_hand14.urdf")

LEFT_HAND_JOINTS_G1 = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]
RIGHT_HAND_JOINTS_G1 = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
LEFT_ARM_JOINTS_G1 = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS_G1 = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

WAIST_JOINTS_G1 = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

WRIST_FRAMES_G1 = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
HAND_FRAMES_G1 = {
    "left_thumb": "left_hand_thumb_2_link",
    "left_index": "left_hand_index_1_link",
    "left_middle": "left_hand_middle_1_link",
    "right_thumb": "right_hand_thumb_2_link",
    "right_index": "right_hand_index_1_link",
    "right_middle": "right_hand_middle_1_link",
}
OFFSETS_G1 = {
    "left_thumb": np.array([0.0, -0.0458, 0.0]),
    "left_index": np.array([0.0458, 0.0, 0.0]),
    "left_middle": np.array([0.0458, 0.0, 0.0]),
    "right_thumb": np.array([0.0, 0.0458, 0.0]),
    "right_index": np.array([0.0458, 0.0, 0.0]),
    "right_middle": np.array([0.0458, 0.0, 0.0]),
    "left_wrist": np.array([0.05, 0.0, 0.0]),
    "right_wrist": np.array([0.05, 0.0, 0.0]),
}

# eepose column key → (frame name, tip offset)
EEPOSE_TARGETS = {
    "wrists.left": (WRIST_FRAMES_G1["left"], OFFSETS_G1["left_wrist"]),
    "wrists.right": (WRIST_FRAMES_G1["right"], OFFSETS_G1["right_wrist"]),
    "hands.left_thumb": (HAND_FRAMES_G1["left_thumb"], OFFSETS_G1["left_thumb"]),
    "hands.left_index": (HAND_FRAMES_G1["left_index"], OFFSETS_G1["left_index"]),
    "hands.left_middle": (HAND_FRAMES_G1["left_middle"], OFFSETS_G1["left_middle"]),
    "hands.right_thumb": (HAND_FRAMES_G1["right_thumb"], OFFSETS_G1["right_thumb"]),
    "hands.right_index": (HAND_FRAMES_G1["right_index"], OFFSETS_G1["right_index"]),
    "hands.right_middle": (HAND_FRAMES_G1["right_middle"], OFFSETS_G1["right_middle"]),
}


def matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    """Same convention as raw_he_to_psi0.matrix_to_rpy."""
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy < 1e-6:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
    else:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


class G1EeposeFK:
    """FK evaluator: 28D upper-body action → 16 eepose vectors (8 targets × xyz/rpy)."""

    def __init__(self, urdf_path: str = DEFAULT_G1_URDF):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Pre-resolve q indices for the 28 action joints (skip any not in URDF)
        joint_names = (LEFT_HAND_JOINTS_G1 + RIGHT_HAND_JOINTS_G1
                       + LEFT_ARM_JOINTS_G1 + RIGHT_ARM_JOINTS_G1)
        self._q_idx = np.full(28, -1, dtype=np.int64)
        for k, name in enumerate(joint_names):
            jid = self.model.getJointId(name)
            if jid == 0:
                raise ValueError(f"Joint {name!r} not found in {urdf_path}")
            j = self.model.joints[jid]
            if j.nq != 1:
                raise ValueError(f"Joint {name!r} has nq={j.nq}, expected 1")
            self._q_idx[k] = j.idx_q

        self._frame_ids = {}
        for label, (frame_name, _) in EEPOSE_TARGETS.items():
            fid = self.model.getFrameId(frame_name)
            if fid >= self.model.nframes:
                raise ValueError(f"Frame {frame_name!r} not found in {urdf_path}")
            self._frame_ids[label] = fid

        # waist joints (for the waist-included variant; mujoco order yaw/roll/pitch)
        self._waist_q_idx = np.array(
            [self.model.joints[self.model.getJointId(nm)].idx_q for nm in WAIST_JOINTS_G1],
            dtype=np.int64)

        self._q_neutral = pin.neutral(self.model)

    def compute(self, action28: np.ndarray, waist_yrp3: np.ndarray | None = None) -> dict:
        """Single frame: (28,) [+ waist (yaw, roll, pitch)] → {label: {'xyz','rpy'}}.

        With waist_yrp3=None the psi convention applies (waist neutral →
        torso-upright frame). Passing the waist angles yields the true
        pelvis-frame pose including torso lean (Unifolm ee semantics).
        """
        q = self._q_neutral.copy()
        q[self._q_idx] = np.asarray(action28, dtype=float).reshape(28)
        if waist_yrp3 is not None:
            q[self._waist_q_idx] = np.asarray(waist_yrp3, dtype=float).reshape(3)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        out = {}
        for label, (_, offset) in EEPOSE_TARGETS.items():
            oMf = self.data.oMf[self._frame_ids[label]]
            pos = oMf.translation + oMf.rotation @ offset
            out[label] = {"xyz": pos.astype(np.float32),
                          "rpy": matrix_to_rpy(oMf.rotation).astype(np.float32)}
        return out

    def compute_batch(self, actions28: np.ndarray,
                      waist_yrp3: np.ndarray | None = None) -> dict:
        """(n, 28) [+ (n, 3) waist] → {'action.<label>.xyz': (n,3), ...rpy...}."""
        n = len(actions28)
        out = {f"action.{label}.{comp}": np.zeros((n, 3), dtype=np.float32)
               for label in EEPOSE_TARGETS for comp in ("xyz", "rpy")}
        for i in range(n):
            poses = self.compute(actions28[i],
                                 None if waist_yrp3 is None else waist_yrp3[i])
            for label, pose in poses.items():
                out[f"action.{label}.xyz"][i] = pose["xyz"]
                out[f"action.{label}.rpy"][i] = pose["rpy"]
        return out
