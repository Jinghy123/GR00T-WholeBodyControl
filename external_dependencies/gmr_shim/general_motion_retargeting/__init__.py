"""Minimal stand-in for the upstream `general_motion_retargeting` package.

Provides only the two symbols `pose_publisher.py` needs:

    XRobotStreamer            — pulls the 22-joint full-body pose that the
                                XRoboToolKit daemon already infers (headset +
                                whatever trackers are paired) and packages it
                                as a {joint_name: (pos_xyz, quat_wxyz)} dict.
    human_head_to_robot_neck  — extracts (yaw, pitch) of the head expressed
                                in the Spine3 frame, decoupling head rotation
                                from torso lean.

Quaternions stay in the daemon's Unity-aligned frame (X right, Y up, Z fwd);
the YXZ Euler decomposition picks yaw around vertical Y and pitch around
lateral X, which matches the robot's 2-DOF neck conventions in
`realsense_server.py` (where `NECK_YAW_SIGN` / `NECK_PITCH_SIGN` already
exist as flip switches).

Usage (point Python at this shim and the XRoboToolKit binding):

    export GR00T_ROOT=/path/to/GR00T-WholeBodyControl
    export PYTHONPATH=$GR00T_ROOT/external_dependencies/gmr_shim:\\
$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64
    export LD_LIBRARY_PATH=\\
$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64
    python pose_publisher.py --bind tcp://0.0.0.0:5559 --hz 50

Prereq: the XRoboToolKit PC Service daemon must already be running
(`sudo bash /opt/apps/roboticsservice/runService.sh`) and producing body
data — `xrt.is_body_data_available()` must return True before
`XRobotStreamer.get_current_frame()` will yield a non-None payload.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import xrobotoolkit_sdk as xrt
except ImportError as e:
    raise ImportError(
        "xrobotoolkit_sdk not importable. Set PYTHONPATH and LD_LIBRARY_PATH "
        "to external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64."
    ) from e


_XROBOT_JOINT_NAMES = (
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head",
    "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist",
)


SmplxData = Dict[str, Tuple[np.ndarray, np.ndarray]]


class XRobotStreamer:
    def __init__(self) -> None:
        xrt.init()
        self._initialized = True

    def get_current_frame(self) -> Tuple[Optional[SmplxData], int]:
        if not xrt.is_body_data_available():
            return None, 0

        ts = int(xrt.get_body_timestamp_ns() or 0)
        body = xrt.get_body_joints_pose()
        if body is None:
            return None, ts

        body = np.asarray(body, dtype=np.float64)
        if body.ndim != 2 or body.shape[1] < 7 or body.shape[0] < len(_XROBOT_JOINT_NAMES):
            return None, ts

        out: SmplxData = {}
        for i, name in enumerate(_XROBOT_JOINT_NAMES):
            pos = body[i, :3].copy()
            qxyzw = body[i, 3:7].copy()
            n = float(np.linalg.norm(qxyzw))
            if not np.isfinite(n) or n < 1e-6:
                return None, ts
            qxyzw /= n
            qwxyz = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])
            out[name] = (pos, qwxyz)
        return out, ts

    def close(self) -> None:
        if getattr(self, "_initialized", False):
            try:
                xrt.close()
            except Exception:
                pass
            self._initialized = False


def _R_from_wxyz(qwxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]])


def human_head_to_robot_neck(smplx_data: SmplxData) -> Tuple[float, float]:
    head_q = smplx_data["Head"][1]
    spine_q = smplx_data["Spine3"][1]
    R_rel = _R_from_wxyz(spine_q).inv() * _R_from_wxyz(head_q)
    yaw, pitch, _roll = R_rel.as_euler("yxz", degrees=False)
    return float(yaw), float(pitch)


__all__ = ["XRobotStreamer", "human_head_to_robot_neck"]
