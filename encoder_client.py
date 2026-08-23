#!/usr/bin/env python3
"""
encoder_client.py

Standalone encoder wrapper for g1 mode (mode 0).

Interface:
    encoder = EncoderClient("gear_sonic_deploy/policy/release/model_encoder.onnx")
    token = encoder.encode(
        joint_pos,   # (10, 29) float32 — 10 future frames step5
        joint_vel,   # (10, 29) float32 — 10 future frames step5
        body_quat,   # (10,  4) float32 — wxyz, 10 future frames step5
    )
    # token: (64,) float32

anchor_orientation 计算的是 inv(body_quat[0]) * body_quat[i]，即参考帧之间的相对旋转。
当你想冻结在当前状态时，10 帧 body_quat 全部填同一个当前机器人的 quat，
结果是 anchor = identity（无旋转信号）。

Encoder input总共 1762 维（所有 mode 的 observation 拼在一起，不用的 mode 填零）。
g1 mode (mode=0) 只填这几段，其余为零：
    [0:4]      encoder_mode_4                    — [1,0,0,0]
    [4:294]    joint_positions × 10 frames       — 29×10
    [294:584]  joint_velocities × 10 frames      — 29×10
    [601:661]  anchor_orientation × 10 frames    — 6D rotation×10（由 body_quat 算出）
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation
import time

# onnxruntime is imported lazily in EncoderClient.__init__ so that modules which
# only need the ZMQ/wire helpers from g1_sonic_client (replay_sonic.py, for one)
# can be imported on a machine without onnxruntime installed.


# ── Quaternion helpers (wxyz convention) ──────────────────────────────────────

def quat_mul(q1, q2):
    """Hamilton product, both (4,) wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_conjugate(q):
    """Conjugate (= inverse for unit quat), wxyz."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_to_rotation_matrix(q):
    """wxyz → 3×3 rotation matrix via scipy."""
    # scipy uses xyzw
    xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    return Rotation.from_quat(xyzw).as_matrix()


def calc_heading_quat(q_wxyz):
    """Yaw-only quaternion of q, wxyz. Mirrors C++ calc_heading_quat_d:
    heading = atan2 of the rotated x-axis, then rotate about z by heading."""
    R = quat_to_rotation_matrix(q_wxyz)
    h = np.arctan2(R[1, 0], R[0, 0])  # rotated x-axis = first column of R
    return np.array([np.cos(h / 2), 0.0, 0.0, np.sin(h / 2)], dtype=np.float64)


def anchor_orientation_6d(ref_quat_wxyz, robot_quat_wxyz):
    """
    Compute 6D rotation representation of the relative rotation
    from robot current quat to reference quat.

    Mirrors C++ GatherMotionAnchorOrientationMutiFrame (orientation_mode=0):
        diff = inv(robot_quat) * ref_quat
        R    = rotation_matrix(diff)
        out  = [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]  (first 2 cols, row-wise)
    """
    q_ref   = np.asarray(ref_quat_wxyz,   dtype=np.float64)
    q_robot = np.asarray(robot_quat_wxyz, dtype=np.float64)

    diff = quat_mul(quat_conjugate(q_robot), q_ref)
    R    = quat_to_rotation_matrix(diff)

    # First 2 columns, row-wise → 6 values
    return np.array([
        R[0, 0], R[0, 1],
        R[1, 0], R[1, 1],
        R[2, 0], R[2, 1],
    ], dtype=np.float32)


def anchor_orientation_heading_6d(ref_quat_wxyz, robot_quat_wxyz):
    """
    Heading-normalized variant used by sonic v1.1
    (motion_anchor_orientation_heading_*). Mirrors C++
    GatherMotionAnchorOrientationMutiFrame with orientation_mode=1:
        diff = inv(heading_quat(robot_quat)) * ref_quat
    i.e. the left side is the robot's yaw-only heading, not its full quat.
    """
    q_ref   = np.asarray(ref_quat_wxyz,   dtype=np.float64)
    q_robot = np.asarray(robot_quat_wxyz, dtype=np.float64)

    diff = quat_mul(quat_conjugate(calc_heading_quat(q_robot)), q_ref)
    R    = quat_to_rotation_matrix(diff)

    # First 2 columns, row-wise → 6 values
    return np.array([
        R[0, 0], R[0, 1],
        R[1, 0], R[1, 1],
        R[2, 0], R[2, 1],
    ], dtype=np.float32)


# ── Encoder client ────────────────────────────────────────────────────────────

class EncoderClient:
    """
    Runs the g1 encoder ONNX model locally via onnxruntime.

    Args:
        model_path: path to model_encoder.onnx
        mode: encoder mode id — 0=g1(wholebody), 1=teleop(vr3pt), 2=smpl
        version: "v1" (release, 1762-dim obs, full-quat anchor) or
                 "v1_1" (sonic v1.1, 1751-dim obs, heading-normalized anchor)
    """

    NUM_JOINTS   = 29
    NUM_FRAMES   = 10  # 10-frame window, step5
    TOKEN_DIM    = 64
    MODE_VEC_DIM = 4   # encoder_mode_4: 4-dim one-hot

    # Per-version encoder input layout. Offsets are for g1 mode (mode=0);
    # segments not listed are zeros. v1_1 drops the 11 root_z dims
    # (motion_root_z_position_10frame_step5 + motion_root_z_position) and
    # switches the anchor to the heading-normalized variant.
    OBS_LAYOUTS = {
        "v1":   {"obs_dim": 1762, "anchor_offset": 601, "heading": False},
        "v1_1": {"obs_dim": 1751, "anchor_offset": 584, "heading": True},
    }

    def __init__(self, model_path: str, mode: int = 0, version: str = "v1"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Encoder model not found: {model_path}")
        if version not in self.OBS_LAYOUTS:
            raise ValueError(f"Unknown encoder version: {version!r} (expected one of {list(self.OBS_LAYOUTS)})")

        self._mode = mode
        self._version = version
        layout = self.OBS_LAYOUTS[version]
        self._obs_dim = layout["obs_dim"]
        self._anchor_offset = layout["anchor_offset"]
        self._anchor_fn = anchor_orientation_heading_6d if layout["heading"] else anchor_orientation_6d

        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, sess_options=sess_opts)

        # Inspect I/O names at load time
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        input_shape       = self._session.get_inputs()[0].shape

        print(f"[EncoderClient] Loaded: {model_path} (version={version})")
        print(f"[EncoderClient] Input : '{self._input_name}' shape={input_shape}")
        print(f"[EncoderClient] Output: '{self._output_name}'")

        # Sanity check: model input dim must match the selected layout,
        # otherwise the version flag and the checkpoint disagree.
        model_dim = input_shape[-1]
        if isinstance(model_dim, int) and model_dim != self._obs_dim:
            raise ValueError(
                f"Encoder input dim mismatch: model expects {model_dim}, "
                f"layout '{version}' builds {self._obs_dim}. "
                f"Wrong --v1.1 flag / model path combination?")

    # Full encoder input layout, ordered by encoder_observations in yaml.
    #
    # v1 (release, 1762 dims):
    #   [0:4]      encoder_mode_4                          4
    #   [4:294]    motion_joint_positions_10frame_step5  290
    #   [294:584]  motion_joint_velocities_10frame_step5 290
    #   [584:594]  motion_root_z_position_10frame_step5   10  (zeros for g1)
    #   [594:595]  motion_root_z_position                  1  (zeros for g1)
    #   [595:601]  motion_anchor_orientation               6  (zeros for g1)
    #   [601:661]  motion_anchor_orientation_10frame_step5 60  ← body_quat→6D
    #   [661:781]  motion_joint_positions_lowerbody...    120  (zeros for g1)
    #   [781:901]  motion_joint_velocities_lowerbody...   120  (zeros for g1)
    #   [901:910]  vr_3point_local_target                  9  (zeros for g1)
    #   [910:922]  vr_3point_local_orn_target             12  (zeros for g1)
    #   [922:1642] smpl_joints_10frame_step1             720  (zeros for g1)
    #   [1642:1702] smpl_anchor_orientation_10frame_step1 60  (zeros for g1)
    #   [1702:1762] motion_joint_positions_wrists_10frame_step1 60 (zeros for g1)
    #
    # v1_1 (sonic v1.1, 1751 dims): same order minus the 11 root_z dims, and
    # anchor orientations are heading-normalized:
    #   [0:4]      encoder_mode_4                          4
    #   [4:294]    motion_joint_positions_10frame_step5  290
    #   [294:584]  motion_joint_velocities_10frame_step5 290
    #   [584:644]  motion_anchor_orientation_heading_10frame_step5 60  ← body_quat→6D
    #   [644:650]  motion_anchor_orientation_heading       6  (zeros for g1)
    #   [650:1751] lowerbody / vr / smpl / wrists            (zeros for g1)

    def _build_obs(
        self,
        joint_pos: np.ndarray,  # (10, 29)
        joint_vel: np.ndarray,  # (10, 29)
        body_quat: np.ndarray,  # (10,  4) wxyz
    ) -> np.ndarray:
        """Build the full encoder observation vector for the selected version."""

        joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(self.NUM_FRAMES, self.NUM_JOINTS)
        joint_vel = np.asarray(joint_vel, dtype=np.float32).reshape(self.NUM_FRAMES, self.NUM_JOINTS)
        body_quat = np.asarray(body_quat, dtype=np.float32).reshape(self.NUM_FRAMES, 4)

        # robot_quat = body_quat[0]（第0帧是当前参考帧，anchor算相对它的旋转）
        robot_quat = body_quat[0]

        obs = np.zeros(self._obs_dim, dtype=np.float32)

        # [0:4] encoder_mode_4: first dim = mode_id as float, rest zero
        # g1=0, teleop=1, smpl=2 (NOT one-hot — matches C++ GatherEncoderMode)
        obs[0] = float(self._mode)

        # [4:294] joint positions
        obs[4:294] = joint_pos.flatten()

        # [294:584] joint velocities
        obs[294:584] = joint_vel.flatten()

        # anchor orientation × 10 frames → 6D rotation each.
        # v1: [601:661] relative to full robot quat; v1_1: [584:644] relative
        # to the robot's yaw-only heading.
        off = self._anchor_offset
        for i in range(self.NUM_FRAMES):
            obs[off + i * 6 : off + i * 6 + 6] = self._anchor_fn(body_quat[i], robot_quat)

        # remaining segments (lowerbody / vr / smpl / wrists) → zeros for g1 mode

        return obs

    def encode(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        body_quat: np.ndarray,
    ) -> np.ndarray:
        """
        Run encoder inference.

        Args:
            joint_pos: (10, 29) float32 — 10 帧关节角（step5）
            joint_vel: (10, 29) float32 — 10 帧关节速度
            body_quat: (10,  4) float32 — wxyz，10 帧参考 body quat
                       冻结当前状态时：10 帧全填同一个当前机器人的 base_quat_measured

        Returns:
            token: (64,) float32
        """
        obs = self._build_obs(joint_pos, joint_vel, body_quat)
        obs_batch = obs.reshape(1, -1).astype(np.float32)  # (1, 644)

        outputs = self._session.run(
            [self._output_name],
            {self._input_name: obs_batch},
        )
        token = np.array(outputs[0], dtype=np.float32).flatten()  # (64,)
        return token


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODEL = "gear_sonic_deploy/policy/release/model_encoder.onnx"

    encoder = EncoderClient(MODEL, mode=0)

    start = time.time()

    # 模拟"冻结在当前状态"：10 帧全填同一个 quat 和 joint_pos
    q_now     = np.array([1., 0., 0., 0.], dtype=np.float32)  # 当前机器人 base_quat_measured
    joint_pos = np.zeros((10, 29), dtype=np.float32)           # 当前 qpos，复制 10 帧
    joint_vel = np.zeros((10, 29), dtype=np.float32)           # 零速度
    body_quat = np.tile(q_now, (10, 1)).astype(np.float32)     # 10 帧全一样

    end = time.time()
    print(f"Build obs time: {(end - start)*1000:.2f} ms")

    token = encoder.encode(joint_pos, joint_vel, body_quat)
    print(f"Token shape: {token.shape}")
