#!/usr/bin/env python3
"""
he_to_sonic.py — convert He-format episode (data.json) to sonic-format
data_sonic.json with per-frame 64-d FSQ-quantized tokens aligned to the
original 30Hz frames.

Pipeline (offline, fully self-contained — no WBC, no ZMQ, no threading):
  1. Load 30Hz data (qpos / quat / hands / body_vel)
       qpos source by actions.sol_q dim:
           14 → arm-from-action, leg-from-states
           29 → full-body-from-action
           else → full-body-from-states
       hands source: actions.{left_angles,right_angles} if present, else
                     states.hand_state
  2. Resample to 50Hz (linear; quat normalized post-lerp)
  3. Auto-switch each 50Hz tick:
       A (idle):   encoder window built directly from data → encode
       B (walk):   planner walks lower body @ data body-vel direction;
                   arms overridden from data; encode
       C (turn):   planner turns in place toward data yaw delta;
                   arms overridden from data; encode
       Phase decision uses smoothed |body_vel_xy| and |yaw_rate| with
       hysteresis + min-dwell (same constants as walk_then_replay.py).
  4. Token at every 50Hz tick → tokens_50 (n_50, 64)
  5. Map to 30Hz: tokens_30[i] = tokens_50[round(i * 5/3)]
  6. Write data_sonic.json (sonic format, per-frame structure):
       { timestamp, states:{qpos, quat, hand_joints},
                    actions:{hand_joints, token}, image }

Usage:
    python data_transforms/he_to_sonic.py <episode_dir>
    python data_transforms/he_to_sonic.py <episode_dir> --out custom.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation


# ── USER CONFIG ───────────────────────────────────────────────────────────────
# Default episode directory — used when the script is run without a CLI arg.
# CLI positional arg overrides this when given.
# EPISODE_DIR = "/home/xiawei/data/walk_towards_a_desk_and_pick_up_the_boiling_pot_from_its_base/episode_0"
EPISODE_DIR = "/home/xiawei/hongyi/Unitree_Robotics/Humanoid-Teleop/teleop/data/g1_1001/Basic/Pick_bottle_and_turn_and_pour_into_cup/episode_15"
OUT_NAME    = "data_sonic.json"   # written into <episode_dir>/<OUT_NAME>
# ──────────────────────────────────────────────────────────────────────────────


# ── Paths ─────────────────────────────────────────────────────────────────────

_GROOT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODER_MODEL = os.path.join(_GROOT_ROOT, "gear_sonic_deploy/policy/release/model_encoder.onnx")
PLANNER_MODEL = os.path.join(_GROOT_ROOT, "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx")


# ── Constants ─────────────────────────────────────────────────────────────────

CONTROL_HZ              = 50
PLANNER_HZ              = 10
REPLAN_INTERVAL_50      = CONTROL_HZ // PLANNER_HZ      # 5
MOTION_LOOK_AHEAD_50HZ  = 2
ENCODER_NUM_FRAMES      = 10
ENCODER_STEP_50HZ       = 5
NUM_JOINTS              = 29

# Phase decision (mirrors walk_then_replay.py)
ENTER_THR        = 0.10                  # m/s walk-enter (max(|vx|,|vy|))
EXIT_THR         = 0.05                  # m/s walk-exit
TURN_RATE_ENTER  = math.radians(8.0)     # rad/s turn-enter
TURN_RATE_EXIT   = math.radians(4.0)     # rad/s turn-exit
SMOOTH_SECS      = 2.0                   # smoothing window for body-vel
YAW_SMOOTH_SECS  = 1.0                   # smoothing window for yaw before gradient
MIN_DWELL_SECS   = 0.2

# Planner inputs
HEIGHT           = 0.788740
PLANNER_MODE     = 1                     # SLOW_WALK
WALK_SPEED_BY_SOL_DIM = {14: 0.3, 29: 0.2}
WALK_SPEED_DEFAULT    = 0.3

# FSQ codebook
FSQ_MIN, FSQ_MAX, FSQ_STEP = -0.625, 0.625, 0.0625

# Mujoco → IsaacLab 29-DOF reorder
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)

# Slot positions of 14 arm joints inside the 29-d isaac vector
ARM_ISAAC_INDICES = np.array(
    [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], dtype=np.int32,
)


# ── FSQ + small helpers ───────────────────────────────────────────────────────

def fsq_quantize(x):
    return np.clip(np.round(np.clip(x, FSQ_MIN, FSQ_MAX) / FSQ_STEP) * FSQ_STEP,
                   FSQ_MIN, FSQ_MAX)


def quat_wxyz_to_yaw(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z)))


def quat_slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return (r / np.linalg.norm(r)).astype(np.float32)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / math.sin(theta_0)
    s1 = math.sin(theta) / math.sin(theta_0)
    return (s0 * q0 + s1 * q1).astype(np.float32)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_to_matrix(q_wxyz):
    xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(xyzw).as_matrix()


def anchor_orientation_6d(ref_quat_wxyz, robot_quat_wxyz):
    """6D rep of inv(robot_quat) * ref_quat (mirrors deploy gather logic)."""
    diff = quat_mul(quat_conjugate(np.asarray(robot_quat_wxyz, dtype=np.float64)),
                    np.asarray(ref_quat_wxyz, dtype=np.float64))
    R = quat_to_matrix(diff)
    return np.array([R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]],
                    dtype=np.float32)


# ── Resample 30Hz → 50Hz ──────────────────────────────────────────────────────

def resample_30hz_to_50hz(arr_30):
    """Linear interp (n, ...) at 30Hz → (n_50, ...) at 50Hz."""
    n = len(arr_30)
    n_50 = int(np.floor(n * 50.0 / 30.0))
    out = np.zeros((n_50,) + arr_30.shape[1:], dtype=np.float32)
    for f_50 in range(n_50):
        f_30 = f_50 * 30.0 / 50.0
        f0 = max(0, min(int(np.floor(f_30)), n - 1))
        f1 = min(f0 + 1, n - 1)
        w1 = f_30 - f0
        out[f_50] = (1.0 - w1) * arr_30[f0] + w1 * arr_30[f1]
    return out


def resample_pred_30hz_to_50hz(pred_30hz):
    """Planner output (T_30, 36) mujoco-order → (pos, quat, joints_isaac) @ 50Hz.
    Mirrors deploy ResampleGeneratedSequence50Hz."""
    T_30 = len(pred_30hz)
    T_50 = max(1, int(np.floor(T_30 / 30.0 * 50)))

    pos = np.zeros((T_50, 3), dtype=np.float32)
    quat = np.zeros((T_50, 4), dtype=np.float32)
    joints = np.zeros((T_50, 29), dtype=np.float32)

    for f_50 in range(T_50):
        f_30 = f_50 * 30.0 / 50.0
        f0 = max(0, min(int(np.floor(f_30)), T_30 - 1))
        f1 = min(f0 + 1, T_30 - 1)
        w1 = max(0.0, min(1.0, f_30 - f0))
        w0 = 1.0 - w1
        pos[f_50] = w0 * pred_30hz[f0, 0:3] + w1 * pred_30hz[f1, 0:3]
        quat[f_50] = (pred_30hz[f0, 3:7] if f0 == f1
                      else quat_slerp(pred_30hz[f0, 3:7], pred_30hz[f1, 3:7], w1))
        joints_mj = w0 * pred_30hz[f0, 7:36] + w1 * pred_30hz[f1, 7:36]
        joints[f_50] = joints_mj[_MUJOCO_TO_ISAACLAB_DOF]
    return pos, quat, joints


# ── Phase-decision helpers ────────────────────────────────────────────────────

def smooth_1d(x, win):
    win = max(1, int(win) | 1)
    half = win // 2
    pad = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    return np.convolve(pad, np.ones(win, dtype=np.float32) / win,
                       mode="valid").astype(np.float32)


def hysteresis_mask(metric, enter_thr, exit_thr, min_dwell):
    out = np.zeros(len(metric), dtype=bool)
    state, dwell = False, 0
    for i, m in enumerate(metric):
        crossed = (m > enter_thr) if not state else (m < exit_thr)
        dwell = dwell + 1 if crossed else 0
        if dwell >= min_dwell:
            state, dwell = (not state), 0
        out[i] = state
    return out


def scale_velocity_xy(v_xy, walk_speed):
    """Dominant local-vel axis → ±walk_speed, the other axis proportional."""
    m = float(max(abs(v_xy[0]), abs(v_xy[1])))
    if m < 1e-6:
        return walk_speed, np.array([1.0, 0.0], dtype=np.float32)
    v_scaled = (v_xy * (walk_speed / m)).astype(np.float32)
    speed = float(np.linalg.norm(v_scaled))
    return speed, (v_scaled / speed).astype(np.float32)


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder:
    """g1-mode (mode_id=0) encoder ONNX wrapper. Builds the 1762-d obs vector
    from (10, 29) joint_pos/vel and (10, 4) wxyz body_quat."""

    OBS_DIM = 1762
    TOKEN_DIM = 64

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, sess_options=opts)
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        print(f"[Encoder] loaded {model_path}")

    def encode(self, joint_pos, joint_vel, body_quat):
        """joint_pos/vel: (10,29) float32, body_quat: (10,4) wxyz → (64,) token."""
        jp = np.asarray(joint_pos, dtype=np.float32).reshape(ENCODER_NUM_FRAMES, NUM_JOINTS)
        jv = np.asarray(joint_vel, dtype=np.float32).reshape(ENCODER_NUM_FRAMES, NUM_JOINTS)
        bq = np.asarray(body_quat, dtype=np.float32).reshape(ENCODER_NUM_FRAMES, 4)

        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        obs[0] = 0.0                                    # mode_id = 0 (g1)
        obs[4:294]   = jp.flatten()
        obs[294:584] = jv.flatten()
        # [601:661] anchor_orientation_10frame_step5: 6D rotation × 10
        anchor = bq[0]
        for i in range(ENCODER_NUM_FRAMES):
            obs[601 + i*6 : 601 + i*6 + 6] = anchor_orientation_6d(bq[i], anchor)

        out = self.sess.run([self.out_name], {self.in_name: obs.reshape(1, -1)})
        return np.asarray(out[0], dtype=np.float32).flatten()


# ── Planner ───────────────────────────────────────────────────────────────────

class Planner:
    IDLE, SLOW_WALK, WALK, RUN = 0, 1, 2, 3

    def __init__(self, model_path, default_height=HEIGHT):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, sess_options=opts)
        self.input_names  = [i.name for i in self.sess.get_inputs()]
        self.input_shapes = {i.name: tuple(i.shape) for i in self.sess.get_inputs()}
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.default_height = default_height
        self.context = None  # (4, 36) mujoco-order
        print(f"[Planner] loaded {model_path}")

    @staticmethod
    def _resolve(shape):
        return [d if isinstance(d, int) and d > 0 else 1 for d in shape]

    def initialize_context(self, base_quat_wxyz, qpos_mujoco29):
        frame = np.zeros(36, dtype=np.float32)
        frame[2]    = self.default_height
        frame[3:7]  = np.asarray(base_quat_wxyz, dtype=np.float32).reshape(4)
        frame[7:36] = np.asarray(qpos_mujoco29, dtype=np.float32).reshape(29)
        self.context = np.tile(frame[None, :], (4, 1))

    def set_context(self, ctx_4x36):
        self.context = np.asarray(ctx_4x36, dtype=np.float32).reshape(4, 36)

    def step(self, mode, target_vel, movement_dir, facing_dir, height=None):
        if height is None:
            height = self.default_height
        feed = {}
        for name in self.input_names:
            shape = self._resolve(self.input_shapes[name])
            if name == "context_mujoco_qpos":
                feed[name] = self.context.reshape(shape).astype(np.float32)
            elif name == "mode":
                feed[name] = np.array([int(mode)], dtype=np.int64).reshape(shape)
            elif name == "target_vel":
                feed[name] = np.array([float(target_vel)], dtype=np.float32).reshape(shape)
            elif name == "movement_direction":
                feed[name] = np.asarray(movement_dir, dtype=np.float32).reshape(shape)
            elif name == "facing_direction":
                feed[name] = np.asarray(facing_dir, dtype=np.float32).reshape(shape)
            elif name == "random_seed":
                feed[name] = np.zeros(shape, dtype=np.int64)
            elif name == "height":
                feed[name] = np.array([float(height)], dtype=np.float32).reshape(shape)
            elif name == "has_specific_target":
                feed[name] = np.zeros(shape, dtype=np.int64)
            elif name == "specific_target_positions":
                feed[name] = np.zeros(shape, dtype=np.float32)
            elif name == "specific_target_headings":
                feed[name] = np.zeros(shape, dtype=np.float32)
            elif name == "allowed_pred_num_tokens":
                arr = np.array([0,0,0,1,1,1,0,0,0,0,0], dtype=np.int64)
                feed[name] = arr[:int(np.prod(shape))].reshape(shape)
            else:
                feed[name] = np.zeros(shape, dtype=np.float32)
        outs = self.sess.run(self.output_names, feed)
        return np.asarray(outs[0], dtype=np.float32).reshape(-1, 36)


# ── 50Hz motion buffer (offline = no lock needed) ─────────────────────────────

class MotionBuffer:
    """50Hz buffer of (pos, quat, joints_isaac); splice-on-replan."""

    def __init__(self):
        self.pos    = np.zeros((0, 3),  dtype=np.float32)
        self.quat   = np.zeros((0, 4),  dtype=np.float32)
        self.joints = np.zeros((0, 29), dtype=np.float32)

    @property
    def length(self):
        return len(self.pos)

    def splice_at(self, gen, new_pos, new_quat, new_joints):
        gen = max(0, min(gen, self.length))
        self.pos    = np.concatenate([self.pos[:gen],    new_pos],    axis=0)
        self.quat   = np.concatenate([self.quat[:gen],   new_quat],   axis=0)
        self.joints = np.concatenate([self.joints[:gen], new_joints], axis=0)

    def read_window(self, frame_50, num_frames, step):
        N = self.length
        if N <= frame_50 + (num_frames - 1) * step + 2:
            return None
        idx = [min(frame_50 + k * step, N - 1) for k in range(num_frames)]
        jp = self.joints[idx].copy()
        bq = self.quat[idx].copy()
        vel = np.zeros((num_frames, NUM_JOINTS), dtype=np.float32)
        for k, j in enumerate(idx):
            vel[k] = (self.joints[min(j + 1, N - 1)] - self.joints[j]) * 50.0
        return jp, bq, vel

    def sample_at_time(self, t):
        if self.length == 0:
            raise RuntimeError("buffer empty")
        f_50 = t * 50.0
        f0 = max(0, min(int(np.floor(f_50)), self.length - 1))
        f1 = min(f0 + 1, self.length - 1)
        w1 = max(0.0, min(1.0, f_50 - f0))
        w0 = 1.0 - w1
        pos    = w0 * self.pos[f0]    + w1 * self.pos[f1]
        joints = w0 * self.joints[f0] + w1 * self.joints[f1]
        quat = self.quat[f0].copy() if f0 == f1 else quat_slerp(self.quat[f0], self.quat[f1], w1)
        return pos, quat, joints

    def build_planner_context(self, gen_frame_50):
        """4-frame ctx at 30Hz spacing from gen_time. Mirrors deploy
        UpdateContextFromMotion. Returns (4, 36) mujoco-order."""
        gen_t = gen_frame_50 / 50.0
        ctx = np.zeros((4, 36), dtype=np.float32)
        for n in range(4):
            pos, quat, joints_isaac = self.sample_at_time(gen_t + n / 30.0)
            ctx[n, 0:3] = pos
            ctx[n, 3:7] = quat
            joints_mj = np.zeros(29, dtype=np.float32)
            joints_mj[_MUJOCO_TO_ISAACLAB_DOF] = joints_isaac
            ctx[n, 7:36] = joints_mj
        return ctx


# ── Episode loader ────────────────────────────────────────────────────────────

def load_episode(episode_dir):
    """Returns dict with raw 30Hz data + source-detection flags. The encoder-
    pipeline uses qpos_isaac_30 (which prefers action sol_q if available);
    output sonic JSON re-uses raw states fields directly (per spec).
    """
    with open(os.path.join(episode_dir, "data.json")) as f:
        d = json.load(f)
    frames = d if isinstance(d, list) else d["frames"]

    a0 = frames[0].get("actions") or {}
    sol0 = a0.get("sol_q")
    sol_dim = len(sol0) if sol0 is not None else 0
    has_action_hands = (
        a0.get("left_angles") is not None and a0.get("right_angles") is not None
        and len(a0["left_angles"]) == 7 and len(a0["right_angles"]) == 7
    )
    print(f"[load] {len(frames)}f @ 30Hz; sol_q dim={sol_dim}; "
          f"action hands={has_action_hands}")

    n = len(frames)
    qpos_mj   = np.zeros((n, 29), dtype=np.float32)         # for encoder pipeline
    states_qpos_mj = np.zeros((n, 29), dtype=np.float32)    # raw states qpos for sonic out
    imu_quat  = np.zeros((n, 4), dtype=np.float32)
    body_vel  = np.zeros((n, 3), dtype=np.float32)
    state_hands  = np.zeros((n, 14), dtype=np.float32)
    action_hands = np.zeros((n, 14), dtype=np.float32) if has_action_hands else None
    images = []
    timestamps = []

    for i, fr in enumerate(frames):
        s = fr["states"]
        a = fr.get("actions") or {}

        # Raw states qpos (always leg + arm from states)
        leg = np.asarray(s.get("leg_states", s.get("leg_state")), dtype=np.float32).reshape(-1)[:15]
        arm_s = np.asarray(s.get("arm_states", s.get("arm_state")), dtype=np.float32).reshape(-1)[:14]
        states_qpos_mj[i, :15] = leg
        states_qpos_mj[i, 15:] = arm_s

        # Encoder-pipeline qpos (prefer action when available)
        sol_q = a.get("sol_q")
        if sol_dim == 29 and sol_q is not None:
            qpos_mj[i] = np.asarray(sol_q, dtype=np.float32).reshape(-1)[:29]
        elif sol_dim == 14 and sol_q is not None:
            qpos_mj[i, :15] = leg
            qpos_mj[i, 15:] = np.asarray(sol_q, dtype=np.float32).reshape(-1)[:14]
        else:
            qpos_mj[i] = states_qpos_mj[i]

        # IMU + velocity (always states)
        q = np.asarray(s["imu"]["quaternion"], dtype=np.float32).reshape(4)
        q = q / max(np.linalg.norm(q), 1e-8)
        if i > 0 and float(np.dot(q, imu_quat[i-1])) < 0.0:
            q = -q
        imu_quat[i] = q
        body_vel[i] = np.asarray(s["odometry"]["velocity"], dtype=np.float32).reshape(3)

        # Hands
        h = s.get("hand_state")
        if h is not None and len(h) >= 14:
            state_hands[i] = np.asarray(h[:14], dtype=np.float32)
        if has_action_hands:
            la = np.asarray(a["left_angles"],  dtype=np.float32).reshape(-1)[:7]
            ra = np.asarray(a["right_angles"], dtype=np.float32).reshape(-1)[:7]
            action_hands[i, :7]  = la
            action_hands[i, 7:]  = ra

        images.append(fr.get("image", ""))
        # Original "time" is float seconds → ns int (sonic format)
        t = fr.get("time")
        timestamps.append(int(t * 1e9) if isinstance(t, (int, float)) else None)

    qpos_isaac_30 = qpos_mj[:, _MUJOCO_TO_ISAACLAB_DOF].astype(np.float32)

    return dict(
        n_30=n,
        sol_dim=sol_dim,
        has_action_hands=has_action_hands,
        qpos_isaac_30=qpos_isaac_30,
        qpos_mujoco_30=qpos_mj,                    # for planner init context
        states_qpos_mujoco_30=states_qpos_mj,      # for sonic output states.qpos
        imu_quat_30=imu_quat,
        body_vel_30=body_vel,
        state_hands_30=state_hands,
        action_hands_30=action_hands,
        images=images,
        timestamps=timestamps,
    )


# ── Encoder window builders ───────────────────────────────────────────────────

def data_encoder_window(qpos_isaac_50, quat_50, pub_frame, n_50):
    """Phase A: window straight from data."""
    idx = [min(pub_frame + k * ENCODER_STEP_50HZ, n_50 - 1)
           for k in range(ENCODER_NUM_FRAMES)]
    jp = qpos_isaac_50[idx].copy()
    bq = quat_50[idx].copy()
    jv = np.zeros((ENCODER_NUM_FRAMES, NUM_JOINTS), dtype=np.float32)
    for k, j in enumerate(idx):
        jv[k] = (qpos_isaac_50[min(j + 1, n_50 - 1)] - qpos_isaac_50[j]) * 50.0
    return jp, bq, jv


def planner_encoder_window(motion_buffer, local_frame, arms_isaac_50, pub_frame, n_50):
    """Phase B/C: window from planner buffer + arm override.

    bq[0] is intentionally NOT overridden to data quat. In online walk_then_replay,
    bq[0] is overridden to real WBC IMU which lags planner output during turning,
    creating the strong "keep rotating" signal in the token. Offline we have no
    real robot — the closest analog is the planner buffer's own quat at this
    frame (= what the planner thinks the body looks like now). Overriding to
    data quat would erase the rotation signal because we steer planner to track
    data yaw, so data quat ≈ planner buffer quat at the same instant → token
    encodes near-identity rotations → decoder produces no turn. Leaving bq[0]
    from the buffer preserves the full rotation trajectory.
    """
    win = motion_buffer.read_window(local_frame, ENCODER_NUM_FRAMES, ENCODER_STEP_50HZ)
    if win is None:
        return None
    jp, bq, jv = win
    d_idx = [min(pub_frame + k * ENCODER_STEP_50HZ, n_50 - 1)
             for k in range(ENCODER_NUM_FRAMES)]
    jp[:, ARM_ISAAC_INDICES] = arms_isaac_50[d_idx]
    arms_v = np.zeros((ENCODER_NUM_FRAMES, len(ARM_ISAAC_INDICES)), dtype=np.float32)
    for k, j in enumerate(d_idx):
        arms_v[k] = (arms_isaac_50[min(j + 1, n_50 - 1)] - arms_isaac_50[j]) * 50.0
    jv[:, ARM_ISAAC_INDICES] = arms_v
    return jp, bq, jv


# ── Main offline processing ───────────────────────────────────────────────────

def process_episode(episode_dir):
    """Compute one 64-d FSQ token per original 30Hz frame. Returns dict with
    tokens_30 and all the raw fields needed to write sonic JSON."""

    ep = load_episode(episode_dir)
    n_30 = ep["n_30"]
    walk_speed = WALK_SPEED_BY_SOL_DIM.get(ep["sol_dim"], WALK_SPEED_DEFAULT)
    print(f"[main] WALK_SPEED={walk_speed} (sol_q dim={ep['sol_dim']})")

    # Resample to 50Hz
    qpos_isaac_50 = resample_30hz_to_50hz(ep["qpos_isaac_30"])
    qpos_mujoco_50 = resample_30hz_to_50hz(ep["qpos_mujoco_30"])
    quat_50 = resample_30hz_to_50hz(ep["imu_quat_30"])
    quat_50 /= np.linalg.norm(quat_50, axis=1, keepdims=True).clip(1e-8)
    body_vel_50 = resample_30hz_to_50hz(ep["body_vel_30"])
    n_50 = len(qpos_isaac_50)

    # Phase masks: smoothed vel + yaw_rate, hysteresis + dwell
    dwell = max(1, int(MIN_DWELL_SECS * CONTROL_HZ))
    vx_s = smooth_1d(body_vel_50[:, 0], int(SMOOTH_SECS * CONTROL_HZ))
    vy_s = smooth_1d(body_vel_50[:, 1], int(SMOOTH_SECS * CONTROL_HZ))
    body_vel_xy_smooth = np.stack([vx_s, vy_s], axis=1)
    walking_mask = hysteresis_mask(np.maximum(np.abs(vx_s), np.abs(vy_s)),
                                   ENTER_THR, EXIT_THR, dwell)
    yaw_50 = np.unwrap([quat_wxyz_to_yaw(q) for q in quat_50]).astype(np.float32)
    yaw_rate_50 = (np.gradient(smooth_1d(yaw_50, int(YAW_SMOOTH_SECS * CONTROL_HZ)))
                   * CONTROL_HZ).astype(np.float32)
    turning_mask = (hysteresis_mask(np.abs(yaw_rate_50),
                                    TURN_RATE_ENTER, TURN_RATE_EXIT, dwell)
                    & ~walking_mask)
    print(f"[main] {n_30}f@30Hz → {n_50}f@50Hz; "
          f"walking {walking_mask.sum()}/{n_50} ({100*walking_mask.mean():.0f}%); "
          f"turning {turning_mask.sum()}/{n_50} ({100*turning_mask.mean():.0f}%)")

    # Arm slice (isaac order) for phase B/C arm-override
    arms_isaac_50 = qpos_isaac_50[:, ARM_ISAAC_INDICES]

    encoder = Encoder(ENCODER_MODEL)
    planner = Planner(PLANNER_MODEL, default_height=HEIGHT)

    # ── State machine ────────────────────────────────────────────────────────
    motion_buffer = None
    first_handover = None
    last_replan_local = 0
    planner_started = False
    current_target_yaw = None
    c_data_yaw_anchor = None
    c_locked_yaw = None
    phase = "A"

    def init_planner_at(pub_frame):
        nonlocal motion_buffer, current_target_yaw
        # Use data quat/qpos at the handover frame (no WBC offline)
        cur_quat = quat_50[pub_frame]
        cur_qpos_mj = qpos_mujoco_50[pub_frame]
        current_target_yaw = quat_wxyz_to_yaw(cur_quat)
        fwd = np.array([np.cos(current_target_yaw), np.sin(current_target_yaw), 0.0],
                       dtype=np.float32)
        print(f"[main] first handover pub_frame={pub_frame}, "
              f"lock yaw={np.degrees(current_target_yaw):.1f}°")

        planner.initialize_context(cur_quat, cur_qpos_mj)
        motion_buffer = MotionBuffer()
        for w in range(4):
            gen = 0 if w == 0 else max(0, motion_buffer.length - 10)
            if w > 0:
                planner.set_context(motion_buffer.build_planner_context(gen))
            pred = planner.step(mode=PLANNER_MODE, target_vel=ENTER_THR,
                                movement_dir=fwd, facing_dir=fwd, height=HEIGHT)
            motion_buffer.splice_at(gen, *resample_pred_30hz_to_50hz(pred))

    def replan(local_frame, mode, speed, mvmt_dir, facing_dir):
        gen = local_frame + MOTION_LOOK_AHEAD_50HZ
        # Extend buffer if planner gen point is past the end
        while motion_buffer.length <= gen:
            tail = max(0, motion_buffer.length - 10)
            planner.set_context(motion_buffer.build_planner_context(tail))
            pred = planner.step(mode=mode, target_vel=speed,
                                movement_dir=mvmt_dir, facing_dir=facing_dir,
                                height=HEIGHT)
            motion_buffer.splice_at(tail, *resample_pred_30hz_to_50hz(pred))
        planner.set_context(motion_buffer.build_planner_context(gen))
        pred = planner.step(mode=mode, target_vel=speed,
                            movement_dir=mvmt_dir, facing_dir=facing_dir,
                            height=HEIGHT)
        motion_buffer.splice_at(gen, *resample_pred_30hz_to_50hz(pred))

    tokens_50 = np.zeros((n_50, 64), dtype=np.float32)

    print(f"[main] processing {n_50} ticks @ 50Hz...")
    for pub_frame in range(n_50):
        walking = bool(walking_mask[pub_frame])
        turning = bool(turning_mask[pub_frame])
        new_phase = "B" if walking else ("C" if turning else "A")

        # ── Phase transitions ────────────────────────────────────────────────
        if new_phase != phase:
            if phase == "A" and new_phase != "A":
                if not planner_started:
                    init_planner_at(pub_frame)
                    first_handover = pub_frame
                    last_replan_local = 0
                    planner_started = True
                else:
                    current_target_yaw = float(yaw_50[pub_frame])
            if new_phase == "C":
                c_data_yaw_anchor = float(yaw_50[pub_frame])
                c_locked_yaw = float(current_target_yaw)
            phase = new_phase

        if phase == "C":
            current_target_yaw = c_locked_yaw + (float(yaw_50[pub_frame]) - c_data_yaw_anchor)

        # ── Build encoder window ─────────────────────────────────────────────
        if phase == "A":
            jp, bq, jv = data_encoder_window(qpos_isaac_50, quat_50, pub_frame, n_50)
        else:
            local_frame = pub_frame - first_handover

            # Replan @ 10Hz with current params
            if local_frame - last_replan_local >= REPLAN_INTERVAL_50:
                facing = np.array([np.cos(current_target_yaw),
                                   np.sin(current_target_yaw), 0.0], dtype=np.float32)
                if phase == "B":
                    speed, vd = scale_velocity_xy(body_vel_xy_smooth[pub_frame], walk_speed)
                    cy, sy = np.cos(current_target_yaw), np.sin(current_target_yaw)
                    mvmt = np.array([cy*vd[0] - sy*vd[1],
                                     sy*vd[0] + cy*vd[1], 0.0], dtype=np.float32)
                    replan(local_frame, PLANNER_MODE, speed, mvmt, facing)
                else:  # C: turn in place
                    replan(local_frame, PLANNER_MODE, 0.0, facing, facing)
                last_replan_local = local_frame

            win = planner_encoder_window(motion_buffer, local_frame,
                                         arms_isaac_50, pub_frame, n_50)
            if win is None:
                # Should not happen since replan() ensures buffer is ready,
                # but guard anyway by extending
                replan(local_frame, PLANNER_MODE, 0.0,
                       np.array([1.0, 0.0, 0.0], dtype=np.float32),
                       np.array([1.0, 0.0, 0.0], dtype=np.float32))
                win = planner_encoder_window(motion_buffer, local_frame,
                                             arms_isaac_50, pub_frame, n_50,
                                             anchor_quat=quat_50[pub_frame])
            jp, bq, jv = win

        # ── Encode + quantize ────────────────────────────────────────────────
        tokens_50[pub_frame] = fsq_quantize(encoder.encode(jp, jv, bq))

        if (pub_frame + 1) % 200 == 0 or pub_frame + 1 == n_50:
            print(f"  [{pub_frame+1:5d}/{n_50}] phase={phase}", end="\r")
    print()

    # ── Map 50Hz tokens back to 30Hz frames ──────────────────────────────────
    idx_30_to_50 = np.round(np.arange(n_30) * 5.0 / 3.0).astype(int).clip(0, n_50 - 1)
    tokens_30 = tokens_50[idx_30_to_50]
    print(f"[main] aligned tokens: {n_30} @ 30Hz (sampled from {n_50} @ 50Hz)")

    return dict(ep, tokens_30=tokens_30)


# ── Sonic-format writer ───────────────────────────────────────────────────────

def write_sonic_json(out_path, ep):
    """Write list-of-frames JSON in sonic format. Output structure per frame:
        timestamp, states:{qpos, quat, hand_joints},
                   actions:{hand_joints, token}, image
    """
    n = ep["n_30"]
    states_qpos = ep["states_qpos_mujoco_30"]   # leg+arm from states
    quat = ep["imu_quat_30"]
    state_hands = ep["state_hands_30"]
    action_hands = ep["action_hands_30"]        # may be None
    tokens = ep["tokens_30"]
    images = ep["images"]
    timestamps = ep["timestamps"]

    out_frames = []
    for i in range(n):
        frame = {
            "states": {
                "qpos":        states_qpos[i].tolist(),
                "quat":        quat[i].tolist(),
                "hand_joints": state_hands[i].tolist(),
            },
            "actions": {
                "hand_joints": (action_hands[i] if action_hands is not None
                                else state_hands[i]).tolist(),
                "token":       tokens[i].tolist(),
            },
            "image": images[i],
        }
        if timestamps[i] is not None:
            frame["timestamp"] = timestamps[i]
        out_frames.append(frame)

    with open(out_path, "w") as f:
        json.dump(out_frames, f)
    print(f"[main] wrote {n} frames → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Convert He-format episode → sonic-format with 30Hz-aligned tokens.")
    p.add_argument("episode_dir", type=str, nargs="?", default=EPISODE_DIR,
                   help=f"Episode directory containing data.json (default: top-of-file EPISODE_DIR)")
    p.add_argument("--out", type=str, default=None,
                   help=f"Output JSON path. Default: <episode_dir>/{OUT_NAME}")
    args = p.parse_args()

    out_path = args.out or os.path.join(args.episode_dir, OUT_NAME)
    print(f"[main] episode_dir = {args.episode_dir}")
    print(f"[main] out_path    = {out_path}")
    ep = process_episode(args.episode_dir)
    write_sonic_json(out_path, ep)


if __name__ == "__main__":
    main()
