#!/usr/bin/env python3
"""
walk_forward_token.py
=====================

匀速向前走路客户端 (zmq pose-streaming 路径), 1:1 复刻 deploy PLANNER 模式
内部的 token 生成 pipeline. 用于:
  ./deploy.sh --input-type zmq sim   + 在 deploy 终端按 Enter 进 streaming
  python walk_forward_token.py --speed 0.3

为什么这么复杂: deploy 内部的 token 是从一个**持续生长的 50Hz motion buffer**
里, 用 step5 的滑窗实时编码出来的. 每个 token 都和前一个 token 平滑演化 ——
policy decoder 是在这种平滑序列上训练的. 之前的简化版本(30Hz native + 每次
replan 重置 buffer) 产生了不连续的 token 序列, decoder 解出"约等于走路"的
关节目标但抬腿幅度不够 → 脚拖地.

本版本严格对齐:
  - 50Hz buffer (Motion50HzBuffer)
  - 每次 replan splice in 新预测, 不重置 buffer
  - 编码窗口 step5 在 50Hz buffer 上, 每 control tick 滑动 1 帧
  - 编码频率 50Hz (= deploy control_dt_)
  - Planner 频率 10Hz (= deploy planner_dt_)
  - 30→50Hz: 关节 + 位置线性插值, quat slerp
  - context 重建: 4 帧, 30Hz 间距, 从 50Hz buffer 采样 (mimic
    UpdateContextFromMotion in localmotion_kplanner.hpp:628)
  - anchor reference: 真实 IMU base_quat (mimic g1_deploy_onnx_ref.cpp:631)
  - joint vel: forward diff over 1 个 50Hz 帧 (= 0.02s)
  - FSQ 量化 token

用法
----
1) 终端 1:
     ./deploy.sh --input-type zmq sim
   等机器人站直, 按一次 Enter 进 streaming.

2) 终端 2:
     python walk_forward_token.py --speed 0.3
"""

import argparse
import json
import math
import os
import signal
import sys
import threading
import time

import msgpack
import numpy as np
import onnxruntime as ort
import zmq

_GROOT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _GROOT_ROOT)

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)
from ours.common.encoder_client import EncoderClient


# ── 常量 ────────────────────────────────────────────────────────────────────────

ENCODER_MODEL = os.path.join(_GROOT_ROOT, "gear_sonic_deploy/policy/release/model_encoder.onnx")
PLANNER_MODEL = os.path.join(_GROOT_ROOT, "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx")

CONTROL_HZ = 50                  # = deploy control_dt_ (g1_deploy_onnx_ref.cpp:2186)
PLANNER_HZ = 10                  # = deploy planner_dt_ (g1_deploy_onnx_ref.cpp:2187)
MOTION_LOOK_AHEAD_50HZ = 2       # = deploy config_.motion_look_ahead_steps (localmotion_kplanner.hpp:215)
ENCODER_NUM_FRAMES = 10
ENCODER_STEP_50HZ = 5            # 10 frames step5 at 50Hz = 1.0s window @ 0.1s spacing

ROOT_DIM = 7
NUM_JOINTS = 29

# Mujoco → IsaacLab 全 29 关节重排 (matches policy_parameters.hpp:103)
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)

# G1 站立默认姿态 (mujoco order)
G1_DEFAULT_ANGLES_MUJOCO = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0,   0.0, 0.0,
     0.2,   0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
     0.2,  -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float32)

# FSQ codebook (matches g1_sonic_client.py:84-108)
FSQ_MIN, FSQ_MAX, FSQ_STEP = -0.625, 0.625, 0.0625

def fsq_quantize(x):
    return np.clip(np.round(np.clip(x, FSQ_MIN, FSQ_MAX) / FSQ_STEP) * FSQ_STEP,
                   FSQ_MIN, FSQ_MAX)


# ── Arm joint mapping (data arm_state in mujoco order → isaac slot positions) ──
#
# joint_isaac[i] = joint_mujoco[_MUJOCO_TO_ISAACLAB_DOF[i]]. Arms occupy mujoco
# indices [15..28]. So if data.arm_state is mujoco-ordered (k → mujoco 15+k):
#   joint_isaac[ARM_ISAAC_INDICES[k]] = data.arm_state[ARM_DATA_INDICES[k]]
ARM_ISAAC_INDICES = np.array([11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], dtype=np.int32)
ARM_DATA_INDICES  = np.array([0,  7,  1,  8,  2,  9,  3,  10, 4,  11, 5,  12, 6,  13], dtype=np.int32)


def load_episode_data(episode_dir):
    """Load episode JSON and return per-frame arm_state(14) + hand_state(14, may be None)
    plus source frequency. Arms are in mujoco order (data convention).
    """
    json_path = os.path.join(episode_dir, "data.json")
    with open(json_path, "r") as f:
        d = json.load(f)
    frames = d if isinstance(d, list) else d["frames"]
    freq = 30 if isinstance(d, list) else d.get("frequency", 30)

    n = len(frames)
    arms_30 = np.zeros((n, 14), dtype=np.float32)
    hands_30 = np.zeros((n, 14), dtype=np.float32)
    hands_valid = False
    for i, fr in enumerate(frames):
        s = fr.get("states") or {}
        a = s.get("arm_states", s.get("arm_state"))
        if a is None or len(a) < 14:
            raise ValueError(f"Frame {i}: arm_state missing or wrong size")
        arms_30[i] = np.asarray(a[:14], dtype=np.float32)
        h = s.get("hand_state")
        if h is not None and len(h) >= 14:
            hands_30[i] = np.asarray(h[:14], dtype=np.float32)
            hands_valid = True
    return arms_30, hands_30 if hands_valid else None, freq


def resample_30hz_to_50hz(arr_30):
    """Linear interp (n, D) at 30Hz → (n_50, D) at 50Hz."""
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


# ── Quaternion helpers (wxyz convention) ────────────────────────────────────────

def quat_wxyz_to_yaw(q):
    """wxyz quat → yaw (rad), atan2 form."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z)))


def quat_slerp(q0, q1, t):
    """SLERP between two wxyz quaternions, t∈[0,1]. Returns float32 (4,)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return (r / np.linalg.norm(r)).astype(np.float32)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    sin_t = math.sin(theta)
    sin_t0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_t / sin_t0
    s1 = sin_t / sin_t0
    return (s0 * q0 + s1 * q1).astype(np.float32)


# ── Planner ONNX wrapper ────────────────────────────────────────────────────────

class PlannerOnnx:
    IDLE, SLOW_WALK, WALK, RUN = 0, 1, 2, 3

    def __init__(self, model_path, default_height=0.788740):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, sess_options=opts)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.input_shapes = {i.name: tuple(i.shape) for i in self.sess.get_inputs()}
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.default_height = default_height
        self.context = None  # (4, 36) mujoco order
        print(f"[Planner] loaded {model_path}")

    @staticmethod
    def _resolve_shape(shape):
        return [d if isinstance(d, int) and d > 0 else 1 for d in shape]

    def initialize_context(self, base_quat_wxyz, qpos_mujoco29):
        """Build 4-frame context all set to current robot state."""
        frame = np.zeros(36, dtype=np.float32)
        frame[2] = self.default_height
        frame[3:7] = np.asarray(base_quat_wxyz, dtype=np.float32).reshape(4)
        frame[7:36] = np.asarray(qpos_mujoco29, dtype=np.float32).reshape(29)
        self.context = np.tile(frame[None, :], (4, 1))

    def set_context(self, ctx_4x36):
        self.context = np.asarray(ctx_4x36, dtype=np.float32).reshape(4, 36)

    def step(self, mode, target_vel, movement_dir, facing_dir, height=None, random_seed=0):
        if height is None:
            height = self.default_height
        feed = {}
        for name in self.input_names:
            shape = self._resolve_shape(self.input_shapes[name])
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
                feed[name] = np.array([int(random_seed)], dtype=np.int64).reshape(shape)
            elif name == "height":
                feed[name] = np.array([float(height)], dtype=np.float32).reshape(shape)
            elif name == "has_specific_target":
                feed[name] = np.zeros(shape, dtype=np.int64)
            elif name == "specific_target_positions":
                feed[name] = np.zeros(shape, dtype=np.float32)
            elif name == "specific_target_headings":
                feed[name] = np.zeros(shape, dtype=np.float32)
            elif name == "allowed_pred_num_tokens":
                arr = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int64)
                feed[name] = arr[: int(np.prod(shape))].reshape(shape)
            else:
                feed[name] = np.zeros(shape, dtype=np.float32)
        outs = self.sess.run(self.output_names, feed)
        return np.asarray(outs[0], dtype=np.float32).reshape(-1, 36)


# ── 30Hz pred → 50Hz buffer 重采样 ──────────────────────────────────────────────

def resample_pred_30hz_to_50hz(pred_30hz):
    """
    (T_30, 36) mujoco order
        → pos (T_50, 3), quat (T_50, 4 wxyz), joints_isaac (T_50, 29)

    线性插值 pos / joints, slerp quat. 与 deploy 的 ResampleGeneratedSequence50Hz
    (localmotion_kplanner.hpp:395-519) 行为一致.
    """
    T_30 = len(pred_30hz)
    motion_seconds = T_30 / 30.0
    T_50 = max(1, int(np.floor(motion_seconds * 50)))

    pos = np.zeros((T_50, 3), dtype=np.float32)
    quat = np.zeros((T_50, 4), dtype=np.float32)
    joints = np.zeros((T_50, 29), dtype=np.float32)

    for f_50 in range(T_50):
        t = f_50 / 50.0
        f_30 = t * 30.0
        f0 = max(0, min(int(np.floor(f_30)), T_30 - 1))
        f1 = min(f0 + 1, T_30 - 1)
        w1 = max(0.0, min(1.0, f_30 - f0))
        w0 = 1.0 - w1

        pos[f_50] = w0 * pred_30hz[f0, 0:3] + w1 * pred_30hz[f1, 0:3]
        if f0 == f1:
            quat[f_50] = pred_30hz[f0, 3:7]
        else:
            quat[f_50] = quat_slerp(pred_30hz[f0, 3:7], pred_30hz[f1, 3:7], w1)
        joints_mj = w0 * pred_30hz[f0, 7:36] + w1 * pred_30hz[f1, 7:36]
        joints[f_50] = joints_mj[_MUJOCO_TO_ISAACLAB_DOF]

    return pos, quat, joints


# ── 50Hz motion buffer ──────────────────────────────────────────────────────────

class Motion50HzBuffer:
    """
    Mimics deploy's planner_motion_50hz_ buffer. Stored fields:
      pos    : (N, 3)  body position xyz
      quat   : (N, 4)  body quaternion wxyz
      joints : (N, 29) joint positions in *isaaclab* order

    Splice semantics: each replan truncates buffer at gen_frame and appends new
    resampled motion — buffer is monotonic (entries before gen_frame are kept).

    Thread-safe: an internal RLock guards mutations (splice_at) and reads
    (read_encoder_window, sample_at_time). Intended for one consumer (encode
    loop @ 50Hz) and one producer (async planner @ 10Hz).
    """

    def __init__(self):
        self.pos = np.zeros((0, 3), dtype=np.float32)
        self.quat = np.zeros((0, 4), dtype=np.float32)
        self.joints = np.zeros((0, 29), dtype=np.float32)
        self._lock = threading.RLock()

    @property
    def length(self):
        return len(self.pos)

    def splice_at(self, gen_frame, new_pos, new_quat, new_joints):
        """Truncate at gen_frame, append new content."""
        with self._lock:
            gen_frame = max(0, min(gen_frame, self.length))
            self.pos = np.concatenate([self.pos[:gen_frame], new_pos], axis=0)
            self.quat = np.concatenate([self.quat[:gen_frame], new_quat], axis=0)
            self.joints = np.concatenate([self.joints[:gen_frame], new_joints], axis=0)

    def read_encoder_window(self, current_frame_50, num_frames, step):
        """Atomically read encoder window + forward-diff vel.
        Returns (joint_pos, body_quat, vel, N) or None if not enough buffer."""
        with self._lock:
            N = self.length
            need = current_frame_50 + (num_frames - 1) * step + 2
            if N <= need:
                return None
            idx = [min(current_frame_50 + k * step, N - 1) for k in range(num_frames)]
            joint_pos = self.joints[idx].copy()
            body_quat = self.quat[idx].copy()
            vel = np.zeros((num_frames, NUM_JOINTS), dtype=np.float32)
            for k, j in enumerate(idx):
                jn = min(j + 1, N - 1)
                vel[k] = (self.joints[jn] - self.joints[j]) * 50.0
            return joint_pos, body_quat, vel, N

    def build_context_locked(self, gen_frame_50):
        """Atomically build planner context (4×36 mujoco-order) from buffer."""
        with self._lock:
            return build_planner_context_from_buffer(self, gen_frame_50)

    def sample_at_time(self, t):
        """Linear/slerp interp at time t (seconds @ 50Hz)."""
        if self.length == 0:
            raise RuntimeError("buffer empty")
        f_50 = t * 50.0
        f0 = max(0, min(int(np.floor(f_50)), self.length - 1))
        f1 = min(f0 + 1, self.length - 1)
        w1 = max(0.0, min(1.0, f_50 - f0))
        w0 = 1.0 - w1
        pos = w0 * self.pos[f0] + w1 * self.pos[f1]
        joints = w0 * self.joints[f0] + w1 * self.joints[f1]
        if f0 == f1:
            quat = self.quat[f0].copy()
        else:
            quat = quat_slerp(self.quat[f0], self.quat[f1], w1)
        return pos, quat, joints


def build_planner_context_from_buffer(motion_buffer, gen_frame_50):
    """
    Mimics deploy UpdateContextFromMotion (localmotion_kplanner.hpp:628):
    sample 4 frames at 30Hz spacing starting from gen_time.
    Returns (4, 36) array, mujoco order: [x,y,z, qw,qx,qy,qz, joints_29_mujoco].
    """
    gen_time = gen_frame_50 / 50.0
    context = np.zeros((4, 36), dtype=np.float32)
    for n in range(4):
        t = gen_time + n / 30.0
        pos, quat, joints_isaac = motion_buffer.sample_at_time(t)
        context[n, 0:3] = pos
        context[n, 3:7] = quat
        # joints_isaac → joints_mujoco (inverse permutation of _MUJOCO_TO_ISAACLAB_DOF)
        joints_mj = np.zeros(29, dtype=np.float32)
        joints_mj[_MUJOCO_TO_ISAACLAB_DOF] = joints_isaac
        context[n, 7:36] = joints_mj
    return context


# ── Async planner thread ───────────────────────────────────────────────────────

class AsyncPlannerThread:
    """
    Decouples planner ONNX inference (~80-200ms in Python) from the 50Hz
    publish loop. Main thread calls request_replan(gen_frame_50); planner
    thread picks it up, runs inference, splices into the shared 50Hz buffer.

    Latest-wins: if main signals faster than planner can keep up, only the
    most recent gen_frame is honored. Older requests are dropped silently.
    """

    def __init__(self, planner, motion_buffer, mode, speed, height,
                 mvmt_dir=(1.0, 0.0, 0.0), facing_dir=(1.0, 0.0, 0.0)):
        self.planner = planner
        self.buffer = motion_buffer
        self.mode = mode
        self.speed = speed
        self.height = height
        self.mvmt_dir = np.asarray(mvmt_dir, dtype=np.float32)
        self.facing_dir = np.asarray(facing_dir, dtype=np.float32)
        self._cv = threading.Condition()
        self._pending_gen = None
        self._stop = threading.Event()
        self._n_replans = 0
        self._last_replan_ms = 0.0
        self._max_replan_ms = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def request_replan(self, gen_frame_50):
        with self._cv:
            self._pending_gen = int(gen_frame_50)
            self._cv.notify()

    def set_mode_speed(self, mode, speed):
        """Hot-swap planner mode/speed. The next replan picks up new values
        (in-flight inference uses old values — at most 1 stale plan)."""
        self.mode = int(mode)
        self.speed = float(speed)

    def set_planner_params(self, mode=None, speed=None, mvmt_dir=None, facing_dir=None):
        """Hot-swap any combination of planner inputs. Next replan picks up
        new values; in-flight inference uses old values (≤1 stale plan)."""
        if mode is not None:
            self.mode = int(mode)
        if speed is not None:
            self.speed = float(speed)
        if mvmt_dir is not None:
            self.mvmt_dir = np.asarray(mvmt_dir, dtype=np.float32)
        if facing_dir is not None:
            self.facing_dir = np.asarray(facing_dir, dtype=np.float32)

    @property
    def n_replans(self):
        return self._n_replans

    @property
    def last_replan_ms(self):
        return self._last_replan_ms

    @property
    def max_replan_ms(self):
        return self._max_replan_ms

    def _loop(self):
        while not self._stop.is_set():
            with self._cv:
                while self._pending_gen is None and not self._stop.is_set():
                    self._cv.wait(timeout=0.1)
                if self._stop.is_set():
                    break
                gen = self._pending_gen
                self._pending_gen = None

            if self.buffer.length < gen + 1:
                continue

            t0 = time.perf_counter()
            ctx = self.buffer.build_context_locked(gen)
            self.planner.set_context(ctx)
            pred = self.planner.step(
                mode=self.mode, target_vel=self.speed,
                movement_dir=self.mvmt_dir, facing_dir=self.facing_dir,
                height=self.height,
            )
            new_pos, new_quat, new_joints = resample_pred_30hz_to_50hz(pred)
            self.buffer.splice_at(gen, new_pos, new_quat, new_joints)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._last_replan_ms = elapsed_ms
            if elapsed_ms > self._max_replan_ms:
                self._max_replan_ms = elapsed_ms
            self._n_replans += 1

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self._thread.join(timeout=1.0)


# ── WBC state subscriber ────────────────────────────────────────────────────────

class WBCStateReader:
    def __init__(self, host="localhost", port=5557, topic="g1_debug"):
        self._tb = topic.encode()
        self._tlen = len(self._tb)
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, self._tb)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.connect(f"tcp://{host}:{port}")
        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[WBC] sub tcp://{host}:{port} topic={topic}")

    def _loop(self):
        while not self._stop.is_set():
            try:
                payload = self._sock.recv()[self._tlen:]
                d = msgpack.unpackb(payload, raw=False)
                with self._lock:
                    self._latest = {
                        "qpos": np.array(d["body_q_measured"], dtype=np.float32),
                        "base_quat": np.array(d["base_quat_measured"], dtype=np.float32),
                    }
            except zmq.Again:
                pass
            except Exception:
                pass

    def get_state(self):
        with self._lock:
            return None if self._latest is None else {k: v.copy() for k, v in self._latest.items()}

    def wait_until_ready(self, timeout_s=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.get_state() is not None:
                return True
            time.sleep(0.05)
        return False

    def close(self):
        self._stop.set()
        self._sock.close()
        self._ctx.term()


# ── Token publisher (pose topic, protocol v4) ──────────────────────────────────

class TokenPublisher:
    def __init__(self, host="*", port=5556, topic="pose"):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://{host}:{port}")
        self.topic = topic
        self.n = 0
        print(f"[Pub] PUB tcp://{host}:{port} topic={topic}")

    def send_command(self, start=False, stop=False, planner=False):
        self.sock.send(build_command_message(start=start, stop=stop, planner=planner))

    def publish(self, hand_joints_14, token_64, body_quat_w=None):
        pose_data = {
            "token_state":       token_64.astype(np.float32).reshape(1, 64),
            "left_hand_joints":  hand_joints_14[:7].astype(np.float32).reshape(1, 7),
            "right_hand_joints": hand_joints_14[7:].astype(np.float32).reshape(1, 7),
        }
        if body_quat_w is not None:
            pose_data["body_quat_w"] = np.asarray(body_quat_w, dtype=np.float32).reshape(1, 4)
        msg = pack_pose_message(pose_data, topic=self.topic, version=4)
        self.sock.send(msg)
        self.n += 1

    def close(self):
        self.sock.close()
        self.ctx.term()


# ── 主流程 ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Walk forward via 50Hz token pipeline (mimics deploy PLANNER mode)")
    p.add_argument("--speed", type=float, default=0.3, help="m/s (SLOW_WALK 0.1~0.8)")
    p.add_argument("--mode", type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument("--zmq-host", type=str, default="*")
    p.add_argument("--zmq-port", type=int, default=5556)
    p.add_argument("--zmq-topic", type=str, default="pose")
    p.add_argument("--wbc-host", type=str, default="localhost")
    p.add_argument("--wbc-port", type=int, default=5557)
    p.add_argument("--wbc-topic", type=str, default="g1_debug")
    p.add_argument("--height", type=float, default=0.788740)
    p.add_argument("--warmup-secs", type=float, default=3.0,
                   help="seconds to wait for IMU to settle before locking init pose")
    p.add_argument("--no-wbc", action="store_true",
                   help="skip WBC; use G1 default standing pose for init (debug only)")
    p.add_argument("--episode-dir", type=str, default=None,
                   help="If given, override arm joints in encoded tokens with arm_state "
                        "from this episode's data.json. Lower body still walks forward as "
                        "normal — only arms come from data. Stops when data ends.")
    p.add_argument("--idle-secs", type=float, default=0.0,
                   help="Stand still (planner mode=IDLE) for this many seconds at start, "
                        "then seamlessly switch to walking. Arms still replay continuously "
                        "from t=0 if --episode-dir is given.")
    args = p.parse_args()

    # Load data arms + hand (resample 30Hz → 50Hz, reorder mujoco→isaac slot order)
    data_arms_isaac_50 = None  # (n_50, 14) — values to plug into joint_pos[:, ARM_ISAAC_INDICES]
    data_hand_50 = None        # (n_50, 14) — full hand state, or None if data has no hands
    if args.episode_dir:
        print(f"[main] loading episode arms from {args.episode_dir}")
        arms_30_mj, hands_30, freq = load_episode_data(args.episode_dir)
        print(f"[main] {len(arms_30_mj)} frames @ {freq}Hz")
        arms_50_mj = resample_30hz_to_50hz(arms_30_mj)        # (n_50, 14) mujoco order
        # Reorder mujoco-order arm_state → isaac slot order so direct slice assign works
        data_arms_isaac_50 = arms_50_mj[:, ARM_DATA_INDICES]  # (n_50, 14)
        if hands_30 is not None:
            data_hand_50 = resample_30hz_to_50hz(hands_30)    # (n_50, 14)
        print(f"[main] resampled to {len(data_arms_isaac_50)} frames @ 50Hz")

    pub = TokenPublisher(host=args.zmq_host, port=args.zmq_port, topic=args.zmq_topic)
    time.sleep(1.0)  # let subscribers connect

    planner = PlannerOnnx(PLANNER_MODEL, default_height=args.height)
    encoder = EncoderClient(ENCODER_MODEL, mode=0)

    if args.no_wbc:
        init_qpos = G1_DEFAULT_ANGLES_MUJOCO.copy()
        init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        wbc = None
    else:
        wbc = WBCStateReader(args.wbc_host, args.wbc_port, args.wbc_topic)
        if not wbc.wait_until_ready(10.0):
            print("[main] WARN: no WBC state — using default standing pose.")
            init_qpos = G1_DEFAULT_ANGLES_MUJOCO.copy()
            init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            time.sleep(args.warmup_secs)
            s = wbc.get_state()
            init_qpos = s["qpos"]
            init_quat = s["base_quat"]

    planner.initialize_context(init_quat, init_qpos)

    # Lock walking direction to robot's current heading at startup, otherwise
    # planner is told to walk world +X and robot turns to face +X first.
    init_yaw = quat_wxyz_to_yaw(init_quat)
    forward_dir = np.array([np.cos(init_yaw), np.sin(init_yaw), 0.0], dtype=np.float32)
    print(f"[main] init yaw={np.degrees(init_yaw):.1f}°")

    # ── 2-phase state machine: IDLE → WALK ──────────────────────────────────────
    # idle: mode=IDLE,      vel=0
    # walk: mode=args.mode, vel=args.speed
    # Both phases use forward_dir for mvmt/face (no turning).
    idle_end_pub = int(args.idle_secs * CONTROL_HZ)

    def phase_at(pub_frame):
        return "idle" if pub_frame < idle_end_pub else "walk"

    def params_for(phase):
        if phase == "idle":
            return PlannerOnnx.IDLE, 0.0
        return args.mode, args.speed

    # ── Initial planner inference + 50Hz buffer pre-roll ─────────────────────────
    # 第一次 planner 上下文 vel=0 (从 standing 起步), 前 ~50 帧是 ramp-up 内容.
    # 紧接着做 N_WARMUP 次 extend (每次 gen=buffer 末尾, 用前一段 plan 的尾部作
    # 上下文), 让 buffer 积累一段"稳态走路"的内容. 主循环再从 WARMUP_SKIP_50
    # 帧开始消费, 跳过 ramp-up.
    N_WARMUP = 3
    WARMUP_SKIP_50 = 50  # skip ramp-up from first plan (1.0s)

    # Initial planner params from phase 0 (whichever phase pub_frame=0 falls in)
    init_mode, init_speed = params_for(phase_at(0))
    print(f"[main] sequence: idle={args.idle_secs:.1f}s, then walk @ {args.speed} m/s")

    motion_buffer = Motion50HzBuffer()
    for w in range(1 + N_WARMUP):
        if w == 0:
            gen = 0
        else:
            gen = max(0, motion_buffer.length - 10)  # forward margin for ctx sampling
            ctx = motion_buffer.build_context_locked(gen)
            planner.set_context(ctx)
        pred = planner.step(
            mode=init_mode, target_vel=init_speed,
            movement_dir=forward_dir, facing_dir=forward_dir,
            height=args.height,
        )
        new_pos, new_quat, new_joints = resample_pred_30hz_to_50hz(pred)
        motion_buffer.splice_at(gen, new_pos, new_quat, new_joints)
    print(f"[main] warmup done: buffer={motion_buffer.length} frames")

    # 把 planner ONNX 推理移到独立线程, 避免阻塞 50Hz 发布
    async_planner = AsyncPlannerThread(planner, motion_buffer,
                                       mode=init_mode, speed=init_speed,
                                       height=args.height,
                                       mvmt_dir=forward_dir, facing_dir=forward_dir)

    pub.send_command(start=True, stop=False, planner=False)

    # ── Main loop @ 50Hz ────────────────────────────────────────────────────────
    REPLAN_INTERVAL_50 = int(round(CONTROL_HZ / PLANNER_HZ))  # = 5

    # Skip warmup ramp-up — start consuming from steady-walking content
    current_frame_50 = WARMUP_SKIP_50
    last_replan_50 = WARMUP_SKIP_50
    hand_joints = np.zeros(14, dtype=np.float32)

    # Track current phase. Start at the phase corresponding to pub_frame=0.
    current_phase = phase_at(0)

    stop_flag = {"stop": False}
    def _on_sigint(*_):
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    print(f"[main] walking forward @ {args.speed} m/s. Ctrl+C to stop.")

    dt = 1.0 / CONTROL_HZ
    n_pub = 0
    try:
        while not stop_flag["stop"]:
            t_start = time.perf_counter()

            # Replan trigger (10Hz, async — does NOT block this tick)
            if current_frame_50 - last_replan_50 >= REPLAN_INTERVAL_50:
                async_planner.request_replan(current_frame_50 + MOTION_LOOK_AHEAD_50HZ)
                last_replan_50 = current_frame_50

            # Read encoder window from shared buffer (atomic)
            window = motion_buffer.read_encoder_window(
                current_frame_50, ENCODER_NUM_FRAMES, ENCODER_STEP_50HZ)
            if window is None:
                time.sleep(0.005)
                continue
            joint_pos_isaac, body_quat, vel_isaac, _ = window

            # Publish frame counter: 0 at first published tick (data t=0)
            pub_frame = current_frame_50 - WARMUP_SKIP_50

            # Phase transition: hot-swap planner mode/speed at boundaries. Next replan
            # picks them up; buffer gradually fills with new motion; encoder slides
            # through and the robot smoothly transitions (stand → walk).
            new_phase = phase_at(pub_frame)
            if new_phase != current_phase:
                m, s = params_for(new_phase)
                async_planner.set_mode_speed(m, s)
                print(f"[main] phase {current_phase} → {new_phase} @ pub_frame={pub_frame}")
                current_phase = new_phase

            # ── Override arms with data (only thing different from pure walk_forward) ──
            if data_arms_isaac_50 is not None:
                n_data = len(data_arms_isaac_50)
                # Stop publishing once we exhaust data (data plays from t=0)
                if pub_frame >= n_data:
                    print(f"\n[main] data exhausted at pub_frame {pub_frame}, stopping")
                    break
                # Encoder window indices into data (clamp at end)
                d_idx = [min(pub_frame + k * ENCODER_STEP_50HZ, n_data - 1)
                         for k in range(ENCODER_NUM_FRAMES)]
                # Override joint_pos arms (10 frames × 14 arm slots)
                joint_pos_isaac[:, ARM_ISAAC_INDICES] = data_arms_isaac_50[d_idx]
                # Override vel arms with forward-diff @ 50Hz of data arms
                arms_vel = np.zeros((ENCODER_NUM_FRAMES, len(ARM_ISAAC_INDICES)), dtype=np.float32)
                for k, j in enumerate(d_idx):
                    jn = min(j + 1, n_data - 1)
                    arms_vel[k] = (data_arms_isaac_50[jn] - data_arms_isaac_50[j]) * 50.0
                vel_isaac[:, ARM_ISAAC_INDICES] = arms_vel
                # Use this frame's hand state for publish
                if data_hand_50 is not None:
                    hand_joints = data_hand_50[min(pub_frame, len(data_hand_50) - 1)]

            # Override anchor reference with real IMU base_quat
            if wbc is not None:
                s = wbc.get_state()
                if s is not None:
                    body_quat[0] = s["base_quat"].astype(np.float32)

            # Encode + FSQ + publish
            token = encoder.encode(joint_pos=joint_pos_isaac,
                                   joint_vel=vel_isaac,
                                   body_quat=body_quat)
            token_qtz = fsq_quantize(token).astype(np.float32)
            pub.publish(hand_joints_14=hand_joints,
                        token_64=token_qtz,
                        body_quat_w=body_quat[0])

            current_frame_50 += 1
            n_pub += 1

            # 50Hz pacing
            sleep_t = dt - (time.perf_counter() - t_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        # Soft stop: publish zero-velocity tokens briefly before sending stop
        try:
            for _ in range(int(0.5 * CONTROL_HZ)):
                N = motion_buffer.length
                if N == 0:
                    break
                idx = [min(current_frame_50 + k * ENCODER_STEP_50HZ, N - 1)
                       for k in range(ENCODER_NUM_FRAMES)]
                jp = motion_buffer.joints[idx]
                bq = motion_buffer.quat[idx].copy()
                if wbc is not None:
                    s = wbc.get_state()
                    if s is not None:
                        bq[0] = s["base_quat"].astype(np.float32)
                jv = np.zeros_like(jp)
                tok = encoder.encode(joint_pos=jp, joint_vel=jv, body_quat=bq)
                tok = fsq_quantize(tok).astype(np.float32)
                pub.publish(hand_joints, tok, body_quat_w=bq[0])
                time.sleep(dt)
            pub.send_command(start=False, stop=True, planner=False)
        finally:
            async_planner.stop()
            pub.close()
            if wbc is not None:
                wbc.close()
            print(f"[main] stopped after {n_pub} tokens")


if __name__ == "__main__":
    main()
