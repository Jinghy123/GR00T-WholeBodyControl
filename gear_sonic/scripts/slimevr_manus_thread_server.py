"""
slimevr_manus_thread_server.py

SlimeVR full-body tracking + Manus gloves → GMR IK → Unitree G1 joint states,
published to the sonic deploy over ZMQ (v1 protocol), plus a 2-DOF neck stream.

This is the SlimeVR counterpart to pico_gmr_thread_server.py / pico_manus_thread_server.py.
The body source is SlimeVR (VMC/OSC) instead of the Pico headset; everything
downstream (GMR retarget, v1 ZMQ contract, neck ZMQ, Manus Dex3 hands) matches
the validated GR00T pipeline so g1_data_server.py / realsense_server.py / the
WBC deploy consume it unchanged.

Pipeline:
  SlimeVR app → VMC/OSC (UDP :39539)
       → SlimevrBodyReader → body_frame {joint: [pos, quat_wxyz]}
       → apply_bvh_like_coordinate_transform → gmr_rename_and_footmod("nokov")
       → GMR IK (src_human="bvh_nokov", tgt_robot="unitree_g1")
       → joint_pos29 (MuJoCo order) + pelvis_quat
       → reorder to IsaacLab → ZMQ v1 (joint_pos/joint_vel/body_quat/frame_index
         + left/right_hand_joints from Manus Dex3 retarget)   on :5556
  body_frame Head vs Spine3 → [neck_yaw, neck_pitch] JSON      on :5570
       (same wire format as pose_publisher.py / pico_manus, consumed by
        realsense_server.py --pose-zmq on the G1)

Neck note: SlimeVR must include a HEAD tracker. Without one the FK Head falls
back to Neck, so Head-vs-Spine3 is ~identity and the neck won't move. Neck
direction is tuned on the G1 via NECK_YAW_SIGN / NECK_PITCH_SIGN.

Controls (in THIS terminal — independent from g1_data_server.py's s/q/d):
  k : toggle send on/off   p : toggle hold (freeze body)   q : exit   e : estop
  ENABLE_KEYBOARD = False runs always-send (no gating); Ctrl-C always exits.

Configuration: edit the CONFIG block below and run with no arguments:
    python gear_sonic/scripts/slimevr_manus_thread_server.py
"""

import json
import os
import sys
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation
import zmq


# ============================================================================
# CONFIG —— 直接改这里的常量，不用命令行参数
# ============================================================================
HUMAN_HEIGHT        = 1.60            # GMR 肢体缩放（米）
FORMAT              = "nokov"         # GMR 源骨架 → src_human=bvh_<FORMAT>
POSE_ADDR           = "tcp://*:5556"  # pose PUB 绑定地址
POSE_TOPIC          = "pose"          # ZMQ topic 前缀
TARGET_FPS          = 50              # 主循环目标帧率
NUM_FRAMES_TO_SEND  = 5               # 每次发布的滑窗帧数
ZMQ_CATCH_UP        = True            # v1 payload catch_up 标志（同 humdex teleop.yaml）

# --- SlimeVR VMC（身体来源）---
VMC_IP              = "0.0.0.0"       # VMC 监听 IP
VMC_PORT            = 39539           # VMC UDP 端口
VMC_TIMEOUT_S       = 0.5             # VMC 帧超时（秒）
VMC_BVH_SCALE       = 0.01            # BVH 全局缩放
BVH_PATH            = ""              # 留空=用自带 assets/bvh-recording.bvh

# --- Manus Dex3（手）---
ENABLE_HAND         = True            # False=只发身体+脖子
LEFT_GLOVE_SN       = "265bba63"      # 左手套 SN，改成你们的
RIGHT_GLOVE_SN      = "81d630d4"      # 右手套 SN，改成你们的

# --- 脖子 ---
ENABLE_NECK_PUB     = True            # False=关闭脖子 [yaw,pitch] PUB
NECK_PUB_PORT       = 5570            # 脖子 PUB 端口
NECK_RETARGET_SCALE = 1.5            # 缩放脖子运动幅度

# --- 键盘 gating（k=送/停, p=hold, q=退, e=急停）---
ENABLE_KEYBOARD     = True            # False=一直发送、无 gating
# ============================================================================


# ── Vendored SlimeVR body module (self-contained under external_dependencies) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_SLIMEVR_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "external_dependencies", "slimevr_body")
)
if _SLIMEVR_DIR not in sys.path:
    sys.path.insert(0, _SLIMEVR_DIR)

from slimevr_adapter import SlimevrBodyConfig, SlimevrBodyReader  # noqa: E402
from pose_transforms import (  # noqa: E402
    apply_bvh_like_coordinate_transform,
    gmr_rename_and_footmod,
)
from keyboard_toggle import KeyboardToggle  # noqa: E402

# ── GMR IK (real package, same import as pico_gmr_thread_server.py) ────────────
from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting  # noqa: E402

# ── ZMQ pose packer (GR00T) ───────────────────────────────────────────────────
try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
except ImportError:
    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")

# ── Manus Dex3 hands — reuse the validated path from pico_manus_thread_server ──
try:
    from pico_manus_thread_server import (
        compute_hand_joints_from_manus,
        init_hand_retargeting,
        init_manus_receiver,
    )
    _MANUS_OK = True
except Exception as _manus_err:  # pragma: no cover - env dependent
    print(f"Warning: Manus hand reuse unavailable ({_manus_err}); running body+neck only.")
    compute_hand_joints_from_manus = init_hand_retargeting = init_manus_receiver = None
    _MANUS_OK = False


# MuJoCo (GMR output) → IsaacLab joint order (29 DOF) — identical to pico_gmr.
_MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)


def _safe_unit_quat_wxyz(q):
    """Normalize a wxyz quaternion; identity on degenerate input. Matches the
    humdex sonic publisher so body_quat is unit-norm on the wire."""
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if (not np.isfinite(n)) or n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _human_head_to_robot_neck(body_frame):
    """[yaw, pitch] of the Head relative to Spine3 (YXZ Euler), inlined from the
    gmr_shim so the neck path is independent of which GMR package is on the path.

    body_frame values are [pos, quat_wxyz]; same convention/result as the neck
    stream in pico_manus_thread_server.py. Returns None if Head/Spine3 missing.
    """
    head = body_frame.get("Head")
    spine = body_frame.get("Spine3")
    if head is None or spine is None:
        return None
    hq = np.asarray(head[1], dtype=np.float64).reshape(4)
    sq = np.asarray(spine[1], dtype=np.float64).reshape(4)

    def _R(qwxyz):
        return Rotation.from_quat([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]])

    R_rel = _R(sq).inv() * _R(hq)
    yaw, pitch, _roll = R_rel.as_euler("yxz", degrees=False)
    return float(yaw), float(pitch)


class SlimevrManusGMRServer:
    def __init__(self):
        # ── SlimeVR body reader (viewer FK + bvh skeleton, matches humdex) ────
        bvh_path = BVH_PATH or os.path.join(_SLIMEVR_DIR, "assets", "bvh-recording.bvh")
        self.body_cfg = SlimevrBodyConfig(
            vmc_ip=VMC_IP,
            vmc_port=VMC_PORT,
            vmc_timeout_s=VMC_TIMEOUT_S,
            vmc_rot_mode="local",
            vmc_use_fk=True,
            vmc_use_viewer_fk=True,
            vmc_fk_skeleton="bvh",
            vmc_bvh_path=bvh_path,
            vmc_bvh_scale=VMC_BVH_SCALE,
        )
        self.body = SlimevrBodyReader(self.body_cfg)
        self.body.initialize()
        print(f"[slimevr] VMC reader on {VMC_IP}:{VMC_PORT}  bvh={bvh_path}")

        # ── GMR retargeter ────────────────────────────────────────────────────
        self.format = FORMAT
        print(f"Building GMR (src_human=bvh_{self.format}, human_height={HUMAN_HEIGHT}m)...")
        self.retargeter = GeneralMotionRetargeting(
            src_human=f"bvh_{self.format}",
            tgt_robot="unitree_g1",
            actual_human_height=HUMAN_HEIGHT,
            verbose=False,
        )
        print("GMR retargeter ready.")

        # ── Manus Dex3 hands ──────────────────────────────────────────────────
        self.manus = None
        self.hand_retgt = None
        if _MANUS_OK and ENABLE_HAND:
            self.manus = init_manus_receiver(
                left_glove_sn=LEFT_GLOVE_SN,
                right_glove_sn=RIGHT_GLOVE_SN,
            )
            self.hand_retgt = init_hand_retargeting()

        # ── ZMQ: pose (v1) on :5556, neck [yaw,pitch] on :5570 ────────────────
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUB)
        # Match humdex zmq_pose_publisher exactly: only keep the latest frame,
        # never backlog. Without CONFLATE the PUB default HWM (1000) queues old
        # frames and the subscriber lags ~1s as the queue drains. MUST be set
        # before bind().
        self.sock.setsockopt(zmq.SNDHWM, 1)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(POSE_ADDR)
        self.topic = POSE_TOPIC
        print(f"[pose] ZMQ PUB (v1) bound to {POSE_ADDR} topic={self.topic}")

        self.neck_pub = None
        if ENABLE_NECK_PUB:
            self.neck_pub = ctx.socket(zmq.PUB)
            self.neck_pub.setsockopt(zmq.SNDHWM, 1)
            self.neck_pub.setsockopt(zmq.LINGER, 0)
            self.neck_pub.bind(f"tcp://*:{NECK_PUB_PORT}")
            print(f"[neck] ZMQ PUB bound to port {NECK_PUB_PORT}")
        self.neck_scale = float(NECK_RETARGET_SCALE)
        self._last_neck_t = 0.0

        # ── Keyboard gating (k/p/q/e). ENABLE_KEYBOARD=False → always-send. ───
        self.kb = KeyboardToggle(
            enable=ENABLE_KEYBOARD,
            toggle_send_key="k",
            hold_key="p",
            exit_key="q",
            emergency_stop_key="e",
        )

        # ── Sliding-window buffers (same window as pico_gmr/pico_manus) ───────
        self.nsend = int(NUM_FRAMES_TO_SEND)
        self._jbuf = deque(maxlen=self.nsend)
        self._qbuf = deque(maxlen=self.nsend)
        self._fbuf = deque(maxlen=self.nsend)
        self._lhbuf = deque(maxlen=self.nsend)
        self._rhbuf = deque(maxlen=self.nsend)
        self._fidx = 0
        # 满周期 sleep（不补偿处理时间），与 humdex stage_step_rate_limiter 一致。
        self._frame_time = 1.0 / max(1, int(TARGET_FPS))
        # hold snapshot
        self._last_jp = None
        self._last_bq = None
        self._last_lh = None
        self._last_rh = None
        # stats
        self._stat_t = time.time()
        self._stat_n = 0
        self._stat_gmr = []

    def _hands(self):
        """(left[7], right[7]) Dex3 joints, zeros if no Manus."""
        if self.hand_retgt is None or self.manus is None:
            z = np.zeros(7, dtype=np.float32)
            return z, z.copy()
        lh, rh = compute_hand_joints_from_manus(self.hand_retgt, self.manus)
        return lh.reshape(-1).astype(np.float32), rh.reshape(-1).astype(np.float32)

    def publish_neck(self, body_frame, now):
        """Rate-limited (50 Hz) [yaw, pitch] JSON — same wire format as pico_manus."""
        if self.neck_pub is None:
            return
        if now - self._last_neck_t < (1.0 / 50.0):
            return
        yp = _human_head_to_robot_neck(body_frame)
        if yp is None:
            return
        yaw = yp[0] * self.neck_scale
        pitch = yp[1] * self.neck_scale
        try:
            self.neck_pub.send(json.dumps([float(yaw), float(pitch)]).encode("utf-8"))
        except Exception:
            return
        self._last_neck_t = now

    def process_and_send(self, body_frame, hold_enabled):
        """Retarget one body_frame and publish a v1 pose message."""
        fr = apply_bvh_like_coordinate_transform(body_frame, pos_unit="m", apply_rotation=True)
        fr = gmr_rename_and_footmod(fr, fmt=self.format)

        t0 = time.perf_counter()
        qpos = self.retargeter.retarget(fr, offset_to_ground=True)
        self._stat_gmr.append(time.perf_counter() - t0)

        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if q.shape[0] < 36:
            return
        # qpos: [x, y, z, qw, qx, qy, qz, joint0..joint28]
        body_quat = _safe_unit_quat_wxyz(q[3:7])
        joint_pos29 = q[7:36].astype(np.float32)[_MUJOCO_TO_ISAACLAB]
        lh, rh = self._hands()

        if hold_enabled and self._last_jp is not None:
            joint_pos29, body_quat = self._last_jp, self._last_bq
            lh, rh = self._last_lh, self._last_rh
        else:
            self._last_jp, self._last_bq = joint_pos29, body_quat
            self._last_lh, self._last_rh = lh, rh

        self._jbuf.append(joint_pos29)
        self._qbuf.append(body_quat)
        self._fbuf.append(self._fidx)
        self._lhbuf.append(lh)
        self._rhbuf.append(rh)
        self._fidx += 1

        N = len(self._fbuf)
        if N >= self.nsend:
            pose_data = {
                "joint_pos": np.stack(list(self._jbuf)),
                "joint_vel": np.zeros((N, 29), dtype=np.float32),
                "body_quat": np.stack(list(self._qbuf)),
                "frame_index": np.array(list(self._fbuf), dtype=np.int64),
                # Match the humdex sonic v1 payload (deploy reads these as optional).
                "timestamp_realtime": np.asarray([time.time()] * N, dtype=np.float64),
                "heading_increment": np.asarray([0.0] * N, dtype=np.float32),
                "catch_up": np.asarray([bool(ZMQ_CATCH_UP)] * N, dtype=bool),
                "left_hand_joints": np.asarray(self._lhbuf[-1], dtype=np.float32),
                "right_hand_joints": np.asarray(self._rhbuf[-1], dtype=np.float32),
            }
            self.sock.send(pack_pose_message(pose_data, topic=self.topic, version=1))

    def _maybe_log(self, now):
        if now - self._stat_t < 5.0 or not self._stat_gmr:
            return
        gmr_ms = np.mean(self._stat_gmr) * 1000.0
        fps = self._stat_n / max(1e-6, now - self._stat_t)
        print(f"[slimevr] frame={self._fidx} gmr={gmr_ms:.1f}ms fps~{fps:.1f}")
        self._stat_gmr.clear()
        self._stat_n = 0
        self._stat_t = now

    def run(self):
        self.kb.start()
        print(
            "Controls (this terminal): k=send on/off  p=hold  q=exit  e=estop  "
            "(Ctrl-C also exits)"
        )
        print("Waiting for SlimeVR VMC frames...")
        try:
            while True:
                send_enabled, hold_enabled, exit_req = self.kb.get_extended_state()
                if exit_req:
                    break
                t = time.time()
                res = self.body.read_frame()
                if not res.get("ok"):
                    time.sleep(0.003)
                    continue
                body_frame = res["body_frame"]

                # Neck tracks the operator's head in all gating states.
                self.publish_neck(body_frame, t)

                if send_enabled:
                    try:
                        self.process_and_send(body_frame, hold_enabled)
                    except Exception as e:
                        print(f"[slimevr] retarget/send error: {e}")

                self._stat_n += 1
                self._maybe_log(t)

                # 关键：与 humdex stage_step_rate_limiter 完全一致 —— 每帧无条件
                # sleep 整个周期，不减去处理时间。补偿式（sleep(period-elapsed)）
                # 会让发送频率明显高于 humdex，deploy 订阅端消费跟不上 → 接收/
                # 回放队列积压 → ~1s 滞后（PUB 端的 CONFLATE 管不到 SUB 积压）。
                if self._frame_time > 0:
                    time.sleep(self._frame_time)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.kb.stop()
            self.sock.close()
            if self.neck_pub is not None:
                self.neck_pub.close()
            try:
                self.body.close()
            except Exception:
                pass
            if self.manus is not None:
                try:
                    self.manus.stop()
                except Exception:
                    pass
            print("[slimevr] shutdown complete")


def main():
    SlimevrManusGMRServer().run()


if __name__ == "__main__":
    main()
