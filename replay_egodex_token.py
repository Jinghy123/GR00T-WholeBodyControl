#!/usr/bin/env python3
"""
replay_egodex_token.py — turn an EgoDex hdf5 episode into mode-1 (teleop /
vr_3pt) sonic tokens on the fly and publish them to the WBC controller over
ZMQ Protocol v4 (same wire format as replay_token.py).

Assumes the human in the EgoDex clip stays in place (no walking) — the
lower-body slots of the encoder obs are zeroed so the decoder will keep the
robot standing.

EgoDex layout (ARKit world frame, +Y up):
    transforms/camera    (T,4,4)  head 6DoF       → mapped to "neck" 3-pt
    transforms/leftHand  (T,4,4)  left wrist 6DoF → vr 3-pt[0]
    transforms/rightHand (T,4,4)  right wrist 6DoF → vr 3-pt[1]
    transforms/hip       (T,4,4)  pelvis / root

Pipeline per frame:
    1. Rotate all 4x4 from ARKit Y-up to robot Z-up (rot_x(-90°)).
    2. Compute root-local position + quat (wxyz) for the 3 keypoints.
    3. Fill the 1762-d encoder obs (mode-1 required slots only — rest stay 0):
         [0:4]      encoder_mode_4   = [1, 0, 0, 0]
         [595:601]  motion_anchor_orientation        (6,   wrt identity)
         [661:781]  lowerbody joint positions ×10    (120, zeros = standing)
         [781:901]  lowerbody joint velocities ×10   (120, zeros)
         [901:910]  vr_3point_local_target           (9)
         [910:922]  vr_3point_local_orn_target       (12, wxyz×3)
    4. Run model_encoder.onnx → 64-d → FSQ quantize.
    5. Publish at 30 Hz over ZMQ pose topic (Protocol v4 token_state).

Usage:
    python replay_egodex_token.py
    python replay_egodex_token.py --hdf5 <path/to/0.hdf5>
    python replay_egodex_token.py --hdf5 ... --dry-run    # encode but don't publish
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import onnxruntime as ort
import zmq
from scipy.spatial.transform import Rotation as sRot


# ── User defaults ─────────────────────────────────────────────────────────────
DEFAULT_HDF5     = "/home/xiawei/hongyi/EgoPoser/data/EgoDex/add_remove_lid/0.hdf5"
DEFAULT_ZMQ_HOST = "*"
DEFAULT_ZMQ_PORT = 5556
DEFAULT_TOPIC    = "pose"
DEFAULT_HZ       = 30   # EgoDex is captured at 30 Hz


# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.abspath(__file__))
ENCODER_MODEL = os.path.join(_REPO, "gear_sonic_deploy/policy/release/model_encoder.onnx")
sys.path.insert(0, _REPO)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (   # noqa: E402
    build_command_message,
    pack_pose_message,
)


# ── Encoder obs layout (1762-d, byte-for-byte matches observation_config.yaml) ─
OBS_DIM = 1762
SLOT_MODE        = slice(0,    4)      # encoder_mode_4   [mode_id, 0, 0, 0]
SLOT_ANCHOR_1F   = slice(595, 601)     # motion_anchor_orientation (6-d, 6D rot)
SLOT_JPOS_LB_10F = slice(661, 781)     # lowerbody joint pos ×10  (120-d)
SLOT_JVEL_LB_10F = slice(781, 901)     # lowerbody joint vel ×10  (120-d)
SLOT_VR3PT_POS   = slice(901, 910)     # 3 pts × xyz             (9-d)
SLOT_VR3PT_ORN   = slice(910, 922)     # 3 pts × quat wxyz       (12-d)

MODE_TELEOP = 1

# FSQ quantization (matches he_to_sonic / encoder spec)
FSQ_MIN, FSQ_MAX, FSQ_STEP = -0.625, 0.625, 0.0625


# ── ARKit (right-handed, Y-up, -Z forward) → Robot (right-handed, Z-up, X-fwd) ─
# Mirrors the Pico→Robot change-of-basis structure in
# pico_manus_thread_server.py:_compute_rel_transform, but with Q rebuilt for
# Apple's ARKit world convention (which is right-handed, unlike Pico's left-
# handed Unity convention).
#
#   robot_x = -arkit_z   (-Z forward → +X forward)
#   robot_y = -arkit_x   (right       → -left)
#   robot_z =  arkit_y   (up          →  up)
Q_ARKIT_TO_ROBOT = np.array(
    [[ 0, 0, -1],
     [-1, 0,  0],
     [ 0, 1,  0]],
    dtype=np.float64,
)

def arkit_T_to_robot(T):
    """4×4 in ARKit world → 4×4 in robot world.

    Position: linear map  p' = Q · p.
    Rotation: change-of-basis  R' = Q · R · Q^T  (same rotation re-expressed in
    the new basis) — matches _compute_rel_transform in pico_manus_thread_server.py.
    """
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Q_ARKIT_TO_ROBOT @ T[:3, :3] @ Q_ARKIT_TO_ROBOT.T
    out[:3, 3]  = Q_ARKIT_TO_ROBOT @ T[:3, 3]
    return out


# ── Quaternion helpers (wxyz convention) ──────────────────────────────────────
def mat_to_quat_wxyz(R):
    xyzw = sRot.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)

def quat_wxyz_to_mat(q):
    return sRot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def anchor_6d(ref_q, robot_q):
    """6D rep of inv(robot_q) * ref_q  (mirrors GatherMotionAnchorOrientationMutiFrame)."""
    diff = quat_mul(quat_conj(robot_q), ref_q)
    R = quat_wxyz_to_mat(diff)
    return np.array(
        [R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]],
        dtype=np.float32,
    )

def fsq_quantize(x):
    return np.clip(
        np.round(np.clip(x, FSQ_MIN, FSQ_MAX) / FSQ_STEP) * FSQ_STEP,
        FSQ_MIN, FSQ_MAX,
    )


def yaw_only_root(R):
    """Project a pelvis rotation matrix to its yaw component, so the resulting
    root frame is strictly vertical (G1 convention: X=forward, Y=left, Z=up).

    Two corrections in one:
      1. Tilt-strip: Apple articulates the pelvis bone, so for sitting subjects
         the raw hip frame is tilted 20-30° even though the spine is nearly
         upright. The encoder was trained on G1 standing data with a vertical
         pelvis, so feeding raw tilted root pushes 3-pt off the manifold.
      2. Convention flip: Apple / USD humanoid pelvis uses (X=back, Y=right,
         Z=up) — still right-handed but rotated 180° in yaw from G1 (X=forward,
         Y=left, Z=up). The user-forward direction is therefore **-pelvis-X**,
         not +pelvis-X. Empirically verified by checking which yaw choice puts
         "rightHand" on body -Y (right side).
    """
    forward_world = -R[:, 0]                       # user-forward = -Apple-pelvis-X
    yaw = np.arctan2(forward_world[1], forward_world[0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


# ── EgoDex loader ─────────────────────────────────────────────────────────────
def load_egodex(hdf5_path):
    """Return (n_frames, head, l_wrist, r_wrist, root) — each (n, 4, 4) in robot frame."""
    with h5py.File(hdf5_path, "r") as f:
        T = f["transforms"]
        head    = T["camera"][:]
        lwrist  = T["leftHand"][:]
        rwrist  = T["rightHand"][:]
        root    = T["hip"][:]
    n = len(head)
    head_r  = np.stack([arkit_T_to_robot(M) for M in head])
    lw_r    = np.stack([arkit_T_to_robot(M) for M in lwrist])
    rw_r    = np.stack([arkit_T_to_robot(M) for M in rwrist])
    root_r  = np.stack([arkit_T_to_robot(M) for M in root])
    return n, head_r, lw_r, rw_r, root_r


# ── Mode-1 obs builder ────────────────────────────────────────────────────────
_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

def build_mode1_obs(head_T, lw_T, rw_T, root_T):
    """Compose the 1762-d encoder input for one frame (teleop mode).

    Order of the 3 keypoints matches the deploy convention used by
    GatherVR3PointPosition / GatherVR3PointOrientation:
        index 0 = left wrist, 1 = right wrist, 2 = head/neck.
    """
    root_pos   = root_T[:3, 3]
    # Yaw-only root: cancel the pelvis-tilt artifact Apple introduces for
    # sitting subjects (see yaw_only_root docstring).
    root_R     = yaw_only_root(root_T[:3, :3])
    root_q     = mat_to_quat_wxyz(root_R)
    root_R_inv = root_R.T

    pts_world = [lw_T[:3, 3],  rw_T[:3, 3],  head_T[:3, 3]]
    quats_world = [
        mat_to_quat_wxyz(lw_T[:3, :3]),
        mat_to_quat_wxyz(rw_T[:3, :3]),
        mat_to_quat_wxyz(head_T[:3, :3]),
    ]

    vr3pt_pos = np.zeros(9,  dtype=np.float32)
    vr3pt_orn = np.zeros(12, dtype=np.float32)
    for i, (p, q) in enumerate(zip(pts_world, quats_world)):
        rel_p = root_R_inv @ (p - root_pos)
        rel_q = quat_mul(quat_conj(root_q), q)
        vr3pt_pos[i * 3:(i + 1) * 3] = rel_p
        vr3pt_orn[i * 4:(i + 1) * 4] = rel_q   # wxyz

    # motion_anchor_orientation: "base_quat" = identity (no real robot here),
    # "ref_root_rot" = current EgoDex root_q in robot frame.
    anchor6 = anchor_6d(ref_q=root_q, robot_q=_IDENTITY_Q)

    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = float(MODE_TELEOP)            # encoder_mode_4: [1, 0, 0, 0]
    obs[SLOT_ANCHOR_1F] = anchor6
    obs[SLOT_VR3PT_POS] = vr3pt_pos
    obs[SLOT_VR3PT_ORN] = vr3pt_orn
    # SLOT_JPOS_LB_10F / SLOT_JVEL_LB_10F intentionally left at zero (standing).
    return obs


# ── Encoder wrapper ───────────────────────────────────────────────────────────
class ModeEncoder:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess     = ort.InferenceSession(model_path, sess_options=opts)
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        in_shape      = self.sess.get_inputs()[0].shape
        print(f"[Encoder] loaded {model_path}  in='{self.in_name}' shape={in_shape}")

    def encode(self, obs):
        o = obs.reshape(1, -1).astype(np.float32)
        out = self.sess.run([self.out_name], {self.in_name: o})[0]
        return np.asarray(out, dtype=np.float32).flatten()


# ── Token publisher (Protocol v4, mirrors replay_token.py) ────────────────────
class TokenPublisher:
    def __init__(self, host, port, topic):
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.bind(f"tcp://{host}:{port}")
        self._topic = topic
        self.n_sent = 0
        print(f"[TokenPublisher] PUB bound to tcp://{host}:{port} topic={topic}")

    def send_command(self, start=False, stop=False, planner=False):
        self._sock.send(build_command_message(start=start, stop=stop, planner=planner))
        print(f"[TokenPublisher] command: start={start} stop={stop} planner={planner}")

    def publish_token(self, token, left_hand=None, right_hand=None):
        pose = {"token_state": token.astype(np.float32).reshape(1, -1)}
        if left_hand is not None:
            pose["left_hand_joints"]  = np.asarray(left_hand,  dtype=np.float32).reshape(1, -1)
        if right_hand is not None:
            pose["right_hand_joints"] = np.asarray(right_hand, dtype=np.float32).reshape(1, -1)
        self._sock.send(pack_pose_message(pose, topic=self._topic, version=4))
        self.n_sent += 1

    def stop(self):
        self._sock.close(); self._ctx.term()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Convert EgoDex hdf5 → mode-1 sonic tokens and stream via ZMQ (Protocol v4)"
    )
    ap.add_argument("--hdf5",  type=str, default=DEFAULT_HDF5,
                    help=f"Path to EgoDex .hdf5 episode (default: {DEFAULT_HDF5})")
    ap.add_argument("--host",  type=str, default=DEFAULT_ZMQ_HOST)
    ap.add_argument("--port",  type=int, default=DEFAULT_ZMQ_PORT)
    ap.add_argument("--topic", type=str, default=DEFAULT_TOPIC)
    ap.add_argument("--hz",    type=int, default=DEFAULT_HZ,
                    help=f"Playback frequency (default {DEFAULT_HZ})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Encode every frame but do not bind ZMQ or send.")
    ap.add_argument("--save-tokens", type=str, default=None,
                    help="Optional .npy path to save the (N, 64) token array.")
    args = ap.parse_args()

    n, head, lw, rw, root = load_egodex(args.hdf5)
    print(f"[ego→token] loaded {n} frames from {args.hdf5}")
    print(f"[ego→token]   head_z[0]={head[0,2,3]:+.3f}  hip_z[0]={root[0,2,3]:+.3f}  "
          f"L_wrist_z[0]={lw[0,2,3]:+.3f}  R_wrist_z[0]={rw[0,2,3]:+.3f}")

    encoder = ModeEncoder(ENCODER_MODEL)

    print(f"[ego→token] encoding {n} frames ...")
    tokens = np.zeros((n, 64), dtype=np.float32)
    t_enc0 = time.perf_counter()
    for i in range(n):
        obs = build_mode1_obs(head[i], lw[i], rw[i], root[i])
        tokens[i] = fsq_quantize(encoder.encode(obs))
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  [{i+1:5d}/{n}]", end="\r")
    print(f"\n[ego→token] encoding took {time.perf_counter()-t_enc0:.2f}s "
          f"({1000*(time.perf_counter()-t_enc0)/max(1,n):.2f} ms/frame)")

    if args.save_tokens:
        np.save(args.save_tokens, tokens)
        print(f"[ego→token] tokens saved → {args.save_tokens}")

    if args.dry_run:
        print("[ego→token] --dry-run: skipping ZMQ publish.")
        return

    pub = TokenPublisher(args.host, args.port, args.topic)
    print("[ego→token] waiting 1.0 s for subscribers ...")
    time.sleep(1.0)
    pub.send_command(start=True, stop=False, planner=True)
    time.sleep(0.2)

    dt = 1.0 / args.hz
    print(f"[ego→token] streaming {n} tokens @ {args.hz} Hz ...")
    try:
        for i in range(n):
            t0 = time.perf_counter()
            pub.publish_token(tokens[i])
            print(f"  [{i+1:5d}/{n}]", end="\r")
            slack = dt - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)
        print()

        # Hold the last token so the robot doesn't snap back to a default pose.
        print("[ego→token] done. Holding last token — Ctrl+C to stop.")
        while True:
            t0 = time.perf_counter()
            pub.publish_token(tokens[-1])
            slack = dt - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print("\n[ego→token] interrupted.")
    finally:
        pub.stop()
        print(f"[ego→token] total sent: {pub.n_sent} frames.")


if __name__ == "__main__":
    main()
