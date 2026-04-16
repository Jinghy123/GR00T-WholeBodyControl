#!/usr/bin/env python3
"""
run_publisher_joints.py

Teleop publisher that converts SAM3D SMPL output → G1 joint positions via GMR IK,
then sends via ZMQ protocol v1 (joint_pos + joint_vel + body_quat) to WBC.

This uses WBC encoder mode 0 ("g1"), identical to replay.py, so that
body_q_measured ≈ the sent reference and can serve as action labels directly.

Pipeline:
  Camera → SAM3D → SMPL (joints + pose + body_quat)
         → GMR IK → joint_pos29 (MuJoCo order) + pelvis_quat
         → reorder to IsaacLab → pack_pose_message v1 → ZMQ → WBC

Usage:
    python run_publisher_joints.py \
        --smpl-model-path checkpoints/smpl/SMPL_NEUTRAL.pkl \
        --gmr-human-height 1.70

    # From video file:
    python run_publisher_joints.py \
        --source video --video foo.mp4 --intrinsics cam.json \
        --smpl-model-path checkpoints/smpl/SMPL_NEUTRAL.pkl
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPH_TREES", "0")

import cv2
import numpy as np
import torch
import zmq
from loguru import logger
from scipy.spatial.transform import Rotation

from fast_mocap.core.gravity_alignment import (
    build_camera_to_world_rotation,
    transform_pose_to_world,
)
from fast_mocap.realtime.interpolator import PoseInterpolator
from fast_mocap.core.mhr_to_smpl_interface import (
    METHOD_FAST_XYZ_JOINT_FIT,
    build_mhr_to_smpl_converter,
)
from fast_mocap.core.setup_estimator import build_default_estimator
from fast_mocap.utils.smpl_render_utils import (
    load_smpl_model,
    smpl_vertices_joints_from_pose,
)
from fast_mocap.utils.video_source import create_video_source

# GMR and smpl_to_joint_states (from smpl_to_joint_states.py in this repo)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smpl_to_joint_states import build_retargeter, smpl_to_joint_states

# WBC ZMQ packing (protocol v1)
_GROOT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _GROOT_ROOT)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)

# ── Constants (same as run_publisher.py) ──────────────────────────────────────
FOV_MODEL_SIZE = "s"
FOV_RESOLUTION_LEVEL = 0
FOV_FIXED_SIZE = 512
FOV_FAST_MODE = True
FOV_DEBUG = False
BACKBONE_MODE = "tensorrt"
BACKBONE_TRT_ENGINE = "./checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine"
BACKBONE_COMPILE_MODE = "default"
YOLO_MODEL_PATH = "checkpoints/yolo_pose/yolo11m-pose.engine"

JOINTS_COORD_TRANSFORM = np.eye(3, dtype=np.float64)

SMPL_BASE_REMOVE_QUAT_XYZW = np.array([-0.5, -0.5, -0.5, 0.5], dtype=np.float64)
GLOBAL_ORIENT_EXTRA_ROT = Rotation.from_euler("y", 90.0, degrees=True) * Rotation.from_euler(
    "x", -90.0, degrees=True
)

# MuJoCo → IsaacLab reordering (same as replay.py)
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def quat_apply(quat, vec):
    qw, qx, qy, qz = quat
    qvec = np.array([qx, qy, qz])
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec + 2.0 * (uv * qw + uuv)


def quat_inverse(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=quat.dtype)


def quat_wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def mujoco29_to_isaaclab29(joint_pos29):
    return np.asarray(joint_pos29, dtype=np.float32).reshape(29)[_MUJOCO_TO_ISAACLAB_DOF].copy()


# ── WBC ZMQ bridge (v1) ────────────────────────────────────────────────────────

class WBCBridge:
    def __init__(self, host="*", port=5556, topic="pose"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")
        self._topic = topic
        self._frame_index = 0

    def send_command(self, start=False, stop=False, planner=False):
        msg = build_command_message(start=start, stop=stop, planner=planner)
        self._socket.send(msg)

    def publish_joints(self, joint_pos_isaaclab29, body_quat_wxyz):
        """
        Send joint positions to WBC via protocol v1 (encoder mode 0).

        Args:
            joint_pos_isaaclab29: (29,) float32 in IsaacLab order
            body_quat_wxyz:       (4,)  float32 WXYZ
        """
        pose_data = {
            "joint_pos":   np.asarray(joint_pos_isaaclab29, dtype=np.float32).reshape(1, 29),
            "joint_vel":   np.zeros((1, 29), dtype=np.float32),
            "body_quat":   np.asarray(body_quat_wxyz, dtype=np.float32).reshape(1, 4),
            "frame_index": np.array([self._frame_index], dtype=np.int64),
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=1)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close()
        self._context.term()


# ── Main publisher class ───────────────────────────────────────────────────────

class RealtimePublisherJoints:
    def __init__(
        self,
        video_source,
        publish_hz,
        interpolate_lag_ms,
        smpl_model_path,
        gmr_human_height=1.70,
        addr="tcp://*:5556",
        topic="pose",
        image_size=512,
        yolo_model_path=YOLO_MODEL_PATH,
        hand_interm_pred_layers=None,
        body_interm_pred_layers=None,
    ):
        logger.info("Initializing RealtimePublisherJoints...")

        logger.info("Loading SAM3D model...")
        self.estimator = build_default_estimator(
            image_size=image_size,
            backbone_mode=BACKBONE_MODE,
            trt_backbone_engine_path=BACKBONE_TRT_ENGINE,
            backbone_compile_mode=BACKBONE_COMPILE_MODE,
            yolo_model_path=yolo_model_path,
            hand_interm_pred_layers=hand_interm_pred_layers,
            body_interm_pred_layers=body_interm_pred_layers,
            fov_model_size=FOV_MODEL_SIZE,
            fov_resolution_level=FOV_RESOLUTION_LEVEL,
            fov_fixed_size=FOV_FIXED_SIZE,
            fov_fast_mode=FOV_FAST_MODE,
            fov_debug=FOV_DEBUG,
        )

        self.video_source = video_source

        logger.info("Warming up SAM3D model...")
        self._warmup()

        self.cam_intrinsics = self.video_source.get_camera_intrinsics()
        self.gravity_direction = self.video_source.get_gravity_direction()
        logger.info(f"Gravity direction: {self.gravity_direction}")
        self.R_world_cam = build_camera_to_world_rotation(self.gravity_direction)
        R_zup_adjustment = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
        self.R_world_cam = R_zup_adjustment @ self.R_world_cam

        logger.info("Loading SMPL converter (fast-xyz-joint-fit)...")
        smpl_model, _, device, num_betas = load_smpl_model(smpl_model_path)
        rest_pose_21 = np.zeros((21, 3), dtype=np.float32)
        _, rest_joints = smpl_vertices_joints_from_pose(
            rest_pose_21, smpl_model, device, num_betas, body_quat=None
        )
        rest_smpl_joints_24 = np.asarray(rest_joints[:24], dtype=np.float32)
        self.converter = build_mhr_to_smpl_converter(
            method=METHOD_FAST_XYZ_JOINT_FIT,
            smpl_model=smpl_model,
            device=device,
            num_betas=num_betas,
            rest_smpl_joints_24=rest_smpl_joints_24,
        )

        logger.info(f"Building GMR retargeter (human height={gmr_human_height}m)...")
        self.retargeter = build_retargeter(actual_human_height=gmr_human_height)
        logger.info("GMR retargeter ready.")

        self.publish_hz = publish_hz
        self.publish_dt = 1.0 / publish_hz
        self.interpolate_lag_s = interpolate_lag_ms / 1000.0

        # Interpolator stores (body_quat, smpl_joints, smpl_pose) tuples,
        # but we intercept in the publish loop and run GMR there.
        self.interpolator = PoseInterpolator()

        # Parse addr: strip "tcp://*:" → host="*", port=int
        host, port_str = addr.replace("tcp://", "").rsplit(":", 1)
        self.bridge = WBCBridge(host=host, port=int(port_str), topic=topic)

        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self._frame_event = threading.Event()

        self.running = False
        self.video_ended = False
        self._final_stats_logged = False
        self._closed = False

        self._pose_clock_lock = threading.Lock()
        self._latest_pose_source_ts = None
        self._latest_pose_perf_ts = None

        self.capture_thread = None
        self.inference_thread = None
        self.publish_thread = None

        self.stats = {
            "capture_count": 0,
            "dropped_capture_count": 0,
            "infer_count": 0,
            "infer_total_time_s": 0.0,
            "convert_total_time_s": 0.0,
            "gmr_total_time_s": 0.0,
            "publish_count": 0,
            "publish_fallback_count": 0,
            "publish_interpolated_count": 0,
            "inference_times": deque(maxlen=100),
            "gmr_times": deque(maxlen=100),
            "publish_intervals": deque(maxlen=500),
        }
        self._live_log_interval_s = 2.0
        self._live_last_log_perf = time.perf_counter()
        self._live_prev_stats = {k: v for k, v in self.stats.items() if isinstance(v, (int, float))}

        logger.success("RealtimePublisherJoints ready.")

    def _warmup(self):
        frame_size = self.video_source.get_frame_size()
        width, height = frame_size if frame_size else (640, 480)
        dummy_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        warmup_bbox = np.array([[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32)
        for _ in range(2):
            self.estimator.process_one_image(dummy_img, bboxes=warmup_bbox, hand_box_source="body_decoder")
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _capture_loop(self):
        while self.running:
            try:
                frame, frame_timestamp = self.video_source.get_frame()
            except Exception as exc:
                if self.running:
                    logger.warning(f"Capture error: {exc}")
                self.video_ended = True
                self._frame_event.set()
                break
            if frame is None:
                self.video_ended = True
                self._frame_event.set()
                break
            if frame_timestamp is None:
                continue
            self.stats["capture_count"] += 1
            with self._latest_frame_lock:
                if self._latest_frame is not None:
                    self.stats["dropped_capture_count"] += 1
                self._latest_frame = (frame, frame_timestamp)
            self._frame_event.set()

    def _prepare_pose(self, result):
        """Same coordinate transform as run_publisher.py._prepare_publish_pose()."""
        joints_w = np.asarray(result.smpl_joints, dtype=np.float64)
        pose = np.asarray(result.smpl_pose, dtype=np.float64)
        q_xyzw = np.asarray(result.body_quat, dtype=np.float64).reshape(4)
        q_cam_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)

        q_world, joints_world = transform_pose_to_world(q_cam_wxyz, joints_w, self.R_world_cam)

        q_world_xyzw = quat_wxyz_to_xyzw(q_world)
        q_world_xyzw = (
            Rotation.from_quat(q_world_xyzw) * Rotation.from_quat(SMPL_BASE_REMOVE_QUAT_XYZW)
        ).as_quat()
        q_world = quat_xyzw_to_wxyz(q_world_xyzw)

        root_pos = joints_world[0]
        joints_local = quat_apply(quat_inverse(q_world), joints_world - root_pos)
        joints_local = joints_local @ JOINTS_COORD_TRANSFORM.T

        q_world_xyzw = quat_wxyz_to_xyzw(q_world)
        q_world_xyzw = (GLOBAL_ORIENT_EXTRA_ROT * Rotation.from_quat(q_world_xyzw)).as_quat()
        q_world = quat_xyzw_to_wxyz(q_world_xyzw)

        return q_world, joints_local, pose

    def _inference_loop(self):
        while self.running:
            self._frame_event.wait(timeout=0.05)
            self._frame_event.clear()

            with self._latest_frame_lock:
                item = self._latest_frame
                self._latest_frame = None

            if item is None:
                if self.video_ended:
                    break
                continue

            frame, frame_timestamp = item
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            outputs, _ = self.estimator.process_one_image(
                frame_rgb,
                cam_int=self.cam_intrinsics,
                hand_box_source="yolo_pose",
                return_timing=True,
            )
            infer_dt = time.perf_counter() - t0
            self.stats["inference_times"].append(infer_dt)
            self.stats["infer_total_time_s"] += infer_dt
            self.stats["infer_count"] += 1

            if len(outputs) == 0:
                continue

            out = outputs[0]
            required = ("pred_joint_coords", "pred_cam_t")
            if any(k not in out for k in required):
                continue

            frame_data = {
                "pred_joint_coords": np.asarray(out["pred_joint_coords"], dtype=np.float32),
                "pred_cam_t": np.asarray(out["pred_cam_t"], dtype=np.float32),
            }
            t1 = time.perf_counter()
            result = self.converter.convert_frame(frame_data)
            self.stats["convert_total_time_s"] += time.perf_counter() - t1

            if result.body_quat is None or result.smpl_joints is None or result.smpl_pose is None:
                logger.warning("Skipping frame: incomplete converter output")
                continue

            body_quat, smpl_joints, smpl_pose = self._prepare_pose(result)

            self.interpolator.add_pose(frame_timestamp, body_quat, smpl_joints, smpl_pose)
            with self._pose_clock_lock:
                self._latest_pose_source_ts = frame_timestamp
                self._latest_pose_perf_ts = time.perf_counter()

    def _publish_loop(self):
        last_publish = None
        next_publish = time.perf_counter()

        while self.running:
            now_perf = time.perf_counter()
            wait_time = next_publish - now_perf
            if wait_time > 0:
                time.sleep(min(wait_time, 0.0015))
                continue

            wall_now = time.time()
            with self._pose_clock_lock:
                latest_pose_source_ts = self._latest_pose_source_ts
                latest_pose_perf_ts = self._latest_pose_perf_ts

            if latest_pose_source_ts is None or latest_pose_perf_ts is None:
                query_ts = wall_now - self.interpolate_lag_s
            else:
                source_now_est = latest_pose_source_ts + (now_perf - latest_pose_perf_ts)
                query_ts = source_now_est - self.interpolate_lag_s

            result = self.interpolator.interpolate(query_ts)
            used_fallback = False
            if result is None:
                result = self.interpolator.get_latest_pose()
                if result is not None:
                    used_fallback = True

            if result is not None:
                body_quat_wxyz, smpl_joints, smpl_pose = result

                # GMR IK: SMPL → robot joint_pos (MuJoCo order)
                t_gmr = time.perf_counter()
                try:
                    joint_pos29_mujoco, pelvis_quat_wxyz = smpl_to_joint_states(
                        self.retargeter, smpl_joints, smpl_pose, body_quat_wxyz,
                        offset_to_ground=True,
                    )
                except Exception as exc:
                    logger.warning(f"GMR retarget failed: {exc}")
                    next_publish += self.publish_dt
                    continue
                gmr_dt = time.perf_counter() - t_gmr
                self.stats["gmr_times"].append(gmr_dt)
                self.stats["gmr_total_time_s"] += gmr_dt

                # Reorder MuJoCo → IsaacLab (required by WBC v1 protocol)
                joint_pos29_isaaclab = mujoco29_to_isaaclab29(joint_pos29_mujoco)

                # pelvis_quat from GMR is already WXYZ (MuJoCo convention)
                self.bridge.publish_joints(joint_pos29_isaaclab, pelvis_quat_wxyz)

                self.stats["publish_count"] += 1
                if used_fallback:
                    self.stats["publish_fallback_count"] += 1
                else:
                    self.stats["publish_interpolated_count"] += 1
                if last_publish is not None:
                    self.stats["publish_intervals"].append(now_perf - last_publish)
                last_publish = now_perf

            next_publish += self.publish_dt
            if next_publish < now_perf - self.publish_dt:
                missed = int((now_perf - next_publish) / self.publish_dt) + 1
                next_publish += missed * self.publish_dt

            if self.video_ended and self.stats["infer_count"] > 0:
                if self.stats["publish_count"] > max(10, int(self.publish_hz * 0.4)):
                    break

            self._maybe_log_live_stats(now_perf)

    def _maybe_log_live_stats(self, now_perf):
        elapsed = now_perf - self._live_last_log_perf
        if elapsed < self._live_log_interval_s:
            return

        n_infer = self.stats["infer_count"] - self._live_prev_stats.get("infer_count", 0)
        n_pub = self.stats["publish_count"] - self._live_prev_stats.get("publish_count", 0)
        n_drop = self.stats["dropped_capture_count"] - self._live_prev_stats.get("dropped_capture_count", 0)

        infer_fps = n_infer / elapsed
        pub_hz = n_pub / elapsed
        gmr_ms = (np.mean(self.stats["gmr_times"]) * 1000) if self.stats["gmr_times"] else float("nan")
        infer_ms = (np.mean(self.stats["inference_times"]) * 1000) if self.stats["inference_times"] else float("nan")

        logger.info(
            f"Live: capture_drop+={n_drop}, "
            f"infer={infer_fps:.1f}fps ({infer_ms:.1f}ms), "
            f"gmr={gmr_ms:.1f}ms, "
            f"publish={pub_hz:.1f}Hz "
            f"[interp+={self.stats['publish_interpolated_count']}, fallback+={self.stats['publish_fallback_count']}]"
        )
        self._live_prev_stats = {k: v for k, v in self.stats.items() if isinstance(v, (int, float))}
        self._live_last_log_perf = now_perf

    def start(self):
        logger.info("Starting (Ctrl+C to stop)...")
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.capture_thread.start()
        self.inference_thread.start()
        self.publish_thread.start()

        while self.running:
            if self.video_ended and not self.inference_thread.is_alive():
                self.running = False
                break
            if not self.capture_thread.is_alive() and not self.inference_thread.is_alive():
                self.running = False
                break
            time.sleep(0.05)

        for t in (self.capture_thread, self.inference_thread, self.publish_thread):
            if t and t.is_alive():
                t.join(timeout=1.0)

    def stop(self):
        if self._closed:
            return
        self.running = False
        try:
            self.video_source.release()
        except Exception:
            pass
        for t in (self.capture_thread, self.inference_thread, self.publish_thread):
            if t and t.is_alive():
                t.join(timeout=1.0)
        self.bridge.stop()
        self._closed = True


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Teleop publisher: SAM3D → GMR IK → joint_pos → WBC (protocol v1)"
    )
    parser.add_argument("--source", type=str, default="camera", choices=["camera", "video"])
    parser.add_argument("--video", type=str, help="Video file path (--source video)")
    parser.add_argument("--intrinsics", type=str, help="Camera intrinsics JSON (--source video)")
    parser.add_argument("--no-loop", dest="no_loop", action="store_true")
    parser.add_argument("--publish-hz", type=float, default=50.0, help="Publish rate in Hz")
    parser.add_argument("--interp-lag-ms", type=float, default=140.0, help="Interpolation lag in ms")
    parser.add_argument("--addr", type=str, default="tcp://*:5556", help="ZMQ bind address")
    parser.add_argument("--topic", type=str, default="pose", help="ZMQ topic")
    parser.add_argument("--image-size", type=int, default=512, choices=[256, 384, 512])
    parser.add_argument("--yolo-model", type=str, default=YOLO_MODEL_PATH)
    parser.add_argument("--smpl-model-path", type=str, required=True, help="SMPL model .pkl path")
    parser.add_argument(
        "--gmr-human-height", type=float, default=1.70,
        help="Actual human height in meters for GMR limb scaling (default: 1.70)"
    )
    parser.add_argument(
        "--hand-interm-pred-layers", type=str, default=None,
        help="Comma-separated layer indices for hand intermediate predictions"
    )
    parser.add_argument(
        "--body-interm-pred-layers", type=str, default=None,
        help="Comma-separated layer indices for body intermediate predictions"
    )
    args = parser.parse_args()

    if args.source == "camera":
        video_source = create_video_source("camera", width=848, height=480, fps=30)
    else:
        if not args.video:
            parser.error("--video required when --source video")
        if not args.intrinsics:
            parser.error("--intrinsics required when --source video")
        video_source = create_video_source(
            "video", video_path=args.video, intrinsics_path=args.intrinsics, loop=not args.no_loop
        )

    hand_layers = None
    if args.hand_interm_pred_layers:
        hand_layers = set(int(x.strip()) for x in args.hand_interm_pred_layers.split(",") if x.strip())
    body_layers = None
    if args.body_interm_pred_layers:
        body_layers = set(int(x.strip()) for x in args.body_interm_pred_layers.split(",") if x.strip())

    publisher = RealtimePublisherJoints(
        video_source=video_source,
        publish_hz=args.publish_hz,
        interpolate_lag_ms=args.interp_lag_ms,
        smpl_model_path=args.smpl_model_path,
        gmr_human_height=args.gmr_human_height,
        addr=args.addr,
        topic=args.topic,
        image_size=args.image_size,
        yolo_model_path=args.yolo_model,
        hand_interm_pred_layers=hand_layers,
        body_interm_pred_layers=body_layers,
    )

    try:
        publisher.start()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        publisher.stop()
        logger.success("Stopped.")


if __name__ == "__main__":
    main()
