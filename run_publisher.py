import argparse
import os
import threading
import time
from collections import deque

# Disable CUDA graphs for torch.compile in multi-threaded environment
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPH_TREES", "0")

import cv2
import numpy as np
import torch
from loguru import logger
from scipy.spatial.transform import Rotation

from fast_mocap.core.gravity_alignment import (
    build_camera_to_world_rotation,
    transform_pose_to_world,
)
from fast_mocap.realtime.interpolator import PoseInterpolator
from fast_mocap.core.mhr_to_smpl_interface import (
    METHOD_FAST_XYZ_JOINT_FIT,
    METHOD_NN_MLP_VERTS,
    build_mhr_to_smpl_converter,
)
from fast_mocap.realtime.publisher import ZMQPublisher
from fast_mocap.core.setup_estimator import build_default_estimator
from fast_mocap.utils.smpl_render_utils import (
    load_smpl_model,
    smpl_vertices_joints_from_pose,
)
from fast_mocap.utils.video_source import create_video_source

from recording_manager_thread_sam import RecordingManagerThreadSAM


FOV_MODEL_SIZE = "s"
FOV_RESOLUTION_LEVEL = 0
FOV_FIXED_SIZE = 512
FOV_FAST_MODE = True
FOV_DEBUG = False
BACKBONE_MODE = "tensorrt"
BACKBONE_TRT_ENGINE = "./checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine"
BACKBONE_COMPILE_MODE = "default"
YOLO_MODEL_PATH = "checkpoints/yolo_pose/yolo11m-pose.engine"
RUN_PUBLISHER_SUPPORTED_METHODS = (
    METHOD_FAST_XYZ_JOINT_FIT,
    METHOD_NN_MLP_VERTS,
)

# Coordinate transform from SMPL body-local frame to protocol frame.
# Keep identity here: after quaternion/base-rotation handling, joints are already
# in the expected protocol local frame.
JOINTS_COORD_TRANSFORM = np.eye(3, dtype=np.float64)

# Match PICO protocol: remove SMPL fixed base rotation from published body quaternion.
# This is the conjugate of [0.5, 0.5, 0.5, 0.5] in xyzw order.
SMPL_BASE_REMOVE_QUAT_XYZW = np.array([-0.5, -0.5, -0.5, 0.5], dtype=np.float64)
# Global orientation adjustment (reversed order): apply X-90 first, then Y+90.
GLOBAL_ORIENT_EXTRA_ROT = Rotation.from_euler("y", 90.0, degrees=True) * Rotation.from_euler(
    "x", -90.0, degrees=True
)


def quat_apply(quat, vec):
    qw, qx, qy, qz = quat
    qvec = np.array([qx, qy, qz])
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec + 2.0 * (uv * qw + uuv)


def quat_inverse(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=quat.dtype)


def quat_wxyz_to_xyzw(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


class RealtimePublisher:
    def __init__(
        self,
        video_source,
        publish_hz,
        interpolate_lag_ms,
        smpl_model_path,
        conversion_method=METHOD_FAST_XYZ_JOINT_FIT,
        nn_model_dir=None,
        smoother_dir=None,
        addr="tcp://*:5556",
        image_size=512,
        yolo_model_path=YOLO_MODEL_PATH,
        hand_interm_pred_layers=None,
        body_interm_pred_layers=None,
    ):
        logger.info("Initializing Realtime Publisher...")

        logger.info("Loading SAM 3D model...")
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
        self.conversion_method = conversion_method
        if self.conversion_method not in RUN_PUBLISHER_SUPPORTED_METHODS:
            raise RuntimeError(
                "run_publisher supports only conversion_method in "
                f"{RUN_PUBLISHER_SUPPORTED_METHODS}, got {self.conversion_method}"
            )

        logger.info("Warming up model...")
        self._warmup()

        self.cam_intrinsics = self.video_source.get_camera_intrinsics()
        if self.cam_intrinsics is not None:
            logger.info(
                f"Using camera intrinsics: fx={self.cam_intrinsics[0,0,0]:.2f}, fy={self.cam_intrinsics[0,1,1]:.2f}"
            )
        else:
            logger.warning("No camera intrinsics provided, will use FOV estimator")

        # Get gravity direction for world frame alignment (required)
        self.gravity_direction = self.video_source.get_gravity_direction()
        logger.info(
            f"Using gravity-aligned world frame: gravity=[{self.gravity_direction[0]:.3f}, {self.gravity_direction[1]:.3f}, {self.gravity_direction[2]:.3f}]"
        )
        self.R_world_cam = build_camera_to_world_rotation(self.gravity_direction)
        R_zup_adjustment = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
        self.R_world_cam = R_zup_adjustment @ self.R_world_cam

        logger.info(
            f"Loading SMPL converter method={self.conversion_method} model={smpl_model_path}..."
        )
        smpl_model, _, device, num_betas = load_smpl_model(smpl_model_path)
        rest_smpl_joints_24 = None
        if self.conversion_method == METHOD_FAST_XYZ_JOINT_FIT:
            rest_pose_21 = np.zeros((21, 3), dtype=np.float32)
            _, rest_joints = smpl_vertices_joints_from_pose(
                rest_pose_21,
                smpl_model,
                device,
                num_betas,
                body_quat=None,
            )
            rest_smpl_joints_24 = np.asarray(rest_joints[:24], dtype=np.float32)
        self.converter = build_mhr_to_smpl_converter(
            method=self.conversion_method,
            smpl_model=smpl_model,
            device=device,
            num_betas=num_betas if self.conversion_method == METHOD_FAST_XYZ_JOINT_FIT else None,
            rest_smpl_joints_24=(
                rest_smpl_joints_24
                if self.conversion_method == METHOD_FAST_XYZ_JOINT_FIT
                else None
            ),
            nn_model_dir=nn_model_dir if self.conversion_method == METHOD_NN_MLP_VERTS else None,
            smoother_dir=smoother_dir if self.conversion_method == METHOD_NN_MLP_VERTS else None,
        )

        self.publish_hz = publish_hz
        self.publish_dt = 1.0 / publish_hz
        self.interpolate_lag_s = interpolate_lag_ms / 1000.0

        self.interpolator = PoseInterpolator()
        self.publisher = ZMQPublisher(addr)

        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self._frame_event = threading.Event()

        self.running = False
        self.video_ended = False
        self._final_stats_logged = False
        self._closed = False

        self._capture_wall_base = None
        self._capture_ts_base = None
        self._pose_clock_lock = threading.Lock()
        self._latest_pose_source_ts = None
        self._latest_pose_perf_ts = None

        self.capture_thread = None
        self.inference_thread = None
        self.publish_thread = None

        self.first_capture_ts = None
        self.first_infer_ts = None

        self.stats = {
            "capture_count": 0,
            "dropped_capture_count": 0,
            "infer_count": 0,
            "inference_times": deque(maxlen=100),
            "infer_total_time_s": 0.0,
            "stage_detect_times": deque(maxlen=100),
            "stage_prepare_times": deque(maxlen=100),
            "stage_cam_times": deque(maxlen=100),
            "stage_model_times": deque(maxlen=100),
            "stage_post_times": deque(maxlen=100),
            "convert_times": deque(maxlen=100),
            "convert_total_time_s": 0.0,
            "publish_count": 0,
            "publish_intervals": deque(maxlen=500),
            "publish_interpolated_count": 0,
            "publish_fallback_count": 0,
        }

        self._live_log_interval_s = 2.0
        self._live_last_log_perf = time.perf_counter()
        self._live_prev_stats = {
            "capture_count": 0,
            "dropped_capture_count": 0,
            "infer_count": 0,
            "infer_total_time_s": 0.0,
            "convert_total_time_s": 0.0,
            "publish_count": 0,
            "publish_interpolated_count": 0,
            "publish_fallback_count": 0,
        }

        logger.success("Publisher ready")

    def _warmup(self):
        frame_size = self.video_source.get_frame_size()
        if frame_size is None:
            width, height = 640, 480
        else:
            width, height = frame_size

        dummy_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        warmup_bbox = np.array(
            [[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32
        )
        for _ in range(2):
            _ = self.estimator.process_one_image(
                dummy_img,
                bboxes=warmup_bbox,
                hand_box_source="body_decoder",
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _capture_loop(self):
        while self.running:
            try:
                frame, frame_timestamp = self.video_source.get_frame()
            except Exception as exc:
                if self.running:
                    logger.warning(f"Capture loop stopped due to source error: {exc}")
                self.video_ended = True
                self._frame_event.set()
                break

            if frame is None:
                self.video_ended = True
                self._frame_event.set()
                break

            if frame_timestamp is None:
                continue

            if self.first_capture_ts is None:
                self.first_capture_ts = frame_timestamp

            if self._capture_wall_base is None:
                self._capture_wall_base = time.perf_counter()
                self._capture_ts_base = frame_timestamp
            else:
                target_wall = self._capture_wall_base + (
                    frame_timestamp - self._capture_ts_base
                )
                now_wall = time.perf_counter()
                delay = target_wall - now_wall
                if delay > 0:
                    time.sleep(delay)

            self.stats["capture_count"] += 1
            with self._latest_frame_lock:
                if self._latest_frame is not None:
                    self.stats["dropped_capture_count"] += 1
                self._latest_frame = (frame, frame_timestamp)
            self._frame_event.set()

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

            if self.first_infer_ts is None:
                self.first_infer_ts = frame_timestamp

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            outputs, timing_ms = self.estimator.process_one_image(
                frame_rgb,
                cam_int=self.cam_intrinsics,
                hand_box_source="yolo_pose",
                return_timing=True,
            )
            infer_dt = timing_ms["total"] / 1000.0
            self.stats["inference_times"].append(infer_dt)
            self.stats["infer_total_time_s"] += infer_dt
            self.stats["stage_detect_times"].append(timing_ms["detect"] / 1000.0)
            self.stats["stage_prepare_times"].append(timing_ms["prepare"] / 1000.0)
            self.stats["stage_cam_times"].append(timing_ms["cam"] / 1000.0)
            self.stats["stage_model_times"].append(timing_ms["model"] / 1000.0)
            self.stats["stage_post_times"].append(timing_ms["post"] / 1000.0)
            self.stats["infer_count"] += 1

            if len(outputs) == 0:
                continue

            t0 = time.perf_counter()
            out = outputs[0]
            if self.conversion_method == METHOD_FAST_XYZ_JOINT_FIT:
                required = ("pred_joint_coords", "pred_cam_t")
                missing = [k for k in required if k not in out]
                if missing:
                    logger.warning(
                        f"Skipping frame missing keys for fast conversion: missing={missing}"
                    )
                    continue
                frame_data = {
                    "pred_joint_coords": np.asarray(out["pred_joint_coords"], dtype=np.float32),
                    "pred_cam_t": np.asarray(out["pred_cam_t"], dtype=np.float32),
                }
            elif self.conversion_method == METHOD_NN_MLP_VERTS:
                required = ("pred_vertices", "pred_cam_t")
                missing = [k for k in required if k not in out]
                if missing:
                    logger.warning(
                        f"Skipping frame missing keys for nn conversion: missing={missing}"
                    )
                    continue
                frame_data = {
                    "pred_vertices": np.asarray(out["pred_vertices"], dtype=np.float32),
                    "pred_cam_t": np.asarray(out["pred_cam_t"], dtype=np.float32),
                }
            else:
                raise RuntimeError(
                    f"Unsupported conversion method in run_publisher: {self.conversion_method}"
                )
            result = self.converter.convert_frame(frame_data)
            if result.body_quat is None or result.smpl_joints is None or result.smpl_pose is None:
                logger.warning("Skipping frame due to incomplete converter output")
                continue
            body_quat, smpl_joints, smpl_pose = self._prepare_publish_pose(result)
            convert_dt = time.perf_counter() - t0
            self.stats["convert_times"].append(convert_dt)
            self.stats["convert_total_time_s"] += convert_dt

            self.interpolator.add_pose(
                frame_timestamp, body_quat, smpl_joints, smpl_pose
            )
            with self._pose_clock_lock:
                self._latest_pose_source_ts = frame_timestamp
                self._latest_pose_perf_ts = time.perf_counter()

    def _prepare_publish_pose(self, result):
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

    def _maybe_log_live_stats(self, now_perf):
        elapsed = now_perf - self._live_last_log_perf
        if elapsed < self._live_log_interval_s:
            return

        curr = {
            "capture_count": self.stats["capture_count"],
            "dropped_capture_count": self.stats["dropped_capture_count"],
            "infer_count": self.stats["infer_count"],
            "infer_total_time_s": self.stats["infer_total_time_s"],
            "convert_total_time_s": self.stats["convert_total_time_s"],
            "publish_count": self.stats["publish_count"],
            "publish_interpolated_count": self.stats["publish_interpolated_count"],
            "publish_fallback_count": self.stats["publish_fallback_count"],
        }
        prev = self._live_prev_stats

        d_capture = curr["capture_count"] - prev["capture_count"]
        d_drop = curr["dropped_capture_count"] - prev["dropped_capture_count"]
        d_infer = curr["infer_count"] - prev["infer_count"]
        d_infer_time = curr["infer_total_time_s"] - prev["infer_total_time_s"]
        d_convert_time = curr["convert_total_time_s"] - prev["convert_total_time_s"]
        d_publish = curr["publish_count"] - prev["publish_count"]
        d_interp = curr["publish_interpolated_count"] - prev["publish_interpolated_count"]
        d_fallback = curr["publish_fallback_count"] - prev["publish_fallback_count"]

        infer_fps = d_infer / elapsed
        infer_ms = (d_infer_time / d_infer * 1000.0) if d_infer > 0 else float("nan")
        convert_ms = (d_convert_time / d_infer * 1000.0) if d_infer > 0 else float("nan")
        publish_hz = d_publish / elapsed
        fallback_pct = (d_fallback / d_publish * 100.0) if d_publish > 0 else 0.0

        infer_ms_str = f"{infer_ms:.1f}" if np.isfinite(infer_ms) else "n/a"
        convert_ms_str = f"{convert_ms:.1f}" if np.isfinite(convert_ms) else "n/a"
        logger.info(
            "Live: "
            f"capture={d_capture/elapsed:.1f}fps drop+={d_drop}, "
            f"infer throughput={infer_fps:.1f}fps model={infer_ms_str}ms, "
            f"convert={convert_ms_str}ms, "
            f"publish={publish_hz:.1f}Hz, "
            f"interp+={d_interp}, fallback+={d_fallback} ({fallback_pct:.1f}%)"
        )

        self._live_prev_stats = curr
        self._live_last_log_perf = now_perf

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
                source_now_est = latest_pose_source_ts + (
                    now_perf - latest_pose_perf_ts
                )
                query_ts = source_now_est - self.interpolate_lag_s

            result = self.interpolator.interpolate(query_ts)
            used_fallback = False
            if result is None:
                latest = self.interpolator.get_latest_pose()
                if latest is not None:
                    result = latest
                    used_fallback = True

            if result is not None:
                self.publisher.publish(*result)
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

            if (
                self.video_ended
                and self.stats["infer_count"] > 0
            ):
                if self.stats["publish_count"] > max(10, int(self.publish_hz * 0.4)):
                    break

            self._maybe_log_live_stats(now_perf)

    def _log_final_stats(self):
        if self._final_stats_logged:
            return

        inf_msg = "Inference n/a"
        stage_msg = ""
        if self.stats["inference_times"]:
            inf_mean = np.mean(self.stats["inference_times"]) * 1000
            inf_fps = 1.0 / np.mean(self.stats["inference_times"])
            inf_msg = f"Inference {inf_mean:.1f}ms ({inf_fps:.1f}fps)"

            stage_msg = (
                f", stage(det={np.mean(self.stats['stage_detect_times']) * 1000:.1f}ms"
                f", prep={np.mean(self.stats['stage_prepare_times']) * 1000:.1f}ms"
                f", cam={np.mean(self.stats['stage_cam_times']) * 1000:.1f}ms"
                f", model={np.mean(self.stats['stage_model_times']) * 1000:.1f}ms"
                f", post={np.mean(self.stats['stage_post_times']) * 1000:.1f}ms)"
            )
            if self.stats["convert_times"]:
                stage_msg += f", convert={np.mean(self.stats['convert_times']) * 1000:.1f}ms"

        pub_msg = "Publish n/a"
        if self.stats["publish_intervals"]:
            pub_hz = 1.0 / np.mean(self.stats["publish_intervals"])
            pub_msg = f"Publish {pub_hz:.1f}Hz (target {self.publish_hz:.1f}Hz)"

        logger.info(
            f"Final stats: {inf_msg}{stage_msg}, {pub_msg}, "
            f"published={self.stats['publish_count']}, "
            f"interp={self.stats['publish_interpolated_count']}, "
            f"fallback={self.stats['publish_fallback_count']}, "
            f"capture_drop={self.stats['dropped_capture_count']}"
        )
        self._final_stats_logged = True

    def _release_video_source_with_timeout(self, timeout_s=1.0):
        release_errors = []

        def _release():
            try:
                self.video_source.release()
            except Exception as exc:
                release_errors.append(exc)

        t = threading.Thread(target=_release, daemon=True)
        t.start()
        t.join(timeout=timeout_s)

        if t.is_alive():
            logger.warning("Timed out while releasing video source; continue shutdown")
        elif release_errors:
            logger.warning(f"Video source release raised: {release_errors[0]}")

    def start(self):
        logger.info("Starting realtime publisher (Press Ctrl+C to stop)")
        self.running = True

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)

        self.capture_thread.start()
        self.inference_thread.start()
        self.publish_thread.start()

        while self.running:
            if (
                self.video_ended
                and not self.inference_thread.is_alive()
            ):
                self.running = False
                break
            if (
                not self.capture_thread.is_alive()
                and not self.inference_thread.is_alive()
            ):
                self.running = False
                break
            time.sleep(0.05)

        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        if self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)

        self._log_final_stats()

    def stop(self):
        if self._closed:
            return

        self.running = False
        self._release_video_source_with_timeout(timeout_s=1.0)

        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.inference_thread is not None and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        if self.publish_thread is not None and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)

        self._log_final_stats()

        self.publisher.close()
        self._closed = True


def main():
    parser = argparse.ArgumentParser(
        description="Publish SAM 3D pose from camera/video stream over ZMQ"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="camera",
        choices=["camera", "video"],
        help="Video source type",
    )
    parser.add_argument(
        "--video", type=str, help="Path to video file (for --source video)"
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        help="Camera intrinsics JSON path (required for --source video)",
    )
    parser.add_argument(
        "--no-loop",
        dest="no_loop",
        action="store_true",
        help="Disable loop video playback (for --source video)",
    )
    parser.add_argument(
        "--publish-hz", type=float, default=50.0, help="Publisher output rate in Hz"
    )
    parser.add_argument(
        "--interp-lag-ms",
        type=float,
        default=140.0,
        help="Interpolation lag in ms to make 10Hz inference interpolatable",
    )
    parser.add_argument(
        "--addr", type=str, default="tcp://*:5556", help="ZMQ bind address"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        choices=[256, 384, 512],
        help="Image size for SAM3D model (must match TensorRT engine)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=YOLO_MODEL_PATH,
        help="YOLO pose model path (e.g., yolo11m-pose.engine or yolo11n-pose.engine)",
    )
    parser.add_argument(
        "--hand-interm-pred-layers",
        type=str,
        default=None,
        help=(
            "Comma-separated layer indices for hand intermediate predictions "
            "(e.g. '0,1,2'). Pass '999' to disable. Default: run every layer."
        ),
    )
    parser.add_argument(
        "--body-interm-pred-layers",
        type=str,
        default=None,
        help=(
            "Comma-separated layer indices for body intermediate predictions "
            "(e.g. '0,1,2'). Pass '999' to disable. Default: run every layer."
        ),
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        required=True,
        help="SMPL model pickle file",
    )
    parser.add_argument(
        "--conversion-method",
        type=str,
        choices=list(RUN_PUBLISHER_SUPPORTED_METHODS),
        default=METHOD_FAST_XYZ_JOINT_FIT,
        help="MHR to SMPL conversion method",
    )
    parser.add_argument(
        "--nn-model-dir",
        type=str,
        default=None,
        help="NN mhr2smpl model directory (contains config.json and best_model.pth)",
    )
    parser.add_argument(
        "--smoother-dir",
        type=str,
        default=None,
        help=(
            "Smoother checkpoint directory (contains smoother_best.pth and smoother_config.json). "
            "Only valid with --conversion-method=nn-mlp-verts."
        ),
    )
    args = parser.parse_args()

    hand_interm_pred_layers = None
    if args.hand_interm_pred_layers is not None:
        hand_interm_pred_layers = set(
            int(x.strip()) for x in args.hand_interm_pred_layers.split(",") if x.strip()
        )
    body_interm_pred_layers = None
    if args.body_interm_pred_layers is not None:
        body_interm_pred_layers = set(
            int(x.strip()) for x in args.body_interm_pred_layers.split(",") if x.strip()
        )

    if args.publish_hz <= 0:
        parser.error("--publish-hz must be > 0")
    if args.interp_lag_ms < 0:
        parser.error("--interp-lag-ms must be >= 0")
    if args.conversion_method == METHOD_NN_MLP_VERTS and not args.nn_model_dir:
        parser.error("--nn-model-dir is required when --conversion-method=nn-mlp-verts")
    if args.conversion_method != METHOD_NN_MLP_VERTS and args.nn_model_dir:
        parser.error("--nn-model-dir can only be used with --conversion-method=nn-mlp-verts")
    if args.smoother_dir and args.conversion_method != METHOD_NN_MLP_VERTS:
        parser.error("--smoother-dir can only be used with --conversion-method=nn-mlp-verts")

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

    publisher = RealtimePublisher(
        video_source=video_source,
        publish_hz=args.publish_hz,
        interpolate_lag_ms=args.interp_lag_ms,
        smpl_model_path=args.smpl_model_path,
        conversion_method=args.conversion_method,
        nn_model_dir=args.nn_model_dir,
        smoother_dir=args.smoother_dir,
        addr=args.addr,
        image_size=args.image_size,
        yolo_model_path=args.yolo_model,
        hand_interm_pred_layers=hand_interm_pred_layers,
        body_interm_pred_layers=body_interm_pred_layers,
    )

    recording_manager = RecordingManagerThreadSAM()
    recording_manager.start()

    try:
        publisher.start()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        publisher.stop()
        recording_manager.stop()
        recording_manager.join(timeout=2.0)
        logger.success("Stopped.")


if __name__ == "__main__":
    main()
