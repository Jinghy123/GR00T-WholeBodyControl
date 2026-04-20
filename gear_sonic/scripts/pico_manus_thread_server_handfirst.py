# Hand-first variant of pico_manus_thread_server.
#
# Goal: when Pico body data is missing/stale, keep publishing pose messages so
# the deploy side (v3 protocol) — which hard-requires body_quat / smpl_* /
# joint_pos / joint_vel — continues to accept frames. Body fields are filled
# from the last cached frame; hand fields come from fresh Manus retargeting.
#
# This file does NOT modify pico_manus_thread_server.py. It imports the base
# module, subclasses PoseStreamer with a cached-body-fields override, and
# monkey-patches base.PoseStreamer so _pose_stream_common / run_pico_manager
# pick up the subclass automatically.

import os
import sys
import time
from collections import deque

import numpy as np

# Make sibling module importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pico_manus_thread_server as base  # noqa: E402


class HandFirstPoseStreamer(base.PoseStreamer):
    """PoseStreamer that keeps publishing when Pico body data stalls.

    Normal path (body fresh): identical to base.PoseStreamer.run_once — fills
    the sliding buffer, packs numpy_data, sends, and caches the sent dict.

    Stale path (body None or not advancing): if a previous frame is cached,
    reuse its body-shaped fields, overwrite hand joints with fresh Manus
    output, bump frame_index monotonically, and send. deploy side is unaware.
    """

    # How long to wait before considering body "stale" and switching to the
    # cached-body path. Chosen a bit above one Pico frame period at 30 Hz.
    STALE_BODY_TIMEOUT_S = 0.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache of the last numpy_data we successfully packed + sent.
        # Initialized to a zero-body snapshot so the stale path works from
        # frame 0 even if Pico never provides body data (hand-only mode).
        self._cached_numpy_data = self._build_zero_body_cache()
        # Wall-clock time of last fresh body sample observed.
        self._last_body_time = 0.0
        # Monotonic counter for stale-path frame_index bumps.
        self._stale_frame_offset = 0

    def _build_zero_body_cache(self) -> dict:
        """Synthesize a zero-body / identity-orientation numpy_data dict.

        Shapes mirror what the body-fresh path emits. Hand joints are zero;
        they'll be overwritten with fresh Manus output before each send.
        """
        N = self.num_frames_to_send
        # Identity unit quaternion (w, x, y, z) — w-first convention.
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        body_quat = np.tile(identity_quat, (N, 1))

        # vr_orientation is a flattened (3, 4) block — 3 joints × wxyz identity.
        vr_orientation = np.tile(identity_quat, 3).astype(np.float32)

        pico_dt = 1.0 / max(1, self.target_fps)
        now_realtime = time.time()
        now_monotonic = time.monotonic()

        return {
            "smpl_pose": np.zeros((N, 21, 3), dtype=np.float32),
            "smpl_joints": np.zeros((N, 24, 3), dtype=np.float32),
            "body_quat_w": body_quat,
            "joint_pos": np.zeros((N, 29), dtype=np.float64),
            "joint_vel": np.zeros((N, 29), dtype=np.float64),
            "vr_position": np.zeros(9, dtype=np.float32),
            "vr_orientation": vr_orientation,
            "frame_index": np.arange(N, dtype=np.int64),
            "left_trigger": np.zeros(1, dtype=np.float32),
            "right_trigger": np.zeros(1, dtype=np.float32),
            "left_grip": np.zeros(1, dtype=np.float32),
            "right_grip": np.zeros(1, dtype=np.float32),
            "pico_dt": np.array([pico_dt], dtype=np.float32),
            "pico_fps": np.array([float(self.target_fps)], dtype=np.float32),
            "timestamp_realtime": np.array([now_realtime], dtype=np.float64),
            "timestamp_monotonic": np.array([now_monotonic], dtype=np.float64),
            "left_hand_joints": np.zeros(7, dtype=np.float32),
            "right_hand_joints": np.zeros(7, dtype=np.float32),
            "toggle_data_collection": np.zeros(1, dtype=bool),
            "toggle_data_abort": np.zeros(1, dtype=bool),
            "heading_increment": np.zeros(1, dtype=np.float32),
        }

    def on_mode_exit(self):
        super().on_mode_exit()
        # Reset cache to zero-body so hand-only streaming still works after
        # a mode re-entry. _stale_frame_offset restarts from 0.
        self._cached_numpy_data = self._build_zero_body_cache()
        self._last_body_time = 0.0
        self._stale_frame_offset = 0

    def run_once(self):
        sample = self.reader.get_latest()
        now = time.time()

        body_fresh = sample is not None and (
            self.prev_stamp_ns is None
            or int(sample.get("timestamp_ns", 0)) > self.prev_stamp_ns
        )

        if body_fresh:
            self._last_body_time = now
            self._run_once_with_body(sample)
            return

        # Body stale or absent. Only take the stale path if we have a cached
        # frame to base the body fields on; otherwise fall back to the parent
        # behavior (which just sleeps until body arrives).
        if self._cached_numpy_data is None:
            time.sleep(0.005)
            return

        # Publish stale-body + fresh-hand frame at target fps.
        elapsed = now - getattr(self, "frame_start", now)
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self._run_once_stale_body()
        self.frame_start = time.time()

    # ----- body-fresh path ------------------------------------------------

    def _run_once_with_body(self, sample):
        """Same as base.PoseStreamer.run_once but caches the packed dict."""
        latest_data = base.compute_from_body_poses(
            self.parent_indices, self.device, sample["body_poses_np"]
        )
        (left_menu_button, left_trigger, right_trigger, left_grip, right_grip) = (
            base.get_controller_inputs()
        )
        a_pressed, b_pressed, x_pressed, y_pressed = base.get_abxy_buttons()

        toggle_data_collection_tmp = a_pressed and left_grip > 0.5
        toggle_data_abort_tmp = b_pressed and left_grip > 0.5
        toggle_data_collection = (
            toggle_data_collection_tmp and not self.toggle_data_collection_last
        )
        toggle_data_abort = toggle_data_abort_tmp and not self.toggle_data_abort_last
        self.toggle_data_collection_last = toggle_data_collection_tmp
        self.toggle_data_abort_last = toggle_data_abort_tmp

        left_hand_joints, right_hand_joints = base.compute_hand_joints_from_manus(
            self.hand_retargeting,
            self.manus_receiver,
        )

        smpl_pose_np = (
            latest_data["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)[0]
        ).astype(np.float32)
        smpl_joints_np = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0].astype(np.float32)
        )
        body_quat_np = (
            latest_data["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)
        )

        curr_stamp_ns = int(sample.get("timestamp_ns", 0))
        step_ns = int(1e9 / max(1, self.target_fps))
        if self.prev_stamp_ns is None:
            self.prev_stamp_ns = curr_stamp_ns
            self.prev_smpl_pose_np = smpl_pose_np
            self.prev_smpl_joints_np = smpl_joints_np
            self.prev_body_quat_np = body_quat_np
            self.next_target_ns = curr_stamp_ns
            return
        if curr_stamp_ns <= self.prev_stamp_ns:
            return
        if self.next_target_ns is None:
            self.next_target_ns = self.prev_stamp_ns + step_ns
        if self.next_target_ns < self.prev_stamp_ns:
            self.next_target_ns = self.prev_stamp_ns
        if self.next_target_ns > curr_stamp_ns:
            return

        denom = float(curr_stamp_ns - self.prev_stamp_ns)
        alpha = float(self.next_target_ns - self.prev_stamp_ns) / denom if denom > 0.0 else 1.0
        alpha = max(0.0, min(1.0, alpha))

        use_joints = (1.0 - alpha) * self.prev_smpl_joints_np + alpha * smpl_joints_np
        use_pose = base._interp_pose_axis_angle(
            self.prev_smpl_pose_np, smpl_pose_np, alpha
        ).astype(np.float32)
        use_body_quat = base._quat_lerp_normalized(
            self.prev_body_quat_np, body_quat_np, alpha
        ).astype(np.float32)

        joint_pos = self._wrist_joint_pos_from_smpl_pose(use_pose)

        smpl_joints_for_vis = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0]
            if self.three_point.enable_smpl_vis
            else None
        )
        vr_3pt_pose = self.three_point.process_smpl_pose(
            sample["body_poses_np"], smpl_joints_local=smpl_joints_for_vis
        )

        self.frame_buffer["smpl_pose"].append(use_pose)
        self.frame_buffer["smpl_joints"].append(use_joints)
        self.frame_buffer["body_quat_w"].append(use_body_quat)
        self.frame_buffer["frame_index"].append(int(self.step))
        self.frame_buffer["joint_pos"].append(joint_pos)
        pico_dt = float(sample.get("dt", 0.0))
        pico_fps = float(sample.get("fps", 0.0))

        buffer_is_full = len(self.frame_buffer["frame_index"]) >= self.num_frames_to_send
        if buffer_is_full and self.buffer_cleared:
            self.buffer_cleared = False

        _, _, rx, _ = base.get_controller_axes()
        self.yaw_accumulator.update(rx, self.frame_time)

        if buffer_is_full and not self.buffer_cleared:
            N = len(self.frame_buffer["frame_index"])
            numpy_data = {
                "smpl_pose": np.stack(self.frame_buffer["smpl_pose"], axis=0),
                "smpl_joints": np.stack(self.frame_buffer["smpl_joints"], axis=0),
                "body_quat_w": np.stack(self.frame_buffer["body_quat_w"], axis=0),
                "joint_pos": np.stack(self.frame_buffer["joint_pos"], axis=0),
                "joint_vel": np.zeros((N, 29)),
                "vr_position": vr_3pt_pose[:, :3].flatten(),
                "vr_orientation": vr_3pt_pose[:, 3:].flatten(),
                "frame_index": np.array(self.frame_buffer["frame_index"], dtype=np.int64),
                "left_trigger": np.array([left_trigger], dtype=np.float32),
                "right_trigger": np.array([right_trigger], dtype=np.float32),
                "left_grip": np.array([left_grip], dtype=np.float32),
                "right_grip": np.array([right_grip], dtype=np.float32),
                "pico_dt": np.array([pico_dt], dtype=np.float32),
                "pico_fps": np.array([pico_fps], dtype=np.float32),
                "timestamp_realtime": np.array(
                    [sample.get("timestamp_realtime", 0.0)], dtype=np.float64
                ),
                "timestamp_monotonic": np.array(
                    [sample.get("timestamp_monotonic", 0.0)], dtype=np.float64
                ),
                "left_hand_joints": left_hand_joints.reshape(-1).astype(np.float32),
                "right_hand_joints": right_hand_joints.reshape(-1).astype(np.float32),
                "toggle_data_collection": np.array([toggle_data_collection], dtype=bool),
                "toggle_data_abort": np.array([toggle_data_abort], dtype=bool),
                "heading_increment": np.array(
                    [self.yaw_accumulator.yaw_angle_change()], dtype=np.float32
                ),
            }

            packed_message = base.pack_pose_message(numpy_data, topic="pose")
            self.socket.send(packed_message)

            # Cache for the stale path. The next stale send will reuse the
            # body-shaped fields and swap in fresh hand joints.
            self._cached_numpy_data = numpy_data
            self._stale_frame_offset = 0

            if self.record_dir:
                out_path = os.path.join(self.record_dir, f"pose_{self.record_idx:06d}.npz")
                np.savez_compressed(out_path, **numpy_data)
                self.record_idx += 1

        self.step += 1
        self.next_target_ns += step_ns
        self.prev_stamp_ns = curr_stamp_ns
        self.prev_smpl_pose_np = smpl_pose_np
        self.prev_smpl_joints_np = smpl_joints_np
        self.prev_body_quat_np = body_quat_np
        self.fps_counter += 1

        current_time = time.time()
        if current_time - self.last_fps_report >= 5.0:
            fps = self.fps_counter / (current_time - self.last_fps_report)
            print(f"[{self.log_prefix}] FPS (body-fresh): {fps:.2f}, Step: {self.step}")
            self.fps_counter = 0
            self.last_fps_report = current_time
        elapsed = time.time() - self.frame_start
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self.frame_start = time.time()

    # ----- stale-body path ------------------------------------------------

    def _run_once_stale_body(self):
        """Publish a pose frame using cached body fields and fresh hand joints."""
        cached = self._cached_numpy_data
        left_hand_joints, right_hand_joints = base.compute_hand_joints_from_manus(
            self.hand_retargeting,
            self.manus_receiver,
        )

        # Keep frame_index monotonically increasing. Add a per-stale-step
        # offset to each element of the cached frame_index array.
        self._stale_frame_offset += 1
        frame_index = cached["frame_index"].astype(np.int64) + self._stale_frame_offset

        # Edge-triggered button states still need polling so the operator can
        # abort / toggle recording even while body is stale.
        a_pressed, b_pressed, _, _ = base.get_abxy_buttons()
        _, _, _, left_grip, _ = base.get_controller_inputs()
        toggle_data_collection_tmp = a_pressed and left_grip > 0.5
        toggle_data_abort_tmp = b_pressed and left_grip > 0.5
        toggle_data_collection = (
            toggle_data_collection_tmp and not self.toggle_data_collection_last
        )
        toggle_data_abort = toggle_data_abort_tmp and not self.toggle_data_abort_last
        self.toggle_data_collection_last = toggle_data_collection_tmp
        self.toggle_data_abort_last = toggle_data_abort_tmp

        _, _, rx, _ = base.get_controller_axes()
        self.yaw_accumulator.update(rx, self.frame_time)

        numpy_data = dict(cached)  # shallow copy is fine; arrays below are replaced
        numpy_data["frame_index"] = frame_index
        numpy_data["left_hand_joints"] = left_hand_joints.reshape(-1).astype(np.float32)
        numpy_data["right_hand_joints"] = right_hand_joints.reshape(-1).astype(np.float32)
        numpy_data["toggle_data_collection"] = np.array([toggle_data_collection], dtype=bool)
        numpy_data["toggle_data_abort"] = np.array([toggle_data_abort], dtype=bool)
        numpy_data["heading_increment"] = np.array(
            [self.yaw_accumulator.yaw_angle_change()], dtype=np.float32
        )

        packed_message = base.pack_pose_message(numpy_data, topic="pose")
        self.socket.send(packed_message)

        if self.record_dir:
            out_path = os.path.join(self.record_dir, f"pose_{self.record_idx:06d}.npz")
            np.savez_compressed(out_path, **numpy_data)
            self.record_idx += 1

        self.step += 1
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_report >= 5.0:
            fps = self.fps_counter / (current_time - self.last_fps_report)
            print(
                f"[{self.log_prefix}] FPS (stale-body+fresh-hand): {fps:.2f}, "
                f"Step: {self.step}, stale_offset: {self._stale_frame_offset}"
            )
            self.fps_counter = 0
            self.last_fps_report = current_time

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _wrist_joint_pos_from_smpl_pose(use_pose: np.ndarray) -> np.ndarray:
        """Port of the joint_pos block in base.PoseStreamer.run_once.

        Extracts G1 wrist roll/pitch/yaw from SMPL elbow+wrist axis-angle.
        """
        from scipy.spatial.transform import Rotation as R  # local import, cheap
        from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa

        joint_pos = np.zeros(29)
        body_pose = use_pose.reshape(-1, 21, 3)

        SMPL_L_ELBOW_IDX = 17
        SMPL_L_WRIST_IDX = 19
        SMPL_R_ELBOW_IDX = 18
        SMPL_R_WRIST_IDX = 20

        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27
        G1_R_WRIST_ROLL_IDX = 24
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28

        smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
        smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
        smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
        smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

        g1_l_elbow_axis = np.array([0, 1, 0])
        _, g1_l_elbow_q_swing = decompose_rotation_aa(smpl_l_elbow_aa, g1_l_elbow_axis)
        g1_r_elbow_axis = np.array([0, 1, 0])
        _, g1_r_elbow_q_swing = decompose_rotation_aa(smpl_r_elbow_aa, g1_r_elbow_axis)

        l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )
        r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )

        l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
        r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

        g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
        g1_l_wrist_pitch = -l_wrist_euler[:, 1]
        g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

        g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
        g1_r_wrist_pitch = -r_wrist_euler[:, 1]
        g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

        joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0]
        joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0]
        joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0]
        joint_pos[G1_R_WRIST_ROLL_IDX] = g1_r_wrist_roll[0]
        joint_pos[G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch[0]
        joint_pos[G1_R_WRIST_YAW_IDX] = g1_r_wrist_yaw[0]

        return joint_pos


# Monkey-patch: both run_pico / run_pico_manager / _pose_stream_common resolve
# PoseStreamer via the base module's globals, so replacing the attribute there
# redirects their construction to the hand-first subclass without touching the
# base file.
base.PoseStreamer = HandFirstPoseStreamer


def run_pico_handonly(
    buffer_size: int = 15,
    port: int = 5556,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """run_pico variant that skips the is_body_data_available() preflight.

    Streaming starts immediately using a synthesized zero-body cache + fresh
    Manus hand joints. If Pico body data eventually arrives, the body-fresh
    path takes over automatically.
    """
    import subprocess  # local — match base.run_pico's pattern

    if base.xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run pose streaming."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    base.xrt.init()
    print("[HandFirst] Skipping body-data preflight; starting hand-only stream.")

    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"[HandFirst] ZMQ socket bound to port {port}")

    if base.build_command_message is not None and base.build_planner_message is not None:
        try:
            socket.send(base.build_command_message(start=False, stop=False, planner=False))
            socket.send(
                base.build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0)
            )
        except Exception as e:
            print(f"[HandFirst] Warning: failed to send initial command/planner messages: {e}")

    try:
        base._pose_stream_common(
            socket=socket,
            buffer_size=buffer_size,
            num_frames_to_send=num_frames_to_send,
            target_fps=target_fps,
            use_cuda=use_cuda,
            record_dir=record_dir,
            record_format=record_format,
            stop_event=None,
            log_prefix="HandFirst",
            enable_vis_vr3pt=enable_vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=enable_waist_tracking,
            enable_smpl_vis=enable_smpl_vis,
        )
    finally:
        socket.close()
        context.term()
        print("[HandFirst] Threads stopped, ZMQ socket closed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pico + Manus pose streamer with stale-body fallback "
        "(hand keeps updating even when Pico body drops)."
    )
    parser.add_argument("--buffer_size", type=int, default=15, help="Sliding window buffer size")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port (default: 5556)")
    parser.add_argument(
        "--num_frames_to_send", type=int, default=5, help="Number of frames to send (default: 5)"
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Target loop FPS (default: 50)")
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA for tensors and model (default: CPU)"
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="Directory to save sent batches (default: disabled)",
    )
    parser.add_argument(
        "--record_format",
        type=str,
        default="npz",
        help="Recording format: 'npz' or 'bin' (default: npz)",
    )
    parser.add_argument(
        "--manager",
        action="store_true",
        help="Run manager with planner and pose threads (interactive)",
    )
    parser.add_argument(
        "--zmq_feedback_host",
        type=str,
        default="localhost",
        help="ZMQ feedback host (default: localhost)",
    )
    parser.add_argument(
        "--zmq_feedback_port",
        type=int,
        default=5557,
        help="ZMQ feedback port (default: 5557)",
    )
    parser.add_argument(
        "--vis_vr3pt",
        action="store_true",
        help="Enable inline VR 3-point pose visualization in pose streaming mode",
    )
    parser.add_argument(
        "--no_g1",
        action="store_true",
        help="Disable G1 robot visualization in VR 3pt pose view",
    )
    parser.add_argument(
        "--waist_tracking",
        action="store_true",
        help="Enable G1 robot waist to follow VR head orientation",
    )
    parser.add_argument(
        "--vis_smpl",
        action="store_true",
        help="Enable SMPL body joint visualization in the VR3pt viewer",
    )
    args = parser.parse_args()

    with_g1_robot = not args.no_g1

    if args.manager:
        base.run_pico_manager(
            port=args.port,
            buffer_size=args.buffer_size,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            zmq_feedback_host=args.zmq_feedback_host,
            zmq_feedback_port=args.zmq_feedback_port,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
        )
    else:
        run_pico_handonly(
            buffer_size=args.buffer_size,
            port=args.port,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
        )
