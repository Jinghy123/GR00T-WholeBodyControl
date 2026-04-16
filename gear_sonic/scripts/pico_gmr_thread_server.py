"""
pico_gmr_thread_server.py

Reads live SMPL body tracking from Pico (XRoboToolkit), retargets to Unitree G1
joint positions via GMR IK, and publishes via ZMQ using pack_pose_message.

Pipeline:
  Pico XRT → body_poses_np (24×7, Unity frame)
           → build_xrobot_frame_dict (Unity → Z-up, XRobot format)
           → GMR IK (src_human="xrobot", tgt_robot="unitree_g1")
           → joint_pos29 (MuJoCo order) + pelvis_quat
           → reorder to IsaacLab → ZMQ v1 protocol (joint_pos/joint_vel/body_quat/frame_index)

Controller:
  A+B+X+Y       : start policy (OFF → PLANNER) / stop policy (any → OFF + exit)
  A+X           : toggle PLANNER ↔ POSE (changes planner flag in command message)
  Left menu btn : hold to pause streaming (POSE_PAUSE), release to resume

Usage:
    python gear_sonic/scripts/pico_gmr_thread_server.py --human-height 1.60
"""

import argparse
import subprocess
import time
from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation
import zmq

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, pack_pose_message
except ImportError:
    def build_command_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_command_message unavailable")
    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")


from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting


# ── Enums ─────────────────────────────────────────────────────────────────────

class StreamMode(IntEnum):
    OFF          = 0
    PLANNER      = 2   # GMR joint-space teleop sent as planner (planner=True)  — default after ABXY
    POSE         = 1   # GMR joint-space teleop (planner=False)  — entered via A+X from PLANNER
    POSE_PAUSE   = 4   # Left menu button held; streaming paused


# ── Constants ─────────────────────────────────────────────────────────────────

# XRobot joint names — rows 0-21 of body_poses_np (24×7)
_XROBOT_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist",
]

# Unity → Z-up robot frame: same as XRobotRecorder.coordinate_transform_unity_data
_ROT_MAT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
_R_UNITY_TO_ROBOT = Rotation.from_matrix(_ROT_MAT)

# MuJoCo (GMR output) → IsaacLab joint order (29 DOF)
_MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)


# ── Controller helpers (mirrors pico_manager_thread_server.py) ────────────────

def get_abxy_buttons():
    """Fetch A, B, X, Y face buttons as booleans (a, b, x, y)."""
    if xrt is None:
        return False, False, False, False
    try:
        return (
            bool(xrt.get_A_button()),
            bool(xrt.get_B_button()),
            bool(xrt.get_X_button()),
            bool(xrt.get_Y_button()),
        )
    except Exception:
        return False, False, False, False


def get_controller_inputs():
    """Fetch trigger/grip/menu from XRoboToolkit."""
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    left_menu_button = xrt.get_left_menu_button()
    return left_menu_button, left_trigger, right_trigger, left_grip, right_grip




# ── Core conversion ───────────────────────────────────────────────────────────

def build_xrobot_frame_dict(body_poses_np):
    """
    Convert raw pico body_poses_np (24×7: [x,y,z,qx,qy,qz,qw] in Unity frame)
    to XRobot-compatible frame dict for GMR retarget().
    Applies the same Unity→Z-up transform as XRobotRecorder.coordinate_transform_unity_data.
    Only joints 0-21 are included (xrobot_to_g1.json uses these 22 joints).
    """
    frame_dict = {}
    for i, name in enumerate(_XROBOT_JOINT_NAMES):
        pos_robot = body_poses_np[i, :3] @ _ROT_MAT.T
        rot_robot = _R_UNITY_TO_ROBOT * Rotation.from_quat(body_poses_np[i, 3:7])  # XYZW in
        frame_dict[name] = [pos_robot.tolist(), rot_robot.as_quat(scalar_first=True).tolist()]
    return frame_dict


# ── Main server ───────────────────────────────────────────────────────────────

class PicoGMRServer:
    def __init__(self, human_height=1.70, zmq_addr="tcp://*:5556", topic="pose"):
        if xrt is None:
            raise RuntimeError("xrobotoolkit_sdk not available")

        print(f"Building GMR retargeter (human_height={human_height}m)...")
        self.retargeter = GeneralMotionRetargeting(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=human_height,
            verbose=False,
        )
        print("GMR retargeter ready.")

        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(zmq_addr)
        self._topic = topic

        self._frame_index = 0
        self._stat_gmr_times = []
        self._last_stat_t = time.time()
        self._frame_time = 0.95 / 50   # same as pico_manager_thread_server (target 50 fps)
        self._num_frames_to_send = 5   # sliding window size (same as pico_manager)
        from collections import deque
        self._joint_buf = deque(maxlen=self._num_frames_to_send)
        self._quat_buf = deque(maxlen=self._num_frames_to_send)
        self._fidx_buf = deque(maxlen=self._num_frames_to_send)

    def _process_and_send(self, body_poses_np):
        frame_dict = build_xrobot_frame_dict(body_poses_np)

        t0 = time.perf_counter()
        qpos = self.retargeter.retarget(frame_dict, offset_to_ground=True)
        self._stat_gmr_times.append(time.perf_counter() - t0)

        # qpos: [x, y, z, qw, qx, qy, qz, joint0..joint28]
        pelvis_quat_wxyz      = qpos[3:7].astype(np.float32)
        joint_pos29_isaaclab  = qpos[7:36].astype(np.float32)[_MUJOCO_TO_ISAACLAB]

        # Sliding window buffer (same as pico_manager_thread_server)
        self._joint_buf.append(joint_pos29_isaaclab)
        self._quat_buf.append(pelvis_quat_wxyz)
        self._fidx_buf.append(self._frame_index)
        self._frame_index += 1

        N = len(self._fidx_buf)
        if N >= self._num_frames_to_send:
            pose_data = {
                "joint_pos":   np.stack(list(self._joint_buf)),
                "joint_vel":   np.zeros((N, 29), dtype=np.float32),
                "body_quat":   np.stack(list(self._quat_buf)),
                "frame_index": np.array(list(self._fidx_buf), dtype=np.int64),
            }
            if self._frame_index % 50 == 0:
                yaw_deg = np.degrees(Rotation.from_quat(pelvis_quat_wxyz[[1,2,3,0]]).as_euler('ZYX')[0])
                print(f"[PicoGMR] frame={self._frame_index} yaw={yaw_deg:.1f}° quat={np.round(pelvis_quat_wxyz, 4)}")
            self.socket.send(pack_pose_message(pose_data, topic=self._topic, version=1))

    def _maybe_log_stats(self):
        now = time.time()
        if now - self._last_stat_t < 5.0 or not self._stat_gmr_times:
            return
        gmr_ms = np.mean(self._stat_gmr_times) * 1000
        print(f"[PicoGMR] frame={self._frame_index}  gmr={gmr_ms:.1f}ms")
        self._stat_gmr_times.clear()
        self._last_stat_t = now

    def run(self):
        subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
        xrt.init()
        print("Waiting for body tracking data...")
        while not xrt.is_body_data_available():
            print("waiting for body data...")
            time.sleep(1)
        print("Body data available! Starting GMR retargeting...")

        print("Controls (zmq mode):")
        print("  WBC side: press ] to start control, then Enter to enable ZMQ streaming")
        print("  Pico side: A+B+X+Y = start/stop sending, Left menu = pause/resume")

        current_mode = StreamMode.OFF
        prev_start_combo = False

        try:
            while True:
                # ── Controller polling ────────────────────────────────────────
                a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()
                left_menu_button, _, _, _, _ = get_controller_inputs()

                start_combo = a_pressed and b_pressed and x_pressed and y_pressed

                # ── State machine ─────────────────────────────────────────────
                new_mode = current_mode

                if current_mode == StreamMode.OFF:
                    if start_combo and not prev_start_combo:
                        new_mode = StreamMode.POSE

                elif current_mode == StreamMode.POSE:
                    if start_combo and not prev_start_combo:
                        new_mode = StreamMode.OFF
                    elif left_menu_button:
                        new_mode = StreamMode.POSE_PAUSE

                elif current_mode == StreamMode.POSE_PAUSE:
                    if start_combo and not prev_start_combo:
                        new_mode = StreamMode.OFF
                    elif not left_menu_button:
                        new_mode = StreamMode.POSE

                # ── Publish one frame if active (rate-limited to ~50 Hz) ─────
                if current_mode == StreamMode.POSE:
                    frame_start = time.time()
                    if xrt.is_body_data_available():
                        try:
                            body_poses = xrt.get_body_joints_pose()
                            self._process_and_send(np.array(body_poses, dtype=np.float64))
                        except Exception as e:
                            print(f"[PicoGMR] retarget/send error: {e}")
                    elapsed = time.time() - frame_start
                    if elapsed < self._frame_time:
                        time.sleep(self._frame_time - elapsed)

                self._maybe_log_stats()

                # ── Mode transition ───────────────────────────────────────────
                if new_mode != current_mode:
                    print(f"[PicoGMR] StreamMode switch: {current_mode.name} -> {new_mode.name}")
                    current_mode = new_mode
                    if current_mode == StreamMode.OFF:
                        exit()

                prev_start_combo = start_combo

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.socket.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pico XRT → GMR IK → G1 joint states → ZMQ"
    )
    parser.add_argument("--human-height", type=float, default=1.70,
                        help="Actual human height in meters for GMR limb scaling (default: 1.70)")
    parser.add_argument("--addr", type=str, default="tcp://*:5556",
                        help="ZMQ bind address (default: tcp://*:5556)")
    parser.add_argument("--topic", type=str, default="pose",
                        help="ZMQ topic prefix (default: pose)")
    args = parser.parse_args()

    server = PicoGMRServer(
        human_height=args.human_height,
        zmq_addr=args.addr,
        topic=args.topic,
    )
    server.run()


if __name__ == "__main__":
    main()
