import os
import time
import threading
import json
from collections import deque

import cv2
import numpy as np
# import requests
import json_numpy

from multiprocessing import Array, Event
from master_whole_body import RobotTaskmaster
from robot_control.compute_tau import GetTauer
import zmq
import eval_utils.policy_server as policy_server
from eval_utils.policy_client import WebsocketClientPolicy

HOST = None
PORT = None

PROMPT = None

FREQ_VLA = 30  
FREQ_CTRL = 60  
ACTION_REPEAT = max(1, int(round(FREQ_CTRL / FREQ_VLA)))


MAX_STEPS = 500
IMAGE_BUFFER_SIZE = 100

VIDEO_FREQ = 30
CAMERA_KEY = "observation/egocentric"

# RELATIVE_OFFSETS = [-23, -16, -8, 0]
RELATIVE_OFFSETS = [-23-1, -16-1, -8-1, 0-1]
ACTION_HORIZON = 24


json_numpy.patch()


class RSCamera:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://192.168.123.164:5556")

    def get_frame(self):
        self.socket.send(b"get_frame")

        rgb_bytes, _, _ = self.socket.recv_multipart()

        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        return rgb_image



def _log_action(actions: np.ndarray, dt: float) -> None:
    """Pretty-print action shape, range, and timing."""
    assert isinstance(actions, np.ndarray), f"Expected numpy array, got {type(actions)}"
    assert actions.ndim == 2, f"Expected 2D array, got shape {actions.shape}"
    assert actions.shape[0] == ACTION_HORIZON, f"Expected {ACTION_HORIZON} actions, got {actions.shape[0]}"
    # assert actions.shape[-1] == 32, (
    #     f"Expected 8 action dims (7 joints + 1 gripper), got {actions.shape[-1]}"
    # )
    print(
        f"  Action shape: {actions.shape}, "
        f"range: [{actions.min():.4f}, {actions.max():.4f}], "
        f"time: {dt:.2f}s"
    )

def main():
    shared_data = {
        "kill_event": Event(),
        "session_start_event": Event(),
        "failure_event": Event(),
        "end_event": Event(),
        "dirname": None,
    }
    kill_event = shared_data["kill_event"]

    robot_shm_array = Array("d", 512, lock=False)
    teleop_shm_array = Array("d", 64, lock=False)

    master = RobotTaskmaster(
        task_name="inference",
        shared_data=shared_data,
        robot_shm_array=robot_shm_array,
        teleop_shm_array=teleop_shm_array,
        robot="g1",
    )

    get_tauer = GetTauer()
    camera = RSCamera()

    pred_action_buffer = {"actions": None, "idx": 0, "prev_rpy": np.array([0.0,0.0,0.0], dtype=np.float32), "prev_height": np.array([0.75], dtype=np.float32)}
    pred_action_lock = threading.Lock()

    image_buffer = deque(maxlen=IMAGE_BUFFER_SIZE)
    image_buffer_lock = threading.Lock()

    state_lock = threading.Lock()
    shared_robot_state = {
        "motor": None,
        "hand": None,
    }

    running = Event()
    running.set()

    sequence_done_event = Event()
    sequence_done_event.set() 
    

    def make_obs(
        frame_indices: list[int],
        prompt: str,
        session_id: str,
        **kwargs,
    ) -> dict:
        """Build an observation dict from real video frames.

        For 1 frame: each image key is (H, W, 3).
        For 4 frames: each image key is (4, H, W, 3).
        """
        obs: dict = {}
        with image_buffer_lock:
            selected = [image_buffer[i].copy() for i in frame_indices]  # (T, H, W, 3)
        selected = np.stack(selected, axis=0) # (T, H, W, 3) or (1, H, W, 3)
        if len(frame_indices) == 1:
            selected = selected[0]  # (H, W, 3)
        obs[CAMERA_KEY] = selected

        obs["observation/left_hand"] = kwargs["hand_joints"][:7]
        obs["observation/right_hand"] = kwargs["hand_joints"][7:]
        obs["observation/left_arm"] = kwargs["arm_joints"][:7]
        obs["observation/right_arm"] = kwargs["arm_joints"][7:]
        obs["observation/rpy"] = kwargs["rpy"]
        obs["observation/height"] = kwargs["height"]
        obs["prompt"] = prompt
        obs["session_id"] = session_id
        return obs

    def get_observation():
        frame = camera.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with image_buffer_lock:
            image_buffer.append(frame)


    def action_request_thread():

        ############## Initialization ##############
        print(f"Connecting to AR_droid server at {HOST}:{PORT}...")
        client = WebsocketClientPolicy(host=HOST, port=PORT)
        metadata = client.get_server_metadata()
        print(f"Server metadata: {metadata}")
        assert isinstance(metadata, dict), "Metadata should be a dict"

        try:
            server_config = policy_server.PolicyServerConfig(**metadata)
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            raise e

        # Validate expected AR_droid configuration
        print(f"Server config: {server_config}")
        assert server_config.n_external_cameras == 0, f"Expected 0 external cameras, got {server_config.n_external_cameras}"
        assert server_config.needs_wrist_camera == False, "Expected wrist camera to be disabled"
        assert server_config.action_space == "joint_position", f"Expected joint_position action space, got {server_config.action_space}"
        
        print("Server configuration validated for AR_droid")

        # Generate unique session ID for this test run
        import uuid
        session_id = str(uuid.uuid4())
        print(f"Session ID: {session_id}")
        #######################################

        try:
            # wait for image buffer to be filled
            while True:              
                with state_lock:
                    motor = shared_robot_state["motor"]
                    hand = shared_robot_state["hand"]

                if motor is None or hand is None:
                    print("[VLA] robot state is empty, waiting for robot state to be updated...")
                    time.sleep(1)
                else:
                    break

            for step in range(MAX_STEPS):
                if not running.is_set():
                    break

                sequence_done_event.wait()

                try:
                    with state_lock:
                        motor = shared_robot_state["motor"].copy()
                        hand = shared_robot_state["hand"].copy()
                    
                    assert motor is not None and hand is not None, "Motor or hand is None"
                    
                    arm_joints = motor[15:29]
                    hand_joints = hand
                    leg_joints = motor[:15]

                    # Step 0: initial single frame
                    if step == 0:
                        print(("=== Initial: frame [0] ==="))
                        frame_indices = [-1]
                        get_observation()

                    else:
                        frame_indices = RELATIVE_OFFSETS # get previous 4 frames relative to current step
                        with image_buffer_lock:
                            len_image_buffer = len(image_buffer)
                        assert len_image_buffer == ACTION_HORIZON, f"Expected {ACTION_HORIZON} frames in image buffer, got {len_image_buffer}"
                    
                    with pred_action_lock:
                        prev_rpy = pred_action_buffer["prev_rpy"].copy()
                        prev_height = pred_action_buffer['prev_height'].copy()
                    obs = make_obs(frame_indices=frame_indices, prompt=PROMPT, session_id=session_id, arm_joints=arm_joints, hand_joints=hand_joints, rpy=prev_rpy, height=prev_height)
                    t0 = time.time()
                    actions = client.infer(obs)
                    dt = time.time() - t0
                    _log_action(actions, dt)


                    if len(actions.shape) != 2 or actions.shape[1] != 36:
                        print("[VLA] invalid action seq:", actions.shape)
                        assert False

                    image_buffer.clear()
                    with pred_action_lock:
                        pred_action_buffer["actions"] = actions
                        pred_action_buffer["idx"] = 0
                    print(f"[VLA] Got sequence of {len(actions)} actions.")
                    sequence_done_event.clear()

                except Exception as e:
                    print(f"[VLA] step {step} failed: {e}")
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n[VLA] Ctrl+C detected, stopping...")
        finally:
            # 无论发生什么（正常结束、报错、Ctrl+C），都会执行这里
            print("[VLA] Sending reset to server to save video...")
            try:
                client.reset({}) 
                print("[VLA] Reset signal sent successfully.")

            except Exception as e:
                print(f"[VLA] Failed to send reset signal: {e}")

            kill_event.set()

        print("[VLA] Finished or stopped. Signaling kill_event.")
        kill_event.set()

    def apply_action_from_buffer():
        print("[DEBUG] apply_action_from_buffer called", flush=True)
        current_lr_arm_q, current_lr_arm_dq = master.get_robot_data()

        with state_lock:
            shared_robot_state["motor"] = master.motorstate.copy()
            shared_robot_state["hand"] = master.handstate.copy()

        with pred_action_lock:
            actions = pred_action_buffer["actions"]
            idx = pred_action_buffer["idx"]

            action = None
            have_vla = False

            if actions is not None:
                real_idx = idx // ACTION_REPEAT
                if real_idx < len(actions):
                    action = actions[real_idx]
                    have_vla = True

                    pred_action_buffer["idx"] += 1

                    next_real_idx = pred_action_buffer["idx"] // ACTION_REPEAT

                    # [0, 7, 15, 23], [24, 31, 39, 47], ...
                    if next_real_idx == real_idx:
                        get_observation()
                    
                    # last action in the sequence
                    if next_real_idx >= len(actions):
                        pred_action_buffer["actions"] = None
                        pred_action_buffer["idx"] = 0
                        pred_action_buffer["prev_rpy"] = action[28:31]
                        pred_action_buffer["prev_height"] = action[31]
                        sequence_done_event.set()
                else:
                    assert False, "what happened here?"
                    # pred_action_buffer["actions"] = None
                    # pred_action_buffer["idx"] = 0
                    # sequence_done_event.set()

        arm_cmd = None
        hand_cmd = None
        if have_vla:
            if action.shape[0] < 36:
                print("[CTRL] Invalid action shape:", action.shape)
            else:

                vx = action[32]
                vy = action[33]
                vyaw = action[34]
                dyaw = action[35]

                vx = 0.6 if vx > 0.25 else 0
                vy = 0 if abs(vy) < 0.3 else 0.5 * (1 if vy > 0 else -1)

                rpyh   = action[28:32]
                arm_cmd = action[14:28]
                hand_cmd = action[:14]

                master.torso_roll   = rpyh[0]
                master.torso_pitch  = rpyh[1]
                master.torso_yaw    = rpyh[2]
                master.torso_height = rpyh[3]

                master.vx = vx
                master.vy = vy
                master.vyaw = vyaw
                master.dyaw = dyaw

                master.prev_torso_roll   = master.torso_roll
                master.prev_torso_pitch  = master.torso_pitch
                master.prev_torso_yaw    = master.torso_yaw
                master.prev_torso_height = master.torso_height

                master.prev_vx   = master.vx
                master.prev_vy  = master.vy
                master.prev_vyaw    = master.vyaw
                master.prev_dyaw = master.dyaw

                master.prev_arm = arm_cmd
                master.prev_hand = hand_cmd

        
        if not have_vla:
            master.torso_roll   = master.prev_torso_roll
            master.torso_pitch  = master.prev_torso_pitch
            master.torso_yaw    = master.prev_torso_yaw
            master.torso_height = master.prev_torso_height

            arm_cmd = master.prev_arm
            hand_cmd = master.prev_hand

            master.vx = 0
            master.vy = 0
            master.vyaw = 0
            master.dyaw = master.prev_dyaw
        
        master.get_ik_observation(record=False)


        pd_target, pd_tauff, raw_action = master.body_ik.solve_whole_body_ik(
            left_wrist=None,
            right_wrist=None,
            current_lr_arm_q=current_lr_arm_q,
            current_lr_arm_dq=current_lr_arm_dq,
            observation=master.observation,
            extra_hist=master.extra_hist,
            is_teleop=False,
        )

        master.last_action = np.concatenate([
            raw_action.copy(),
            (master.motorstate - master.default_dof_pos)[15:] / master.action_scale,
        ])

        if arm_cmd is not None:
            pd_target[15:] = arm_cmd
            tau_arm = np.asarray(get_tauer(arm_cmd), dtype=np.float64).reshape(-1)
            pd_tauff[15:] = tau_arm

        if hand_cmd is not None:
            with master.dual_hand_data_lock:
                master.hand_shm_array[:] = hand_cmd

        master.body_ctrl.ctrl_whole_body(
            pd_target[15:], pd_tauff[15:], pd_target[:15], pd_tauff[:15]
        )

        return pd_target


    def control_loop_thread():
        dt = 1.0 / FREQ_CTRL
        while running.is_set() and not kill_event.is_set():
            try:
                apply_action_from_buffer()
            except Exception as e:
                print("[CTRL] loop error:", e)
            time.sleep(dt)
        print("[CTRL] Control loop stopped.")

    try:
        stabilize_thread = threading.Thread(target=master.maintain_standing, daemon=True)
        stabilize_thread.start()
        master.episode_kill_event.set()
        print("[MAIN] Initialize with standing pose...")

        time.sleep(30)
        master.episode_kill_event.clear() 

        master.reset_yaw_offset = True

        t_req = threading.Thread(target=action_request_thread, daemon=True)
        t_ctrl = threading.Thread(target=control_loop_thread, daemon=True)
        t_req.start()
        t_ctrl.start()

        print("[MAIN] Running. Ctrl+C to stop.")
        while not kill_event.is_set():
            time.sleep(0.5)

        print("[MAIN] kill_event set, preparing to stop...")
        running.clear()
        time.sleep(0.5)

        master.episode_kill_event.set()
        print("[MAIN] Returning to standing pose for 5s...")
        time.sleep(5)
        master.episode_kill_event.clear()

    except KeyboardInterrupt:
        print("[MAIN] Caught Ctrl+C, exiting...")
        running.clear()
        kill_event.set()
    finally:
        shared_data["end_event"].set()
        master.stop()
        print("[MAIN] Shutdown complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--prompt", required=True, type=str)
    
    args = parser.parse_args()
    HOST = args.host
    PORT = args.port
    PROMPT = args.prompt

    print("HOST: {}".format(HOST))
    print("PORT: {}".format(PORT))
    print("PROMPT: {}".format(PROMPT))
    main()