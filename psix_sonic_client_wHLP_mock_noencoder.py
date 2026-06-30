#!/usr/bin/env python3
"""PsiX Sonic + HLP client with only the Sonic encoder mocked out.

This wrapper reuses psix_sonic_client_wHLP.py for camera, robot state, HLP,
policy HTTP, action repacking, and token publishing. The only substitution is
EncoderClient: instead of loading gear_sonic_deploy/policy/release/model_encoder.onnx,
it returns a deterministic 64-D zero token when the publish loop asks for the
between-chunk freeze token.

Use this for client/HLP/VLA integration testing when model_encoder.onnx is not
available. It is not a physically meaningful Sonic hold token.
"""

import argparse
import json
import os

import numpy as np

import psix_sonic_client_wHLP as client_impl


class MockEncoderClient:
    """Drop-in stand-in for encoder_client.EncoderClient."""

    def __init__(self, model_path: str, mode: int = 0):
        self._model_path = model_path
        self._mode = mode
        self._calls = 0
        print(
            "[MockEncoderClient] Sonic encoder is mocked; "
            f"skipping ONNX load: {model_path!r} (mode={mode})"
        )

    def encode(self, joint_pos, joint_vel, body_quat):
        self._calls += 1
        print(
            "[MockEncoderClient] encode() call "
            f"{self._calls}: returning zero {client_impl.TOKEN_DIM}-D freeze token"
        )
        return np.zeros(client_impl.TOKEN_DIM, dtype=np.float32)


class MockRobotStateSubscriber:
    """Optional stand-in for RobotStateSubscriber for offline client smoke tests."""

    def __init__(self, host="localhost", port=5557, topic="g1_debug"):
        self._state = {
            "body_q_measured": np.zeros(29, dtype=np.float32),
            "left_hand_q_measured": np.zeros(7, dtype=np.float32),
            "right_hand_q_measured": np.zeros(7, dtype=np.float32),
            "base_quat_measured": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        print(
            "[MockRobotStateSubscriber] Robot state is mocked; "
            f"not subscribing to tcp://{host}:{port} topic={topic!r}"
        )

    def get_state(self):
        return {k: v.copy() for k, v in self._state.items()}

    def stop(self):
        pass


class MockCamera:
    """Optional non-blocking camera for offline HLP/VLA smoke tests."""

    FRAME_PATH = None
    _frame = None

    def __init__(self, address="tcp://192.168.123.164:5558"):
        self._address = address
        if MockCamera._frame is None:
            MockCamera._frame = self._load_frame()
        source = MockCamera.FRAME_PATH or "black 480x640 frame"
        print(f"[MockCamera] Camera is mocked; not connecting to {address}; source={source}")

    @staticmethod
    def _load_frame():
        if MockCamera.FRAME_PATH:
            frame = client_impl.cv2.imread(MockCamera.FRAME_PATH, client_impl.cv2.IMREAD_COLOR)
            if frame is not None:
                return frame.astype(np.uint8)
            print(f"[MockCamera] WARNING: failed to read {MockCamera.FRAME_PATH}; using black frame")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_frame(self):
        return MockCamera._frame.copy()

    def close(self):
        pass


class DebugHlpController(client_impl.HlpController):
    """HlpController that only prints HLP switch/done decision changes."""

    ENABLED = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_last = None

    @staticmethod
    def _short(value, limit=240):
        text = "" if value is None else str(value)
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "...<truncated>"

    def _print_decision_change(self, out):
        decision = out.get("decision")
        if not self.ENABLED or decision not in ("switch", "done"):
            return
        stage = out.get("stage")
        subtask = out.get("next_subtask") or out.get("instruction")
        summary = (decision, stage, subtask)
        if summary == self._debug_last:
            return
        self._debug_last = summary
        if decision == "switch":
            print(f"[HLP] switch subtask[{stage}]: {self._short(subtask)}")
        else:
            detail = f": {self._short(subtask)}" if subtask else ""
            print(f"[HLP] done stage={stage}{detail} (VLA gated)")

    def _poll_once(self):
        ego = self._ego_rgb()
        with self._lock:
            gen, is_initial = self._gen, self._is_initial
        body = {"task": self._task, "is_initial": is_initial}
        if ego is not None:
            body["ego_image"] = ego
        try:
            out = self._post("/hlp", body)
        except Exception as e:
            print(f"[HLP] poll failed (ignored): {e}")
            return
        instr, decision, stage = out.get("instruction"), out.get("decision"), out.get("stage")
        with self._lock:
            if gen != self._gen:
                return
            if decision == "switch" or (instr and "Subtask:" in instr):
                self._is_initial = False
            if decision == "done":
                self._instruction = None
            elif instr:
                self._instruction = instr
            if stage is not None:
                self._sg.goto(stage)
        self._print_decision_change(out)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Chunk-based PsiX + HLP non-RTC HTTP client with only the Sonic "
            "encoder mocked out"
        )
    )
    parser.add_argument("--host", type=str, default="localhost",
                        help="VLA policy server host")
    parser.add_argument("--port", type=int, default=8014,
                        help="VLA policy server port")
    parser.add_argument("--zmq-host", type=str, default="localhost",
                        help="ZMQ host for robot state subscriber")
    parser.add_argument("--zmq-pub-port", type=int, default=5556,
                        help="ZMQ PUB port for sending pose to WBC")
    parser.add_argument("--zmq-sub-port", type=int, default=5557,
                        help="ZMQ SUB port for receiving robot state")
    parser.add_argument("--zmq-topic", type=str, default="pose",
                        help="ZMQ topic for pose messages")
    parser.add_argument("--zmq-sub-topic", type=str, default="g1_debug",
                        help="ZMQ topic for robot state subscription")
    parser.add_argument("--camera-address", type=str, default="tcp://192.168.123.164:5558",
                        help="Camera ZMQ address")
    parser.add_argument("--episode-dir", type=str,
                        default="/home/xiawei/data/multi-task/put_chip_can_into_plate/episode_0",
                        help="Episode folder containing color/ and color_subgoal/ for subgoal images")
    parser.add_argument("--prompts-json", type=str,
                        default="/home/xiawei/data/multi-task/prompts.json",
                        help="JSON mapping task-key -> {task_description, subtasks[]}")
    parser.add_argument("--task-key", type=str, default="put_chip_can_into_plate",
                        help="Key into prompts.json; selects task_description and subtasks")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override task instruction (else taken from prompts.json[task-key])")
    parser.add_argument("--include-neck", action="store_true",
                        help="Neck variant: states 45-dim, action chunk 80-dim")
    parser.add_argument("--neck-pub-host", type=str,
                        default=client_impl.DEFAULT_NECK_PUB_HOST,
                        help=f"Neck PUB bind host (default: {client_impl.DEFAULT_NECK_PUB_HOST})")
    parser.add_argument("--neck-pub-port", type=int,
                        default=client_impl.DEFAULT_NECK_PUB_PORT,
                        help=f"Neck PUB port (default: {client_impl.DEFAULT_NECK_PUB_PORT})")
    parser.add_argument("--neck-state-zmq", type=str,
                        default=client_impl.DEFAULT_NECK_STATE_ZMQ,
                        help=f"Neck-state SUB address (default: {client_impl.DEFAULT_NECK_STATE_ZMQ})")
    parser.add_argument("--hlp-host", type=str, default="localhost",
                        help="HLP server host")
    parser.add_argument("--hlp-port", type=int, default=8015,
                        help="HLP server port")
    parser.add_argument("--hlp-timeout", type=float, default=30.0,
                        help="HLP HTTP request timeout (s)")
    parser.add_argument("--hlp-period", type=float, default=0.7,
                        help="HLP poll period (s); 0 = as fast as possible")
    parser.add_argument("--no-hlp", action="store_true",
                        help="Disable the HLP poller — manual stdin steering only")
    parser.add_argument("--mock-robot-state", action="store_true",
                        help="Offline smoke-test mode: return zero robot state instead of "
                             "subscribing to ZMQ g1_debug")
    parser.add_argument("--mock-camera", action="store_true",
                        help="Offline smoke-test mode: return a local image instead of blocking "
                             "on the camera ZMQ server")
    parser.add_argument("--mock-camera-frame", type=str, default=None,
                        help="Image file to use for --mock-camera; default is the first image in "
                             "--episode-dir/color, falling back to black")
    parser.add_argument("--no-hlp-debug", action="store_true",
                        help="Suppress HLP response debug prints in this mock client")
    return parser.parse_args()


def _default_mock_camera_frame(episode_dir):
    color_dir = os.path.join(episode_dir, "color")
    if not os.path.isdir(color_dir):
        return None
    for name in sorted(os.listdir(color_dir)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(color_dir, name)
    return None


def main():
    args = parse_args()

    client_impl.EncoderClient = MockEncoderClient
    if args.mock_robot_state:
        client_impl.RobotStateSubscriber = MockRobotStateSubscriber
        if not args.mock_camera:
            print(
                "[mock-client] WARNING: --mock-robot-state is set but --mock-camera is not; "
                "the HLP poller will still wait on the real camera ZMQ server."
            )
    if args.mock_camera:
        MockCamera.FRAME_PATH = args.mock_camera_frame or _default_mock_camera_frame(args.episode_dir)
        client_impl.RSCamera = MockCamera
        client_impl.ZedNeckCamera = MockCamera
    DebugHlpController.ENABLED = not args.no_hlp_debug
    client_impl.HlpController = DebugHlpController

    task_instruction = args.instruction
    subtasks = []
    if args.task_key:
        with open(args.prompts_json) as f:
            prompts = json.load(f)
        if args.task_key not in prompts:
            raise SystemExit(
                f"[MAIN] task-key '{args.task_key}' not found in {args.prompts_json}; "
                f"available: {list(prompts)}"
            )
        entry = prompts[args.task_key]
        if task_instruction is None:
            task_instruction = entry.get("task_description")
        subtasks = entry.get("subtasks", [])
    if task_instruction is None:
        task_instruction = client_impl.TASK_INSTRUCTION

    server_url = f"http://{args.host}:{args.port}/act"
    hlp_url = None if args.no_hlp else f"http://{args.hlp_host}:{args.hlp_port}"
    client_impl.main(
        server_url=server_url,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_sub_port=args.zmq_sub_port,
        zmq_topic=args.zmq_topic,
        zmq_sub_topic=args.zmq_sub_topic,
        camera_address=args.camera_address,
        episode_dir=args.episode_dir,
        task_instruction=task_instruction,
        subtasks=subtasks,
        include_neck=args.include_neck,
        neck_pub_host=args.neck_pub_host,
        neck_pub_port=args.neck_pub_port,
        neck_state_zmq=args.neck_state_zmq,
        hlp_url=hlp_url,
        hlp_timeout=args.hlp_timeout,
        hlp_period=args.hlp_period,
    )


if __name__ == "__main__":
    main()
