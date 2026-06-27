#!/usr/bin/env python3
"""RTC PsiX Sonic + HLP client with hardware/encoder mocks for smoke tests.

This wrapper reuses psix_rtc_sonic_client_wHLP.py for the WebSocket transport,
HLP control flow, action repacking, and token publishing. Optional mocks let us
exercise the HLP/VLA path without the Sonic encoder, robot state ZMQ, or camera
ZMQ. HLP logging is intentionally quiet: it only prints switch/done decisions.
"""

import argparse
import json
import os
import sys
import types

import numpy as np

import ours.clients.psix_rtc_sonic_client_wHLP as client_impl


class MockEncoderClient:
    """Drop-in stand-in for encoder_client.EncoderClient."""

    def __init__(self, model_path: str, mode: int = 0):
        self._calls = 0
        print(
            "[MockEncoderClient] Sonic encoder is mocked; "
            f"skipping ONNX load: {model_path!r} (mode={mode})"
        )

    def encode(self, joint_pos, joint_vel, body_quat):
        self._calls += 1
        print(
            "[MockEncoderClient] encode() call "
            f"{self._calls}: returning zero {client_impl.TOKEN_DIM}-D token"
        )
        return np.zeros(client_impl.TOKEN_DIM, dtype=np.float32)


class MockRobotStateSubscriber:
    """Optional stand-in for RobotStateSubscriber for offline RTC smoke tests."""

    def __init__(self, host="localhost", port=5557, topic="g1_debug"):
        zeros_hand = np.zeros(7, dtype=np.float32)
        self._state = {
            "body_q_measured": np.zeros(29, dtype=np.float32),
            "left_hand_q": zeros_hand,
            "right_hand_q": zeros_hand,
            "left_hand_q_measured": zeros_hand,
            "right_hand_q_measured": zeros_hand,
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


class QuietDecisionHlpController(client_impl.HlpController):
    """HlpController that only prints HLP switch/done decision changes."""

    ENABLED = True
    VLA_RESET_URL = None

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

    def _reset_vla(self):
        if not self.VLA_RESET_URL:
            return
        try:
            resp = client_impl.requests.post(self.VLA_RESET_URL, timeout=3.0)
            resp.raise_for_status()
            print(f"[VLA] reset -> {self.VLA_RESET_URL}")
        except Exception as e:
            print(f"[VLA] reset failed: {e}")

    def restart(self):
        super().restart()
        self._reset_vla()

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


class ResetAwareRTCWebSocketClient(client_impl.RTCWebSocketClient):
    """RTC client that drops stale actions while HLP gates the VLA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_action_log = 0.0
        self._last_drop_log = 0.0

    def _on_message(self, ws, message):
        self.start_time = client_impl.time.time()
        try:
            data = json.loads(message)
            action_data = data.get("action")
            version = data.get("version", -1)
            if action_data is None:
                return

            action = client_impl.convert_numpy_in_dict(action_data, client_impl.numpy_deserialize)
            if not isinstance(action, np.ndarray):
                return

            if self._hlp.current_instruction() is None:
                now = client_impl.time.time()
                if now - self._last_drop_log > 2.0:
                    print(f"[client] gated: dropping stale action version={version}")
                    self._last_drop_log = now
                return

            self.execute_action(action)
            now = client_impl.time.time()
            if now - self._last_action_log > 2.0:
                print(f"[client] action stream version={version}, shape={action.shape}")
                self._last_action_log = now
        except Exception as e:
            print(f"[client] Message processing error: {e}")


def _install_mock_encoder():
    mod = sys.modules.get("encoder_client")
    if mod is None:
        mod = types.ModuleType("encoder_client")
        sys.modules["encoder_client"] = mod
    mod.EncoderClient = MockEncoderClient


def _default_mock_camera_frame(episode_dir):
    color_dir = os.path.join(episode_dir, "color")
    if not os.path.isdir(color_dir):
        return None
    for name in sorted(os.listdir(color_dir)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(color_dir, name)
    return None


def _ws_to_http_reset_url(ws_url):
    if ws_url.startswith("ws://"):
        base = "http://" + ws_url[len("ws://"):]
    elif ws_url.startswith("wss://"):
        base = "https://" + ws_url[len("wss://"):]
    else:
        base = ws_url
    if base.endswith("/ws"):
        base = base[:-3]
    return base.rstrip("/") + "/reset"


def parse_args():
    p = argparse.ArgumentParser(description="RTC sonic client + HLP with mock state/camera/encoder")
    p.add_argument("--host", type=str, default="localhost", help="VLA policy server host")
    p.add_argument("--port", type=int, default=8014, help="VLA policy server port")
    p.add_argument("--zmq-host", type=str, default="localhost")
    p.add_argument("--zmq-pub-port", type=int, default=5556)
    p.add_argument("--zmq-sub-port", type=int, default=5557)
    p.add_argument("--zmq-topic", type=str, default="pose")
    p.add_argument("--zmq-sub-topic", type=str, default="g1_debug")
    p.add_argument("--camera-address", type=str, default="tcp://192.168.123.164:5558")
    p.add_argument("--data-dir", type=str, default=None,
                   help="task folder; episode defaults to <data-dir>/episode_<eps-idx>")
    p.add_argument("--episode-dir", type=str, default=None,
                   help="direct episode folder containing color/ and color_subgoal/")
    p.add_argument("--eps-idx", type=int, default=0)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--subtasks", type=str, default=None,
                   help="optional ';'-separated subtask text")
    p.add_argument("--prompts-json", type=str, default=None,
                   help="optional JSON mapping task-key -> {task_description, subtasks[]}")
    p.add_argument("--task-key", type=str, default=None,
                   help="key into --prompts-json")
    p.add_argument("--hlp-host", type=str, default="localhost")
    p.add_argument("--hlp-port", type=int, default=8015)
    p.add_argument("--hlp-period", type=float, default=0.7)
    p.add_argument("--hlp-timeout", type=float, default=30.0)
    p.add_argument("--no-hlp", action="store_true")
    p.add_argument("--include-neck", action="store_true")
    p.add_argument("--neck-pub-host", type=str, default=client_impl.DEFAULT_NECK_PUB_HOST)
    p.add_argument("--neck-pub-port", type=int, default=client_impl.DEFAULT_NECK_PUB_PORT)
    p.add_argument("--neck-state-zmq", type=str, default=client_impl.DEFAULT_NECK_STATE_ZMQ)
    p.add_argument("--mock-robot-state", action="store_true")
    p.add_argument("--mock-camera", action="store_true")
    p.add_argument("--mock-camera-frame", type=str, default=None)
    p.add_argument("--no-hlp-debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.episode_dir is None and args.data_dir is None:
        raise SystemExit("Provide --episode-dir or --data-dir")

    episode_dir = args.episode_dir
    data_dir = args.data_dir
    if episode_dir is None:
        data_dir = os.path.normpath(data_dir)
        episode_dir = os.path.join(data_dir, f"episode_{args.eps_idx}")
    else:
        episode_dir = os.path.normpath(episode_dir)
        data_dir = args.data_dir or os.path.dirname(episode_dir)

    task_instruction = args.task
    subtasks = [s.strip() for s in args.subtasks.split(";") if s.strip()] if args.subtasks else []
    if args.prompts_json and args.task_key:
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
        if not subtasks:
            subtasks = entry.get("subtasks", [])
    if task_instruction is None:
        task_instruction = os.path.basename(os.path.normpath(data_dir)).replace("_", " ").strip()
    if not task_instruction:
        task_instruction = client_impl.TASK_INSTRUCTION

    _install_mock_encoder()
    if args.mock_robot_state:
        client_impl.RobotStateSubscriber = MockRobotStateSubscriber
        if not args.mock_camera:
            print(
                "[mock-client] WARNING: --mock-robot-state is set but --mock-camera is not; "
                "the HLP poller will still wait on the real camera ZMQ server."
            )
    if args.mock_camera:
        MockCamera.FRAME_PATH = args.mock_camera_frame or _default_mock_camera_frame(episode_dir)
        client_impl.RSCamera = MockCamera
        client_impl.ZedNeckCamera = MockCamera
    QuietDecisionHlpController.ENABLED = not args.no_hlp_debug
    QuietDecisionHlpController.VLA_RESET_URL = _ws_to_http_reset_url(f"ws://{args.host}:{args.port}/ws")
    client_impl.HlpController = QuietDecisionHlpController
    client_impl.RTCWebSocketClient = ResetAwareRTCWebSocketClient

    server_url = f"ws://{args.host}:{args.port}/ws"
    hlp_url = None if args.no_hlp else f"http://{args.hlp_host}:{args.hlp_port}"
    client_impl.main(
        server_url=server_url,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_sub_port=args.zmq_sub_port,
        zmq_topic=args.zmq_topic,
        zmq_sub_topic=args.zmq_sub_topic,
        camera_address=args.camera_address,
        episode_dir=episode_dir,
        task_instruction=task_instruction,
        subtasks=subtasks,
        hlp_url=hlp_url,
        hlp_period=args.hlp_period,
        hlp_timeout=args.hlp_timeout,
        no_hlp=args.no_hlp,
        include_neck=args.include_neck,
        neck_pub_host=args.neck_pub_host,
        neck_pub_port=args.neck_pub_port,
        neck_state_zmq=args.neck_state_zmq,
    )


if __name__ == "__main__":
    main()
