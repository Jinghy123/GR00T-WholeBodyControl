import os
import sys
import time
import threading
import json
import signal
import struct

import cv2
import numpy as np
import zmq
import msgpack
from websocket import WebSocketApp

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message,
    build_command_message,
)

# ---------------- Configuration ----------------
TASK_INSTRUCTION = "Hug box and move."

# ---------------- Serialization utilities ----------------
from base64 import b64encode, b64decode
from numpy.lib.format import dtype_to_descr, descr_to_dtype


def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(dct["shape"]) if dct["shape"] else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    else:
        return data


# ---------------- RSCamera ----------------
class RSCamera:
    def __init__(self, address="tcp://192.168.123.164:5558"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)

    def get_frame(self):
        self.socket.send(b"get_frame")
        rgb_bytes, _, _ = self.socket.recv_multipart()
        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        return rgb_image


# ---------------- RobotStateSubscriber ----------------
class RobotStateSubscriber:
    """Subscribe to robot state published by g1_deploy_onnx_ref on ZMQ PUB port."""

    def __init__(self, host="localhost", port=5557, topic="g1_debug"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout
        self._socket.setsockopt(zmq.RCVHWM, 1)
        
        self._topic = topic
        self._lock = threading.Lock()
        self._latest_state = None
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                msg = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break

            # Strip topic prefix
            topic_bytes = self._topic.encode("utf-8")
            if msg.startswith(topic_bytes):
                payload = msg[len(topic_bytes):]
            else:
                payload = msg

            try:
                state = msgpack.unpackb(payload, raw=False)
                with self._lock:
                    self._latest_state = state
            except Exception as e:
                print(f"[StateSubscriber] Unpack error: {e}")

    def get_state(self):
        """Return the latest robot state dict, or None if not yet received."""
        with self._lock:
            return self._latest_state

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._socket.close()
        self._context.term()


# ---------------- WBCBridge ----------------
class WBCBridge:
    """Publish Protocol V1 pose messages to g1_deploy_onnx_ref via ZMQ PUB."""

    def __init__(self, host="*", port=5556, topic="pose"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")
        self._topic = topic
        self._frame_index = 0

    def send_command(self, start=False, stop=False, planner=False):
        """Send a command message (start/stop/mode)."""
        msg = build_command_message(start=start, stop=stop, planner=planner)
        self._socket.send(msg)
        print(f"[WBCBridge] Sent command: start={start}, stop={stop}, planner={planner}")

    def publish_joints(self, joint_pos, body_quat):
        """
        Publish target joint positions to the WBC controller.

        Args:
            joint_pos: np.ndarray of shape (29,) — target joint positions (MuJoCo order)
            body_quat: np.ndarray of shape (4,) — base quaternion (pass-through from state)
        """
        pose_data = {
            "joint_pos": joint_pos.astype(np.float32).reshape(1, 29),
            "joint_vel": np.zeros((1, 29), dtype=np.float32),
            "body_quat": body_quat.astype(np.float32).reshape(1, 4),
            "frame_index": np.array([self._frame_index], dtype=np.int64),
        }
        msg = pack_pose_message(pose_data, topic=self._topic, version=1)
        self._socket.send(msg)
        self._frame_index += 1

    def stop(self):
        self._socket.close()
        self._context.term()


# ---------------- Quaternion helpers [w, x, y, z] ----------------
def _quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)

# ---------------- Joint reordering (MuJoCo → IsaacLab DOF order) ----------------
_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)

# ---------------- Global state ----------------
running = threading.Event()
running.set()


# ---------------- RTCWebSocketClient ----------------
class RTCWebSocketClient:
    def __init__(self, server_url, state_subscriber, camera, wbc_bridge):
        self.server_url = server_url
        self._running = True
        self._connected = threading.Event()
        self._ws = None
        self._send_lock = threading.Lock()
        self.start_time = time.time()

        self._state_sub = state_subscriber
        self._camera = camera
        self._wbc_bridge = wbc_bridge
        self._q_ref = None       # set once on first robot state, shared by send and execute
        self._prev_quat = None   # for sign-continuity in execute_action

    def execute_action(self, action):
        """
        Publish VLA action (33D: 29 joint targets + 4 quat) to WBC bridge.
        action[0:29] — joint positions in MuJoCo order (relative to training)
        action[29:33] — quaternion relative to q_ref [w, x, y, z]
        """
        if action.ndim > 1:
            action = action[0]

        # if action.shape[0] != 33:
        #     print(f"[client] Unexpected action shape: {action.shape}, expected (33,)")
        #     return

        joint_action = action[:29].astype(np.float32)
        q_action_rel = action[29:33].astype(np.float32)

        # Reorder joints from MuJoCo order (policy output) to IsaacLab DOF order (WBC expects)
        action_isaac = joint_action[_MUJOCO_TO_ISAACLAB_DOF]

        # Compose absolute quat: q_abs = q_ref * q_action_rel
        if self._q_ref is None:
            print("[client] q_ref not set yet, skipping action")
            return
        body_quat = _quat_mul(self._q_ref, q_action_rel).astype(np.float64)

        # Sign-continuity fix: prevent 180-degree flip from double-cover ambiguity
        if self._prev_quat is not None and np.dot(body_quat, self._prev_quat) < 0:
            body_quat = -body_quat
        self._prev_quat = body_quat.copy()

        self._wbc_bridge.publish_joints(action_isaac, body_quat)

    def _on_open(self, ws):
        print("[client] Connected!")
        self._connected.set()

    def _on_message(self, ws, message):
        interval = time.time() - self.start_time
        self.start_time = time.time()
        print(f"[client] recv_action interval: {interval:.3f}s")

        try:
            data = json.loads(message)
            action_data = data.get("action")
            version = data.get("version", -1)

            if action_data is not None:
                action = convert_numpy_in_dict(action_data, numpy_deserialize)
                if isinstance(action, np.ndarray):
                    self.execute_action(action)
                    print(f"[client] Received action, version={version}, shape={action.shape}")

        except Exception as e:
            print(f"[client] Message processing error: {e}")

    def _on_error(self, ws, error):
        print(f"[client] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[client] Connection closed: {close_status_code} - {close_msg}")
        self._running = False
        running.clear()

    def _send_thread(self):
        print("[client] Send thread started, waiting for connection...")
        self._connected.wait()
        print("[client] Connected, starting observation loop")

        prev_tick = time.perf_counter()

        while self._running and running.is_set():
            try:
                # Get robot state
                state = self._state_sub.get_state()
                if state is None:
                    print("[client] No robot state yet, waiting...")
                    time.sleep(0.1)
                    continue

                body_q    = np.array(state["body_q_measured"],   dtype=np.float32)
                q_measured = np.array(state["base_quat_measured"], dtype=np.float32)

                # Record reference quat on first frame (same value used in execute_action)
                if self._q_ref is None:
                    self._q_ref = q_measured.copy()
                    print(f"[client] q_ref set: {self._q_ref}")

                # Compute relative quat (matches training data format)
                q_rel = _quat_mul(_quat_inv(self._q_ref), q_measured)

                states = np.concatenate((body_q, q_rel), axis=0)

                # Get camera frame
                frame = self._camera.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.uint8)

                # Build observation payload
                img_obs = {"video": frame}
                state_obs = {"states": states}

                payload = {
                    "image": img_obs,
                    "state": state_obs,
                    "gt_action": None,
                    "dataset_name": None,
                    "instruction": TASK_INSTRUCTION,
                    "history": None,
                    "condition": None,
                    "timestamp": None,
                }
                payload = convert_numpy_in_dict(payload, numpy_serialize)
                message = json.dumps(payload)

                # Send (thread-safe)
                with self._send_lock:
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send(message)
                    else:
                        print("[client] WebSocket not connected, skipping send")
                        break

            except Exception as e:
                print(f"[client] Send error: {e}")
                break

            now = time.perf_counter()
            interval = now - prev_tick
            prev_tick = now
            print(f"[client] send interval: {interval:.3f}s")

        print("[client] Send thread stopped")

    def run(self):
        print(f"[client] Connecting to {self.server_url}")

        self._ws = WebSocketApp(
            self.server_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        send_thread = threading.Thread(target=self._send_thread, daemon=True)
        send_thread.start()

        self._ws.run_forever()

        self._running = False
        send_thread.join(timeout=2.0)
        print("[client] Client stopped")

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()


# ---------------- Main ----------------
def main(server_url, zmq_host, zmq_pub_port, zmq_sub_port, zmq_topic, zmq_sub_topic,
         camera_address):
    print("[MAIN] Initializing components...")

    # 1. Initialize WBC bridge (ZMQ PUB)
    wbc_bridge = WBCBridge(host="*", port=zmq_pub_port, topic=zmq_topic)
    print(f"[MAIN] WBC bridge bound on port {zmq_pub_port}, topic='{zmq_topic}'")

    # 2. Initialize robot state subscriber (ZMQ SUB)
    state_sub = RobotStateSubscriber(host=zmq_host, port=zmq_sub_port, topic=zmq_sub_topic)
    print(f"[MAIN] State subscriber connected to {zmq_host}:{zmq_sub_port}, topic='{zmq_sub_topic}'")

    # 3. Initialize camera
    camera = RSCamera(address=camera_address)
    print(f"[MAIN] Camera connected to {camera_address}")

    # 4. Wait briefly for ZMQ PUB socket to establish connections
    time.sleep(1.0)

    # 5. Send start command (streamed motion mode)
    wbc_bridge.send_command(start=True, stop=False, planner=False)

    # 6. Wait for first robot state
    print("[MAIN] Waiting for robot state...")
    for i in range(30):
        state = state_sub.get_state()
        if state is not None:
            print(f"[MAIN] Got robot state with keys: {list(state.keys())}")
            body_q = np.array(state.get("body_q_measured", []))
            print(f"[MAIN] body_q_measured shape: {body_q.shape}")
            break
        time.sleep(0.5)
    else:
        print("[MAIN] WARNING: No robot state received after 15s, proceeding anyway...")

    # 7. Start WebSocket client
    client = RTCWebSocketClient(
        server_url=server_url,
        state_subscriber=state_sub,
        camera=camera,
        wbc_bridge=wbc_bridge,
    )

    def websocket_thread():
        client.run()
        print("[WS] WebSocket thread stopped")

    t_ws = threading.Thread(target=websocket_thread, daemon=True)
    t_ws.start()

    print("[MAIN] Running. Ctrl+C to stop.")

    # 8. Wait for shutdown
    def signal_handler(sig, frame):
        print("\n[MAIN] Caught signal, shutting down...")
        running.clear()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[MAIN] Caught Ctrl+C, exiting...")
        running.clear()

    # 9. Shutdown
    print("[MAIN] Shutting down...")
    client.stop()
    time.sleep(0.5)

    # Send stop command
    wbc_bridge.send_command(start=False, stop=True, planner=False)
    time.sleep(0.5)

    state_sub.stop()
    wbc_bridge.stop()
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLA Policy Inference with WBC Stabilization")
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
    parser.add_argument("--instruction", type=str, default=None,
                        help="Task instruction for VLA policy")

    args = parser.parse_args()

    if args.instruction:
        TASK_INSTRUCTION = args.instruction

    server_url = f"ws://{args.host}:{args.port}/ws"
    main(
        server_url=server_url,
        zmq_host=args.zmq_host,
        zmq_pub_port=args.zmq_pub_port,
        zmq_sub_port=args.zmq_sub_port,
        zmq_topic=args.zmq_topic,
        zmq_sub_topic=args.zmq_sub_topic,
        camera_address=args.camera_address,
    )
