"""
view_pose_stream.py

Kinematics-only MuJoCo viewer for the retargeted G1 pose stream published by
slimevr_pico_thread_server.py / slimevr_manus_thread_server.py / pico_gmr (the
sonic v1 "pose" topic on :5556).

It subscribes to the ZMQ stream, unpacks the v1 message ([topic][1280-byte JSON
header][binary fields]), converts joint_pos back from IsaacLab to MuJoCo order,
and poses the GMR unitree_g1 model with mj_forward only — no physics, no deploy.
The wire carries body_quat but no root translation, so the pelvis is pinned at a
fixed height and only rotates.

Run (same interpreter as the servers):
    .venv_teleop/bin/python gear_sonic/scripts/view_pose_stream.py

Needs a display (MuJoCo passive viewer). Close the window or Ctrl-C to exit.
"""

import json
import os
import sys
import time

import numpy as np
import zmq

# ── CONFIG ────────────────────────────────────────────────────────────────────
POSE_HOST   = "localhost"  # where the server's PUB is bound
POSE_PORT   = 5556
POSE_TOPIC  = "pose"
VIEW_FPS    = 50           # viewer refresh rate (stream is ~50 Hz)
ROOT_HEIGHT = 0.793        # fixed pelvis height (root pos is not on the wire)
# ──────────────────────────────────────────────────────────────────────────────

# GMR package lives in-repo; put it on the path like start_sonic.sh does.
_HERE = os.path.dirname(os.path.abspath(__file__))
_GMR_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "GMR"))
if _GMR_DIR not in sys.path:
    sys.path.insert(0, _GMR_DIR)

from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer  # noqa: E402

HEADER_SIZE = 1280  # matches gear_sonic.utils.teleop.zmq.zmq_planner_sender

# Same table as the servers (MuJoCo → IsaacLab); invert it to undo the reorder.
_MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)
_ISAACLAB_TO_MUJOCO = np.argsort(_MUJOCO_TO_ISAACLAB)

_DTYPES = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64, "bool": np.bool_}


def parse_pose_message(raw: bytes, topic: str = POSE_TOPIC) -> dict:
    """Decode a v1 [topic][1280-byte JSON header][binary] message into arrays."""
    tl = len(topic)
    if raw[:tl] != topic.encode("utf-8"):
        raise ValueError(f"Unexpected topic prefix in {len(raw)}-byte message")
    header = json.loads(raw[tl:tl + HEADER_SIZE].split(b"\x00", 1)[0])
    payload = raw[tl + HEADER_SIZE:]
    out, off = {}, 0
    for f in header["fields"]:
        dt = np.dtype(_DTYPES[f["dtype"]]).newbyteorder("<")
        n = int(np.prod(f["shape"])) if f["shape"] else 1
        out[f["name"]] = np.frombuffer(payload, dtype=dt, count=n, offset=off).reshape(f["shape"])
        off += n * dt.itemsize
    return out


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, POSE_TOPIC)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(f"tcp://{POSE_HOST}:{POSE_PORT}")
    print(f"[viewer] SUB tcp://{POSE_HOST}:{POSE_PORT} topic={POSE_TOPIC!r}")

    viewer = RobotMotionViewer(robot_type="unitree_g1", motion_fps=VIEW_FPS)
    print("[viewer] waiting for pose frames... (server must be sending, key 'k')")

    root_pos = np.array([0.0, 0.0, ROOT_HEIGHT])
    dof_pos = np.zeros(29)
    root_rot = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz, matches qpos[3:7]
    got_first = False
    n_frames = 0
    stat_t = time.time()

    try:
        while viewer.viewer.is_running():
            # Drain to the latest message; CONFLATE keeps the queue at 1 anyway.
            raw = None
            while sock.poll(timeout=0):
                raw = sock.recv(zmq.NOBLOCK)
            if raw is not None:
                fields = parse_pose_message(raw)
                jp = np.asarray(fields["joint_pos"], dtype=np.float64)
                bq = np.asarray(fields["body_quat"], dtype=np.float64)
                if jp.ndim == 2:  # sliding window — take the newest frame
                    jp, bq = jp[-1], bq[-1]
                dof_pos = jp[_ISAACLAB_TO_MUJOCO]
                root_rot = bq
                if not got_first:
                    got_first = True
                    print(f"[viewer] first frame (frame_index={int(fields['frame_index'][-1])})")
                n_frames += 1

            viewer.step(root_pos, root_rot, dof_pos, rate_limit=True)

            now = time.time()
            if now - stat_t >= 5.0:
                print(f"[viewer] stream ~{n_frames / (now - stat_t):.1f} msg/s")
                n_frames = 0
                stat_t = now
    except KeyboardInterrupt:
        print("\n[viewer] stopping...")
    finally:
        viewer.close()
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
