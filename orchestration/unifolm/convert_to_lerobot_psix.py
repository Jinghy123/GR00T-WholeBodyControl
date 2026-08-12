#!/usr/bin/env python3
"""Convert the Unifolm LeRobot-v3 dataset into the LeRobot PsiX (v2.1) format.

Source (LeRobot v3, e.g. /data/Unifolm/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher):
  data/chunk-XXX/file-XXX.parquet     — concatenated episodes, columns:
      observation.state.robot_q_current (36) / .hand_state (12) / .ee_state (12)
      action.robot_q_desired (36) / .hand_cmd (12) / .ee_action (12)
      timestamp / frame_index / episode_index / index / task_index
  meta/episodes/chunk-XXX/file-XXX.parquet — per-episode frame ranges + per-video
      file indices and from/to timestamps
  videos/<key>/chunk-XXX/file-XXX.mp4 — concatenated episodes (AV1)

Target (mirrors psi/scripts/data/raw_he_raw_to_lerobot_psix.py output, v2.1):
  <out>/data/chunk-XXX/episode_XXXXXX.parquet   (one file per episode)
  <out>/videos/chunk-XXX/observation.images.egocentric/episode_XXXXXX.mp4 (h264)
  <out>/meta/{info.json,episodes.jsonl,tasks.jsonl,episodes_stats.jsonl,modality.json}

Field mapping:
  robot_q_* 36 = root pos xyz [0:3] + root quat wxyz [3:7] + 29 joints [7:36]
                 (29 joints in MuJoCo serial order: Lleg6, Rleg6, waist3, Larm7, Rarm7)
  state 43 = hand14(state) + arm14(robot_q_current[22:36]) + leg15(robot_q_current[7:22])
  action 36 = hand14(cmd) + arm14(robot_q_desired[22:36]) + torso8(zeros, he_raw convention)
  hand14: BrainCo 12 → per-hand closure scalar (max finger / 0.8) → Dex3 close
          preset (same synergy the teleop trigger path uses)
  eepose (action.wrists.* / action.hands.*, 48 dims over 16 columns): zeros — TODO
  action.body_token 64: offline sonic v1.1 encoding of robot_q_desired
          (clean kinematic reference), same pipeline as orchestration/offline_record.py:
          mujoco→isaac reorder, 30→50Hz resample, 10-frame step5 future window,
          finite-difference velocities, FSQ quantization, map back to 30Hz.

Usage:
  .venv_teleop/bin/python orchestration/unifolm/convert_to_lerobot_psix.py \
      [--src DIR] [--out DIR] [--episodes N] [--no-video] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from encoder_client import EncoderClient  # noqa: E402
from walk_forward_token import (  # noqa: E402
    _MUJOCO_TO_ISAACLAB_DOF, fsq_quantize, resample_30hz_to_50hz,
)
from walk_then_replay import ENCODER_MODEL_V1_1, build_data_window  # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SRC = "/data/Unifolm/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher"
DEFAULT_OUT = "/data/Unifolm/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher_lerobot_psix/g1"
SRC_VIDEO_KEY = "observation.images.head_stereo_left"  # → egocentric

CODE_VERSION = "v2.1"
FPS = 30
CHUNKS_SIZE = 1000
EGO_VIDEO_KEY = "observation.images.egocentric"
BODY_TOKEN_KEY = "action.body_token"
BODY_TOKEN_DIM = 64
DONE_SUBTASK_SENTINEL = "__done__"
EMBODIMENT_TAG = "psix_he_g1_sonic"  # body tokens present → sonic tag

TASK_NAME = "G1_WBT_Brainco_Collect_Plates_Into_Dishwasher"
TASK_CATEGORY = "Unifolm"

# ── Joint name tables (identical to raw_he_raw_to_lerobot_psix.py g1 spec) ────

G1_HAND_NAMES = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
G1_ARM_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
G1_LEG_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]
TORSO_BLOCK_NAMES = [
    "torso_roll", "torso_pitch", "torso_yaw",
    "torso_height", "base_vx", "base_vy", "base_vyaw", "base_target_yaw",
]

STATE_NAMES = G1_HAND_NAMES + G1_ARM_NAMES + G1_LEG_NAMES            # 43
ACTION_NAMES = G1_HAND_NAMES + G1_ARM_NAMES + TORSO_BLOCK_NAMES      # 36
STATE_DIM, ACTION_DIM = len(STATE_NAMES), len(ACTION_NAMES)

# eepose columns (48 dims total) — zeros for now, TODO fill from ee_state/FK
EEPOSE_KEYS = [
    f"action.{part}.{comp}"
    for part in (
        "wrists.left", "wrists.right",
        "hands.left_thumb", "hands.left_index", "hands.left_middle",
        "hands.right_thumb", "hands.right_index", "hands.right_middle",
    )
    for comp in ("xyz", "rpy")
]

# ── BrainCo → Dex3 hand mapping ───────────────────────────────────────────────
# BrainCo per-hand: [thumb open/close, thumb lateral, index, middle, ring, little],
# commanded range is trigger*0.8 (pipeline cap). closure = max(fingers)/0.8.
# Dex3 close preset = teleop trigger synergy (G1GripperInverseKinematicsSolver
# middle-close); left hand, right = negation. Open = zeros.
_BRAINCO_CMD_MAX = 0.8
_BRAINCO_FINGER_DIMS = [0, 2, 3, 4, 5]
_DEX3_CLOSE_LEFT = np.array([0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5], dtype=np.float32)


def brainco12_to_dex3_14(hand12: np.ndarray) -> np.ndarray:
    """(N, 12) BrainCo → (N, 14) Dex3 [left7, right7] via per-hand closure.

    Closure = mean over the 5 moving finger dims. For hand_cmd the dims are
    strictly identical so mean == max; for hand_state (measured) mean preserves
    the partial-closure contact signal (fingers blocked by the grasped object
    stop early), whereas max would track the one unblocked finger and nearly
    duplicate the command.
    """
    left_c = np.clip(hand12[:, :6][:, _BRAINCO_FINGER_DIMS].mean(axis=1) / _BRAINCO_CMD_MAX, 0.0, 1.0)
    right_c = np.clip(hand12[:, 6:][:, _BRAINCO_FINGER_DIMS].mean(axis=1) / _BRAINCO_CMD_MAX, 0.0, 1.0)
    left = left_c[:, None] * _DEX3_CLOSE_LEFT[None, :]
    right = right_c[:, None] * (-_DEX3_CLOSE_LEFT)[None, :]
    return np.concatenate([left, right], axis=1).astype(np.float32)


# ── Sonic v1.1 token encoding (mirrors offline_record.encode_episode) ─────────

def encode_tokens_30hz(qpos29_mujoco_30: np.ndarray, quat_wxyz_30: np.ndarray,
                       encoder: EncoderClient) -> np.ndarray:
    """(n, 29) mujoco qpos + (n, 4) quat at 30Hz → (n, 64) FSQ tokens at 30Hz."""
    n_30 = len(qpos29_mujoco_30)
    qpos_isaac_30 = qpos29_mujoco_30[:, _MUJOCO_TO_ISAACLAB_DOF].astype(np.float32)

    quat = quat_wxyz_30.astype(np.float32).copy()
    quat /= np.linalg.norm(quat, axis=1, keepdims=True).clip(1e-8)
    for i in range(1, n_30):  # sign continuity
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]

    qpos_50 = resample_30hz_to_50hz(qpos_isaac_30)
    quat_50 = resample_30hz_to_50hz(quat)
    quat_50 /= np.linalg.norm(quat_50, axis=1, keepdims=True).clip(1e-8)
    n_50 = len(qpos_50)

    tokens_50 = np.zeros((n_50, BODY_TOKEN_DIM), dtype=np.float32)
    for pub_frame in range(n_50):
        jp, jv, bq = build_data_window(qpos_50, quat_50, pub_frame)
        tokens_50[pub_frame] = fsq_quantize(
            encoder.encode(joint_pos=jp, joint_vel=jv, body_quat=bq)
        ).astype(np.float32)

    idx = np.round(np.arange(n_30) * 5.0 / 3.0).astype(int).clip(0, n_50 - 1)
    return tokens_50[idx]


# ── Parquet schema / writing ──────────────────────────────────────────────────

def _f32_list(dim):  # noqa: ANN001
    return pa.list_(pa.float32(), -1)


def build_episode_table(ep_idx: int, n: int, state: np.ndarray, action: np.ndarray,
                        tokens: np.ndarray, task_index: int, task_description: str,
                        global_index_start: int) -> pa.Table:
    cols: dict[str, pa.Array] = {}
    cols["observation.state"] = pa.array(list(state), type=_f32_list(STATE_DIM))
    cols["action"] = pa.array(list(action), type=_f32_list(ACTION_DIM))
    cols["timestamp"] = pa.array((np.arange(n) / FPS).astype(np.float32), type=pa.float32())
    cols["frame_index"] = pa.array(np.arange(n, dtype=np.int64), type=pa.int64())
    cols["episode_index"] = pa.array(np.full(n, ep_idx, dtype=np.int64), type=pa.int64())
    cols["index"] = pa.array(np.arange(global_index_start, global_index_start + n, dtype=np.int64),
                             type=pa.int64())
    cols["task_index"] = pa.array(np.full(n, task_index, dtype=np.int64), type=pa.int64())
    cols["task_description"] = pa.array([task_description] * n, type=pa.string())
    done = np.zeros(n, dtype=bool)
    done[-1] = True
    cols["next.done"] = pa.array(done, type=pa.bool_())

    # Annotation columns — unannotated defaults (same as reference converter)
    cols["memory_desc"] = pa.array([""] * n, type=pa.string())
    cols["subtask_prompt"] = pa.array([""] * n, type=pa.string())
    cols["next_subtask"] = pa.array([""] * n, type=pa.string())
    cols["prev_subtask"] = pa.array([""] * n, type=pa.string())
    cols["memory_items"] = pa.array(["[]"] * n, type=pa.string())
    cols["sub_task_index"] = pa.array(np.zeros(n, dtype=np.int64), type=pa.int64())
    cols["sub_goal_frame_index"] = pa.array(np.zeros(n, dtype=np.int64), type=pa.int64())
    cols["sub_task_delta"] = pa.array(np.zeros(n, dtype=np.float32), type=pa.float32())
    cols["sub_goal_image_path"] = pa.array([""] * n, type=pa.string())

    # eepose — zeros, TODO: fill from ee_state / FK (48 dims over 16 columns)
    zero3 = np.zeros((n, 3), dtype=np.float32)
    for key in EEPOSE_KEYS:
        cols[key] = pa.array(list(zero3), type=_f32_list(3))

    cols[BODY_TOKEN_KEY] = pa.array(list(tokens), type=_f32_list(BODY_TOKEN_DIM))

    return pa.table(cols)


# ── Meta builders (contract of raw_he_raw_to_lerobot_psix.py) ─────────────────

def build_features_dict() -> dict:
    feats: dict = {
        EGO_VIDEO_KEY: {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": 30.0,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": ACTION_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "task_description": {"dtype": "string", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "memory_desc": {"dtype": "string", "shape": [1]},
        "subtask_prompt": {"dtype": "string", "shape": [1]},
        "next_subtask": {"dtype": "string", "shape": [1]},
        "prev_subtask": {"dtype": "string", "shape": [1]},
        "memory_items": {"dtype": "string", "shape": [1]},
        "sub_task_index": {"dtype": "int64", "shape": [1]},
        "sub_goal_frame_index": {"dtype": "int64", "shape": [1]},
        "sub_task_delta": {"dtype": "float32", "shape": [1]},
        "sub_goal_image_path": {
            "dtype": "string",
            "shape": [1],
            "image_info": {
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "encoding": "jpeg",
                "fps": 30,
            },
        },
    }
    for key in EEPOSE_KEYS:
        feats[key] = {"dtype": "float32", "shape": [3]}
    feats[BODY_TOKEN_KEY] = {"dtype": "float32", "shape": [BODY_TOKEN_DIM]}
    return feats


def build_modality_json() -> dict:
    arm_end = 14 + 14
    action_mod = {
        "hand_joints": {"original_key": "action", "start": 0, "end": 14,
                        "rotation_type": "euler_angles_rpy", "absolute": True, "dtype": "float32"},
        "arm_joints": {"original_key": "action", "start": 14, "end": arm_end,
                       "rotation_type": "euler_angles_rpy", "absolute": True, "dtype": "float32"},
        "torso_rpy": {"original_key": "action", "start": arm_end, "end": arm_end + 3,
                      "rotation_type": "euler_angles_rpy", "absolute": True, "dtype": "float32"},
        "base_height": {"original_key": "action", "start": arm_end + 3, "end": arm_end + 4,
                        "rotation_type": None, "absolute": True, "dtype": "float32"},
        "base_vx": {"original_key": "action", "start": arm_end + 4, "end": arm_end + 5,
                    "rotation_type": None, "absolute": False, "dtype": "float32"},
        "base_vy": {"original_key": "action", "start": arm_end + 5, "end": arm_end + 6,
                    "rotation_type": None, "absolute": False, "dtype": "float32"},
        "base_vyaw": {"original_key": "action", "start": arm_end + 6, "end": arm_end + 7,
                      "rotation_type": None, "absolute": False, "dtype": "float32"},
        "target_yaw": {"original_key": "action", "start": arm_end + 7, "end": arm_end + 8,
                       "rotation_type": None, "absolute": True, "dtype": "float32"},
    }
    for key in EEPOSE_KEYS:
        short = key[len("action."):]
        action_mod[short] = {"original_key": key, "start": 0, "end": 3,
                             "rotation_type": ("euler_angles_rpy" if key.endswith("rpy") else None),
                             "absolute": True, "dtype": "float32"}
    action_mod["body_token"] = {"original_key": BODY_TOKEN_KEY, "start": 0, "end": BODY_TOKEN_DIM,
                                "rotation_type": None, "absolute": True, "dtype": "float32"}
    return {
        "state": {
            "joint_positions": {"original_key": "observation.state", "start": 0, "end": STATE_DIM,
                                "rotation_type": "euler_angles_rpy", "absolute": True,
                                "dtype": "float32"},
        },
        "action": action_mod,
        "video": {"egocentric": {"original_key": EGO_VIDEO_KEY}},
        "subgoal": {"egocentric": {"original_key": "sub_goal_image_path"}},
        "annotation": {
            "human_instruction": {"original_key": "task_description"},
            "memory_desc": {"original_key": "memory_desc"},
            "sub_task": {"original_key": "subtask_prompt"},
            "next_sub_task": {"original_key": "next_subtask"},
            "prev_sub_task": {"original_key": "prev_subtask"},
            "memory_items": {"original_key": "memory_items"},
            "sub_task_delta": {"original_key": "sub_task_delta"},
        },
        "meta": {k: {} for k in (
            "quality", "speed", "control_mode", "embodiment_tag", "failure_type",
            "scene_type", "data_source", "operator_id", "task_description",
            "episode_length", "number_of_tasks",
        )},
    }


def episode_stats(action: np.ndarray, n: int) -> dict:
    a_min = action.min(0)
    a_max = action.max(0)
    a_mean = action.mean(0)
    a_std = action.std(0)
    return {
        "action": {
            "min": a_min.tolist(), "max": a_max.tolist(),
            "mean": a_mean.tolist(), "std": a_std.tolist(), "count": [n],
        },
        "timestamp": {
            "min": [0.0], "max": [(n - 1) / FPS],
            "mean": [((n - 1) / 2) / FPS],
            "std": [n / (2 * FPS * math.sqrt(3))],
            "count": [n],
        },
    }


# ── Video segment transcode ───────────────────────────────────────────────────

def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def cut_video(src_mp4: Path, t_from: float, t_to: float, out_mp4: Path, ff: str) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.with_suffix(".tmp.mp4")
    cmd = [
        ff, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{t_from:.6f}", "-to", f"{t_to:.6f}", "-i", str(src_mp4),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", "-f", "mp4", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp, out_mp4)


# ── Main conversion ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=str, default=DEFAULT_SRC)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--episodes", type=int, default=None,
                    help="Convert only the first N episodes (debug)")
    ap.add_argument("--no-video", action="store_true", help="Skip video transcodes")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-convert episodes that already have output")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    ff = None if args.no_video else ffmpeg_exe()

    # Source meta
    with open(src / "meta" / "info.json") as f:
        src_info = json.load(f)
    tasks_df = pd.read_parquet(src / "meta" / "tasks.parquet")
    task_description = str(tasks_df.iloc[0]["task"])

    ep_meta_files = sorted((src / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(p) for p in ep_meta_files]).sort_values("episode_index")
    if args.episodes is not None:
        ep_meta = ep_meta.iloc[: args.episodes]

    src_data_files = {}  # (chunk, file) → DataFrame cache (one at a time)

    def load_data_file(chunk_idx: int, file_idx: int) -> pd.DataFrame:
        key = (chunk_idx, file_idx)
        if key not in src_data_files:
            src_data_files.clear()  # keep at most one cached (files are ~500MB in RAM)
            path = src / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
            src_data_files[key] = pd.read_parquet(path)
        return src_data_files[key]

    print(f"[convert] encoder: {ENCODER_MODEL_V1_1} (v1_1)")
    encoder = EncoderClient(ENCODER_MODEL_V1_1, mode=0, version="v1_1")

    episodes_rows, stats_rows = [], []
    total_frames = 0
    t_start = time.perf_counter()

    for _, ep in ep_meta.iterrows():
        ep_idx = int(ep["episode_index"])
        n = int(ep["length"])
        chunk_id = ep_idx // CHUNKS_SIZE

        pq_out = out / "data" / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}.parquet"
        mp4_out = out / "videos" / f"chunk-{chunk_id:03d}" / EGO_VIDEO_KEY / f"episode_{ep_idx:06d}.mp4"

        df = load_data_file(int(ep["data/chunk_index"]), int(ep["data/file_index"]))
        rows = df[df["episode_index"] == ep_idx]
        if len(rows) != n:
            raise ValueError(f"episode {ep_idx}: meta length {n} != parquet rows {len(rows)}")

        q_cur = np.stack(rows["observation.state.robot_q_current"].values).astype(np.float32)   # (n,36)
        q_des = np.stack(rows["action.robot_q_desired"].values).astype(np.float32)               # (n,36)
        hand_state = np.stack(rows["observation.state.hand_state"].values).astype(np.float32)    # (n,12)
        hand_cmd = np.stack(rows["action.hand_cmd"].values).astype(np.float32)                   # (n,12)

        # state 43 = hand14 + arm14 + leg15 (mujoco 29 = legs[0:12] waist[12:15] arms[15:29])
        cur29 = q_cur[:, 7:36]
        state = np.concatenate(
            [brainco12_to_dex3_14(hand_state), cur29[:, 15:29], cur29[:, 0:15]], axis=1)
        # action 36 = hand14 + arm14 + torso8(zeros)
        des29 = q_des[:, 7:36]
        action = np.concatenate(
            [brainco12_to_dex3_14(hand_cmd), des29[:, 15:29],
             np.zeros((n, 8), dtype=np.float32)], axis=1)

        if pq_out.exists() and (args.no_video or mp4_out.exists()) and not args.overwrite:
            print(f"[convert] episode {ep_idx:3d}: exists, skipping")
        else:
            tokens = encode_tokens_30hz(des29, q_des[:, 3:7], encoder)  # (n, 64)

            table = build_episode_table(
                ep_idx, n, state, action, tokens,
                task_index=0, task_description=task_description,
                global_index_start=total_frames)
            pq_out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, pq_out)

            if not args.no_video:
                v_chunk = int(ep[f"videos/{SRC_VIDEO_KEY}/chunk_index"])
                v_file = int(ep[f"videos/{SRC_VIDEO_KEY}/file_index"])
                t_from = float(ep[f"videos/{SRC_VIDEO_KEY}/from_timestamp"])
                t_to = float(ep[f"videos/{SRC_VIDEO_KEY}/to_timestamp"])
                src_mp4 = src / "videos" / SRC_VIDEO_KEY / f"chunk-{v_chunk:03d}" / f"file-{v_file:03d}.mp4"
                cut_video(src_mp4, t_from, t_to, mp4_out, ff)

            done = len(episodes_rows) + 1
            elapsed = time.perf_counter() - t_start
            print(f"[convert] episode {ep_idx:3d}: {n} frames ok "
                  f"({done}/{len(ep_meta)}, {elapsed:.0f}s elapsed)")

        episodes_rows.append({
            "episode_index": ep_idx,
            "episode_unique_id": f"{TASK_CATEGORY}/{TASK_NAME}/episode_{ep_idx}",
            "tasks": [0],
            "length": n,
            "dataset_from_index": total_frames,
            "dataset_to_index": total_frames + n - 1,
            "task_description": task_description,
            "episode_length": n,
            "number_of_tasks": 1,
            "speed": 1.0,
            "quality": 1.0,
            "control_mode": "",
            "embodiment_tag": EMBODIMENT_TAG,
            "failure_type": "",
            "scene_type": TASK_CATEGORY,
            "data_source": "teleop",
            "operator_id": "",
        })
        stats_rows.append({"episode_index": ep_idx, "stats": episode_stats(action, n)})
        total_frames += n

    # ── meta files ────────────────────────────────────────────────────────────
    n_eps = len(episodes_rows)
    info = {
        "codebase_version": CODE_VERSION,
        "robot_type": "g1",
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0 if args.no_video else n_eps,
        "total_chunks": max(1, math.ceil(n_eps / CHUNKS_SIZE)),
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": build_features_dict(),
    }
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for row in episodes_rows:
            f.write(json.dumps(row) + "\n")
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({
            "task_index": 0, "task": TASK_NAME,
            "category": TASK_CATEGORY, "description": task_description,
        }) + "\n")
    with open(out / "meta" / "episodes_stats.jsonl", "w") as f:
        for row in stats_rows:
            f.write(json.dumps(row) + "\n")
    with open(out / "meta" / "modality.json", "w") as f:
        json.dump(build_modality_json(), f, indent=4)
    # images/ dir exists for layout parity (subgoal jpgs would live here)
    (out / "images" / "observation.images.subgoal").mkdir(parents=True, exist_ok=True)

    print(f"[convert] DONE: {n_eps} episodes, {total_frames} frames → {out}")


if __name__ == "__main__":
    main()
