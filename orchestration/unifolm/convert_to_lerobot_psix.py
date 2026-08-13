#!/usr/bin/env python3
"""Convert Unifolm LeRobot-v3 datasets into a single LeRobot PsiX (v2.1) dataset.

Handles all three Unifolm dataset families under one roof:

  wbt_brainco  G1_WBT_Brainco_*  36+12+12 whole-body, BrainCo Revo2 hand
               hand layout [thumbOC, thumbLat, index, middle, ring, little],
               cmd = trigger * 0.8 (hard cap), thumb lateral pinned at 0.8
  wbt_inspire  G1_WBT_Inspire_*  36+12+12 whole-body, Inspire five-finger hand
               hand layout [index, middle, ring, little, thumbOC, thumbLat],
               finger cmd capped at 0.9, thumbOC at 0.8, thumb lateral pinned 0
  dex3_upper   G1_Dex3_*  28-dim upper body only ([arm14, dex3-hand14]);
               legs/waist filled with the default standing pose and identity
               root quat (same approach as orchestration/offline_record.py)

Output (mirrors psi/scripts/data/raw_he_raw_to_lerobot_psix.py, v2.1):
  <out>/data/chunk-XXX/episode_XXXXXX.parquet   (one file per episode)
  <out>/videos/chunk-XXX/observation.images.egocentric/episode_XXXXXX.mp4 (h264)
  <out>/meta/{info.json,episodes.jsonl,tasks.jsonl,episodes_stats.jsonl,modality.json}

All source datasets merge into ONE psix g1 dataset: episodes are renumbered
globally (sorted dataset order × source episode order — keep the dataset set
stable across resumed runs), one tasks.jsonl row per source dataset.

Field mapping (common):
  state 43 = hand14(state) + arm14(measured) + leg15(measured | standing default)
  action 36 = hand14(cmd) + arm14(desired) + torso8: torso_rpy = desired waist
          (roll, pitch, yaw) for WBT datasets (zeros for Dex3), rest zeros
  hand14: five-finger hands → per-hand closure scalar (open/close endpoint
          normalization per family, direction-aware) → virtual-Dex3 close
          preset; Dex3 datasets pass through unchanged
  eepose (16 cols / 48 dims): fixed-base upper-body FK on assets/g1 URDF,
          WITH the desired waist angles filled in (pelvis-frame pose incl.
          torso lean — matches Unifolm ee semantics; Dex3 sets: waist zero)
  action.body_token 64:       sonic v1 (release tokenizer) offline encoding
  action.body_token_v1_1 64:  sonic v1.1 encoding
          (both: desired joints, mujoco→isaac, 30→50Hz, 10-frame step5 future
          window, FSQ quantize)

Usage:
  .venv_teleop/bin/python orchestration/unifolm/convert_to_lerobot_psix.py \
      --src-root /data/hongyi/data/Unifolm --out <dir>/g1 \
      [--datasets G1_WBT_Brainco_Make_The_Bed,...] [--episodes N] \
      [--no-video] [--video-workers 4] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from encoder_client import EncoderClient  # noqa: E402
from walk_forward_token import (  # noqa: E402
    _MUJOCO_TO_ISAACLAB_DOF, G1_DEFAULT_ANGLES_MUJOCO, fsq_quantize,
    resample_30hz_to_50hz,
)
from walk_forward_token import ENCODER_MODEL as ENCODER_MODEL_V1  # noqa: E402
from walk_then_replay import ENCODER_MODEL_V1_1, build_data_window  # noqa: E402
from orchestration.unifolm.g1_eepose import DEFAULT_G1_URDF, G1EeposeFK  # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SRC_ROOT = "/data/hongyi/data/Unifolm"
DEFAULT_OUT = "/data/hongyi/data/Unifolm_lerobot_psix/g1"

CODE_VERSION = "v2.1"
FPS = 30
CHUNKS_SIZE = 1000
EGO_VIDEO_KEY = "observation.images.egocentric"
BODY_TOKEN_KEY = "action.body_token"          # sonic v1 (release tokenizer)
BODY_TOKEN_V11_KEY = "action.body_token_v1_1"  # sonic v1.1
BODY_TOKEN_DIM = 64
EMBODIMENT_TAG = "psix_he_g1_sonic"  # body tokens present → sonic tag
TASK_CATEGORY = "Unifolm"

# Curated prompts: the Dex3 datasets' tasks.parquet strings are terse shorthands
# ("stack three block", "pour water"); these come from each dataset README's
# Task Objective. Datasets not listed here use their tasks.parquet string.
TASK_DESCRIPTION_OVERRIDES = {
    "G1_Dex3_BlockStacking_Dataset":
        "Stack the three cubic blocks on the desktop from bottom to top in the "
        "order of red, yellow, and blue on the black tape affixed to the desktop.",
    "G1_Dex3_CameraPackaging_Dataset":
        "Place the RealSense D-405 camera into the mounting case and secure the lid.",
    "G1_Dex3_ObjectPlacement_Dataset":
        "Pick up the toothpaste and trash bag on the desk, and place them into "
        "the blue storage container.",
    "G1_Dex3_Pouring_Dataset":
        "Take the water bottle and simulate pouring water into a transparent glass cup.",
    "G1_Dex3_ToastedBread_Dataset":
        "Place the bread from the serving tray into the toaster. After the bread "
        "is toasted, hand the toast to the human.",
}

# Source camera → egocentric, first present wins (per-dataset):
#   head_stereo_left: WBT Brainco/Inspire; cam_left_high: Dex3; cam_0: MainCamOnly
SRC_VIDEO_KEYS = [
    "observation.images.head_stereo_left",
    "observation.images.cam_left_high",
    "observation.images.cam_0",
]

# Standing default for legs+waist (mujoco order [Lleg6, Rleg6, waist3]) — used
# for the upper-body-only Dex3 datasets, mirrors offline_record's stable-leg path.
LEG15_DEFAULT = G1_DEFAULT_ANGLES_MUJOCO[:15].astype(np.float32)

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

EEPOSE_KEYS = [
    f"action.{part}.{comp}"
    for part in (
        "wrists.left", "wrists.right",
        "hands.left_thumb", "hands.left_index", "hands.left_middle",
        "hands.right_thumb", "hands.right_index", "hands.right_middle",
    )
    for comp in ("xyz", "rpy")
]

# ── Hand adapters ─────────────────────────────────────────────────────────────
# Five-finger hands → per-hand closure scalar → virtual-Dex3 close preset (the
# teleop trigger synergy from G1GripperInverseKinematicsSolver middle-close).
# closure uses ABSOLUTE normalization (dim / family close cap), NOT per-dataset
# min-max: demos that keep a light finger curl stay lightly curled on Dex3.
_DEX3_CLOSE_LEFT = np.array([0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5], dtype=np.float32)

# per-family: (moving dim indices within a 6-dim hand, open value, close value)
# per moving dim; closure = (val - open) / (close - open), so direction-agnostic.
HAND_CALIB = {
    # [thumbOC, thumbLat, index, middle, ring, little]; thumbLat pinned at 0.8.
    # 0 = open, cmd = trigger * 0.8 (verified: 0.8/256 quantization grid).
    "brainco": (np.array([0, 2, 3, 4, 5]),
                np.zeros(5, dtype=np.float32),
                np.array([0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float32)),
    # [index, middle, ring, little, thumbOC, thumbLat]; thumbLat pinned at 0.
    # REVERSED direction (native Inspire convention, despite the dataset README):
    # fingers rest/open at 0.9 (episode-start value and mode), close toward 0;
    # thumbOC open at 0.8.
    "inspire": (np.array([0, 1, 2, 3, 4]),
                np.array([0.9, 0.9, 0.9, 0.9, 0.8], dtype=np.float32),
                np.zeros(5, dtype=np.float32)),
}


def hand12_to_dex3_14(hand12: np.ndarray, family: str) -> np.ndarray:
    """(N, 12) five-finger [left6, right6] → (N, 14) virtual Dex3 [left7, right7]."""
    dims, open_v, close_v = HAND_CALIB[family]
    span = close_v - open_v
    left_c = np.clip(((hand12[:, :6][:, dims] - open_v) / span).mean(axis=1), 0.0, 1.0)
    right_c = np.clip(((hand12[:, 6:][:, dims] - open_v) / span).mean(axis=1), 0.0, 1.0)
    left = left_c[:, None] * _DEX3_CLOSE_LEFT[None, :]
    right = right_c[:, None] * (-_DEX3_CLOSE_LEFT)[None, :]
    return np.concatenate([left, right], axis=1).astype(np.float32)


# ── Profile detection ─────────────────────────────────────────────────────────

def detect_profile(ds_dir: Path, info: dict) -> str:
    """Return 'wbt_brainco' | 'wbt_inspire' | 'dex3_upper' from schema + stats."""
    feats = info["features"]
    if "action" in feats and feats["action"].get("shape") == [28]:
        return "dex3_upper"
    if "action.robot_q_desired" not in feats:
        raise ValueError(f"{ds_dir.name}: unrecognized schema "
                         f"(no robot_q_desired, no 28-dim action)")

    # Brainco vs Inspire family: hand_cmd stats signature is authoritative —
    # Brainco pins thumb-lateral (dims 1,7) at 0.8; Inspire/Dex5 pin dims 5,11 at 0.
    sig = None
    stats_path = ds_dir / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            hc = json.load(f).get("action.hand_cmd")
        if hc:
            mn, mx = np.array(hc["min"]), np.array(hc["max"])
            if np.allclose(mn[[1, 7]], 0.8, atol=1e-3) and np.allclose(mx[[1, 7]], 0.8, atol=1e-3):
                sig = "wbt_brainco"
            elif np.allclose(mn[[5, 11]], 0.0, atol=1e-3) and np.allclose(mx[[5, 11]], 0.0, atol=1e-3):
                sig = "wbt_inspire"

    name = ds_dir.name.lower()
    by_name = "wbt_brainco" if "brainco" in name else \
              "wbt_inspire" if ("inspire" in name or "dex5" in name) else None

    if sig and by_name and sig != by_name:
        raise ValueError(f"{ds_dir.name}: stats signature says {sig} but name says {by_name}")
    profile = sig or by_name
    if profile is None:
        raise ValueError(f"{ds_dir.name}: cannot determine hand family "
                         f"(no stats signature, name has no brainco/inspire)")
    return profile


def pick_video_key(info: dict) -> str:
    for key in SRC_VIDEO_KEYS:
        if key in info["features"]:
            return key
    raise ValueError(f"none of {SRC_VIDEO_KEYS} present "
                     f"(cameras: {[k for k in info['features'] if 'image' in k]})")


def read_task_string(ds_dir: Path) -> str:
    """tasks.parquet appears with either [task_index, task] columns or the task
    string as the index — handle both; fall back to the dataset name."""
    df = pd.read_parquet(ds_dir / "meta" / "tasks.parquet")
    if "task" in df.columns:
        s = str(df.iloc[0]["task"])
    else:
        s = str(df.index[0])
    return s if s and s not in ("0", "nan") else ds_dir.name


# ── Per-profile episode extraction ────────────────────────────────────────────

def extract_wbt(rows: pd.DataFrame, family: str):
    """WBT 36+12+12 → (state43, action36, qpos29_mujoco_desired, quat_wxyz,
    waist_yrp3_desired). torso_rpy in action36 carries the desired waist
    (roll, pitch, yaw); mujoco waist order is [yaw, roll, pitch] = des29[12:15]."""
    n = len(rows)
    q_cur = np.stack(rows["observation.state.robot_q_current"].values).astype(np.float32)
    q_des = np.stack(rows["action.robot_q_desired"].values).astype(np.float32)
    hand_state = np.stack(rows["observation.state.hand_state"].values).astype(np.float32)
    hand_cmd = np.stack(rows["action.hand_cmd"].values).astype(np.float32)

    cur29, des29 = q_cur[:, 7:36], q_des[:, 7:36]
    state = np.concatenate(
        [hand12_to_dex3_14(hand_state, family), cur29[:, 15:29], cur29[:, 0:15]], axis=1)
    torso8 = np.zeros((n, 8), dtype=np.float32)
    torso8[:, 0:3] = des29[:, [13, 14, 12]]  # torso_roll, torso_pitch, torso_yaw
    action = np.concatenate(
        [hand12_to_dex3_14(hand_cmd, family), des29[:, 15:29], torso8], axis=1)
    return state, action, des29, q_des[:, 3:7], des29[:, 12:15]


def extract_dex3(rows: pd.DataFrame):
    """Dex3 28-dim [arm14, hand14] → same tuple; legs = standing default,
    quat = identity, waist = None (upper-body-only recordings)."""
    n = len(rows)
    st28 = np.stack(rows["observation.state"].values).astype(np.float32)
    ac28 = np.stack(rows["action"].values).astype(np.float32)

    leg_tile = np.tile(LEG15_DEFAULT, (n, 1))
    state = np.concatenate([st28[:, 14:28], st28[:, 0:14], leg_tile], axis=1)
    action = np.concatenate(
        [ac28[:, 14:28], ac28[:, 0:14], np.zeros((n, 8), dtype=np.float32)], axis=1)

    qpos29 = np.concatenate([leg_tile, ac28[:, 0:14]], axis=1)  # mujoco order
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    return state, action, qpos29, quat, None


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
                        tokens_v1: np.ndarray, tokens_v11: np.ndarray, eepose: dict,
                        task_index: int, task_description: str,
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

    for key in EEPOSE_KEYS:
        cols[key] = pa.array(list(eepose[key]), type=_f32_list(3))

    cols[BODY_TOKEN_KEY] = pa.array(list(tokens_v1), type=_f32_list(BODY_TOKEN_DIM))
    cols[BODY_TOKEN_V11_KEY] = pa.array(list(tokens_v11), type=_f32_list(BODY_TOKEN_DIM))
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
    feats[BODY_TOKEN_V11_KEY] = {"dtype": "float32", "shape": [BODY_TOKEN_DIM]}
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
    action_mod["body_token_v1_1"] = {"original_key": BODY_TOKEN_V11_KEY, "start": 0,
                                     "end": BODY_TOKEN_DIM, "rotation_type": None,
                                     "absolute": True, "dtype": "float32"}
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
    return {
        "action": {
            "min": action.min(0).tolist(), "max": action.max(0).tolist(),
            "mean": action.mean(0).tolist(), "std": action.std(0).tolist(), "count": [n],
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
    subprocess.run(cmd, check=True, capture_output=True)
    os.replace(tmp, out_mp4)


# ── Dataset discovery ─────────────────────────────────────────────────────────

def discover_datasets(src_root: Path, only: list[str] | None) -> list[Path]:
    if (src_root / "meta" / "info.json").exists():  # src-root is a single dataset
        return [src_root]
    out = []
    for child in sorted(src_root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "meta" / "info.json").exists():
            out.append(child)
    if only:
        keep = set(only)
        out = [d for d in out if d.name in keep]
        missing = keep - {d.name for d in out}
        if missing:
            raise ValueError(f"--datasets not found under {src_root}: {sorted(missing)}")
    if not out:
        raise ValueError(f"no datasets with meta/info.json under {src_root}")
    return out


# ── Main conversion ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src-root", type=str, default=DEFAULT_SRC_ROOT,
                    help="Root with Unifolm datasets, or a single dataset dir")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--datasets", type=str, default=None,
                    help="Comma-separated dataset names to convert (default: all). "
                         "NOTE: episode numbering depends on the dataset set — keep "
                         "it stable across resumed runs.")
    ap.add_argument("--episodes", type=int, default=None,
                    help="Convert only the first N episodes per dataset (debug)")
    ap.add_argument("--no-video", action="store_true", help="Skip video transcodes")
    ap.add_argument("--video-workers", type=int, default=4,
                    help="Parallel ffmpeg transcodes (default: 4)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-convert episodes that already have output")
    ap.add_argument("--urdf", type=str, default=DEFAULT_G1_URDF,
                    help="G1 URDF (body29+hand14) for eepose FK")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out = Path(args.out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    ff = None if args.no_video else ffmpeg_exe()

    datasets = discover_datasets(
        src_root, args.datasets.split(",") if args.datasets else None)
    print(f"[convert] {len(datasets)} datasets under {src_root}")

    print(f"[convert] encoder v1:   {ENCODER_MODEL_V1}")
    encoder_v1 = EncoderClient(ENCODER_MODEL_V1, mode=0, version="v1")
    print(f"[convert] encoder v1.1: {ENCODER_MODEL_V1_1}")
    encoder_v11 = EncoderClient(ENCODER_MODEL_V1_1, mode=0, version="v1_1")
    print(f"[convert] eepose FK urdf: {args.urdf}")
    fk = G1EeposeFK(args.urdf)

    # Resume state: previously written episode records keyed by episode_index
    episodes_path = out / "meta" / "episodes.jsonl"
    stats_path = out / "meta" / "episodes_stats.jsonl"
    done_eps: dict[int, dict] = {}
    done_stats: dict[int, dict] = {}
    if episodes_path.exists() and not args.overwrite:
        with open(episodes_path) as f:
            for line in f:
                row = json.loads(line)
                done_eps[row["episode_index"]] = row
        if stats_path.exists():
            with open(stats_path) as f:
                for line in f:
                    row = json.loads(line)
                    done_stats[row["episode_index"]] = row
        print(f"[convert] resume: {len(done_eps)} episodes already recorded")

    pool = ThreadPoolExecutor(max_workers=max(1, args.video_workers))
    episodes_rows: dict[int, dict] = {}
    stats_rows: dict[int, dict] = {}
    tasks_rows = []
    global_ep = 0
    total_frames = 0
    t_start = time.perf_counter()
    n_converted = 0

    for task_index, ds_dir in enumerate(datasets):
        with open(ds_dir / "meta" / "info.json") as f:
            info = json.load(f)
        profile = detect_profile(ds_dir, info)
        family = "brainco" if profile == "wbt_brainco" else "inspire"
        src_video_key = pick_video_key(info)
        task_description = TASK_DESCRIPTION_OVERRIDES.get(ds_dir.name) or read_task_string(ds_dir)
        tasks_rows.append({
            "task_index": task_index, "task": ds_dir.name,
            "category": TASK_CATEGORY, "description": task_description,
        })

        ep_meta_files = sorted((ds_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        ep_meta = pd.concat([pd.read_parquet(p) for p in ep_meta_files]).sort_values("episode_index")
        if args.episodes is not None:
            ep_meta = ep_meta.iloc[: args.episodes]
        print(f"[convert] == {ds_dir.name}: profile={profile} cam={src_video_key} "
              f"episodes={len(ep_meta)}")

        data_cache: dict = {}

        def load_data_file(chunk_idx: int, file_idx: int) -> pd.DataFrame:
            key = (chunk_idx, file_idx)
            if key not in data_cache:
                data_cache.clear()  # keep at most one file in RAM
                path = ds_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
                data_cache[key] = pd.read_parquet(path)
            return data_cache[key]

        pending = []  # (ep_row, stats_row, video_future | None)

        for _, ep in ep_meta.iterrows():
            src_ep_idx = int(ep["episode_index"])
            n = int(ep["length"])
            ep_idx = global_ep
            chunk_id = ep_idx // CHUNKS_SIZE
            global_ep += 1

            pq_out = out / "data" / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}.parquet"
            mp4_out = out / "videos" / f"chunk-{chunk_id:03d}" / EGO_VIDEO_KEY / f"episode_{ep_idx:06d}.mp4"

            already = (ep_idx in done_eps and pq_out.exists()
                       and (args.no_video or mp4_out.exists()))
            if already and not args.overwrite:
                episodes_rows[ep_idx] = done_eps[ep_idx]
                if ep_idx in done_stats:
                    stats_rows[ep_idx] = done_stats[ep_idx]
                total_frames += n
                continue

            df = load_data_file(int(ep["data/chunk_index"]), int(ep["data/file_index"]))
            rows = df[df["episode_index"] == src_ep_idx]
            if len(rows) != n:
                raise ValueError(f"{ds_dir.name} ep {src_ep_idx}: meta length {n} "
                                 f"!= parquet rows {len(rows)}")

            if profile == "dex3_upper":
                state, action, qpos29, quat, waist_yrp = extract_dex3(rows)
            else:
                state, action, qpos29, quat, waist_yrp = extract_wbt(rows, family)

            tokens_v1 = encode_tokens_30hz(qpos29, quat, encoder_v1)
            tokens_v11 = encode_tokens_30hz(qpos29, quat, encoder_v11)
            eepose = fk.compute_batch(action[:, :28], waist_yrp)

            table = build_episode_table(
                ep_idx, n, state, action, tokens_v1, tokens_v11, eepose,
                task_index=task_index, task_description=task_description,
                global_index_start=total_frames)
            pq_out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, pq_out)

            fut = None
            if not args.no_video:
                v_chunk = int(ep[f"videos/{src_video_key}/chunk_index"])
                v_file = int(ep[f"videos/{src_video_key}/file_index"])
                t_from = float(ep[f"videos/{src_video_key}/from_timestamp"])
                t_to = float(ep[f"videos/{src_video_key}/to_timestamp"])
                src_mp4 = ds_dir / "videos" / src_video_key / f"chunk-{v_chunk:03d}" / f"file-{v_file:03d}.mp4"
                fut = pool.submit(cut_video, src_mp4, t_from, t_to, mp4_out, ff)

            ep_row = {
                "episode_index": ep_idx,
                "episode_unique_id": f"{TASK_CATEGORY}/{ds_dir.name}/episode_{src_ep_idx}",
                "tasks": [task_index],
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
            }
            pending.append((ep_row, {"episode_index": ep_idx, "stats": episode_stats(action, n)}, fut))
            total_frames += n
            n_converted += 1
            elapsed = time.perf_counter() - t_start
            print(f"[convert] ep {ep_idx:5d} ({ds_dir.name} #{src_ep_idx}): {n} frames "
                  f"({n_converted} converted, {elapsed:.0f}s)", flush=True)

        # Drain this dataset's video jobs, then commit its meta rows
        for ep_row, st_row, fut in pending:
            if fut is not None:
                fut.result()  # raises on ffmpeg failure
            episodes_rows[ep_row["episode_index"]] = ep_row
            stats_rows[st_row["episode_index"]] = st_row
        data_cache.clear()

        # Persist meta incrementally so an interrupted run can resume
        _write_meta(out, episodes_rows, stats_rows, tasks_rows, args.no_video)

    pool.shutdown(wait=True)
    _write_meta(out, episodes_rows, stats_rows, tasks_rows, args.no_video)
    print(f"[convert] DONE: {len(episodes_rows)} episodes, "
          f"{sum(r['length'] for r in episodes_rows.values())} frames → {out}")


def _write_meta(out: Path, episodes_rows: dict, stats_rows: dict,
                tasks_rows: list, no_video: bool) -> None:
    n_eps = len(episodes_rows)
    total_frames = sum(r["length"] for r in episodes_rows.values())
    info = {
        "codebase_version": CODE_VERSION,
        "robot_type": "g1",
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": len(tasks_rows),
        "total_videos": 0 if no_video else n_eps,
        "total_chunks": max(1, math.ceil(n_eps / CHUNKS_SIZE)),
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": build_features_dict(),
    }
    meta = out / "meta"
    with open(meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    with open(meta / "episodes.jsonl", "w") as f:
        for k in sorted(episodes_rows):
            f.write(json.dumps(episodes_rows[k]) + "\n")
    with open(meta / "episodes_stats.jsonl", "w") as f:
        for k in sorted(stats_rows):
            f.write(json.dumps(stats_rows[k]) + "\n")
    with open(meta / "tasks.jsonl", "w") as f:
        for row in tasks_rows:
            f.write(json.dumps(row) + "\n")
    with open(meta / "modality.json", "w") as f:
        json.dump(build_modality_json(), f, indent=4)
    (out / "images" / "observation.images.subgoal").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
