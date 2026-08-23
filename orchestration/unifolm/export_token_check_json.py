#!/usr/bin/env python3
"""Export one Unifolm episode to a sonic-style data.json for token verification.

Runs the exact same pipeline as convert_to_lerobot_psix.py (profile detection,
hand mapping, sonic v1.1 token encoding) on a single episode and writes
<out-dir>/<dataset>__episode_XXX/data.json in the format replay_token.py reads:

    [{"timestamp": t,
      "states":  {"qpos": 29 (mujoco: leg15+arm14, measured),
                  "quat": 4 (wxyz), "hand_joints": 14 (dex3, measured/mapped)},
      "actions": {"qpos": 29 (mujoco, desired),
                  "hand_joints": 14 (dex3 cmd, mapped),
                  "token": 64 (sonic v1.1, FSQ-quantized)}}, ...]

Verify on robot/sim:  python replay_token.py --episode-dir <out-dir>/<name>

Usage:
  .venv_teleop/bin/python orchestration/unifolm/export_token_check_json.py \
      --dataset /data/Unifolm/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
      [--episode 0] [--out-dir /data/Unifolm/token_check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from encoder_client import EncoderClient  # noqa: E402
from walk_forward_token import ENCODER_MODEL as ENCODER_MODEL_V1  # noqa: E402
from walk_then_replay import ENCODER_MODEL_V1_1  # noqa: E402
from orchestration.unifolm.convert_to_lerobot_psix import (  # noqa: E402
    FPS, LEG15_DEFAULT, detect_profile, encode_tokens_30hz, extract_dex3,
    extract_wbt, hand12_to_dex3_14, read_task_string,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=str, required=True, help="Source dataset dir")
    ap.add_argument("--episode", type=int, default=0, help="Source episode index")
    ap.add_argument("--out-dir", type=str, default="/data/Unifolm/token_check")
    ap.add_argument("--encoder-version", type=str, choices=["v1", "v1_1"], default="v1_1",
                    help="Sonic tokenizer: v1 (release checkpoint) or v1_1 (default)")
    args = ap.parse_args()

    ds_dir = Path(args.dataset)
    with open(ds_dir / "meta" / "info.json") as f:
        info = json.load(f)
    profile = detect_profile(ds_dir, info)
    family = "brainco" if profile == "wbt_brainco" else "inspire"
    task = read_task_string(ds_dir)
    print(f"[export] {ds_dir.name}: profile={profile} task={task!r}")

    ep_meta_files = sorted((ds_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(p) for p in ep_meta_files])
    ep = ep_meta[ep_meta["episode_index"] == args.episode].iloc[0]
    n = int(ep["length"])

    data_path = (ds_dir / "data" / f"chunk-{int(ep['data/chunk_index']):03d}"
                 / f"file-{int(ep['data/file_index']):03d}.parquet")
    df = pd.read_parquet(data_path)
    rows = df[df["episode_index"] == args.episode]
    assert len(rows) == n, f"meta length {n} != rows {len(rows)}"

    if profile == "dex3_upper":
        state, action, qpos29_des, quat, _waist = extract_dex3(rows)
        st28 = np.stack(rows["observation.state"].values).astype(np.float32)
        qpos29_meas = np.concatenate([np.tile(LEG15_DEFAULT, (n, 1)), st28[:, 0:14]], axis=1)
        hand_state14 = st28[:, 14:28]
    else:
        state, action, qpos29_des, quat, _waist = extract_wbt(rows, family)
        q_cur = np.stack(rows["observation.state.robot_q_current"].values).astype(np.float32)
        qpos29_meas = q_cur[:, 7:36]
        hand_state14 = hand12_to_dex3_14(
            np.stack(rows["observation.state.hand_state"].values).astype(np.float32), family)
    hand_cmd14 = action[:, :14]

    model_path = ENCODER_MODEL_V1_1 if args.encoder_version == "v1_1" else ENCODER_MODEL_V1
    print(f"[export] encoding {n} frames with sonic {args.encoder_version} ...")
    encoder = EncoderClient(model_path, mode=0, version=args.encoder_version)
    tokens = encode_tokens_30hz(qpos29_des, quat, encoder)  # (n, 64)

    quat_n = quat / np.linalg.norm(quat, axis=1, keepdims=True).clip(1e-8)
    frames = []
    for i in range(n):
        frames.append({
            "timestamp": i / FPS,
            "states": {
                "qpos": qpos29_meas[i].astype(float).round(6).tolist(),
                "quat": quat_n[i].astype(float).round(6).tolist(),
                "hand_joints": hand_state14[i].astype(float).round(6).tolist(),
            },
            "actions": {
                "qpos": qpos29_des[i].astype(float).round(6).tolist(),
                "hand_joints": hand_cmd14[i].astype(float).round(6).tolist(),
                "token": tokens[i].astype(float).round(6).tolist(),
            },
        })

    out_dir = Path(args.out_dir) / f"{ds_dir.name}__episode_{args.episode:03d}_tok{args.encoder_version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data.json", "w") as f:
        json.dump(frames, f)
    print(f"[export] wrote {n} frames → {out_dir / 'data.json'}")
    print(f"[export] replay check:  python replay_token.py --episode-dir {out_dir}")


if __name__ == "__main__":
    main()
