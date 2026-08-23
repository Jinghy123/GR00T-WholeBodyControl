#!/usr/bin/env python3
"""
lerobot_eval_recorder.py

Record one client evaluation as a LeRobot v2.0 dataset laid out exactly like
`g1_sonic_lerobot_0810_merged_val`, so an eval run can be loaded by the same psi
dataloader, replayed, or scored against the training distribution without any
conversion step.

Why the stats are copied rather than computed
---------------------------------------------
`meta/stats.json` (and `meta/stats_psi0.json`, which psi actually normalizes with --
see `data.transform.field.stat_path` in a run_config.json) define the normalization
the policy was TRAINED with. An eval episode is a handful of frames from one task, so
statistics computed from it would be meaningless and, worse, silently renormalize the
data away from what the model expects. Both files are therefore copied verbatim from
the source dataset; only info/episodes/tasks are regenerated for the new episodes.

Layout produced (v2.0, global stats, one chunk):

    <root>/data/chunk-000/episode_%06d.parquet
    <root>/videos/chunk-000/<image_key>/episode_%06d.mp4
    <root>/meta/{info,episodes,tasks}.json(l)      <- generated
    <root>/meta/{stats,stats_psi0,modality,...}.json  <- copied from --src

Recording runs entirely off the caller's control loop: `add_frame()` only puts on a
queue, and a worker thread does the JPEG encode. Video/parquet encoding happens once,
at `close()`, after the episode is over.

Used by psi_rtc_sonic_client.py --record-lerobot; can also be run standalone to
finalize a raw directory left behind by an interrupted run:

    python lerobot_eval_recorder.py --finalize .eval_datasets/<run>/.raw/episode_000000
"""

import json
import os
import shutil
import subprocess
import time
from queue import Queue, Empty
from threading import Thread

import cv2
import numpy as np

# Dataset whose meta/ defines the format and the training normalization.
DEFAULT_SRC_DATASET = ".data/g1_sonic_lerobot_0810_merged_val"

# Copied verbatim from the source dataset when present. stats.json / stats_psi0.json
# MUST be copied (train/inference consistency); the rest describe the embodiment and
# are equally run-independent.
COPIED_META = (
    "stats.json",
    "stats_psi0.json",
    "modality.json",
    "embodiment.json",
    "g1_sonic_mapping.json",
)

CODEBASE_VERSION = "v2.0"
ROBOT_TYPE = "unitree_g1"
DEFAULT_IMAGE_KEY = "observation.images.head"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


# ---------------------------------------------------------------- state/action layout
def dataset_state(body_q, left_hand_q, right_hand_q, neck=None):
    """Assemble observation.state in the DATASET's layout: qpos(29) + hands(14) [+ neck(2)].

    Note this is NOT the order the clients feed the policy -- they send
    [left_hand(7), right_hand(7), arm(14), leg(15)] (+neck). The dataset layout is the
    one declared in meta/modality.json (qpos 0:29, hand_joints 29:43, neck 43:45), and
    it is what meta/stats*.json is computed over, so a recorded eval must use it.
    """
    parts = [np.asarray(body_q, np.float32).reshape(-1),
             np.asarray(left_hand_q, np.float32).reshape(-1),
             np.asarray(right_hand_q, np.float32).reshape(-1)]
    if neck is not None:
        parts.append(np.asarray(neck, np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def dataset_action(hand_joints, token, neck=None):
    """Assemble action in the DATASET's layout: hand(14) + neck(2) + token(64).

    The psi server returns token(64) + hand(14) + neck(2); this is the inverse of the
    repack permutation (`action_keys: ["action[16:80]", "action[:14]", "action[14:16]"]`).
    Pass the QUANTIZED token, i.e. what was actually published -- recorded dataset tokens
    sit exactly on the FSQ grid.
    """
    parts = [np.asarray(hand_joints, np.float32).reshape(-1)]
    if neck is not None:
        parts.append(np.asarray(neck, np.float32).reshape(-1))
    parts.append(np.asarray(token, np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------- recorder
def _require_parquet_writer():
    """Fail before the episode, not after it.

    pyarrow is only needed at the very last step (`_write_parquet`), so a missing
    install used to surface after a full recording -- video and tasks.jsonl already
    written, parquet/info/episodes missing -- and the client swallows shutdown-path
    errors. Check the import up front instead."""
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is required to write LeRobot parquet files but is not installed "
            f"in this interpreter ({e}). Install it with: pip install pyarrow"
        ) from e


class EvalDatasetRecorder:
    """Buffer one evaluation episode, then write it as a LeRobot episode."""

    def __init__(self, root, task, src_dataset=DEFAULT_SRC_DATASET, fps=30,
                 image_key=DEFAULT_IMAGE_KEY, jpeg_quality=95, keep_raw=False):
        self.root = os.path.abspath(root)
        self.task = task
        self.src = src_dataset
        self.fps = int(fps)
        self.image_key = image_key
        self.jpeg_quality = int(jpeg_quality)
        self.keep_raw = keep_raw

        _require_parquet_writer()

        self.episode_index = _next_episode_index(self.root)
        self.raw_dir = os.path.join(self.root, ".raw", f"episode_{self.episode_index:06d}")
        self.frames_dir = os.path.join(self.raw_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        self._queue = Queue(maxsize=600)   # ~20 s of slack at 30 Hz before back-pressure
        self._states, self._actions = [], []
        self._n_written = 0
        self._dropped = 0
        self._stop = False
        self._worker = Thread(target=self._process_queue, daemon=True)
        self._worker.start()
        print(f"[LeRobotRec] episode {self.episode_index} -> {self.root} (src meta: {self.src})")

    # -- called from the control loop; must stay cheap -------------------------------
    def add_frame(self, image_bgr, state, action):
        """Queue one (observation, action) pair. Never blocks: a full queue drops the
        frame and is reported at close, because stalling a 30 Hz control loop to keep a
        recording complete is the wrong trade on a real robot."""
        if self._stop:
            return
        try:
            self._queue.put_nowait((image_bgr, np.asarray(state, np.float32),
                                    np.asarray(action, np.float32)))
        except Exception:
            self._dropped += 1

    def _process_queue(self):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while not self._stop or not self._queue.empty():
            try:
                image, state, action = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                cv2.imwrite(os.path.join(self.frames_dir, f"frame_{self._n_written:06d}.jpg"),
                            image, params)
                self._states.append(state)
                self._actions.append(action)
                self._n_written += 1
            except Exception as e:
                print(f"[LeRobotRec] frame write failed: {e}")
            finally:
                self._queue.task_done()

    # -- end of episode ---------------------------------------------------------------
    def close(self, save=True):
        self._stop = True
        self._worker.join(timeout=30.0)
        if self._dropped:
            print(f"[LeRobotRec] WARNING: dropped {self._dropped} frames (queue full)")
        if not save or self._n_written == 0:
            print(f"[LeRobotRec] nothing saved ({self._n_written} frames buffered)")
            return None

        np.savez(os.path.join(self.raw_dir, "arrays.npz"),
                 state=np.stack(self._states), action=np.stack(self._actions))
        with open(os.path.join(self.raw_dir, "meta.json"), "w") as f:
            json.dump({"root": self.root, "episode_index": self.episode_index,
                       "task": self.task, "fps": self.fps, "image_key": self.image_key,
                       "src_dataset": self.src}, f, indent=2)
        try:
            return finalize(self.raw_dir, keep_raw=self.keep_raw)
        except Exception as e:
            # The raw dir is complete and self-contained; keep it and say how to retry.
            print(f"[LeRobotRec] finalize failed: {e}\n"
                  f"[LeRobotRec] frames+arrays are intact; recover with:\n"
                  f"    python {os.path.basename(__file__)} --finalize {self.raw_dir}")
            raise


# ---------------------------------------------------------------- finalization
def _next_episode_index(root):
    info_path = os.path.join(root, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            return int(json.load(f)["total_episodes"])
    return 0


def _encode_video(frames_dir, out_path, fps):
    """JPEG sequence -> h264/yuv420p mp4, matching the source dataset's video_info."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(frames_dir, "frame_%06d.jpg"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    try:
        subprocess.run(cmd, check=True)
        return "h264"
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"[LeRobotRec] ffmpeg failed ({e}); falling back to OpenCV mp4v. The video "
              f"is still decodable but info.json will report the real codec.")
        first = cv2.imread(os.path.join(frames_dir, "frame_000000.jpg"))
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for name in sorted(os.listdir(frames_dir)):
            vw.write(cv2.imread(os.path.join(frames_dir, name)))
        vw.release()
        return "mp4v"


def _write_parquet(path, state, action, task, task_index, episode_index, index0, fps):
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(state)
    schema = pa.schema([
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("annotation.task", pa.large_string()),
        ("timestamp", pa.float32()),
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])
    table = pa.table({
        "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
        "action": pa.array(list(action), type=pa.list_(pa.float32())),
        "annotation.task": pa.array([task] * n, type=pa.large_string()),
        # float32 k/fps, exactly how lerobot lays out a fixed-rate episode
        "timestamp": pa.array((np.arange(n, dtype=np.float32) / fps), type=pa.float32()),
        "episode_index": pa.array(np.full(n, episode_index, np.int64)),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "index": pa.array(np.arange(index0, index0 + n, dtype=np.int64)),
        "task_index": pa.array(np.full(n, task_index, np.int64)),
    }, schema=schema)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)


def _task_index(root, task):
    """Resolve `task` against meta/tasks.jsonl, appending it when new."""
    path = os.path.join(root, "meta", "tasks.jsonl")
    tasks = []
    if os.path.exists(path):
        with open(path) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
    for entry in tasks:
        if entry["task"] == task:
            return entry["task_index"]
    idx = len(tasks)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
    return idx


def _append_episode(root, episode_index, task, length, index0):
    path = os.path.join(root, "meta", "episodes.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps({"episode_index": episode_index, "tasks": [task],
                            "length": length, "dataset_from_index": index0,
                            "dataset_to_index": index0 + length - 1}) + "\n")


def _write_info(root, src, image_key, video_shape, codec, state_dim, action_dim,
                fps, total_episodes, total_frames, total_tasks):
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": ROBOT_TYPE,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "fps": float(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "chunks_size": CHUNKS_SIZE,
        "features": {
            image_key: {
                "dtype": "video",
                "shape": list(video_shape),
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": float(fps), "video.codec": codec,
                               "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                               "has_audio": False},
            },
            "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
            "action": {"dtype": "float32", "shape": [action_dim], "names": None},
            "annotation.task": {"dtype": "string", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        },
        "total_videos": total_episodes,
        "total_chunks": 1,
        "recorded_by": os.path.basename(__file__),
        "stats_copied_from": os.path.abspath(src) if src else None,
    }
    with open(os.path.join(root, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    return info


def _copy_meta(root, src):
    """Copy the run-independent meta files, stats above all (see module docstring)."""
    if not src:
        return []
    copied = []
    for name in COPIED_META:
        s = os.path.join(src, "meta", name)
        d = os.path.join(root, "meta", name)
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copyfile(s, d)
            copied.append(name)
    missing = [n for n in ("stats.json", "stats_psi0.json")
               if not os.path.exists(os.path.join(root, "meta", n))]
    if missing:
        print(f"[LeRobotRec] WARNING: {missing} not found under {src}/meta — the dataset "
              f"has no training-consistent normalization stats")
    return copied


def finalize(raw_dir, keep_raw=False):
    """Turn a raw episode directory into a LeRobot episode under its dataset root."""
    with open(os.path.join(raw_dir, "meta.json")) as f:
        meta = json.load(f)
    arrays = np.load(os.path.join(raw_dir, "arrays.npz"))
    state, action = arrays["state"], arrays["action"]
    root, ep, fps = meta["root"], meta["episode_index"], meta["fps"]
    frames_dir = os.path.join(raw_dir, "frames")

    n_frames = len(sorted(os.listdir(frames_dir)))
    n = min(n_frames, len(state))
    if n != n_frames or n != len(state):
        print(f"[LeRobotRec] truncating to {n} (frames={n_frames}, rows={len(state)})")
    state, action = state[:n], action[:n]

    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    chunk = ep // CHUNKS_SIZE

    video_path = os.path.join(root, VIDEO_PATH.format(episode_chunk=chunk,
                                                      video_key=meta["image_key"],
                                                      episode_index=ep))
    codec = _encode_video(frames_dir, video_path, fps)
    first = cv2.imread(os.path.join(frames_dir, "frame_000000.jpg"))
    video_shape = (first.shape[0], first.shape[1], 3)

    # Totals from the existing info.json, so several eval runs can share one root.
    info_path = os.path.join(root, "meta", "info.json")
    prev = json.load(open(info_path)) if os.path.exists(info_path) else None
    index0 = int(prev["total_frames"]) if prev else 0

    task_index = _task_index(root, meta["task"])
    _write_parquet(os.path.join(root, DATA_PATH.format(episode_chunk=chunk, episode_index=ep)),
                   state, action, meta["task"], task_index, ep, index0, fps)
    _append_episode(root, ep, meta["task"], n, index0)

    with open(os.path.join(root, "meta", "tasks.jsonl")) as f:
        total_tasks = sum(1 for line in f if line.strip())
    _write_info(root, meta.get("src_dataset"), meta["image_key"], video_shape, codec,
                int(state.shape[1]), int(action.shape[1]), fps,
                total_episodes=ep + 1, total_frames=index0 + n, total_tasks=total_tasks)
    _copy_meta(root, meta.get("src_dataset"))

    if not keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
    print(f"[LeRobotRec] episode {ep}: {n} frames @ {fps} Hz ({n / fps:.1f}s) -> {root}")
    return root


def main():
    import argparse
    p = argparse.ArgumentParser(description="Finalize a raw eval episode into a LeRobot dataset.")
    p.add_argument("--finalize", required=True, help="Raw episode dir (…/.raw/episode_%%06d)")
    p.add_argument("--keep-raw", action="store_true", help="Keep the JPEG frames after encoding")
    args = p.parse_args()
    finalize(args.finalize, keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
