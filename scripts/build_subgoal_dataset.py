"""Rebuild the color_subgoal/ + prompts.json folder layout (consumed by
psix_sonic_client.py's SubgoalManager) from a converted lerobot dataset.

Source: lerobot v3 dataset (task per episode, sub_goal_image_path column pointing
        at images/observation.images.subgoal/episode_XXXXXX/segment_NN.jpg)
Output: <DST>/<episode_key>/color_subgoal/segment_NN.jpg
        <DST>/prompts.json   ->  { <episode_key>: {task_description, subtasks[]} }

One key/folder per episode: each episode has its own subtask ordering and subgoal
images, so prompts and images cannot be shared. Folder name == prompts.json key,
so the client runs with `--task-key <k> --episode-dir <DST>/<k>`.
"""
import json
import os
import shutil

import pandas as pd

SRC = "/home/hongyi/data/cleanup_table_2026-07-01/g1"
DST = "/home/hongyi/data/real_clean_up_table"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    tasks = {t["task_index"]: t for t in load_jsonl(os.path.join(SRC, "meta/tasks.jsonl"))}
    episodes = load_jsonl(os.path.join(SRC, "meta/episodes.jsonl"))

    episodes_by_task = {}
    for ep in episodes:
        task_index = ep["tasks"][0]
        episodes_by_task.setdefault(task_index, []).append(ep["episode_index"])

    prompts = {}

    for task_index, episode_indices in sorted(episodes_by_task.items()):
        task_name = tasks[task_index]["task"].replace(".", "_")

        episodes_stages = []
        for episode_index in sorted(episode_indices):
            parquet_path = os.path.join(SRC, f"data/chunk-000/episode_{episode_index:06d}.parquet")
            df = pd.read_parquet(
                parquet_path,
                columns=["sub_task_index", "subtask_prompt", "task_description", "sub_goal_image_path"],
            )
            episodes_stages.append(df.drop_duplicates(subset=["sub_task_index"]).sort_values("sub_task_index"))

        # Shared subtasks across all episodes -> one key + episode_<i> subfolders.
        # Otherwise -> one key + folder per episode.
        shared = len({tuple(s["subtask_prompt"]) for s in episodes_stages}) == 1

        for out_idx, stages in enumerate(episodes_stages):
            if shared:
                key = task_name
                ep_dir = os.path.join(DST, task_name, f"episode_{out_idx}", "color_subgoal")
            else:
                key = f"{task_name}_episode_{out_idx}"
                ep_dir = os.path.join(DST, key, "color_subgoal")

            os.makedirs(ep_dir, exist_ok=True)
            for rel_path in stages["sub_goal_image_path"]:
                src_path = os.path.join(SRC, rel_path)
                shutil.copy2(src_path, os.path.join(ep_dir, os.path.basename(rel_path)))

            prompts[key] = {
                "task_description": stages["task_description"].iloc[0],
                "subtasks": stages["subtask_prompt"].tolist(),
            }

        print(f"[{task_name}] {len(episodes_stages)} episodes, "
              f"{'shared' if shared else 'per-episode'} subtasks")

    with open(os.path.join(DST, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Wrote {os.path.join(DST, 'prompts.json')}")


if __name__ == "__main__":
    main()
