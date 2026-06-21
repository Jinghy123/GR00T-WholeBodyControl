"""Rebuild the color_subgoal/ + prompts.json folder layout (consumed by
psix_sonic_client.py's SubgoalManager) from a converted lerobot dataset.

Source: /home/hongyi/data/g1_neck_0617/g1 (lerobot v3, task per episode,
sub_goal_image_path column pointing at images/observation.images.subgoal/episode_XXXXXX/segment_NN.jpg)

Output: /home/hongyi/data/real_g1_neck_0617/<task_folder>/episode_<i>/color_subgoal/segment_NN.jpg
        /home/hongyi/data/real_g1_neck_0617/prompts.json
"""
import json
import os
import shutil

import pandas as pd

SRC = "/home/hongyi/data/g1_neck_0617/g1"
DST = "/home/hongyi/data/real_g1_neck_0617"


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
        task_dir = os.path.join(DST, task_name)
        os.makedirs(task_dir, exist_ok=True)

        task_description = None
        subtasks = None

        for out_idx, episode_index in enumerate(sorted(episode_indices)):
            parquet_path = os.path.join(SRC, f"data/chunk-000/episode_{episode_index:06d}.parquet")
            df = pd.read_parquet(
                parquet_path,
                columns=["sub_task_index", "subtask_prompt", "task_description", "sub_goal_image_path"],
            )
            stages = df.drop_duplicates(subset=["sub_task_index"]).sort_values("sub_task_index")

            if task_description is None:
                task_description = stages["task_description"].iloc[0]
                subtasks = stages["subtask_prompt"].tolist()

            ep_dir = os.path.join(task_dir, f"episode_{out_idx}", "color_subgoal")
            os.makedirs(ep_dir, exist_ok=True)
            for rel_path in stages["sub_goal_image_path"]:
                src_path = os.path.join(SRC, rel_path)
                shutil.copy2(src_path, os.path.join(ep_dir, os.path.basename(rel_path)))

        prompts[task_name] = {
            "task_description": task_description,
            "subtasks": subtasks,
        }
        print(f"[{task_name}] {len(episode_indices)} episodes, task='{task_description}'")

    with open(os.path.join(DST, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Wrote {os.path.join(DST, 'prompts.json')}")


if __name__ == "__main__":
    main()
