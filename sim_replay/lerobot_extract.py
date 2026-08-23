"""Pull one LeRobot episode's action stream out of its parquet -> npz.

Run with a python that has pyarrow (the repo venvs don't). The action layout in
meta/modality.json is hand_joints(14) + neck(2) + token(64) = 80, i.e. exactly
record_sonic.py's include_neck layout, so no re-ordering is needed.
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq

try:
    from tqdm import tqdm

    def _progress(total, desc):
        return tqdm(total=total, desc=desc, unit="frame", dynamic_ncols=True)

except ImportError:  # no tqdm in this interpreter - print a simple percentage bar

    class _progress:
        def __init__(self, total, desc):
            self.total, self.desc, self.n = max(total, 1), desc, 0

        def __enter__(self):
            self.update(0)
            return self

        def update(self, k):
            self.n += k
            filled = int(30 * self.n / self.total)
            print(f"\r{self.desc} |{'#' * filled}{'.' * (30 - filled)}| "
                  f"{self.n}/{self.total}", end="", file=sys.stderr, flush=True)

        def __exit__(self, *exc):
            print(file=sys.stderr)

root = sys.argv[1]
ep = int(sys.argv[2])
out = sys.argv[3]

info = json.load(open(f"{root}/meta/info.json"))
fps = float(info["fps"])
path = f"{root}/" + info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
# Read in batches so the (slow) python-list conversion can drive a progress bar.
pf = pq.ParquetFile(path)
total = pf.metadata.num_rows
cols = ["action", "observation.state", "timestamp"]
chunks = {c: [] for c in cols}
with _progress(total, f"extract ep{ep}") as bar:
    for batch in pf.iter_batches(batch_size=256, columns=cols):
        for c in cols:
            chunks[c].append(batch.column(c).to_pylist())
        bar.update(batch.num_rows)

action = np.array([r for b in chunks["action"] for r in b], dtype=np.float32)
state = np.array([r for b in chunks["observation.state"] for r in b], dtype=np.float32)
ts = np.array([r for b in chunks["timestamp"] for r in b], dtype=np.float64)

task = ""
for line in open(f"{root}/meta/episodes.jsonl"):
    d = json.loads(line)
    if d["episode_index"] == ep:
        task = d["tasks"][0]
        break

np.savez(out, action=action, state=state, t=ts, fps=fps, task=task)
print(f"episode {ep}: {len(action)} frames @ {fps} Hz ({len(action)/fps:.1f}s)")
print(f"action {action.shape}, token range [{action[:,16:].min():.3f}, {action[:,16:].max():.3f}]")
print(f"task: {task}")
