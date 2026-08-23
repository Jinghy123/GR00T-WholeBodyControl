"""Pull one LeRobot episode's action stream out of its parquet -> npz.

    usage: lerobot_extract.py <dataset_root> <episode_index> <out.npz>
                              [--max-frames N]

Run with a python that has pyarrow (the repo venvs don't). The action layout in
meta/modality.json is hand_joints(14) + neck(2) + token(64) = 80, i.e. exactly
record_sonic.py's include_neck layout, so no re-ordering is needed.

--max-frames keeps only the first N frames of the episode. The parquet is read
lazily in batches, so this also STOPS reading once N rows are in hand rather
than decoding the whole episode and slicing afterwards.
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

argv = sys.argv[1:]
max_frames = None
if "--max-frames" in argv:
    i = argv.index("--max-frames")
    try:
        max_frames = int(argv[i + 1])
    except (IndexError, ValueError):
        sys.exit("lerobot_extract.py: --max-frames needs an integer argument")
    if max_frames <= 0:
        sys.exit("lerobot_extract.py: --max-frames must be > 0")
    del argv[i:i + 2]
if len(argv) != 3:
    sys.exit("usage: lerobot_extract.py <dataset_root> <episode_index> "
             "<out.npz> [--max-frames N]")

root = argv[0]
ep = int(argv[1])
out = argv[2]

info = json.load(open(f"{root}/meta/info.json"))
fps = float(info["fps"])
path = f"{root}/" + info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
# Read in batches so the (slow) python-list conversion can drive a progress bar.
pf = pq.ParquetFile(path)
num_rows = pf.metadata.num_rows
total = min(num_rows, max_frames) if max_frames else num_rows
cols = ["action", "observation.state", "timestamp"]
chunks = {c: [] for c in cols}
read = 0
with _progress(total, f"extract ep{ep}") as bar:
    for batch in pf.iter_batches(batch_size=256, columns=cols):
        take = batch.num_rows if max_frames is None else min(batch.num_rows, total - read)
        if take < batch.num_rows:
            batch = batch.slice(0, take)
        for c in cols:
            chunks[c].append(batch.column(c).to_pylist())
        read += take
        bar.update(take)
        if max_frames is not None and read >= total:
            break

action = np.array([r for b in chunks["action"] for r in b], dtype=np.float32)
state = np.array([r for b in chunks["observation.state"] for r in b], dtype=np.float32)
ts = np.array([r for b in chunks["timestamp"] for r in b], dtype=np.float64)
if max_frames is not None:
    print(f"--max-frames {max_frames}: keeping {len(action)} of {num_rows} frames")

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
