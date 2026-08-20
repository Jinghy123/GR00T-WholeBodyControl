"""npz -> record_sonic.py-style pickle that replay_sonic.py can stream.

replay_sonic.py needs: include_neck, freq, ticks{t, action}. The remaining
record_sonic fields (measured qpos/hands/quat) are filled from the dataset's
observation.state so the pickle stays self-describing.
"""
import pickle, sys
import numpy as np

npz = np.load(sys.argv[1], allow_pickle=True)
out = sys.argv[2]

action = npz["action"].astype(np.float32)
state = npz["state"].astype(np.float32)
t = npz["t"].astype(np.float64)
fps = float(npz["fps"])

data = {
    "include_neck": True,               # action layout is hand(14)+neck(2)+token(64)
    "action_dim": int(action.shape[1]),
    "freq": int(round(fps)),
    "ticks": {
        "t": t,
        "action": action,
        "qpos": state[:, 0:29],
        "left_hand_q": state[:, 29:36],
        "right_hand_q": state[:, 36:43],
        "base_quat": np.tile(np.array([1, 0, 0, 0], np.float32), (len(t), 1)),
        "neck_state": state[:, 43:45],
        "is_repeat": np.zeros(len(t), bool),
    },
    "commands": [],
    "task": str(npz["task"]),
}
with open(out, "wb") as f:
    pickle.dump(data, f, protocol=4)
print(f"wrote {out}: {len(t)} ticks @ {data['freq']} Hz ({t[-1]-t[0]:.1f}s)")
