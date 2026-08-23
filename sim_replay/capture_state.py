"""Capture WBC measured joint state (port 5557) to npz for replay-tracking checks."""
import sys, time, zmq, msgpack, numpy as np

dur = float(sys.argv[1]); out = sys.argv[2]
ctx = zmq.Context(); s = ctx.socket(zmq.SUB)
s.connect("tcp://localhost:5557"); s.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
s.setsockopt(zmq.RCVTIMEO, 2000)
ts, qs, quats = [], [], []
t0 = time.perf_counter()
while time.perf_counter() - t0 < dur:
    try:
        raw = s.recv()
    except zmq.Again:
        continue
    d = msgpack.unpackb(raw[len(b"g1_debug"):], raw=False)
    if "body_q_measured" not in d:
        continue
    ts.append(time.perf_counter() - t0)
    qs.append(np.asarray(d["body_q_measured"], dtype=np.float32))
    quats.append(np.asarray(d.get("base_quat_measured", [np.nan]*4), dtype=np.float32))
np.savez(out, t=np.array(ts), q=np.array(qs), quat=np.array(quats))
print(f"captured {len(ts)} frames over {dur}s -> {out}")
