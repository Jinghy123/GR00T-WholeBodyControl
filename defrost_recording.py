#!/usr/bin/env python3
"""
defrost_recording.py

Strip the frozen (WAITING-phase) ticks from a record_sonic.py recording,
producing a version with only the fresh action-chunk ticks. The kept ticks are
re-timed to a continuous `freq` Hz timeline so replay plays back-to-back with
no pauses between chunks.

Usage:
    python defrost_recording.py --in recordings/run4.pkl --out recordings/run4_nofreeze.pkl
"""

import argparse
import pickle

import numpy as np


def main():
    p = argparse.ArgumentParser(description="Remove frozen ticks from a recording.")
    p.add_argument("--in", dest="in_path", type=str, required=True)
    p.add_argument("--out", dest="out_path", type=str, required=True)
    args = p.parse_args()

    with open(args.in_path, "rb") as f:
        d = pickle.load(f)

    ticks = d["ticks"]
    keep = ~ticks["is_repeat"]
    n_in = len(ticks["t"])
    n_out = int(keep.sum())
    freq = d["freq"]

    new_ticks = {}
    for k, v in ticks.items():
        new_ticks[k] = v[keep]
    # Re-time kept ticks to a continuous timeline so replay has no gaps.
    new_ticks["t"] = (np.arange(n_out, dtype=np.float64) / freq)

    out = dict(d)
    out["ticks"] = new_ticks
    out["commands"] = []  # original command timestamps no longer align

    with open(args.out_path, "wb") as f:
        pickle.dump(out, f)

    print(f"[Defrost] {n_in} -> {n_out} ticks "
          f"({n_in - n_out} frozen dropped), {n_out / freq:.1f}s @ {freq}Hz")
    print(f"[Defrost] Saved to {args.out_path}")


if __name__ == "__main__":
    main()
