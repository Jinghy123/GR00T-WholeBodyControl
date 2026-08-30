"""Rebuild frozen states.neck from actions.neck.

BUG2 kills the neck state stream mid-session: states.neck holds one constant
value while actions.neck keeps working and the neck keeps moving. In healthy
episodes states tracks actions closely, so the frozen episodes can be
reconstructed instead of thrown away.

Model, fitted per dim on this dataset's own healthy episodes (calibration
differs between recording sessions, so a same-session fit beats a shared one):

    state(t) = clip( sum_k w_k * action(t - LAG + k) + b , -1, 1 )

LAG=8 frames of servo delay, a 21-tap centered window, and the clip because the
neck saturates at +-1 rad while commands run past it.

    python fix_neck_states.py --root <dataset> --out <dataset_fixed>
    python fix_neck_states.py --root <dataset> --report-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

LAG = 8
HALF_WIDTH = 10
TAPS = list(range(-HALF_WIDTH, HALF_WIDTH + 1))
CLIP = 1.0


def episodes(root: Path) -> list[Path]:
    return sorted((p for p in root.iterdir()
                   if p.is_dir() and p.name.startswith("episode_")
                   and (p / "data.json").is_file()),
                  key=lambda p: int(p.name.split("_")[1]))


def neck(frames, group):
    return np.array([f[group]["neck"] for f in frames], dtype=np.float64)


def is_frozen(states: np.ndarray) -> bool:
    return states.std(axis=0).max() == 0.0


def design(act: np.ndarray, dim: int) -> np.ndarray:
    x = act[:, dim]
    n = len(x)
    cols = []
    for t in TAPS:
        sh = LAG + t
        if sh > 0:
            cols.append(np.concatenate([np.full(sh, x[0]), x[:n - sh]]))
        elif sh < 0:
            cols.append(np.concatenate([x[-sh:], np.full(-sh, x[-1])]))
        else:
            cols.append(x)
    cols.append(np.ones(n))
    return np.stack(cols, axis=1)


def fit(samples, dim):
    X = np.concatenate([design(a, dim) for _, a in samples])
    Y = np.concatenate([s[:, dim] for s, _ in samples])
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def predict(act, W):
    return np.clip(np.stack([design(act, d) @ W[d] for d in range(2)], axis=1),
                   -CLIP, CLIP)


def repair_spikes(states: np.ndarray):
    """Replace out-of-range frames with the nearest in-range value (BUG3)."""
    bad = (np.abs(states) > CLIP).any(axis=1)
    if not bad.any():
        return states, []
    good = np.flatnonzero(~bad)
    if len(good) == 0:
        return states, []
    out = states.copy()
    for i in np.flatnonzero(bad):
        out[i] = states[good[np.argmin(np.abs(good - i))]]
    return out, np.flatnonzero(bad).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="destination (omit with --report-only)")
    ap.add_argument("--report-only", action="store_true",
                    help="fit and report accuracy, write nothing")
    ap.add_argument("--copy", action="store_true",
                    help="copy frames instead of hardlinking (needed across filesystems)")
    args = ap.parse_args()

    root = args.root.resolve()
    eps = episodes(root)
    if not eps:
        sys.exit(f"no episodes under {root}")

    healthy, frozen = [], []
    for ep in eps:
        frames = json.loads((ep / "data.json").read_text())
        s, a = neck(frames, "states"), neck(frames, "actions")
        (frozen if is_frozen(s) else healthy).append((ep, s, a))

    n_spikes = sum(1 for _, s, _ in healthy if (np.abs(s) > CLIP).any())
    print(f"{root.name}: {len(eps)} episodes -- {len(healthy)} healthy, "
          f"{len(frozen)} frozen, {n_spikes} with out-of-range spikes")
    if not frozen and not n_spikes:
        print("nothing to fix")
        return 0

    W = None
    if frozen:
        if len(healthy) < 3:
            sys.exit(f"only {len(healthy)} healthy episodes -- not enough to fit a model")
        order = np.random.RandomState(0).permutation(len(healthy))
        k = max(2, int(len(healthy) * 0.7))
        tr = [(healthy[i][1], healthy[i][2]) for i in order[:k]]
        te = [(healthy[i][1], healthy[i][2]) for i in order[k:]]
        Wtr = [fit(tr, d) for d in range(2)]
        print(f"\nheld-out check ({len(tr)} fit / {len(te)} test episodes):")
        worst = 0.0
        for d in range(2):
            e = np.concatenate([predict(a, Wtr)[:, d] - s[:, d] for s, a in te])
            rng = np.concatenate([s[:, d] for s, _ in te])
            span = rng.max() - rng.min()
            worst = max(worst, float(np.abs(e).max()))
            print(f"  dim{d}: rms={np.sqrt((e ** 2).mean()):.4f} rad "
                  f"({np.sqrt((e ** 2).mean()) / span:.2%} of range), "
                  f"max={np.abs(e).max():.4f} rad")
        print(f"  worst single-frame error: {np.degrees(worst):.2f} deg")
        W = [fit([(s, a) for _, s, a in healthy], d) for d in range(2)]

    if args.report_only:
        print("\n[report-only] nothing written")
        return 0
    if not args.out:
        sys.exit("--out is required unless --report-only")
    out = args.out.resolve()
    if out.exists():
        sys.exit(f"{out} already exists")
    if not args.copy and out.parent.stat().st_dev != root.stat().st_dev:
        sys.exit(f"{out} is on a different filesystem than {root}; pass --copy")

    out.mkdir(parents=True)
    manifest = {"source": str(root), "lag": LAG, "taps": TAPS,
                "coefficients": ({f"dim{d}": W[d].tolist() for d in range(2)}
                                 if W else None),
                "reconstructed": [], "unchanged": [], "spikes_repaired": []}

    for ep in eps:
        frames = json.loads((ep / "data.json").read_text())
        s, a = neck(frames, "states"), neck(frames, "actions")
        dst = out / ep.name
        (dst / "color").mkdir(parents=True)
        for img in sorted((ep / "color").iterdir()):
            (shutil.copy2 if args.copy else os.link)(img, dst / "color" / img.name)

        if is_frozen(s):
            rec = predict(a, W)
            for i, f in enumerate(frames):
                f["states"]["neck"] = [float(rec[i, 0]), float(rec[i, 1])]
            (dst / "neck_reconstructed.json").write_text(json.dumps(
                {"reconstructed": True, "frames": len(frames),
                 "frozen_value": s[0].tolist()}))
            manifest["reconstructed"].append(ep.name)
            tag = f"RECONSTRUCTED  range [{rec.min():+.3f}, {rec.max():+.3f}]"
        else:
            fixed, spikes = repair_spikes(s)
            if spikes:
                for i in spikes:
                    frames[i]["states"]["neck"] = [float(fixed[i, 0]), float(fixed[i, 1])]
                manifest["spikes_repaired"].append(
                    {"episode": ep.name, "frames": spikes,
                     "was": [s[i].tolist() for i in spikes],
                     "now": [fixed[i].tolist() for i in spikes]})
                tag = f"SPIKE REPAIRED  frames {spikes} -> {fixed[spikes[0]].round(6).tolist()}"
            else:
                manifest["unchanged"].append(ep.name)
                tag = "unchanged"
        (dst / "data.json").write_text(json.dumps(frames))
        print(f"  {ep.name:<13} {tag}")

    for extra in root.iterdir():          # carry root-level files (manifests etc.)
        if extra.is_file():
            shutil.copy2(extra, out / extra.name)

    (out / "NECK_RECONSTRUCTION.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out}")
    print(f"  {len(manifest['reconstructed'])} reconstructed, "
          f"{len(manifest['unchanged'])} unchanged")
    print(f"  frames {'copied' if args.copy else 'hardlinked'}; "
          f"NECK_RECONSTRUCTION.json records the model and which episodes were touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
