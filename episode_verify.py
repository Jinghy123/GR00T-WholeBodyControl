"""Neck sanity checks for a recorded episode.

BUG1  actions.neck stuck at [0,0] away from the start -- neck command stream dropped
      (NeckActionReader returned None). A [0,0] run at frame 0 is normal startup.
BUG2  states.neck frozen at one value -- neck state stream dead.
BUG3  states.neck beyond +-NECK_LIMIT -- bad read (seen: [-2.91, -5.02]).

    python episode_verify.py <episode_dir> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NECK_LIMIT = 1.2      # teleop overshoot reaches 1.06; real faults start at 1.44
FROZEN_RUN = 150      # identical consecutive frames = stuck (~5 s at 30 Hz)
ZERO_RUN = 10         # [0,0] action frames away from the start
STARTUP = 120         # a [0,0] run beginning inside this many frames is startup
MIN_FRAMES = 30


class VerifyReport:
    def __init__(self, episode: str, n_frames: int):
        self.episode = episode
        self.n_frames = n_frames
        self.problems: list[tuple[str, str]] = []   # (bug_id, message)

    @property
    def ok(self) -> bool:
        return not self.problems

    def lines(self) -> list[str]:
        if self.ok:
            return [f"[verify] {self.episode}: OK ({self.n_frames} frames)"]
        out = [f"[verify] {self.episode}: {len(self.problems)} PROBLEM(S) "
               f"in {self.n_frames} frames"]
        out += [f"    !! {bug}  {msg}" for bug, msg in self.problems]
        return out


def _neck(frames: list[dict], group: str) -> list[list[float]] | None:
    out = []
    for fr in frames:
        src = fr.get(group)
        if not isinstance(src, dict) or not isinstance(src.get("neck"), (list, tuple)):
            return None
        out.append([float(x) for x in src["neck"]])
    return out or None


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """(start, end_exclusive) runs of True."""
    out, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _longest_constant(col: list[list[float]]) -> tuple[int, int]:
    best_len = best_start = 0
    run_len = run_start = 0
    for i in range(1, len(col)):
        if col[i] == col[i - 1]:
            if run_len == 0:
                run_start = i - 1
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0
    return (best_len + 1 if best_len else 0), best_start


def verify_frames(frames: list[dict], episode: str = "episode",
                  neck_limit: float = NECK_LIMIT, frozen_run: int = FROZEN_RUN,
                  zero_run: int = ZERO_RUN) -> VerifyReport:
    rep = VerifyReport(episode, len(frames))
    if len(frames) < MIN_FRAMES:
        rep.problems.append(("BUG?", f"only {len(frames)} frames -- too short"))
        return rep

    act = _neck(frames, "actions")
    if act is not None:
        for s, e in _runs([r == [0.0, 0.0] for r in act]):
            if s < STARTUP or e - s < zero_run:
                continue
            rep.problems.append((
                "BUG1", f"actions.neck stuck at [0,0] for {e - s} frames "
                        f"({s}-{e - 1}) -- neck command stream dropped"))

    st = _neck(frames, "states")
    if st is not None:
        run_len, run_start = _longest_constant(st)
        if run_len >= frozen_run:
            whole = "the whole episode" if run_len == len(st) else \
                    f"{run_len} frames ({run_start}-{run_start + run_len - 1})"
            rep.problems.append((
                "BUG2", f"states.neck frozen at {st[run_start]} for {whole} "
                        f"-- neck state stream dead"))

        bad = [i for i, r in enumerate(st) if any(abs(x) > neck_limit for x in r)]
        if bad:
            runs = _runs([i in set(bad) for i in range(len(st))])
            where = ", ".join(f"{s}" if e - s == 1 else f"{s}-{e - 1}" for s, e in runs[:5])
            lo = min(min(r) for r in st)
            hi = max(max(r) for r in st)
            rep.problems.append((
                "BUG3", f"states.neck exceeds +-{neck_limit} on {len(bad)} frame(s) "
                        f"at [{where}]: range [{lo:.3f}, {hi:.3f}] -- bad read"))
    return rep


def verify_episode_dir(ep_dir: str | Path, **kw) -> VerifyReport:
    ep_dir = Path(ep_dir)
    try:
        frames = json.loads((ep_dir / "data.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        rep = VerifyReport(ep_dir.name, 0)
        rep.problems.append(("BUG?", f"data.json unreadable: {e}"))
        return rep
    if not isinstance(frames, list):
        rep = VerifyReport(ep_dir.name, 0)
        rep.problems.append(("BUG?", "data.json is not a list of frames"))
        return rep
    kw.setdefault("episode", ep_dir.name)
    return verify_frames(frames, **kw)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: episode_verify.py <episode_dir> [...]")
    rc = 0
    for d in sys.argv[1:]:
        r = verify_episode_dir(d)
        print("\n".join(r.lines()))
        rc |= 0 if r.ok else 1
    sys.exit(rc)
