"""Versioned semantic-profile loader + fail-closed canonicalizer.

The profile JSON (profiles/<name>_v<N>.json, built from the training packs by
psi/docs/tmp/wm_serve_v2/g2_vocab_wm/build_cleanup_fine_profile.py) is the ONLY
vocabulary the active pipeline may execute: raw HLP text is snapped to a
canonical label by exact -> approved-alias -> normalized-exact matching, and a
miss or ambiguity FAILS CLOSED (returns no label; the orchestrator holds).
Raw text is never passed through to the WM or the VLA in active mode.

Pure stdlib on purpose: imported by the robot client, the mock-matrix tests,
and offline eval probes alike.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

PROFILE_SCHEMA_VERSION = "wm-semantic-profile/1"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def profile_hash(profile: Dict[str, Any]) -> str:
    """sha256 over the profile body with its own hash field excluded.
    Byte-compatible with psi.deploy.wire_contracts.profile_hash."""
    body = {k: v for k, v in profile.items() if k != "profile_hash"}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MatchResult:
    kind: str                 # "exact" | "alias" | "normalized" | "miss" | "ambiguous"
    canonical: Optional[str]  # None unless matched
    why: str = ""

    @property
    def ok(self) -> bool:
        return self.canonical is not None


class SemanticProfile:
    def __init__(self, data: Dict[str, Any], path: str = "<inline>"):
        if data.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {data.get('schema_version')!r} "
                             f"!= {PROFILE_SCHEMA_VERSION!r}")
        stored = data.get("profile_hash")
        recomputed = profile_hash(data)
        if stored != recomputed:
            raise ValueError(f"{path}: profile_hash mismatch (stored {stored!r} "
                             f"!= recomputed {recomputed!r}) — profile was edited "
                             f"without rebuilding")
        self.data = data
        self.path = path
        self.name: str = data["profile_name"]
        self.version: int = int(data["profile_version"])
        self.hash: str = stored
        self.task_text: str = data["task_text"]
        self.labels: List[str] = list(data["labels"])
        self.done_sentinel: str = data.get("done_sentinel", "__done__")
        self._norm_cfg: Dict[str, bool] = dict(data.get("normalize", {}))
        self._instruction_template: str = data["vla_instruction_template"]

        # exact index
        self._exact: Set[str] = set(self.labels)
        # normalized index (fail loudly if two canonicals collide)
        self._normed: Dict[str, str] = {}
        for lab in self.labels:
            n = self.normalize(lab)
            if n in self._normed and self._normed[n] != lab:
                raise ValueError(f"{path}: labels {lab!r} and {self._normed[n]!r} "
                                 f"collide under normalize()")
            self._normed[n] = lab
        # approved aliases (alias -> canonical), matched on normalized form
        self._aliases: Dict[str, str] = {}
        for alias, canon in dict(data.get("aliases", {})).items():
            if canon not in self._exact:
                raise ValueError(f"{path}: alias {alias!r} -> unknown label {canon!r}")
            self._aliases[self.normalize(alias)] = canon

        # grammar
        g = data.get("grammar", {})
        self.pairs: List[Dict[str, str]] = list(g.get("pairs", []))
        self.terminal: Optional[str] = g.get("terminal")
        self._place_after_pick: Dict[str, str] = {p["pick"]: p["place"] for p in self.pairs}
        self._pick_labels: Set[str] = {p["pick"] for p in self.pairs}
        self._place_labels: Set[str] = {p["place"] for p in self.pairs}

    @classmethod
    def load(cls, path: str, expected_hash: Optional[str] = None) -> "SemanticProfile":
        with open(path) as f:
            data = json.load(f)
        prof = cls(data, path=path)
        if expected_hash is not None and prof.hash != expected_hash:
            raise ValueError(f"{path}: profile_hash {prof.hash} != pinned {expected_hash}")
        return prof

    # -- normalization + matching ------------------------------------------
    def normalize(self, text: str) -> str:
        t = str(text)
        if self._norm_cfg.get("strip_whitespace", True):
            t = t.strip()
        if self._norm_cfg.get("collapse_internal_spaces", True):
            t = re.sub(r"\s+", " ", t)
        if self._norm_cfg.get("strip_terminal_punctuation", True):
            t = t.rstrip(".!。！ ")
        if self._norm_cfg.get("casefold_for_match", True):
            t = t.casefold()
        return t

    def match(self, raw: Optional[str]) -> MatchResult:
        """Snap raw HLP text to a canonical label. Miss/ambiguous fail closed."""
        if raw is None or not str(raw).strip():
            return MatchResult("miss", None, "empty")
        raw = str(raw)
        if raw in self._exact:
            return MatchResult("exact", raw)
        n = self.normalize(raw)
        if n in self._aliases:
            return MatchResult("alias", self._aliases[n])
        if n in self._normed:
            return MatchResult("normalized", self._normed[n])
        return MatchResult("miss", None, f"no canonical label for {raw!r}")

    def canonical_instruction(self, canonical_subtask: str) -> str:
        """Byte-matches the HLP server's composition: task lowercased, subtask
        verbatim. G2c pins this against the deployed VLA checkpoint."""
        if canonical_subtask not in self._exact:
            raise ValueError(f"not a canonical label: {canonical_subtask!r}")
        return self._instruction_template.format(
            task_lower=self.task_text.strip().lower(),
            canonical_subtask=canonical_subtask,
        )

    # -- grammar -------------------------------------------------------------
    def valid_next(self, prev: Optional[str], done_objects: Set[str]) -> Set[str]:
        """Allowed canonical labels after `prev` given already-completed objects.
        prev=None means episode start. done_objects = objects whose (pick, place)
        pair is fully complete."""
        remaining_picks = {p["pick"] for p in self.pairs if p["object"] not in done_objects}
        if prev is None:
            return set(remaining_picks)
        if prev in self._place_after_pick:          # prev is a pick
            return {self._place_after_pick[prev]}
        if prev in self._place_labels:              # prev is a place
            out = set(remaining_picks)
            if self.terminal and done_objects:
                out.add(self.terminal)
            return out
        if prev == self.terminal:
            return {self.done_sentinel}
        return set()


class TrajectoryTracker:
    """Feeds committed canonical labels through the profile grammar; rejects
    out-of-grammar transitions (the '零越级 committed switch' check)."""

    def __init__(self, profile: SemanticProfile):
        self.profile = profile
        self.committed: List[str] = []
        self.done_objects: Set[str] = set()
        self._obj_of = {}
        for p in profile.pairs:
            self._obj_of[p["pick"]] = p["object"]
            self._obj_of[p["place"]] = p["object"]

    @property
    def current(self) -> Optional[str]:
        return self.committed[-1] if self.committed else None

    def admissible(self, label: str) -> bool:
        return label in self.profile.valid_next(self.current, self.done_objects)

    def commit(self, label: str) -> None:
        if not self.admissible(label):
            raise ValueError(
                f"grammar violation: {label!r} after {self.current!r} "
                f"(done={sorted(self.done_objects)})")
        if label in self.profile._place_labels:
            self.done_objects.add(self._obj_of[label])
        self.committed.append(label)

    def retreat(self) -> Optional[str]:
        """Undo the last commit (mirrors HLP /prev)."""
        if not self.committed:
            return None
        last = self.committed.pop()
        if last in self.profile._place_labels:
            self.done_objects.discard(self._obj_of[last])
        return self.current
