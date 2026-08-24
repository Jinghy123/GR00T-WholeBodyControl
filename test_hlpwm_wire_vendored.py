"""Frozen golden vectors for the VENDORED wire contract (robot-client side).

Mirror of psi/tests/test_wire_contracts.py, asserting the SAME digests in
``wire_contracts_golden.json`` against the vendored psix_wire_contracts.py.
Run with plain pytest on the robot machine (needs numpy only). If this fails
while psi's twin passes (or vice versa), the two copies have drifted — never
fix by editing one side alone; see psix_wire_contracts.py's editing rules.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from psix_wire_contracts import (
    CONDITION_HASH_VERSION,
    CONDITION_SCHEMA_VERSION,
    condition_hash,
)

_GOLDEN = json.loads((Path(__file__).parent / "wire_contracts_golden.json").read_text())


def _goal_for(shape: list[int], rng: np.random.Generator) -> np.ndarray:
    if shape[0] == 0:
        return np.zeros(shape, np.uint8)
    return rng.integers(0, 256, size=tuple(shape), dtype=np.uint8)


def test_versions_pinned():
    assert _GOLDEN["hash_version"] == CONDITION_HASH_VERSION
    assert _GOLDEN["schema_version"] == CONDITION_SCHEMA_VERSION


def test_condition_hash_golden_vectors():
    rng = np.random.default_rng(_GOLDEN["generator_seed"])
    for vec in _GOLDEN["vectors"]:
        goal = _goal_for(vec["shape"], rng)
        assert condition_hash(vec["instruction"], goal) == vec["digest"], (
            f"vendored condition_hash drifted for vector {vec['name']!r} — "
            "psix_wire_contracts.py no longer matches psi's wire_contracts.py."
        )


if __name__ == "__main__":
    # Executed directly by scripts/deploy/smoke_deploy_config.sh, which cannot
    # assume pytest exists in the robot-side venv. Without this the gate ran the
    # file as a script, defined the tests, and exited 0 on ANY drift.
    test_versions_pinned()
    test_condition_hash_golden_vectors()
    print("wire-contract goldens OK")
