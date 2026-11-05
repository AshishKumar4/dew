"""Scratch probe: check 1's worst-row rule on equally exact pairs, per layout and tail class (row 3684)."""
import json

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gate_null_runs  # noqa: E402

LAYOUTS = [(rows, width, seeds, shared) for rows in (1, 4, 16, 256) for width in (1, 2, 16) for seeds in (8, 64)
           for shared in (False, True)]


@pytest.mark.parametrize("chunk", range(4))
@pytest.mark.parametrize("rows, width, seeds, shared", LAYOUTS)
def test_probe(rows, width, seeds, shared, chunk):
    verdicts, statistics = gate_null_runs.row_controls(width, rows, seeds, 500, chunk, amplified=True,
                                                       shared=shared)
    reds = sum(verdict != "pass" for verdict, _ in verdicts)
    print("\nNULLS " + json.dumps({"rows": rows, "width": width, "seeds": seeds, "shared": shared, "chunk": chunk,
                                   "runs": len(verdicts), "reds": reds}))
