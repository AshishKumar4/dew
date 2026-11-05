"""Scratch probe (row 3684, G17 review): the worst-row rule's false-red rate through the gate where one value
rules rows, about RULED_SHARE too (small largest k), at 32,000 runs a layout."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gate_null_runs  # noqa: E402

LAYOUTS = [(rows, width, shared, steepest) for rows in (1, 4) for width in (3, 4, 16) for shared in (False, True)
           for steepest in (4, 6, 24)]


@pytest.mark.parametrize("chunk", range(32))
@pytest.mark.parametrize("rows, width, shared, steepest", LAYOUTS)
def test_probe(rows, width, shared, steepest, chunk):
    verdicts, _ = gate_null_runs.row_controls(width, rows, 8, 1000, chunk, amplified=True, shared=shared,
                                              steepest=steepest)
    reds = sum(verdict != "pass" for verdict, _ in verdicts)
    print("\nCALIB " + json.dumps({"rows": rows, "width": width, "shared": shared, "steepest": steepest,
                                   "runs": len(verdicts), "reds": reds}))
