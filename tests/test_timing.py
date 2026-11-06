"""The measurement lane's timing harness (tools/timing.py), run whole on the CPU with the wall clock: a window
builds every arm (one process per tree and env, each revision in a checkout of its own), times them in rounds
whose arm order reverses each round, beside the reference's A/A arm, and writes a row a call."""

import json
import subprocess
import sys
from pathlib import Path

from tool_support import git

TIMING = Path(__file__).resolve().parents[1] / "tools/timing.py"

BUILDER = '''import jax.numpy as jnp

SIGN = {sign}


def built(cell, implementation):
    size = cell["size"]
    pick = {{"sin": jnp.sin, "cos": jnp.cos}}[implementation]
    return (lambda x: pick(x) * SIGN), (lambda index: (jnp.full((size,), float(index)),))
'''


def test_a_window_times_every_arm_in_reversed_rounds_beside_its_a_a_each_revision_in_its_own_tree(
        tmp_path):
    """Two cells, three arms (two implementations in the window's tree, and the first again at an earlier
    commit whose builder differs), and the reference's A/A: each arm built once (its output's digest kept,
    the A/A's the reference's, the earlier commit's its own), then timed 2 rounds of 2 calls, the arms in one
    order the first round and the reverse the second, every row naming its tree."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "src/cells.py").write_text(BUILDER.format(sign=-1.0))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "earlier")
    earlier = git(repo, "rev-parse", "HEAD").strip()
    (repo / "src/cells.py").write_text(BUILDER.format(sign=1.0))
    git(repo, "commit", "-qam", "head")
    head = git(repo, "rev-parse", "HEAD").strip()
    manifest = {"clock": "wall", "cases": [{
        "name": "cells", "builder": "cells:built", "cells": [{"size": 8}, {"size": 16}], "rounds": 2,
        "repeats": 2, "draws": 3,
        "arms": [{"name": "sin", "tree": None, "implementation": "sin"},
                 {"name": "cos", "tree": None, "implementation": "cos"},
                 {"name": "sin-earlier", "tree": earlier, "implementation": "sin"}]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    subprocess.run([sys.executable, str(TIMING), "run", str(tmp_path / "manifest.json"), "--out",
                    str(tmp_path / "out")], cwd=repo, check=True)
    rows = [json.loads(line) for line in (tmp_path / "out/timing.jsonl").read_text().splitlines()]
    arms = ["sin", "cos", "sin-earlier", "sin~aa"]
    assert len(rows) == 2 * 2 * len(arms) * 2
    for cell in (8, 16):
        first = [row["arm"] for row in rows if row["round"] == 0 and row["cell"]["size"] == cell][::2]
        second = [row["arm"] for row in rows if row["round"] == 1 and row["cell"]["size"] == cell][::2]
        assert first == arms and second == arms[::-1]
    digests = {row["arm"]: row["digest"] for row in rows if row["cell"]["size"] == 8}
    assert digests["sin~aa"] == digests["sin"]
    assert len({digests["sin"], digests["cos"], digests["sin-earlier"]}) == 3
    assert {row["tree"] for row in rows if row["arm"] == "sin-earlier"} == {earlier}
    assert {row["tree"] for row in rows if row["arm"] != "sin-earlier"} == {head}
    assert all(row["wall_ms"] > 0 and row["device_ms"] is None and row["cache"] == "cold" for row in rows)
    assert [row["order"] for row in rows][::2] == sorted({row["order"] for row in rows})
