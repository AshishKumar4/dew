"""The measurement lane's timing harness (tools/timing.py), run whole on the CPU with the wall clock: a window
builds every arm (one process per tree and env, each revision in a checkout of its own), times them in rounds
whose arm order reverses each round, beside the reference's A/A arm, writes a row a call and its verdict, and
refuses a revision no node could fetch before it starts."""

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
    repo, origin = tmp_path / "repo", tmp_path / "origin.git"
    (repo / "src").mkdir(parents=True)
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(repo, "init", "-q")
    git(repo, "remote", "add", "origin", str(origin))
    (repo / "src/cells.py").write_text(BUILDER.format(sign=-1.0))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "earlier")
    earlier = git(repo, "rev-parse", "HEAD").strip()
    (repo / "src/cells.py").write_text(BUILDER.format(sign=1.0))
    git(repo, "commit", "-qam", "head")
    head = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "push", "-q", "origin", "HEAD:refs/heads/work")
    git(repo, "fetch", "-q", "origin")
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
    # Its verdict: each arm's ratios to the reference against the A/A's band, and whether its output is the
    # reference's; the run order and each round's GPU beside them.
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    (case,) = summary["cases"]
    assert case["measure"] == "wall_ms" and case["reference"] == "sin" and len(case["cells"]) == 2
    for cell in case["cells"]:
        arms = {arm["arm"]: arm for arm in cell["arms"]}
        assert len(arms) == 4
        assert arms["sin"]["ratio_range"] == [1.0, 1.0] and arms["sin"]["differs"] is False
        assert arms["sin~aa"]["digest_is_reference"] and not arms["cos"]["digest_is_reference"]
        assert len(cell["noise_band"]) == 2
    assert len(summary["run_order"]) == 2 * 2 * 4
    assert "| cell | arm |" in (tmp_path / "out/summary.md").read_text()
    # A revision held here and pushed nowhere is refused before any arm is built.
    (repo / "src/cells.py").write_text(BUILDER.format(sign=2.0))
    git(repo, "commit", "-qam", "unpushed")
    unpushed = git(repo, "rev-parse", "HEAD").strip()
    manifest["cases"][0]["arms"][2]["tree"] = unpushed
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    refused = subprocess.run([sys.executable, str(TIMING), "run", str(tmp_path / "manifest.json"), "--out",
                              str(tmp_path / "refused")], cwd=repo, capture_output=True, text=True)
    assert refused.returncode != 0 and "on no remote branch: push it" in refused.stderr
    assert not (tmp_path / "refused").exists()


def test_the_dense_cells_build_every_product_for_each_implementation_on_the_cpu():
    """The dense builder's cells compile on the CPU through XLA's dot (a GPU's own launches are its window's), and
    every product's draw meets its call."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("timing_cases", TIMING.with_name("timing_cases.py"))
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    cases.TOKENS = 16
    cases.DENSE = {"tiny": (32, 48)}
    shapes = {"forward": (16, 48), "input_gradient": (16, 32), "kernel_gradient": (32, 48)}
    for product, shape in shapes.items():
        call, draw = cases.dense({"shape": "tiny", "product": product}, "xla")
        assert call(*draw(0)).shape == shape and draw(0) is not None

