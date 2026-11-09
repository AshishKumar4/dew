"""The measurement lane's timing harness (tools/timing.py), run whole on the CPU with the wall clock: a window
builds every arm (one process per tree and env, each revision in a checkout of its own), times them in rounds
whose arm order reverses each round, beside the reference's A/A arm, writes a row a call and its verdict, and
refuses a revision no node could fetch before it starts."""

import json
import os
import subprocess
import sys
from pathlib import Path

from tool_support import StandInSession, git, load

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
    the A/A's the reference's, the earlier commit's its own), then timed a warm-up round and 6 rounds of 2
    calls, the arms in one order each round and the reverse the next, every row naming its tree. The warm-up's
    rows are kept, marked, and decide nothing: the verdict judges the 6 rounds' ratios, each printed, against
    the A/A's by Dew's one judgment (dew.telemetry.benchmark.judged)."""
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
        "name": "cells", "builder": "cells:built", "cells": [{"size": 8}, {"size": 16}], "rounds": 6,
        "repeats": 2, "draws": 3,
        "arms": [{"name": "sin", "tree": None, "implementation": "sin"},
                 {"name": "cos", "tree": None, "implementation": "cos"},
                 {"name": "sin-earlier", "tree": earlier, "implementation": "sin"}]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    subprocess.run([sys.executable, str(TIMING), "run", str(tmp_path / "manifest.json"), "--out",
                    str(tmp_path / "out")], cwd=repo, check=True)
    rows = [json.loads(line) for line in (tmp_path / "out/timing.jsonl").read_text().splitlines()]
    arms = ["sin", "cos", "sin-earlier", "sin~aa"]
    assert len(rows) == 7 * 2 * len(arms) * 2
    assert {row["round"] for row in rows if row["warmup"]} == {-1}
    assert {row["round"] for row in rows if not row["warmup"]} == set(range(6))
    for cell in (8, 16):
        warm = [row["arm"] for row in rows if row["round"] == -1 and row["cell"]["size"] == cell][::2]
        assert warm == arms[::-1]
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
    from dew.telemetry import benchmark

    summary = json.loads((tmp_path / "out/summary.json").read_text())
    (case,) = summary["cases"]
    assert case["measure"] == "wall_ms" and case["reference"] == "sin" and len(case["cells"]) == 2
    for cell in case["cells"]:
        arms = {arm["arm"]: arm for arm in cell["arms"]}
        assert len(arms) == 4
        # The reference is the band's own end, never judged against it.
        assert arms["sin"]["ratio_median"] == 1.0 and arms["sin"]["differs"] is None
        assert all(len(arm["ratios"]) == 6 for arm in arms.values())
        assert arms["sin~aa"]["digest_is_reference"] and not arms["cos"]["digest_is_reference"]
        # The other arms judged, their rounds' interval against the A/A's, the band the A/A's own interval.
        judgment = benchmark.judged(arms["cos"]["ratios"], arms["sin~aa"]["ratios"])
        assert arms["cos"]["ratio_interval"] == list(judgment.interval) and cell["noise_band"] == list(
            judgment.alone)
        assert arms["cos"]["differs"] is judgment.differs and arms["sin~aa"]["differs"] is None
        assert cell["unjudged"] is None
    assert len(summary["run_order"]) == 7 * 2 * 4
    # A revision held here and pushed nowhere is refused before any arm is built.
    (repo / "src/cells.py").write_text(BUILDER.format(sign=2.0))
    git(repo, "commit", "-qam", "unpushed")
    unpushed = git(repo, "rev-parse", "HEAD").strip()
    manifest["cases"][0]["arms"][2]["tree"] = unpushed
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    refused = subprocess.run([sys.executable, str(TIMING), "run", str(tmp_path / "manifest.json"), "--out",
                              str(tmp_path / "refused")], cwd=repo, capture_output=True, text=True)
    assert refused.returncode == 1 and not (tmp_path / "refused").exists()


def test_a_windows_verdict_pairs_each_arms_round_with_the_references_and_leaves_its_warm_up_out(tmp_path):
    """A window's verdict from its rows: each arm's round median over the reference's of the same round, six
    rounds past the warm-up judged against the A/A's by `dew.telemetry.benchmark.judged` at 0.95 (its least
    and greatest round, rank 1): here a fast warm-up round of the arm, which judged would hold 1, decides
    nothing, and the arm's rounds lie apart from 1 and from the A/A's."""
    timing = load("timing")
    # Values a binary float holds exactly, so every ratio is its literal: the reference 8 ms each round, the
    # A/A about it, the arm about 10, its warm-up round (index 0) at 4.
    measured = {"ref": [8.0] * 7, "ref~aa": [8.0, 8.0, 7.5, 8.5, 8.0, 9.0, 7.0],
                "arm": [4.0, 10.0, 10.5, 9.5, 11.0, 12.0, 10.0]}
    rows = [{"case": "c", "cell": {"n": 1}, "arm": arm, "round": index - 1, "warmup": index == 0,
             "repeat": repeat, "order": index, "wall_ms": value, "device_ms": None, "digest": "d",
             "gpu": "cpu"}
            for arm, values in measured.items() for index, value in enumerate(values) for repeat in (0, 1)]
    (tmp_path / "timing.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    manifest = {"clock": "wall", "cases": [{"name": "c", "cells": [{"n": 1}], "reference": "ref",
                                            "arms": [{"name": "ref"}, {"name": "arm"}]}]}
    timing.summarized(manifest, tmp_path)
    (cell,) = json.loads((tmp_path / "summary.json").read_text())["cases"][0]["cells"]
    arms = {found["arm"]: found for found in cell["arms"]}
    assert arms["arm"]["ratios"] == [1.25, 1.3125, 1.1875, 1.375, 1.5, 1.25]
    assert arms["arm"]["ratio_interval"] == [1.1875, 1.5] and cell["noise_band"] == [0.875, 1.125]
    assert arms["arm"]["differs"] is True
    assert arms["ref"]["differs"] is None and cell["unjudged"] is None


def test_the_dense_cells_build_every_product_for_each_implementation_on_the_cpu():
    """The dense builder's cells compile on the CPU through XLA's dot (a GPU's own launches are its window's),
    and every product's draw meets its call."""
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


def test_one_window_a_gpu_kind_is_a_timing_job_running_its_kinds_manifest(tmp_path):
    """`jobs` writes one measurement window a GPU kind, as `node submit` takes it: a timing, alone on its GPU,
    at the ref, running the harness over that kind's manifest, its minutes grown with its timed calls."""
    small = {"cases": [{"name": "a", "builder": "b:c", "cells": [{}], "arms": [{"name": "x"}], "rounds": 1,
                        "repeats": 1}]}
    large = {"cases": [{**small["cases"][0], "cells": [{}] * 50, "arms": [{"name": "x"}, {"name": "y"}],
                        "rounds": 5, "repeats": 5}]}
    for name, manifest in (("a100.json", small), ("h100.json", large)):
        (tmp_path / name).write_text(json.dumps(manifest))
    written = subprocess.run([sys.executable, str(TIMING), "jobs", "--ref", "abc", "--name", "W",
                              f"a100={tmp_path / 'a100.json'}", f"h100={tmp_path / 'h100.json'}"],
                             capture_output=True, text=True, check=True)
    a100, h100 = json.loads(written.stdout)["jobs"]
    assert (a100["name"], a100["gpu"], a100["kind"], a100["ref"]) == ("W-a100", "a100", "timing", "abc")
    assert "$SESSION/extra/a100.json" in a100["run"] and "$SESSION/extra/h100.json" in h100["run"]
    assert h100["minutes"] > a100["minutes"] >= 15


CALLED = '''import os
import time

import jax.numpy as jnp


def built(cell, implementation):
    pick = {"sin": jnp.sin, "cos": jnp.cos}[implementation]

    def call(x):
        with open(os.environ["CALLS"], "a") as calls:
            calls.write(f"{time.time()}\\n")
        return pick(x)

    return call, (lambda index: (jnp.full((cell["size"],), float(index)),))
'''
"""A builder whose every call, a build's and a round's, says when it was made."""


def windowed_run(tmp_path: Path, case_extra: dict, session: StandInSession) -> list[float]:
    """A one-case window run whole in a scratch repository, as a node's session runs a timing job: its
    $SESSION and $OUT are `session`'s, and a stand-in `ncu` on its PATH says when it ran and whether the
    window was open (`$PROFILED`). When each call of the case's builder was made, a build's and a round's."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "src/cells.py").write_text(CALLED)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "cells")
    stand_ins = tmp_path / "bin"
    stand_ins.mkdir()
    (stand_ins / "ncu").write_text(
        '#!/bin/sh\necho "$(date +%s.%N) $([ -e "$OUT/window.granted" ] && echo open || echo shut)" '
        '>> "$PROFILED"\n')
    (stand_ins / "ncu").chmod(0o755)
    manifest = {"clock": "wall", "cases": [{
        "name": "cells", "builder": "cells:built", "cells": [{"size": 8}], "rounds": 2, "repeats": 2,
        "arms": [{"name": "sin", "tree": None, "implementation": "sin"},
                 {"name": "cos", "tree": None, "implementation": "cos"}], **case_extra}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    environment = {**os.environ, **session.environment, "CALLS": str(tmp_path / "calls"),
                   "PROFILED": str(tmp_path / "profiled"),
                   "PATH": f"{stand_ins}{os.pathsep}{os.environ['PATH']}"}
    subprocess.run([sys.executable, str(TIMING), "run", str(tmp_path / "manifest.json"), "--out",
                    str(tmp_path / "out")], cwd=repo, env=environment, check=True)
    return [float(line) for line in (tmp_path / "calls").read_text().split()]


def test_a_window_asks_its_session_for_one_window_around_its_rounds_and_builds_outside_it(tmp_path):
    """A window run as a node's session runs a timing job builds every arm (each build calls its builder once,
    the reference's A/A arm's among them) before it asks for its window, and times every round (a warm-up
    round and two, two calls each, of three arms) inside the one window it asks for, which it has ended by
    the time it writes its verdict."""
    with StandInSession(tmp_path) as session:
        calls = windowed_run(tmp_path, {}, session)
    assert session.asked == [""] and len(session.granted) == len(session.released) == 1
    opened, closed = session.granted[0], session.released[0]
    assert [call < opened for call in calls] == [True] * 3 + [False] * 18
    assert all(opened <= call <= closed for call in calls[3:])
    assert (tmp_path / "out/summary.json").is_file() and not (session.out / "window").exists()


def test_a_cases_profiles_run_inside_one_more_window(tmp_path):
    """A case that asks for profiles has them run (Nsight Compute, a stand-in here, a process an arm) after
    its rounds, all inside a window of their own: the second window the run asks for."""
    with StandInSession(tmp_path) as session:
        calls = windowed_run(tmp_path, {"profile": {"kernel": "sin"}}, session)
    assert session.asked == ["", ""] and len(session.granted) == len(session.released) == 2
    assert all(session.granted[0] <= call <= session.released[0] for call in calls[3:])
    profiled = [line.split() for line in (tmp_path / "profiled").read_text().splitlines()]
    assert [state for _, state in profiled] == ["open", "open"]
    assert all(session.granted[1] <= float(at) <= session.released[1] for at, _ in profiled)

