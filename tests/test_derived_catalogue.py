"""The derivations a built package ships (tools/derived_catalogue.py, `dew.algebra.stored.shipped`): a step's
lowering for a generation derives what that step compiled ahead for a GPU of it reads."""

import json
import os
import subprocess
import sys

from lane_environment import outside_suite
from tool_support import REPO_ROOT

STEP = "step finetune-1.7b 1x2048"
"""The perf gate's step the catalogue lowers here: its cheapest to compile ahead with attention to derive."""

COMPILED_AHEAD = """
import sys
sys.path[:0] = ["tools", "tests"]
import perf_gate
from benchmark_cases import Case
from benchmark_models import ahead_of
ahead_of(Case(**perf_gate.STEP_CASES[sys.argv[1]]), "sm90", 96 * 2**30)
"""
"""The step compiled ahead for one compile-only sm90 GPU (`benchmark_models.ahead_of`, a run's prepare)."""


def test_a_step_compiled_ahead_asks_for_no_derivation_its_catalogue_lacks(tmp_path):
    """The catalogue's lowering of a step for a generation (`TrainerAdvanced(lower_for=...)`, never compiled)
    derives every derivation that step compiled ahead for a GPU of the generation asks for (`compile_ahead`):
    compiled ahead in a fresh process with an empty cache and no read-only store, each entry it writes is
    one of the catalogue's (which holds more: a lowering's base rung derives what the plan the compile
    keeps does not ask for)."""
    out, cache = tmp_path / "derived", tmp_path / "cache"
    env = outside_suite(os.environ)
    built = subprocess.run([sys.executable, "tools/derived_catalogue.py", "--steps", STEP, "--generations",
                            "sm90", "--out", str(out), "--jobs", "1"], cwd=REPO_ROOT, env=env,
                           capture_output=True, text=True, check=True)
    catalogue = {entry.name for entry in out.glob("*.json")}
    assert json.loads(built.stdout.strip().splitlines()[-1])["entries"] == len(catalogue) > 0
    subprocess.run([sys.executable, "-c", COMPILED_AHEAD, STEP], cwd=REPO_ROOT, capture_output=True,
                   text=True, check=True,
                   env={**env, "XDG_CACHE_HOME": str(cache), "DEW_DERIVATIONS": str(tmp_path / "none")})
    asked = {entry.name for entry in (cache / "dew" / "algebra" / "derived").glob("*.json")}
    assert asked and asked <= catalogue


def test_a_staged_union_is_weighed_by_the_one_rule_a_generations_entries_are(tmp_path):
    """What a package may carry (tools/derived_weight.py, the rule the catalogue weighs a generation's entries
    by): run on a directory as a release's package step runs it on the union it staged (tools/ci/package.sh),
    entries each under the limit refused where together they pass it, and within it, weighed."""
    staged = tmp_path / "derived"
    staged.mkdir()
    for name in ("a", "b"):
        (staged / f"{name}.json").write_bytes(b"x" * (9 * 2**20))
    (staged / "notes.txt").write_bytes(b"x" * (9 * 2**20))  # Not an entry: never weighed.
    weigh = [sys.executable, str(REPO_ROOT / "tools/derived_weight.py"), str(staged)]
    past = subprocess.run(weigh, capture_output=True, text=True)
    assert past.returncode == 1
    assert json.loads(past.stdout) == {"entries": 2, "mib": 18.0, "limit_mib": 16.0}
    (staged / "b.json").unlink()
    within = subprocess.run(weigh, capture_output=True, text=True, check=True)
    assert json.loads(within.stdout) == {"entries": 1, "mib": 9.0, "limit_mib": 16.0}
