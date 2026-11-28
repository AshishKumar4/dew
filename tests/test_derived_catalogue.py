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


def test_a_releases_generations_are_staged_as_one_union_weighed_once(tmp_path):
    """The one staging of a release's derivations (tools/derived_weight.py `stage`, which tools/ci/package.sh
    calls): generations each within the package's limit are refused where their union is past it; an entry two
    generations hold alike is one entry, counted once; one name two generations hold differently is refused,
    as is a generation holding none."""
    derived = tmp_path / "derived"
    for generation, names in (("sm80", ("a", "shared")), ("sm90", ("b", "shared"))):
        (derived / generation).mkdir(parents=True)
        for name in names:
            (derived / generation / f"{name}.json").write_bytes(b"x" * (5 * 2**20))

    def stage(into: str, *generations: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(REPO_ROOT / "tools/derived_weight.py"), "stage",
                               str(tmp_path / into), str(derived), *generations],
                              capture_output=True, text=True)
    # Each 10 MiB; their union three entries, `shared` once: 15 MiB, within 16.
    within = stage("within", "sm80", "sm90")
    assert within.returncode == 0, within.stderr
    assert json.loads(within.stdout) == {"generations": {"sm80": 2, "sm90": 2}, "entries": 3, "mib": 15.0,
                                         "limit_mib": 16.0}
    # A third generation, itself 5 MiB: the union past the limit, refused though each generation is within it.
    (derived / "sm100").mkdir()
    (derived / "sm100" / "c.json").write_bytes(b"x" * (5 * 2**20))
    assert stage("past", "sm80", "sm90", "sm100").returncode == 1
    (derived / "sm100" / "c.json").unlink()
    assert stage("empty", "sm80", "sm100").returncode == 1
    (derived / "sm90" / "shared.json").write_bytes(b"y" * (5 * 2**20))
    assert stage("differs", "sm80", "sm90").returncode == 1
