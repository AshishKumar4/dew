"""The accuracy gate's arms as the algebra pass meets them (docs/design/algebra.md §3): a check's reference
and its float64 truth run their programs unrewritten, the candidate as its users run it."""

import json
import os
import subprocess
import sys
from pathlib import Path

FAULTY_REWRITE = """
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from egglog import i64, rewrite, var
from float64_terms import float64_terms
from reference_error import (GateFailure, as_given, assert_as_exact, dew_path, dew_programs, over_windows,
                             run_to_run, windows)
from rounding_bound import rounding_scale

from dew import algebra

from dew.algebra import rules
from dew.algebra.terms import Term
from dew.nn.kernels import options

x, params = var("x", Term), var("params", i64)
faulty = rules.RuleFamily("the exponential as its argument", (rewrite(x.unary("exp", params)).to(x),),
                      (jnp.exp, lambda x: x, None))
rules.register(rules.Rules.declared("faulty exponential", "reorders", (faulty,), "a test's"))


def sums(x, y):
    return jnp.sum(jnp.exp(x), -1) + jnp.sum(jnp.tanh(y), -1)


exponential = options.program(sums)


def draw(seed):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal((6, 34)), jnp.float32) for _ in range(2))


arm = dew_programs(lambda x, y: exponential(x, y))
moved = float(jnp.max(jnp.abs(arm(*draw(0)) - sums(*draw(0)))))
try:
    assert_as_exact(arm, arm, draw, "a faulty rewrite against its program as written", seeds=8)
    failed = False
except GateFailure:
    failed = True
assert_as_exact(dew_programs(lambda x, y: sums(x, y)), arm, draw,
                "the program as written against the reference run unrewritten", seeds=8, truth_fn=arm)
# One of Dew's own paths runs under the rewrite as Dew's arm: the faulty rewrite on both, on par; within
# a window loop around a host arm, and declared outside a run-to-run adapter, alike.
assert_as_exact(arm, dew_path(arm), draw, "a path of Dew's own under its rewrite", seeds=8, truth_fn=arm)
host = as_given(lambda x, y: arm(x, y))
assert_as_exact(over_windows(host), over_windows(dew_path(host)), windows(draw, 2),
                "a windowed path of Dew's own", seeds=8, truth_fn=over_windows(arm))
assert_as_exact(host, dew_path(run_to_run(host, "a test's")), draw, "a varying path of Dew's own", seeds=8,
                truth_fn=arm)
# A truth the gate takes from a declared reference runs unrewritten: the reference, rewritten, is not it.
log = os.path.join(os.environ["OUT"], "stages.jsonl")
os.environ["DEW_CHECK1_STAGES"] = log
assert_as_exact(arm, dew_path(arm), draw, "a path of Dew's own, its truth taken", seeds=8)
exact = [json.loads(line)["exact"] for line in open(log)]
# The gate's evaluators of a truth's terms and a reference's rounding scale trace their program as written.
rewritten = algebra.rewritten(sums)
with jax.enable_x64(new_val=True):
    wide = [np.asarray(leaf, np.float64) for leaf in draw(0)]
    written = np.asarray(sums(*wide))
    termed = float(np.max(np.abs(np.asarray(float64_terms(rewritten, *wide)[0]) - written)))
scaled = float(np.max(np.abs(np.asarray(rounding_scale(rewritten, *draw(0))[0]) - written)))
print(json.dumps({"moved": moved, "failed": failed, "termed": termed, "scaled": scaled, "exact": exact}))
"""
"""A process whose registry holds a faulty rule set (an exponential rewritten to its argument: a real
rewrite the engine saturates and extracts, a leaf being cheaper than a transcendental), which checks a
program of Dew's it rewrites (two sums, one of a bounded term, a window the pass reads) against itself."""


def test_a_reference_and_its_truth_run_their_programs_unrewritten(tmp_path):
    """The gate runs a check's reference arm and its float64 truth with the algebra pass off and the
    candidate as its users run it, the pass on (docs/design/algebra.md §3): a program a faulty rule set
    rewrites, passed as both arms of a check running Dew's own programs, fails it, its reference computing
    the program as written, which its truth, the same program, agrees with; declared one of Dew's own paths
    (`dew_path`), the reference runs under the rewrite too and the check of invariance passes, within a
    window loop around a host arm and outside a run-to-run adapter alike, while a truth taken from it runs
    unrewritten (its reference never the truth); and the evaluators of a truth's terms and a reference's
    rounding scale trace a rewritten function as written."""
    tests = Path(__file__).resolve().parent
    done = subprocess.run([sys.executable, "-c", FAULTY_REWRITE], capture_output=True, text=True, check=True,
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "OUT": str(tmp_path),
                               "PYTHONPATH": os.pathsep.join([str(tests), str(tests.parent / "src")])})
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found["moved"] > 0.1
    assert found["failed"]
    assert found["termed"] < 1e-9 and found["scaled"] < 1e-3
    assert found["exact"] == [False]
