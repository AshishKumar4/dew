"""The accuracy gate's arms as the algebra pass meets them (docs/design/algebra.md §3): a check's reference
and its float64 truth run their programs unrewritten, the candidate as its users run it."""

import json
import os
import subprocess
import sys
from pathlib import Path

FAULTY_REWRITE = """
import json

import jax
import jax.numpy as jnp
import numpy as np
from egglog import i64, rewrite, var
from float64_terms import float64_terms
from reference_error import GateFailure, assert_as_exact, dew_programs
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
# The gate's evaluators of a truth's terms and a reference's rounding scale trace their program as written.
rewritten = algebra.rewritten(sums)
with jax.enable_x64(new_val=True):
    wide = [np.asarray(leaf, np.float64) for leaf in draw(0)]
    written = np.asarray(sums(*wide))
    termed = float(np.max(np.abs(np.asarray(float64_terms(rewritten, *wide)[0]) - written)))
scaled = float(np.max(np.abs(np.asarray(rounding_scale(rewritten, *draw(0))[0]) - written)))
print(json.dumps({"moved": moved, "failed": failed, "termed": termed, "scaled": scaled}))
"""
"""A process whose registry holds a faulty rule set (an exponential rewritten to its argument: a real
rewrite the engine saturates and extracts, a leaf being cheaper than a transcendental), which checks a
program of Dew's it rewrites (two sums, one of a bounded term, a window the pass reads) against itself."""


def test_a_reference_and_its_truth_run_their_programs_unrewritten():
    """The gate runs a check's reference arm and its float64 truth with the algebra pass off and the
    candidate as its users run it, the pass on (docs/design/algebra.md §3): a program a faulty rule set
    rewrites, passed as both arms of a check running Dew's own programs, fails it, its reference computing
    the program as written, which its truth, the same program, agrees with; and the evaluators of a truth's
    terms and a reference's rounding scale trace a rewritten function as written."""
    tests = Path(__file__).resolve().parent
    done = subprocess.run([sys.executable, "-c", FAULTY_REWRITE], capture_output=True, text=True, check=True,
                          env={**os.environ, "JAX_PLATFORMS": "cpu",
                               "PYTHONPATH": os.pathsep.join([str(tests), str(tests.parent / "src")])})
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found["moved"] > 0.1
    assert found["failed"]
    assert found["termed"] < 1e-9 and found["scaled"] < 1e-3
