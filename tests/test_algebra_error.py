"""Error-aware extraction (`dew.algebra.error.choose`, docs/design/algebra.md §1.5): each candidate form's
error measured against the function at float64 on its cases' draws, and the cheapest whose error meets the
policy chosen: the log-sum-exp fold, `log1p` and `expm1` where the form as written loses digits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew import algebra
from dew.algebra.error import Cases, choose

SHAPE = (8, 16)


def softmax_with_lse(s, m, t):
    """A softmax's probabilities through its saved log-sum-exp, the logits large: `m + log t` rounds into a
    value of the logits' size (a derived attention backward's `exp(s - lse)`)."""
    return jnp.exp(s - (m + jnp.log(t)))


def logits(seed):
    """Logits near a thousand with their rows' maximum and a total of a hundred terms or fewer."""
    rng = np.random.default_rng(seed)
    m = np.full(SHAPE, 1000.0, np.float32)
    s = (m - rng.uniform(0.0, 8.0, SHAPE)).astype(np.float32)
    t = rng.uniform(1.0, 100.0, SHAPE).astype(np.float32)
    return jnp.asarray(s), jnp.asarray(m), jnp.asarray(t)


def small(seed):
    """Values a millionth in size, whose digits `1 + x` rounds away."""
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.uniform(1e-7, 1e-5, SHAPE).astype(np.float32)),)


OPERANDS = (jax.ShapeDtypeStruct(SHAPE, jnp.float32),)


def test_the_accurate_policy_takes_the_log_sum_exp_fold_whose_error_its_written_form_misses():
    """Under `--precision accurate` the softmax through its saved log-sum-exp takes `exp(s - m) / t`, which
    never rounds the logarithm of the total into a logit-sized value: no logarithm left, its error a small
    share of the form as written's; under default numerics the cheapest within check 1's parity of the form
    as written is taken."""
    cases = Cases(OPERANDS * 3, logits, seeds=8)
    accurate = choose(softmax_with_lse, cases, accurate=True, saturation=algebra.SATURATION)
    assert accurate.chosen.error < 0.1
    traced = jax.make_jaxpr(accurate.chosen.function)(*logits(0))
    assert "log" not in {equation.primitive.name for equation in traced.jaxpr.eqns}
    default = choose(softmax_with_lse, cases, accurate=False, saturation=algebra.SATURATION)
    admitted = [candidate for candidate in default.candidates if candidate.error <= 1.5]
    assert default.chosen.seconds == min(candidate.seconds for candidate in admitted)


@pytest.mark.parametrize(("function", "stable"), [
    (lambda x: jnp.log(1.0 + x), "log1p"), (lambda x: jnp.exp(x) - 1.0, "expm1")])
def test_a_small_values_digits_are_kept_by_its_stable_form(function, stable):
    """`log(1 + x)` and `exp(x) - 1` of a millionth lose most of its digits; the accurate policy takes
    `log1p(x)` and `expm1(x)`, whose error is a small share of theirs."""
    choice = choose(function, Cases(OPERANDS, small, seeds=8), accurate=True, saturation=algebra.SATURATION)
    assert choice.chosen.error < 0.01
    traced = jax.make_jaxpr(choice.chosen.function)(*small(0))
    assert stable in {equation.primitive.name for equation in traced.jaxpr.eqns}
