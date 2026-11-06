"""Error-aware extraction (`dew.algebra.error.choose`, docs/design/algebra.md §1.5): the candidate forms the
e-graph proposes, each admitted by the accuracy gate (check 1, `assert_as_exact`, the one owner of the
policy), and the cheapest admitted chosen: the log-sum-exp fold, `log1p` and `expm1` where the form as
written loses digits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from check1_statistics import GateFailure
from reference_error import assert_as_exact

from dew import algebra
from dew.algebra.error import choose

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


def gate(function, draw, seeds: int = 8):
    """The gate a candidate is admitted by (`choose`'s `admits`): whether it is as exact as a reference from
    `function`'s float64 truth over `draw`'s seeds (check 1)."""
    def admits(candidate, reference) -> bool:
        try:
            assert_as_exact(jax.jit(candidate), jax.jit(reference), draw, "a candidate form", seeds=seeds,
                            truth_fn=function)
        except GateFailure:
            return False
        return True

    return admits


def test_the_accurate_policy_takes_the_log_sum_exp_fold_whose_error_its_written_form_misses():
    """Under `--precision accurate` the softmax through its saved log-sum-exp takes `exp(s - m) / t`, which
    never rounds the logarithm of the total into a logit-sized value: no logarithm left, as exact as the form
    as written and the form as written not as exact as it; under default numerics the cheapest candidate the
    gate admits against the form as written is taken."""
    admits = gate(softmax_with_lse, logits)
    accurate = choose(softmax_with_lse, OPERANDS * 3, accurate=True, saturation=algebra.SATURATION,
                      admits=admits)
    assert not admits(softmax_with_lse, accurate.chosen.function)
    default = choose(softmax_with_lse, OPERANDS * 3, accurate=False, saturation=algebra.SATURATION,
                     admits=admits)
    admitted = [candidate for candidate in default.candidates
                if candidate.name == "as written" or admits(candidate.function, softmax_with_lse)]
    assert default.chosen.seconds == min(candidate.seconds for candidate in admitted)


@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
def test_the_candidates_are_proposed_cheapest_first_and_the_first_admitted_chosen(accurate):
    """The softmax through its log-sum-exp: its candidates, each root's forms enumerated beside the
    cheapest and the most stable extractions, come in order of price, and the chosen is the first the gate
    admits in that order (docs/design/algebra.md §6.5), where three fixed proposals came as written first."""
    admits = gate(softmax_with_lse, logits)
    choice = choose(softmax_with_lse, OPERANDS * 3, accurate=accurate, saturation=algebra.SATURATION,
                    admits=admits)
    prices = [candidate.seconds for candidate in choice.candidates]
    assert prices == sorted(prices)
    references = choice.candidates if accurate else [
        next(candidate for candidate in choice.candidates if candidate.name == "as written")]
    first = next(candidate for candidate in choice.candidates
                 if all(candidate is reference or admits(candidate.function, reference.function)
                        for reference in references))
    assert choice.chosen is first


@pytest.mark.parametrize("target", [None, "v5e"], ids=["here", "a TPU"])
@pytest.mark.parametrize("function", [lambda x: jnp.log(1.0 + x), lambda x: jnp.exp(x) - 1.0],
                         ids=["log of one plus", "exp minus one"])
def test_a_small_values_digits_are_kept_by_its_stable_form(function, target):
    """`log(1 + x)` and `exp(x) - 1` of a millionth lose most of its digits; the accurate policy takes a form
    the one as written is not as exact as (the gate's, check 1), priced here or for a TPU, whose vector
    units' rate no fact states: the stable proposal's risk is seconds of its own."""
    admits = gate(function, small)
    with algebra.passes(target=target):
        choice = choose(function, OPERANDS, accurate=True, saturation=algebra.SATURATION, admits=admits)
    assert not admits(function, choice.chosen.function)


@pytest.mark.parametrize("function", [lambda x: (jnp.log(1.0 + x), jnp.log(1.0 + x)),
                                      lambda x: jnp.log(1.0 + x)], ids=["repeated outputs", "one output"])
def test_a_chosen_form_returns_the_functions_outputs(function):
    """A function returning one value twice, and one returning a single array: the chosen form returns the
    function's outputs in their structure, on par with it (check 1)."""
    choice = choose(function, OPERANDS, accurate=True, saturation=algebra.SATURATION,
                    admits=gate(function, small))
    assert_as_exact(jax.jit(choice.chosen.function), jax.jit(function), small, "a chosen form's outputs",
                    seeds=8)


def test_a_form_an_equation_outside_the_language_reads_is_the_one_its_candidate_computes():
    """`log(1 + x)` read by a strided slice (an equation outside the language, `terms.Region.opaque`): the
    accurate policy prices the slice's input in a stable form and its candidate computes it, keeping a
    millionth's digits (the form as written is not as exact as it, check 1)."""
    def sliced(x):
        return jnp.log(1.0 + x)[:, ::2]

    admits = gate(sliced, small)
    choice = choose(sliced, OPERANDS, accurate=True, saturation=algebra.SATURATION, admits=admits)
    assert not admits(sliced, choice.chosen.function)
