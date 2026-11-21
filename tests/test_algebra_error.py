"""Error-aware extraction (`dew.algebra.error.choose`, docs/design/algebra.md §1.5): the candidate forms the
e-graph proposes, each admitted by the accuracy gate (check 1, `assert_as_exact`, the one owner of the
policy), and the cheapest admitted chosen: the log-sum-exp fold, `log1p` and `expm1` where the form as
written loses digits."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew import algebra
from dew.algebra.error import at_least, choose, product_precision
from dew.nn import programs
from dew.parity import GateFailure

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
    cheapest and the most stable extractions, come in order of price, the compensated forms after every
    uncompensated one, each at its form's price, and the chosen is the first the gate admits in that order
    (docs/design/algebra.md §6.5): against the form as written, or under the accurate policy against it and
    every candidate as exact as it, where three fixed proposals came as written first."""
    admits = gate(softmax_with_lse, logits)
    choice = choose(softmax_with_lse, OPERANDS * 3, accurate=accurate, saturation=algebra.SATURATION,
                    admits=admits)
    prices = [(candidate.compensated, candidate.seconds) for candidate in choice.candidates]
    assert prices == sorted(prices)
    assert any(candidate.compensated for candidate in choice.candidates)
    written = next(candidate for candidate in choice.candidates if candidate.name == "as written")
    references = [written]
    if accurate:
        references += [candidate for candidate in choice.candidates
                       if candidate is not written and admits(candidate.function, written.function)]
    first = next(candidate for candidate in choice.candidates
                 if (not accurate or any(candidate is reference for reference in references))
                 and all(candidate is reference or admits(candidate.function, reference.function)
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



def _factors(shape: tuple[int, int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A seed's fp32 factors of a `[rows, depth] @ [depth, columns]` product, standard normal."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape[:2]).astype(np.float32),
            rng.standard_normal(shape[1:]).astype(np.float32))


def _product(preset: jax.lax.DotAlgorithmPreset):
    """A product of two factors in the dot algorithm `preset`, compiled."""
    return jax.jit(lambda x, y: jnp.dot(x, y, precision=preset))


def test_a_products_precision_is_a_stated_algorithm_its_sites_gate_admits():
    """A product's dot algorithm chosen for an accuracy asked (`product_precision`, row 5473), by a site
    whose gate is check 1 itself: an algorithm is admitted where this backend runs the product in it and
    the product is as exact as the IEEE fp32 one from the float64 truth. The choice is an algorithm that
    gate admits, and the product computed in it holds the accuracy asked; a gate that admits none is
    refused, never answered with a cheaper algorithm; FP8 operands, whose algorithms name no operand
    types of their own, are given one JAX lowers for them; and an algorithm converting its operands is
    never given, whatever a backend states it runs and its gate admits: an fp16 product of fp32 operands
    past fp16's range, its digits TF32's, is infinite."""
    shape = (64, 32, 16)
    draw = functools.partial(_factors, shape)

    def admits(preset):
        try:
            _product(preset)(*draw(0))
        except ValueError:  # this backend's dot_general refuses the algorithm
            return False
        try:
            assert_as_exact(_product(preset), _product(jax.lax.DotAlgorithmPreset.F32_F32_F32), draw,
                            f"a product in {preset.name}", seeds=8)
        except GateFailure:
            return False
        return True

    chosen = product_precision(*shape, jnp.float32, admits=admits)
    assert_as_exact(_product(chosen), _product(jax.lax.DotAlgorithmPreset.F32_F32_F32), draw,
                    "the product in the algorithm chosen", seeds=8)
    with pytest.raises(ValueError):
        product_precision(*shape, jnp.float32, admits=lambda preset: False)
    stated = (jax.lax.DotAlgorithmPreset.F16_F16_F16, jax.lax.DotAlgorithmPreset.F32_F32_F32)
    wide = product_precision(*shape, jnp.float32, admits=lambda preset: True, stated=stated)
    assert_as_exact(_product(wide), _product(jax.lax.DotAlgorithmPreset.F32_F32_F32),
                    lambda seed: tuple(1e6 * factor for factor in draw(seed)), "a product past fp16's range",
                    seeds=8)
    eight = product_precision(*shape, jnp.float8_e4m3fn, admits=lambda preset: True)
    assert programs.dot_arithmetic(eight, jnp.float8_e4m3fn, jnp.float8_e4m3fn).operand_bits == (8, 8)


def test_an_algorithm_at_least_another_is_as_exact_as_it():
    """A site's gate stated as an algorithm (`at_least`, row 5473) admits, of the algorithms this backend
    runs fp32 operands in, only ones whose product is as exact as the algorithm asked's from the float64
    truth (check 1 between them, over every pair), by what each states of its arithmetic alone: its
    operands' significand digits, its parts and its accumulation. Every algorithm is at least itself, so a
    site asking its upstream's algorithm can always take it."""
    draw = functools.partial(_factors, (64, 128, 32))
    runs = []
    for preset in jax.lax.DotAlgorithmPreset:
        try:
            _product(preset)(*draw(0))
        except ValueError:  # this backend's dot_general refuses the algorithm
            continue
        if programs.dot_arithmetic(preset, jnp.float32, jnp.float32) is not None:
            runs.append(preset)
    for asked in runs:
        admits = at_least(asked, jnp.float32)
        assert admits(asked)
        for chosen in filter(admits, runs):
            assert_as_exact(_product(chosen), _product(asked), draw, f"{chosen.name}, at least {asked.name}",
                            seeds=8)


def cancelled(seed):
    """Terms near a hundred million, a term of unit size, and the first's negation: a sum whose large terms
    cancel exactly, leaving the small one, which float32 rounds away beside the large."""
    rng = np.random.default_rng(seed)
    large = (rng.uniform(1e7, 1e8, SHAPE) * rng.choice([-1.0, 1.0], SHAPE)).astype(np.float32)
    small = rng.standard_normal(SHAPE).astype(np.float32)
    return jnp.asarray(large), jnp.asarray(small), jnp.asarray(-large)


def three_terms(a, b, c):
    """`(a + b) + c`, its first sum rounded before the last term is added."""
    return (a + b) + c


def test_the_accurate_policy_takes_a_compensated_sum_where_every_form_rounds_its_terms_away():
    """`(a + b) + c` of `c = -a` large: every form the e-graph holds (each order of each sum) rounds `b`
    away beside `a`, and the e-graph holds no compensated form (its equality is the reals', in which an
    addition's rounding error is zero); the chooser's error-free derivation does (`error.compensated`), and
    under the accurate policy it is taken, the form as written not as exact as it (check 1)."""
    admits = gate(three_terms, cancelled)
    choice = choose(three_terms, OPERANDS * 3, accurate=True, saturation=algebra.SATURATION, admits=admits)
    assert choice.chosen.compensated
    assert not admits(three_terms, choice.chosen.function)


def ranged(seed):
    """Operands whose product rounds in range and is brought back by its divisor, `a = c` of `1e-20`, `b`
    of `1e20`: the form as written, `(a b) / c`, is finite, where the regrouped `a (b / c)` passes float32's
    largest."""
    rng = np.random.default_rng(seed)
    small = (rng.uniform(1.0, 2.0, SHAPE) * 1e-20).astype(np.float32)
    large = (rng.uniform(1.0, 2.0, SHAPE) * 1e20).astype(np.float32)
    return jnp.asarray(small), jnp.asarray(large), jnp.asarray(small)


def product_over(a, b, c):
    """`(a b) / c`, the stable set's regrouping's left side."""
    return (a * b) / c


def test_the_accurate_policy_chooses_past_a_proposal_that_leaves_the_floats_range():
    """`(a b) / c` where `a (b / c)` overflows: the regrouped proposal is not as exact as the form as
    written, so it raises no bar under the accurate policy, which chooses a form as exact as the written
    (it raised, every finite form failing against the infinite one)."""
    admits = gate(product_over, ranged)
    choice = choose(product_over, OPERANDS * 3, accurate=True, saturation=algebra.SATURATION, admits=admits)
    assert admits(choice.chosen.function, product_over)
