"""Check 1's conditioning (tests/reference_error.py's `assert_as_exact(..., conditioning=True)`, ledger
1714): where the unweighted check stays undecided, each arm's error weighed by the reference program's own
rounding there (tests/rounding_bound.py's `rounding_scale`), so a fault on the elements a program computes
well shows beside the ones it amplifies.

The operation holds both in one row: `amplified`, a value pushed through 2^20 and back, whose rounding is
2^20 times its own size's, and `kept`, one product, rounded once. Unweighed, a row's error is its amplified
half's, and a product eight units of its rounding off hides in it; weighed, every element's error is in
units of its own rounding, and it shows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import (
    GateFailure,
    Inconclusive,
    _Conditioning,
    _site_judged,
    assert_as_exact,
    null_rate_bound,
    null_runs,
)
from rounding_bound import rounding_scale

ROWS, HALF = 8, 16

WEIGHED_SEEDS = 8
"""Check 1's declared seeds for the weighed sites here (`seeds_for` with `conditioning`, paired)."""

NULL_RUNS = 32
"""The null control's runs: the reference against itself on disjoint draws, weighed."""


def draw(seed: int):
    """A seed's two `[ROWS, HALF]` standard normal operands, float32."""
    rng = np.random.default_rng([0x1714, seed])
    return tuple(rng.standard_normal((ROWS, HALF)).astype(np.float32) for _ in range(2))


def amplified(a):
    """`a` through 2^20 and back: exact in float64, its float32 rounding 2^-4 at the most."""
    big = jnp.asarray(2.0 ** 20, a.dtype)
    return jax.lax.optimization_barrier(a + big) - big


def reference(a, b):
    """Each row's amplified half beside its product half."""
    return jnp.concatenate([amplified(a), a * b], axis=-1)


def biased(a, b):
    """The reference with its product half eight units of float32's rounding off: 2^-21 of each."""
    return jnp.concatenate([amplified(a), (a * b) * jnp.float32(1 + 2.0 ** -21)], axis=-1)


def weighed(dew, label: str, against=reference):
    """The weighed check alone (`assert_as_exact`'s second check) of `dew` against `against`."""
    _site_judged(jax.jit(dew), jax.jit(against), draw, f"{label}, conditioned", seeds=WEIGHED_SEEDS,
                 truth_fn=None, orders=None, library_fn=None, library_row=None, library_leaves=(), also=None,
                 terms=False, examples=False, weighing=_Conditioning(jax.jit(against)))


def test_a_fault_on_the_elements_a_program_computes_well_hides_unweighed_and_shows_weighed():
    """A product half eight units of its rounding off passes check 1 unweighed, its row's error the
    amplified half's, a decided pass the site takes as it is; the weighed check, each element's error in
    units of its own rounding, decides it a failure."""
    assert_as_exact(jax.jit(biased), jax.jit(reference), draw, "a biased product beside an amplified value",
                    seeds=WEIGHED_SEEDS, conditioning=True)
    with pytest.raises(GateFailure):
        weighed(biased, "a biased product beside an amplified value")


def test_an_arm_as_exact_as_the_reference_passes_weighed():
    """The reference's own math, its product's operands taken in the other order, passes weighed; so does
    a reformulation that cancels as the reference does, the amplified value through 2^20 subtracted
    before it is added back."""
    def swapped(a, b):
        return jnp.concatenate([amplified(a), b * a], axis=-1)

    def reformulated(a, b):
        big = jnp.asarray(2.0 ** 20, a.dtype)
        return jnp.concatenate([-(jax.lax.optimization_barrier(big - a) - big), a * b], axis=-1)

    weighed(swapped, "the product swapped")
    weighed(reformulated, "the amplified value cancelled the other way")


def test_an_element_no_rounding_reaches_must_be_exact_in_every_arm():
    """An element no rounding of the reference's program reaches (`a` copied) must be exact in every arm,
    weighed: an arm one unit of its rounding off there fails (GateFailure's AssertionError, by name)."""
    def copied(a, b):
        return jnp.concatenate([a, a * b], axis=-1)

    def nudged(a, b):
        return jnp.concatenate([a * jnp.float32(1 + 2.0 ** -23), a * b], axis=-1)

    weighed(copied, "a copy", against=copied)
    with pytest.raises(AssertionError) as raised:
        weighed(nudged, "a copy nudged", against=copied)
    assert not isinstance(raised.value, Inconclusive)


def test_the_weighed_check_passes_the_reference_against_itself():
    """The null control: the reference against itself on disjoint draws, judged as a weighed site is,
    passes every one of `NULL_RUNS` runs, each verdict naming the check that decided it, its false-failure
    rate bounded by `null_rate_bound`."""
    verdicts = null_runs(jax.jit(reference), draw, "the reference against itself, weighed",
                         seeds=WEIGHED_SEEDS, runs=NULL_RUNS, conditioning=True)
    reds = sum(not verdict.startswith("pass") for verdict, _, _ in verdicts)
    assert reds == 0, (reds, null_rate_bound(reds, NULL_RUNS), verdicts)


def test_a_rounding_reaches_an_element_its_first_derivative_does_not():
    """((a + 2^20) - 2^20 - a)^2 is zero in float64 and 2^-20 or so in float32: its rounding's first
    derivative is zero there, and the scale, read by moving each rounding a rounding's size, is not."""
    def cancelled(a):
        big = jnp.asarray(2.0 ** 20, a.dtype)
        return jnp.square(jax.lax.optimization_barrier(a + big) - big - a)

    value, scale = rounding_scale(cancelled, np.full((4,), 1 + 2.0 ** -10, np.float32))
    assert np.all(value == 0) and np.all(scale > 0)


def test_a_rounding_behind_a_stop_gradient_reaches_its_output():
    """stop_gradient(a * b) holds its product's rounding, which the scale reads, a stop gradient moving
    nothing of a value's own."""
    a, b = draw(0)
    _, scale = rounding_scale(lambda a, b: jax.lax.stop_gradient(a * b), a, b)
    assert np.all(scale > 0)


def test_an_integer_rounds_where_its_float_is_too_narrow_to_hold_it():
    """An int32 past 2^24 converts to float32 with rounding the scale reads; one below converts exactly,
    at no scale."""
    _, scale = rounding_scale(lambda n: n.astype(jnp.float32), np.asarray([2 ** 24 + 1, 3], np.int32))
    assert scale[0] > 0 and scale[1] == 0


def test_a_programs_integer_output_has_no_scale_and_its_float_one_does():
    """A program of a float and an integer output (a product and an argmax) scales its float alone."""
    a, b = draw(0)
    _, (product, chosen) = rounding_scale(lambda a, b: (a * b, jnp.argmax(a, axis=-1)), a, b)
    assert chosen is None and np.all(product > 0)


def test_a_bfloat16_program_is_weighed_at_its_own_rounding():
    """A bfloat16 product's scale is bfloat16's rounding, 2^16 times float32's for the same values."""
    a, b = draw(0)
    _, narrow = rounding_scale(lambda a, b: a * b, a.astype(jnp.bfloat16), b.astype(jnp.bfloat16))
    _, wide = rounding_scale(lambda a, b: a * b, a, b)
    ratio = np.median(np.asarray(narrow, np.float64) / np.asarray(wide, np.float64))
    assert 2.0 ** 15 < ratio < 2.0 ** 17
