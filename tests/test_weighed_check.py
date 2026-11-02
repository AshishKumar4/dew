"""Check 1's conditioning (tests/reference_error.py's `assert_as_exact(..., conditioning=True)`, ledger
1714): each arm's error weighed by the reference program's own rounding there (tests/rounding_bound.py's
`rounding_scale`), so a fault on the elements a program computes well shows beside the ones it amplifies.

The operation holds both in one row: `amplified`, a value pushed through 2^20 and back, whose rounding is
2^20 times its own size's, and `kept`, one product, rounded once. Unweighed, a row's error is its amplified
half's, and a product eight units of its rounding off hides in it; weighed, every element's error is in
units of its own rounding, and it shows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, null_rate_bound, null_runs

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


def test_a_fault_on_the_elements_a_program_computes_well_hides_unweighed_and_shows_weighed():
    """A product half eight units of its rounding off passes check 1 unweighed, its row's error the
    amplified half's, and fails it weighed, each element's error in units of its own rounding."""
    assert_as_exact(jax.jit(biased), jax.jit(reference), draw, "a biased product beside an amplified value",
                    seeds=WEIGHED_SEEDS)
    with pytest.raises(AssertionError):
        assert_as_exact(jax.jit(biased), jax.jit(reference), draw,
                        "a biased product beside an amplified value, weighed", seeds=WEIGHED_SEEDS,
                        conditioning=True)


def test_an_arm_as_exact_as_the_reference_passes_weighed():
    """The reference's own math, its product's operands taken in the other order, passes weighed."""
    def swapped(a, b):
        return jnp.concatenate([amplified(a), b * a], axis=-1)

    assert_as_exact(jax.jit(swapped), jax.jit(reference), draw, "the product swapped, weighed",
                    seeds=WEIGHED_SEEDS, conditioning=True)


def test_an_element_the_reference_does_not_round_must_be_exact_in_every_arm():
    """An element the reference's program takes no rounding at (`a` copied) must be exact in every arm,
    weighed: an arm one unit of its rounding off there fails, by name."""
    def copied(a, b):
        return jnp.concatenate([a, a * b], axis=-1)

    def nudged(a, b):
        return jnp.concatenate([a * jnp.float32(1 + 2.0 ** -23), a * b], axis=-1)

    assert_as_exact(jax.jit(copied), jax.jit(copied), draw, "a copy, weighed", seeds=WEIGHED_SEEDS,
                    conditioning=True)
    with pytest.raises(AssertionError, match="takes no rounding"):
        assert_as_exact(jax.jit(nudged), jax.jit(copied), draw, "a copy nudged, weighed",
                        seeds=WEIGHED_SEEDS, conditioning=True)


def test_a_weighed_site_sizes_no_row_by_its_terms():
    """A weighed site reads every row at its unit size, so its terms are refused (ValueError)."""
    with pytest.raises(ValueError):
        assert_as_exact(jax.jit(reference), jax.jit(reference), draw, "terms, weighed", seeds=WEIGHED_SEEDS,
                        conditioning=True, terms=True)


def test_the_weighed_check_passes_the_reference_against_itself():
    """The null control: the reference against itself on disjoint draws, weighed, passes every one of
    `NULL_RUNS` runs, its false-failure rate bounded by `null_rate_bound`."""
    verdicts = null_runs(jax.jit(reference), draw, "the reference against itself, weighed",
                         seeds=WEIGHED_SEEDS, runs=NULL_RUNS, conditioning=True)
    reds = sum(verdict != "pass" for verdict, _, _ in verdicts)
    assert reds == 0, (reds, null_rate_bound(reds, NULL_RUNS), verdicts)
