"""Float-range and nested-derivative contracts of the samplers' seams: the flow time shift, a Gaussian
transition's quadratic term and KL, a guided prediction (ledger 2641; the harness is
tests/contract_support.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, assert_drawn_contract, normal_draw

from dew.diffusion.schedules.flow import time_shift


def positive_draw(seed: int | tuple[int, ...], shape: tuple, scale: float) -> jax.Array:
    """A float32 variance a row, positive, at `scale`: a draw's magnitude, at least a tenth of the scale."""
    rng = np.random.default_rng(seed)
    return jnp.asarray((np.abs(rng.standard_normal(shape)) + 0.1) * scale, jnp.float32)


@pytest.mark.parametrize("edge", ["normal", "near the least normal"])
def test_a_time_shift_holds_every_order(edge):
    """`schedules.flow.time_shift` (its derivative in the regular form): every order finite wherever float64
    is a normal float32 and the plain shifted time, shift t / (1 + (shift - 1) t), is, over times in (0, 1],
    its domain, and near its least normal; a time past 1 is no time of a schedule's."""
    def plain(t):
        return 3.0 * t / (1 + 2.0 * t)

    def inputs(seed, scale):
        """A seed's times in (0, 1] and their direction at `scale`, and its cotangent."""
        drawn = np.random.default_rng((2644, seed))
        times = jnp.asarray(drawn.uniform(1e-3, 1.0, (9,)) * scale, jnp.float32)
        along = jnp.asarray(drawn.standard_normal(9) * scale, jnp.float32)
        return (times,), (along,), jnp.asarray(drawn.standard_normal(9), jnp.float32)

    assert_drawn_contract(lambda t: time_shift(3.0, t), inputs, EDGES[edge], f"a time shift {edge}",
                          truth_fn=plain, plain_fn=plain)


def half_squared_over(value, mean, variance):
    """`|value - mean|**2 / (2 variance)` a row, as written: a Gaussian transition's quadratic term, and its
    KL to a reference of the same variance."""
    return jnp.sum(jnp.square(value - mean), axis=tuple(range(1, value.ndim))) / (2 * variance)


@pytest.mark.parametrize("edge", EDGES)
def test_a_transitions_quadratic_term_holds_every_order(edge):
    """`flow._half_mahalanobis` (a custom JVP whose tangent is the plain formula's, scaled to range): every
    order finite wherever float64 is a normal float32, past where the plain formula overflows, at values,
    means and variances near float32's ends, a value equal to its mean among them (ledger 2383); there,
    at zero, at one and near float32's least normal, under tangents of their own scales and far from them
    (ledger 3331), the second forward derivative and the gradient's forward derivative are the plain
    formula's, |t|**2 / v and t / v, and a first derivative past the old rule's product's range, exactly
    (each a power of two)."""
    from dew.sampling import flow

    def inputs(seed, scale):
        """A seed's value, mean and variance and their direction at `scale`, a value equal to its mean
        among them, and its cotangent."""
        value, centre = normal_draw((2283, seed), (3, 4), (3, 4), scale=scale)
        along = (*normal_draw((2285, seed), (3, 4), (3, 4), scale=scale),
                 positive_draw((2286, seed), (3,), scale))
        return ((value.at[0].set(centre[0]), centre, positive_draw((2284, seed), (3,), scale)), along,
                normal_draw((2287, seed), (3,))[0])

    assert_drawn_contract(flow._half_mahalanobis, inputs, EDGES[edge],
                          f"a transition's quadratic term {edge}", truth_fn=half_squared_over)

    unit = jnp.ones((1,), jnp.float32)
    # A tangent far from its operands' scale among them: a value and mean of 2**-120 under 2**10
    # (ledger 3331).
    for level, scale in ((0.0, 1.0), (1.0, 2.0**10), (2.0**-100, 2.0**-40), (2.0**-120, 2.0**10)):
        at, along = jnp.full((1, 3), level, jnp.float32), jnp.full((1, 3), scale, jnp.float32)

        def term(value, at=at):
            return flow._half_mahalanobis(value, at, unit)

        def moved(value, term=term, along=along):
            return jax.jvp(term, (value,), (along,))[1]

        np.testing.assert_array_equal(np.asarray(jax.jvp(moved, (at,), (along,))[1]), [3 * scale**2])
        gradient = jax.grad(lambda value, term=term: term(value).sum())
        np.testing.assert_array_equal(np.asarray(jax.jvp(gradient, (at,), (along,))[1]),
                                      np.full((1, 3), scale))
    # A residual of 2**100 under a variance of 2**100 along a tangent of 2**100: 3 * 2**100, exactly.
    wide, far = jnp.full((1,), 2.0**100, jnp.float32), jnp.full((1, 3), 2.0**100, jnp.float32)
    first = jax.jvp(lambda value: flow._half_mahalanobis(value, jnp.zeros_like(far), wide), (far,), (far,))[1]
    np.testing.assert_array_equal(np.asarray(first), [3 * 2.0**100])
    # A small coordinate beside a large one, moved alone by a large tangent: 2**-40, exactly, kept.
    mixed = jnp.asarray([[2.0**100, 2.0**-40]], jnp.float32)
    first = jax.jvp(lambda value: flow._half_mahalanobis(value, jnp.zeros_like(mixed), wide), (mixed,),
                    (jnp.asarray([[0.0, 2.0**100]], jnp.float32),))[1]
    np.testing.assert_array_equal(np.asarray(first), [2.0**-40])


@pytest.mark.parametrize("edge", EDGES)
def test_a_transitions_kl_holds_every_order(edge):
    """`GaussianTransition.kl` (`flow._gaussian_kl`, a custom JVP whose tangent is one linear map: its
    forward mode `_kl_tangent`'s, its reverse mode `_kl_pullback`'s, each formed in range): every order
    finite wherever float64 is a normal float32 and the KL as written is, in the mean, the reference and
    the variance, a forward derivative of its gradient included (ledger 2383)."""
    from dew.sampling import flow

    def kl(mean, reference, variance):
        return flow.GaussianTransition(mean, variance).kl(reference)

    def inputs(seed, scale):
        """A seed's mean, reference and variance and their direction at `scale`, and its cotangent."""
        centre, held = normal_draw((2288, seed), (3, 4), (3, 4), scale=scale)
        along = (*normal_draw((2290, seed), (3, 4), (3, 4), scale=scale),
                 positive_draw((2291, seed), (3,), scale))
        drawn = (centre, held, positive_draw((2289, seed), (3,), scale))
        return drawn, along, normal_draw((2292, seed), (3,))[0]

    assert_drawn_contract(kl, inputs, EDGES[edge], f"a transition's KL {edge}", truth_fn=half_squared_over,
                          plain_fn=half_squared_over)


@pytest.mark.parametrize("edge", EDGES)
def test_a_guided_prediction_holds_every_order(edge):
    """`guidance.guide` (`_guided`, a custom JVP: the conditional exactly at a unit scale, the affine
    combination's derivative at every scale): every order finite wherever float64 is a normal float32 and the
    combination as written, unconditional + scale (conditional - unconditional), is, a row's scale one."""
    from dew.sampling import guidance

    scale = jnp.asarray([1.0, 3.5, -0.5, 7.5], jnp.float32)

    def affine(conditional, unconditional, scale):
        return unconditional + scale[:, None] * (conditional - unconditional)

    def inputs(seed, at):
        """A seed's predictions and their direction at `at`, its rows' scales as fixed, and its cotangent."""
        drawn, unguided = normal_draw((2298, seed), (4, 5), (4, 5), scale=at)
        along = (*normal_draw((2299, seed), (4, 5), (4, 5), scale=at), *normal_draw((2300, seed), (4,)))
        return (drawn, unguided, scale), along, normal_draw((2301, seed), (4, 5))[0]

    assert_drawn_contract(guidance.guide, inputs, EDGES[edge], f"a guided prediction {edge}", truth_fn=affine,
                          plain_fn=affine)


def test_a_guided_predictions_derivative_along_its_scale_and_conditional_is_the_affine_ones_at_a_unit_scale():
    """`guidance.guide`'s derivative along its scale and its conditional prediction is the affine
    combination's, one, at every scale, a unit one included, where its value is the conditional exactly; and
    along its scale and its unconditional, minus one (a branch on the unit scale made both zero there: 512
    of 1024 unit-scale elements of a mixed derivative missed)."""
    from dew.sampling import guidance

    conditional, unconditional = normal_draw((2302, 0), (4, 5), (4, 5))
    scale = jnp.asarray([1.0, 3.5, -0.5, 1.0], jnp.float32)
    ones, zeros = jnp.ones_like(conditional), jnp.zeros_like(conditional)

    def along(conditional_direction, unconditional_direction):
        def tangent(scale):
            return jax.jvp(lambda c, u: guidance.guide(c, u, scale), (conditional, unconditional),
                           (conditional_direction, unconditional_direction))[1]
        return jax.jvp(tangent, (scale,), (jnp.ones_like(scale),))[1]

    np.testing.assert_array_equal(np.asarray(along(ones, zeros)), np.ones((4, 5), np.float32))
    np.testing.assert_array_equal(np.asarray(along(zeros, ones)), -np.ones((4, 5), np.float32))


def test_a_guided_predictions_direction_stays_finite_where_its_affine_combination_is():
    """`guidance.guide`'s direction is the affine combination's in its own order, the directions' difference
    formed first, finite wherever that is: two directions of 1e38 at a scale of 4 give 1e38 (weighed apart,
    4 times one less 3 times the other overflows); at a unit scale it is the conditional's direction exactly,
    an unconditional one that is not finite weighed nothing."""
    from dew.sampling import guidance

    huge = jnp.full((2, 3), 1e38, jnp.float32)
    zeros = jnp.zeros((2, 3), jnp.float32)
    scaled = jax.jvp(lambda c, u: guidance.guide(c, u, jnp.asarray([4.0, 1.0], jnp.float32)),
                     (zeros, zeros), (huge, huge.at[1].set(jnp.inf)))[1]
    np.testing.assert_array_equal(np.asarray(scaled), np.asarray(huge))
