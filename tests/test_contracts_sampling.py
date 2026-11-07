"""Float-range and nested-derivative contracts of the samplers' seams: the flow time shift, a Gaussian
transition's quadratic term and KL, a guided prediction (ledger 2641; the harness is
tests/contract_support.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, assert_contract, normal_draw

from dew.diffusion.schedules.flow import time_shift


def positive_draw(seed: int, shape: tuple, scale: float) -> jax.Array:
    """A float32 variance a row, positive, at `scale`: a draw's magnitude, at least a tenth of the scale."""
    rng = np.random.default_rng(seed)
    return jnp.asarray((np.abs(rng.standard_normal(shape)) + 0.1) * scale, jnp.float32)


@pytest.mark.parametrize("edge", ["normal", "near the least normal"])
def test_a_time_shift_holds_every_order(edge):
    """`schedules.flow.time_shift` (its derivative in the regular form): every order finite wherever float64
    is a normal float32 and the plain shifted time, shift t / (1 + (shift - 1) t), is, over times in (0, 1],
    its domain, and near its least normal; a time past 1 is no time of a schedule's."""
    rng = np.random.default_rng(2644)
    t = jnp.asarray(rng.uniform(1e-3, 1.0, (9,)) * EDGES[edge], jnp.float32)
    direction = jnp.asarray(rng.standard_normal(t.shape) * EDGES[edge], jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal(t.shape), jnp.float32)

    def plain(t):
        return 3.0 * t / (1 + 2.0 * t)

    assert_contract(lambda t: time_shift(3.0, t), (t,), (direction,), cotangent, f"a time shift {edge}",
                    truth_fn=plain, plain_fn=plain)


def half_squared_over(value, mean, variance):
    """`|value - mean|**2 / (2 variance)` a row, as written: a Gaussian transition's quadratic term, and its
    KL to a reference of the same variance."""
    return jnp.sum(jnp.square(value - mean), axis=tuple(range(1, value.ndim))) / (2 * variance)


@pytest.mark.parametrize("edge", EDGES)
def test_a_transitions_quadratic_term_and_kl_hold_every_order(edge):
    """`GaussianTransition.kl` and the quadratic term of its `log_prob`, one map (`flow._gaussian_kl`, a
    custom JVP whose tangent is one linear map: its forward mode `_kl_tangent`'s, its reverse mode
    `_kl_pullback`'s, each formed in range): every order finite wherever float64 is a normal float32 and the
    term as written is, in the mean, the reference and the variance, past where the plain formula
    overflows, a value equal to its mean among them, a forward derivative of its gradient included (ledger
    2383)."""
    from dew.sampling import flow

    mean, reference = normal_draw(2288, (3, 4), (3, 4), scale=EDGES[edge])
    mean = mean.at[0].set(reference[0])
    variance = positive_draw(2289, (3,), EDGES[edge])
    direction = (*normal_draw(2290, (3, 4), (3, 4), scale=EDGES[edge]),
                 positive_draw(2291, (3,), EDGES[edge]))
    (cotangent,) = normal_draw(2292, (3,))

    def kl(mean, reference, variance):
        return flow.GaussianTransition(mean, variance).kl(reference)

    assert_contract(kl, (mean, reference, variance), direction, cotangent, f"a transition's KL {edge}",
                    truth_fn=half_squared_over, plain_fn=half_squared_over)


@pytest.mark.parametrize(("level", "scale"), [(0.0, 1.0), (1.0, 2.0**10), (2.0**-100, 2.0**-40),
                                              (2.0**-120, 2.0**10), (2.0**100, 2.0**100)])
def test_a_transitions_quadratic_term_differentiates_exactly_at_a_tangent_far_from_its_operands(level,
                                                                                               scale):
    """The quadratic term `|value - mean|**2 / 2` (`flow._gaussian_kl` at a unit variance), at a value equal
    to its mean `level` under tangents of `scale`, far from its operands' scale among them (ledger 3331: a
    mean of 2**-120 under a tangent of 2**10, the old rule's second derivative infinite; a mean of 2**100
    under 2**100): its second forward derivative and its gradient's forward derivative the plain formula's,
    |t|**2 and t, exactly (each a power of two times three or one); at a variance of 2**100 its first
    derivative along the value, 2**100 from its mean, 3 t 2**100 / 2**100, exactly."""
    from dew.sampling import flow

    unit = jnp.ones((1,), jnp.float32)
    at, along = jnp.full((1, 3), level, jnp.float32), jnp.full((1, 3), scale, jnp.float32)

    def term(value):
        return flow._gaussian_kl(value, at, unit)

    def moved(value):
        return jax.jvp(term, (value,), (along,))[1]

    np.testing.assert_array_equal(np.asarray(jax.jvp(moved, (at,), (along,))[1]), [3 * scale**2])
    gradient = jax.grad(lambda value: term(value).sum())
    np.testing.assert_array_equal(np.asarray(jax.jvp(gradient, (at,), (along,))[1]), np.full((1, 3), scale))
    wide = jnp.full((1,), 2.0**100, jnp.float32)
    first = jax.jvp(lambda value: flow._gaussian_kl(value, jnp.zeros((1, 3), jnp.float32), wide),
                    (jnp.full((1, 3), 2.0**100, jnp.float32),), (along,))[1]
    np.testing.assert_array_equal(np.asarray(first), [3 * scale])


@pytest.mark.parametrize("edge", EDGES)
def test_a_guided_prediction_holds_every_order(edge):
    """`guidance.guide` (`_guided`, a custom JVP: the conditional exactly at a unit scale, the affine
    combination's derivative at every scale): every order finite wherever float64 is a normal float32 and the
    combination as written, unconditional + scale (conditional - unconditional), is, a row's scale one."""
    from dew.sampling import guidance

    conditional, unconditional = normal_draw(2298, (4, 5), (4, 5), scale=EDGES[edge])
    scale = jnp.asarray([1.0, 3.5, -0.5, 7.5], jnp.float32)
    direction = (*normal_draw(2299, (4, 5), (4, 5), scale=EDGES[edge]), *normal_draw(2300, (4,)))
    (cotangent,) = normal_draw(2301, (4, 5))

    def affine(conditional, unconditional, scale):
        return unconditional + scale[:, None] * (conditional - unconditional)

    assert_contract(guidance.guide, (conditional, unconditional, scale), direction, cotangent,
                    f"a guided prediction {edge}", truth_fn=affine, plain_fn=affine)
