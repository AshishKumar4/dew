"""Float-range and nested-derivative contracts of the samplers' seams: the flow time shift, a Gaussian
transition's quadratic term and KL, the KL's forward mode, a guided prediction (ledger 2641; the harness
is tests/contract_support.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, GRADIENT_ORDERS, REVERSE_ORDERS, assert_contract, normal_draw

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
def test_a_transitions_quadratic_term_holds_every_order(edge):
    """`flow._half_mahalanobis` (a custom JVP whose tangent is the plain formula's, scaled to range): every
    order finite wherever float64 is a normal float32, past where the plain formula overflows, at values,
    means and variances near float32's ends. A value equal to its mean, whose second forward derivative is
    NaN, is ledger 2383's, whose head adds it."""
    from dew.sampling import flow

    value, mean = normal_draw(2283, (3, 4), (3, 4), scale=EDGES[edge])
    variance = positive_draw(2284, (3,), EDGES[edge])
    direction = (*normal_draw(2285, (3, 4), (3, 4), scale=EDGES[edge]),
                 positive_draw(2286, (3,), EDGES[edge]))
    (cotangent,) = normal_draw(2287, (3,))
    assert_contract(flow._half_mahalanobis, (value, mean, variance), direction, cotangent,
                    f"a transition's quadratic term {edge}", truth_fn=half_squared_over)


@pytest.mark.parametrize("edge", EDGES)
def test_a_transitions_kl_holds_every_order(edge):
    """`GaussianTransition.kl` (`flow._gaussian_kl`, Dew's seam over `generation.linear_tangent`: its forward
    mode `_directional`'s, its reverse mode `_kl_pullback`'s): every order it takes finite wherever float64 is
    a normal float32 and the KL as written is, in the mean, the reference and the variance; its gradient takes
    no derivative (ledger 2383)."""
    from dew.sampling import flow

    mean, reference = normal_draw(2288, (3, 4), (3, 4), scale=EDGES[edge])
    variance = positive_draw(2289, (3,), EDGES[edge])
    direction = (*normal_draw(2290, (3, 4), (3, 4), scale=EDGES[edge]),
                 positive_draw(2291, (3,), EDGES[edge]))
    (cotangent,) = normal_draw(2292, (3,))

    def kl(mean, reference, variance):
        return flow.GaussianTransition(mean, variance).kl(reference)

    assert_contract(kl, (mean, reference, variance), direction, cotangent, f"a transition's KL {edge}",
                    truth_fn=half_squared_over, plain_fn=half_squared_over, without=GRADIENT_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_a_kls_forward_mode_holds_its_orders(edge):
    """`flow._directional` (the KL's forward mode, a custom JVP whose tangent reads its tangents'
    exponents): each forward order finite wherever float64 is a normal float32 and the KL as written is; it
    is forward mode alone by design, its reverse mode `_kl_pullback`'s, which the KL's contract holds."""
    from dew.sampling import flow

    mean, reference = normal_draw(2293, (3, 4), (3, 4), scale=EDGES[edge])
    variance = positive_draw(2294, (3,), EDGES[edge])
    direction = (*normal_draw(2295, (3, 4), (3, 4), scale=EDGES[edge]),
                 positive_draw(2296, (3,), EDGES[edge]))
    (cotangent,) = normal_draw(2297, (3,))
    assert_contract(flow._directional, (mean, reference, variance), direction, cotangent,
                    f"a KL's forward mode {edge}", truth_fn=half_squared_over, plain_fn=half_squared_over,
                    without=REVERSE_ORDERS)


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
