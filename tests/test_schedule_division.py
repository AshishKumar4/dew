"""A diffusion schedule's division of a state by a rate, and an array's by a constant
(`dew.nn.precision.divided`): IEEE's correctly rounded quotient on every backend, as the PyTorch
sources divide, where XLA's division
rounds twice on a CPU (a broadcast divisor's reciprocal, openxla/xla#51148) and approximately on a
GPU; and the schedule sites that divide, on par with their PyTorch float32 forms (check 1)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference_error import as_given, assert_as_exact

from dew.diffusion.transforms import EpsilonPredictionTransform, from_clean
from dew.nn.precision import divided
from dew.sampling.solvers.sigma import _euler_step

SHAPE = (64, 4, 4, 3)
"""A batch of latents, each row its own rate: enough rows that every draw spans the rates."""

DIVISION_SEEDS = 16
"""Check 1's declared seeds for each schedule site, sized by `seeds_for` on armada."""


def draw(seed):
    """Latents, a clean prediction, a noise prediction and each row's signal and noise rates, the
    rates uniform in log over (0.01, 1), 64 rows of them: a quotient's error scales with one over
    its rate, so every draw spans the magnitudes rather than leaving its error to its few smallest
    rates."""
    rng = np.random.default_rng([23, seed])
    states = [rng.standard_normal(SHAPE).astype(np.float32) for _ in range(3)]
    rates = [np.exp(rng.uniform(np.log(0.01), 0.0, (SHAPE[0], 1, 1, 1))).astype(np.float32) for _ in range(2)]
    return (*states, *rates)


def test_a_schedules_division_keeps_zeros_infinities_and_mixed_operands():
    """A signed zero, a zero divisor of either sign, an overflowing quotient and exact quotients by
    negative divisors are IEEE's, bit for bit, as is a float32 state over a float16 rate or a Python
    float; a float64 state over a broadcast rate is the plain float64 division, which keeps its exact
    quotients and its zeros and infinities, wherever the backend rounds an inexact one."""
    numerator = np.asarray([-0.0, 0.0, 1.0, 2.0 ** 127, 3.0, -2.0, np.inf, 1.0, -1.0, 6.0, -1.0, 7.0],
                           np.float32)
    rate = np.asarray([1.0, -1.0, 0.0, 0.25, np.inf, 0.0, 2.0, -1.0, -1.0, -3.0, 3.0, -0.0], np.float32)
    got = np.asarray(jax.jit(divided)(numerator, rate))
    with np.errstate(divide="ignore", over="ignore"):
        np.testing.assert_array_equal(got.view(np.uint32), (numerator / rate).view(np.uint32))
    states, _, _, rates, _ = draw(1)
    half = np.asarray(rates, np.float16)
    np.testing.assert_array_equal(np.asarray(jax.jit(divided)(states, half)).view(np.uint32),
                                  (states / half.astype(np.float32)).view(np.uint32))
    np.testing.assert_array_equal(np.asarray(divided(states, 0.3)).view(np.uint32),
                                  (states / np.float32(0.3)).view(np.uint32))
    with jax.enable_x64(new_val=True):
        wide = np.asarray([[6.0, -0.0, np.inf], [1.0, 0.0, -2.0]])
        by = np.asarray([[3.0], [-4.0]])
        np.testing.assert_array_equal(np.asarray(jax.jit(divided)(wide, by)), wide / by)


def signed_draw(seed):
    """`draw`'s latents and first rate with a random sign on each row's rate, and a quarter of the
    latents an exact quotient's numerator: the rate times a power of two of either sign, so
    their quotients are exact (1, -1, 4, -0.5, ...) and their residuals zero."""
    numerator, _, _, rate, _ = draw(seed)
    rng = np.random.default_rng([29, seed])
    rate = (rate * rng.choice([-1, 1], rate.shape)).astype(np.float32)
    exact = rng.random(SHAPE) < 0.25
    multiple = rng.choice([1.0, -1.0, 4.0, -0.5, 2.0 ** -20], SHAPE).astype(np.float32)
    numerator = np.where(exact, np.broadcast_to(rate, SHAPE) * multiple, numerator).astype(np.float32)
    return numerator, rate


def test_a_schedules_division_rounds_once_at_the_normal_edge_and_below_it():
    """Quotients of normal operands at the normal range's edge and below it are IEEE's, bit for bit,
    as NumPy and the certificates' exact quotient (tests/op_certificates.py `exact_div`) give them: an
    exact one just above the least normal whose significand is odd (2^-126 + 2^-149, where a half
    spacing underflows), one at the least normal, and subnormal ones exact, rounded and tied, which
    round once onto the subnormal grid, whatever a lane's flush does to its arithmetic."""
    import op_certificates as oc

    numerator = np.asarray([2.0 ** -125 + 2.0 ** -148, 2.0 ** -125, 2.0 ** -120, 2.0 ** -126,
                            3 * 2.0 ** -126, -(2.0 ** -100), 3 * 2.0 ** -126, 2.0 ** -126, 1.0], np.float32)
    rate = np.asarray([2.0, 2.0, 2.0 ** 20, 3.0, 2.0 ** 22, 3.0 * 2.0 ** 40, 2.0 ** 24, 2.0 ** 23,
                       2.0 ** 127], np.float32)
    got = np.asarray(jax.jit(divided)(numerator, rate))
    with np.errstate(under="ignore"):
        want = numerator / rate
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    certificate = oc.exact_div(numerator, rate, oc.FLOAT32).astype(np.float32)
    np.testing.assert_array_equal(got.view(np.uint32), certificate.view(np.uint32))
    assert np.any(np.abs(want) < np.finfo(np.float32).tiny) and np.all(want != 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("broadcast", [True, False], ids=["per-row", "full"])
def test_a_schedules_division_is_ieee_division(broadcast, dtype):
    """The quotient of every element, a per-row rate broadcast over a state's or a full-shape
    divisor, compiled, is IEEE's correctly rounded fp32 quotient, bit for bit; a bfloat16 or float16
    one that quotient of the widened operands rounded to its dtype, as PyTorch's opmath divides.
    The rates take either sign and a quarter of the quotients are exact (`signed_draw`)."""
    numerator, rate = signed_draw(0)
    if not broadcast:
        rate = np.broadcast_to(rate, SHAPE).copy()
    numerator, rate = (np.asarray(array.astype(dtype)) for array in (numerator, rate))
    got = np.asarray(jax.jit(divided)(numerator, rate))
    expected = (numerator.astype(np.float32) / rate.astype(np.float32)).astype(numerator.dtype)
    assert got.dtype == expected.dtype
    bits = f"uint{8 * got.itemsize}"
    np.testing.assert_array_equal(got.view(bits), expected.view(bits))


def torch_wide(fn, dtype):
    """`fn` over the draw's arrays as torch tensors of `dtype`, its result as numpy."""
    @as_given
    def run(*drawn):
        return fn(*(torch.from_numpy(np.asarray(array, np.float64)).to(dtype) for array in drawn)).numpy()
    return run


SITES = {
    # epsilon to x_0, as diffusers' DDIM recovers the clean sample.
    "epsilon": (lambda x_t, _, eps, signal, noise: EpsilonPredictionTransform().backward_diffusion(
                    x_t, eps, (signal, noise))[0],
                lambda x_t, _, eps, signal, noise: (x_t - eps * noise) / signal),
    # x_0 to epsilon, as a v- or x_0-space parameterization recovers the noise.
    "clean": (lambda x_t, x_0, _, signal, noise: from_clean(x_t, x_0, (signal, noise))[1],
              lambda x_t, x_0, _, signal, noise: (x_t - x_0 * signal) / noise),
    # An Euler step's derivative in sigma, k-diffusion's (x - x_0) / sigma at alpha 1.
    "euler": (lambda x_t, x_0, _, signal, noise: _euler_step(
                  x_t, x_0, (jnp.ones_like(noise), noise), (jnp.ones_like(signal), noise * signal))[1],
              lambda x_t, x_0, _, signal, noise: (x_t - x_0) / noise),
}


@pytest.mark.parametrize("site", sorted(SITES))
def test_each_division_site_is_on_par_with_its_pytorch_form(site):
    """Each schedule site that divides by a rate, compiled, on par with its PyTorch source's float32
    form over drawn states and rates, from that form in float64 (check 1)."""
    dew, source = SITES[site]
    assert_as_exact(jax.jit(dew), torch_wide(source, torch.float32), draw, f"{site} division",
                    seeds=DIVISION_SEEDS, truth_fn=torch_wide(source, torch.float64))


@pytest.mark.parametrize("site", ["unit range", "magnitude-preserving SiLU", "magnitude-preserving sum"])
def test_a_division_by_a_constant_is_ieee_division_under_jit(site):
    """The library's divisions of an array by a constant are IEEE's, bit for bit with NumPy's, under
    jit, where XLA takes `x / c` as `x * RN(1 / c)` (row 1402): pixels' `(p - 127.5) / 127.5`
    (`dew.inputs.unit_range`) over every float32 the 0-255 range's halves give, and EDM2's
    `silu(x) / 0.596` (`dew.nn.mp.mp_silu`) over a normal draw, and its interpolation's
    `(a + t (b - a)) / sqrt((1 - t)**2 + t**2)` (`dew.nn.mp.mp_sum`) at t = 0.5, where a = b = 0.75
    showed it (train 208's review)."""
    from dew.inputs import unit_range
    from dew.nn.activations import silu
    from dew.nn.mp import mp_silu, mp_sum

    if site == "unit range":
        pixels = np.arange(0, 255.5, 0.5, dtype=np.float32)
        got = np.asarray(jax.jit(unit_range)(pixels))
        expected = (pixels - np.float32(127.5)) / np.float32(127.5)
    elif site == "magnitude-preserving SiLU":
        values = np.random.default_rng(1402).standard_normal(1 << 16).astype(np.float32)
        got = np.asarray(jax.jit(mp_silu)(values))
        expected = np.asarray(jax.jit(silu)(values)) / np.float32(0.596)
    else:
        values = np.concatenate([np.float32([0.75]), np.random.default_rng(1805).standard_normal(1 << 16)])
        values = values.astype(np.float32)
        got = np.asarray(jax.jit(lambda a: mp_sum(a, a, 0.5))(values))
        expected = values / np.float32(math.sqrt(0.5))
    np.testing.assert_array_equal(got.view(np.uint32), expected.astype(np.float32).view(np.uint32))
