"""A diffusion schedule's division of a state by a rate (`dew.diffusion.schedules.divided`): IEEE's
correctly rounded quotient on every backend, as the PyTorch sources divide, where XLA's division
rounds twice on a CPU (a broadcast divisor's reciprocal, openxla/xla#51148) and approximately on a
GPU; and the schedule sites that divide, on par with their PyTorch float32 forms (check 1)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference_error import assert_as_exact

from dew.diffusion.schedules import divided
from dew.diffusion.transforms import EpsilonPredictionTransform, from_clean
from dew.sampling.solvers.sigma import _euler_step

SHAPE = (4, 8, 8, 3)
"""A batch of latents, each row its own rate."""

DIVISION_SEEDS = 16
"""Check 1's declared seeds for each schedule site (to size)."""


def draw(seed):
    """Latents, a clean prediction, a noise prediction and each row's signal and noise rates, the
    rates in (0.01, 1) as a schedule's are."""
    rng = np.random.default_rng([23, seed])
    states = [rng.standard_normal(SHAPE).astype(np.float32) for _ in range(3)]
    rates = [rng.uniform(0.01, 1.0, (SHAPE[0], 1, 1, 1)).astype(np.float32) for _ in range(2)]
    return (*states, *rates)


@pytest.mark.parametrize("broadcast", [True, False], ids=["per-row", "full"])
def test_a_schedules_division_is_ieee_division(broadcast):
    """The quotient of every element, a per-row rate broadcast over a state's or a full-shape
    divisor, compiled, is IEEE's correctly rounded fp32 quotient, bit for bit."""
    numerator, _, _, rate, _ = draw(0)
    if not broadcast:
        rate = np.broadcast_to(rate, SHAPE).copy()
    got = np.asarray(jax.jit(divided)(numerator, rate))
    np.testing.assert_array_equal(got.view(np.uint32), (numerator / rate).view(np.uint32))


def torch_wide(fn, dtype):
    """`fn` over the draw's arrays as torch tensors of `dtype`, its result as numpy."""
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
