"""The guidance rules against their references: APG against Diffusers'
`normalized_guidance`, autoguidance against NVlabs/edm2's sampler
(`tools/guidance_reference.py`), and CFG++ against its paper's update."""

import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion import (
    DirectPredictionTransform,
    EpsilonPredictionTransform,
    FlowMatchingScheduler,
    FlowMatchPredictionTransform,
    LinearNoiseScheduler,
    Process,
    expand,
)
from dew.sampling import APG, DDIM, Autoguidance, CFGPlusPlus, Euler, Heun, sample

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "guidance"


class Velocity(nn.Module):
    """The flow model `tools/guidance_reference.py` walks APG over."""

    @nn.compact
    def __call__(self, x, time, label):
        return jnp.sin(x) * 0.07 + expand(time, x) * 0.001 + expand(label, x)


@pytest.mark.parametrize("name", ["plain", "projected", "momentum"])
def test_apg_walks_diffusers_adaptive_projected_guidance(name):
    """Momentum, the norm clip and the projection onto the conditional
    velocity, over a whole Euler walk; `plain` (eta 1, nothing clipped, no
    momentum) is classifier-free guidance."""
    arrays = np.load(FIXTURES / "apg.npz")
    case = json.loads(str(arrays["cases"]))[name]
    rows = arrays["x_T"].shape[0]
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    denoise = process.denoiser(Velocity(), {}, {"label": jnp.full((rows,), float(arrays["label"]))},
                               {"label": jnp.zeros((rows,))})
    walked = sample(denoise, jnp.asarray(arrays["x_T"]), int(arrays["steps"]), solver=Euler(),
                    guidance=APG(**case), key=jax.random.PRNGKey(0), final_denoise=False)
    assert_as_exact_as_the_reference(walked, arrays[f"{name}.result32"], arrays[f"{name}.result"], name)


def test_apg_momentum_rests_where_guidance_is_off():
    """Diffusers updates the momentum buffer only while guidance is on: the
    steps past the interval leave the running average as it was."""
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    denoise = process.denoiser(Velocity(), {}, {"label": jnp.full((2,), 0.7)}, {"label": jnp.zeros((2,))})
    walk = APG(6.0, eta=0.2, norm_threshold=0.8, momentum=-0.5, interval=(0.0, 0.5)).walk(denoise)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))
    average = walk.init(x)
    history = []
    for t in np.asarray(process.times(7))[:-1]:
        _, average = walk.step(x, jnp.full((2,), t), average)
        history.append(np.asarray(average))
    assert not np.array_equal(history[1], history[0])
    for rest in history[4:]:
        np.testing.assert_array_equal(rest, history[3])


def test_apg_momentum_needs_a_walk():
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    with pytest.raises(ValueError, match="momentum"):
        APG(4.0, momentum=-0.5)(process.denoiser(Velocity(), {}, {"label": jnp.ones((1,))},
                                                 {"label": jnp.zeros((1,))}))


class Strength(nn.Module):
    """EDM's closed-form D(x; sigma) at a strength, as the reference's main
    network and its weaker guide."""

    strength: float

    @nn.compact
    def __call__(self, x, sigma):
        sigma = expand(sigma, x)
        return jnp.tanh(x / jnp.sqrt(1 + sigma ** 2)) * self.strength + 0.1 * jnp.sin(sigma)


def test_autoguidance_walks_edm2s_guided_sampler():
    from dew.diffusion.schedules.source_grids import SigmaGrid

    arrays = np.load(FIXTURES / "autoguidance.npz")
    settings = json.loads(str(arrays["settings"]))
    steps = settings["num_steps"]
    ramp = np.arange(steps, dtype=np.float64) / (steps - 1)
    sigmas = np.append((80.0 ** (1 / 7) + ramp * (0.002 ** (1 / 7) - 80.0 ** (1 / 7))) ** 7, 0.0)
    process = Process(SigmaGrid(sigmas, sigmas, 80.0), DirectPredictionTransform())
    denoise = process.denoiser(Strength(settings["strength"]), {"guide": {}}, {})
    guidance = Autoguidance(settings["guidance"], Strength(settings["guide_strength"]))
    x_T = jnp.asarray(arrays["noise"] * np.float32(80.0))
    walked = sample(denoise, x_T, solver=Heun(), guidance=guidance, key=jax.random.PRNGKey(0),
                    times=np.arange(steps, -1, -1, dtype=np.float32), final_denoise=False)
    assert_as_exact_as_the_reference(walked, arrays["result32"], arrays["result"], "autoguidance")


class Epsilon(nn.Module):
    @nn.compact
    def __call__(self, x, time, label):
        return jnp.tanh(x) * 0.3 + expand(label, x) * jnp.cos(expand(time, x) / 300)


def test_cfg_plus_plus_takes_the_papers_ddim_update():
    """Algorithm 1 of Chung et al.: x_{t-1} = sqrt(abar_{t-1}) x0(eps_lambda)
    + sqrt(1 - abar_{t-1}) eps_uncond, eps_lambda = eps_u + lambda (eps_c -
    eps_u), written out step by step over the DDIM grid."""
    process = Process(LinearNoiseScheduler(1000), EpsilonPredictionTransform())
    rows, scale = 2, 0.6
    conditioned, blank = {"label": jnp.full((rows,), 0.8)}, {"label": jnp.zeros((rows,))}
    denoise = process.denoiser(Epsilon(), {}, conditioned, blank)
    x_T = jax.random.normal(jax.random.PRNGKey(1), (rows, 4, 4, 3))
    walked = sample(denoise, x_T, 12, solver=DDIM(), guidance=CFGPlusPlus(scale),
                    key=jax.random.PRNGKey(0), final_denoise=False)

    schedule = process.schedule

    def walk(dtype):
        x = np.asarray(x_T, dtype)
        times = np.asarray(process.times(12))
        for t, following in itertools.pairwise(times):
            t_row = jnp.full((rows,), t)
            abar = dtype(schedule.rates(t_row)[0][0]) ** 2
            abar_next = dtype(schedule.rates(jnp.full((rows,), following))[0][0]) ** 2
            time = np.asarray(schedule.model_time(t_row), dtype).reshape(rows, 1, 1, 1)
            eps_c = np.tanh(x) * dtype(0.3) + dtype(0.8) * np.cos(time / dtype(300))
            eps_u = np.tanh(x) * dtype(0.3)
            eps = eps_u + dtype(scale) * (eps_c - eps_u)
            clean = (x - np.sqrt(1 - abar) * eps) / np.sqrt(abar)
            x = np.sqrt(abar_next) * clean + np.sqrt(1 - abar_next) * eps_u
        return x

    assert_as_exact_as_the_reference(walked, walk(np.float32), walk(np.float64), "cfg++")
