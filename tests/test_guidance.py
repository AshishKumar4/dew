"""The guidance rules against their references: APG against Diffusers'
`AdaptiveProjectedGuidance` guider, autoguidance against NVlabs/edm2's
sampler, interval-limited CFG against Kynkaanniemi et al.'s own sampler
(`tools/guidance_reference.py`), and CFG++ against its paper's update."""

import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffusion_support import ConditionalSourceOracle
from flax import linen as nn
from reference_error import assert_as_exact, assert_as_exact_as_the_reference

from dew.diffusion import (
    DirectPredictionTransform,
    EpsilonPredictionTransform,
    FlowMatchingScheduler,
    FlowMatchPredictionTransform,
    LinearNoiseScheduler,
    Process,
    expand,
)
from dew.sampling import APG, CFG, DDIM, Autoguidance, CFGPlusPlus, Euler, Heun, sample

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "guidance"


@pytest.mark.parametrize("name", ["plain", "projected", "momentum", "interval"])
def test_apg_walks_diffusers_adaptive_projected_guidance(name):
    """Momentum, the norm clip and the projection onto the conditional
    velocity, over a whole Euler walk; `plain` (eta 1, nothing clipped, no
    momentum) is classifier-free guidance. `interval` guides only the
    guider's steps [int(start N), int(stop N)) of ten, 2 through 6, which is
    `interval=(2/10, 6/10)`, the momentum resting on the steps either side.

    Plain guidance has no order-dependent reduction. Its 1024 independent
    states resolve the RMS error instead: default/AVX/SDE-ICX ratios are
    1.03/1.11/1.03; disjoint subsets of 128 states range from 0.98 to 1.17.
    Two states put an AVX subset at 2.18 despite the same computation.

    After the compensated projection, projected ratios over 52 spatial
    orders range from 1.47 to 1.80 on default/SDE-ICX and 1.33 to 1.67 on
    AVX, so its single-draw rule keeps a margin at the factor of two.
    """
    ensemble = name == "plain"
    arrays = np.load(FIXTURES / ("apg_ensemble.npz" if ensemble else "apg.npz"))
    case = json.loads(str(arrays["case"])) if ensemble else json.loads(str(arrays["cases"]))[name]
    steps = case["steps"]
    first, after = int(case["start"] * steps), int(case["stop"] * steps)
    rows = arrays["x_T"].shape[0]
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    condition = {"label": jnp.full((rows,), float(arrays["label"]))}
    denoise = process.denoiser(ConditionalSourceOracle(), {}, condition, {"label": jnp.zeros((rows,))})
    guidance = APG(case["scale"], eta=case["eta"], norm_threshold=case["norm_threshold"],
                   momentum=case["momentum"], interval=(first / steps, (after - 1) / steps))
    walked = sample(denoise, jnp.asarray(arrays["x_T"]), steps + 1, solver=Euler(),
                    guidance=guidance, key=jax.random.PRNGKey(0), final_denoise=False)
    prefix = "" if ensemble else f"{name}."
    assert_as_exact_as_the_reference(walked, arrays[f"{prefix}result32"], arrays[f"{prefix}result"], name)


class RawPrediction(nn.Module):
    """Given predictions isolate the guidance from a model's rounding."""

    def __call__(self, x, time, *, prediction):
        return prediction


@pytest.mark.parametrize("name", ["ordinary", "large", "tiny", "different_scales"])
def test_apg_keeps_diffusers_projection_precision_on_raw_predictions(name):
    """Diffusers projects in float64 and rounds each component to fp32.
    Nearly parallel raw predictions expose the rounding of the orthogonal
    component without an Euler walk's conversion through a clean prediction."""
    arrays = np.load(FIXTURES / "apg_projection.npz")
    conditional, unconditional = (jnp.asarray(arrays[f"{name}.{key}"])
                                   for key in ("conditional", "unconditional"))
    process = Process(FlowMatchingScheduler(), DirectPredictionTransform())

    @jax.jit
    def projected(conditional, unconditional):
        denoise = process.denoiser(RawPrediction(), {}, {"prediction": conditional},
                                   {"prediction": unconditional})
        walk = APG(6.0, eta=0.0, norm_threshold=0.0).walk(denoise, 1)
        x = jnp.zeros_like(conditional)
        # At alpha=0 and sigma=1, DirectPredictionTransform returns the raw
        # prediction as x0, so the denoiser's conversion introduces no rounding.
        return walk.step(x, jnp.ones(x.shape[0]), 0, walk.init(x))[0][0]

    prediction = projected(conditional, unconditional)
    assert_as_exact_as_the_reference(prediction, arrays[f"{name}.result32"], arrays[f"{name}.result"],
                                     f"APG projection {name}")


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


class Labelled(nn.Module):
    """EDM's closed-form D(x; sigma) at the condition's strength, the
    conditional and unconditional networks of the paper's guided walk."""

    @nn.compact
    def __call__(self, x, sigma, label):
        sigma = expand(sigma, x)
        return jnp.tanh(x / jnp.sqrt(1 + sigma ** 2)) * expand(label, x) + 0.1 * jnp.sin(sigma)


@pytest.mark.parametrize("name", ["middle", "tail"])
def test_interval_cfg_walks_the_papers_sampler(name):
    """Kynkaanniemi et al.'s `edm_sampler` guides the steps whose index lies
    in `guidance_interval`, closed at both ends, and both evaluations of a
    guided step, Heun's corrector included, take the guided prediction. Its
    [3, 6] and [7, 9] of ten steps are `interval=(0.3, 0.6)` and
    `(0.7, 0.9)`: step 6's corrector is guided and step 2's is not, and the
    last step, which has no corrector, is guided in `tail`."""
    from dew.diffusion.schedules.source_grids import SigmaGrid

    arrays = np.load(FIXTURES / "interval.npz")
    case = json.loads(str(arrays["cases"]))[name]
    steps, (first, last) = case["num_steps"], case["guidance_interval"]
    ramp = np.arange(steps, dtype=np.float64) / (steps - 1)
    sigmas = np.append((80.0 ** (1 / 7) + ramp * (0.002 ** (1 / 7) - 80.0 ** (1 / 7))) ** 7, 0.0)
    process = Process(SigmaGrid(sigmas, sigmas, 80.0), DirectPredictionTransform())
    rows = arrays["noise"].shape[0]
    conditional, unconditional = (float(strength) for strength in arrays["strengths"])
    denoise = process.denoiser(Labelled(), {}, {"label": jnp.full((rows,), conditional)},
                               {"label": jnp.full((rows,), unconditional)})
    walked = sample(denoise, jnp.asarray(arrays["noise"] * np.float32(80.0)), solver=Heun(),
                    guidance=CFG(case["G"], interval=(first / steps, last / steps)),
                    key=jax.random.PRNGKey(0), times=np.arange(steps, -1, -1, dtype=np.float32),
                    final_denoise=False)
    assert_as_exact_as_the_reference(walked, arrays[f"{name}.result32"], arrays[f"{name}.result"], name)


class Epsilon(nn.Module):
    @nn.compact
    def __call__(self, x, time, label):
        return jnp.tanh(x) * 0.3 + expand(label, x) * jnp.cos(expand(time, x) / 300)


CFG_PLUS_PLUS_SEEDS = 16
"""Check 1's declared seeds for the CFG++ walk, sized by `seeds_for` (2026-10-21, on armada)."""


@pytest.mark.parametrize("interval", [(0.0, 1.0), (0.0, 0.5)])
def test_cfg_plus_plus_takes_the_papers_ddim_update(interval):
    """Algorithm 1 of Chung et al.: x_{t-1} = sqrt(abar_{t-1}) x0(eps_lambda)
    + sqrt(1 - abar_{t-1}) eps_uncond, eps_lambda = eps_u + lambda (eps_c -
    eps_u), written out step by step over the DDIM grid, Dew's walk on par
    with it over drawn starting noise (check 1); a step outside the guidance
    interval takes the conditional denoiser's pair, x0(eps_c) and eps_c."""
    process = Process(LinearNoiseScheduler(1000), EpsilonPredictionTransform())
    rows, scale = 2, 0.6
    conditioned, blank = {"label": jnp.full((rows,), 0.8)}, {"label": jnp.zeros((rows,))}
    denoise = process.denoiser(Epsilon(), {}, conditioned, blank)

    def walked(x_T):
        return sample(denoise, jnp.asarray(x_T), 12, solver=DDIM(),
                      guidance=CFGPlusPlus(scale, interval=interval), key=jax.random.PRNGKey(0),
                      final_denoise=False)

    schedule = process.schedule

    def walk(x_T):
        dtype = x_T.dtype.type
        x = x_T
        times = np.asarray(process.times(12))
        for index, (t, following) in enumerate(itertools.pairwise(times)):
            t_row = jnp.full((rows,), t)
            abar = dtype(schedule.rates(t_row)[0][0]) ** 2
            abar_next = dtype(schedule.rates(jnp.full((rows,), following))[0][0]) ** 2
            time = np.asarray(schedule.model_time(t_row), dtype).reshape(rows, 1, 1, 1)
            eps_c = np.tanh(x) * dtype(0.3) + dtype(0.8) * np.cos(time / dtype(300))
            eps_u = np.tanh(x) * dtype(0.3)
            if interval[0] <= index / (len(times) - 1) <= interval[1]:
                eps, noise = eps_u + dtype(scale) * (eps_c - eps_u), eps_u
            else:
                eps, noise = eps_c, eps_c
            clean = (x - np.sqrt(1 - abar) * eps) / np.sqrt(abar)
            x = np.sqrt(abar_next) * clean + np.sqrt(1 - abar_next) * noise
        return x

    def noise(seed):
        return (np.asarray(jax.random.normal(jax.random.PRNGKey(seed), (rows, 4, 4, 3))),)

    assert_as_exact(walked, walk, noise, f"cfg++ {interval}", seeds=CFG_PLUS_PLUS_SEEDS)
