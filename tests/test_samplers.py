"""End-to-end sampler tests against an analytic oracle denoiser.

For gaussian data x0 ~ N(0, s^2 I), the optimal denoiser has a closed form:
E[x0 | x_t] = alpha * s^2 / (alpha^2 s^2 + sigma^2) * x_t. A solver
integrating the reverse process with this oracle must produce samples with
mean 0 and std s. This catches integrator bugs, step-domain bugs, and
shape bugs in one place, with no training involved.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from dew.diffusion import (
    CosineNoiseScheduler, EpsilonPredictionTransform, FlowMatchingScheduler,
    FlowMatchPredictionTransform, KarrasPredictionTransform, KarrasVENoiseScheduler, Process,
    broadcast_rates, expand,
)
from dew.registry import samplers
from dew.sampling import (
    CFG, DDIM, DDPM, Euler, EulerAncestral, Heun, MultiStepDPM, RK4, SimplifiedEuler, sample,
)

DATA_STD = 0.3


class VPOracle(nn.Module):
    """Optimal epsilon-predictor for x0 ~ N(0, DATA_STD^2) on a VP schedule.

    The model is conditioned on the raw timestep for discrete schedules, so
    rates are recomputed from the schedule inside the module.
    """
    schedule: CosineNoiseScheduler

    @nn.compact
    def __call__(self, x, temb):
        alpha, sigma = broadcast_rates(self.schedule, temb, x)
        return x * sigma / (alpha**2 * DATA_STD**2 + sigma**2)


class KarrasOracle(nn.Module):
    """Optimal raw-F predictor under the Karras preconditioning (VE, alpha=1).

    Receives x * c_in and temb = log(sigma)/4; must output F such that
    c_skip * x + c_out * F = E[x0 | x] = s^2/(s^2 + sigma^2) * x.
    """
    sigma_data: float = 0.5

    @nn.compact
    def __call__(self, x_scaled, temb):
        sigma = expand(jnp.exp(4.0 * temb), x_scaled)
        sd = self.sigma_data
        c_in = 1 / jnp.sqrt(sigma**2 + sd**2)
        c_skip = sd**2 / (sd**2 + sigma**2)
        c_out = sigma * sd / jnp.sqrt(sd**2 + sigma**2)
        x = x_scaled / c_in
        x0 = DATA_STD**2 / (DATA_STD**2 + sigma**2) * x
        return (x0 - c_skip * x) / c_out


def vp_process():
    schedule = CosineNoiseScheduler(1000)
    return Process(schedule, EpsilonPredictionTransform()), VPOracle(schedule=schedule)


def karras_process():
    schedule = KarrasVENoiseScheduler(sigma_max=80, rho=7, sigma_data=0.5)
    return Process(schedule, KarrasPredictionTransform(sigma_data=0.5)), KarrasOracle()


def assert_gaussian_stats(samples, std=DATA_STD, tol=0.05):
    assert jnp.all(jnp.isfinite(samples)), "sampler produced non-finite values"
    assert abs(float(jnp.mean(samples))) < tol
    assert abs(float(jnp.std(samples)) - std) < tol


def generate(process, model, solver, steps=100, count=256, shape=(8, 8, 3), seed=2):
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *shape)), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    key = jax.random.PRNGKey(seed)
    x_T = process.noise(jax.random.fold_in(key, 0), (count, *shape))
    return sample(denoise, x_T, steps, solver=solver, key=jax.random.fold_in(key, 1))


@pytest.mark.parametrize("solver", [Euler(), DDIM(), DDPM()], ids=lambda s: type(s).__name__)
def test_vp_sampler_converges(solver):
    process, model = vp_process()
    assert_gaussian_stats(generate(process, model, solver))


@pytest.mark.parametrize(
    "solver", [Euler(), EulerAncestral(), DDIM(), Heun(), MultiStepDPM(), DDPM(), SimplifiedEuler()],
    ids=lambda s: type(s).__name__)
def test_karras_sampler_converges(solver):
    process, model = karras_process()
    assert_gaussian_stats(generate(process, model, solver))


def test_sampling_starts_from_sigma_max():
    """x_T is drawn at the top of the schedule, sigma_max on the Karras grid,
    so a trajectory that started lower would come out too narrow."""
    process, _ = karras_process()
    x_T = process.noise(jax.random.PRNGKey(0), (4096, 4))
    assert abs(float(jnp.std(x_T)) - 80.0) < 1.5


@pytest.mark.parametrize("solver", [DDIM(), Euler()], ids=lambda s: type(s).__name__)
def test_video_samples(solver):
    process, model = karras_process()
    samples = generate(process, model, solver, steps=50, count=2, shape=(3, 8, 8, 3))
    assert samples.shape == (2, 3, 8, 8, 3)
    assert jnp.all(jnp.isfinite(samples))


def test_ddpm_sampler_converges_at_every_step():
    process, model = vp_process()
    assert_gaussian_stats(generate(process, model, DDPM(), steps=1000))


@pytest.mark.parametrize("t,s", [(700, 500), (300, 200), (100, 99)])
def test_ddpm_step_is_the_vp_posterior(t, s):
    """One step from t to s is the DDPM posterior q(x_s | x_t, x_0) written in
    signal and noise rates, which under alpha^2 + sigma^2 = 1 has std
    sqrt(sigma_s^2 / sigma_t^2 * (1 - alpha_t^2 / alpha_s^2)).
    """
    process, _ = vp_process()
    schedule = process.schedule
    key = jax.random.PRNGKey(3)
    x0 = jax.random.normal(jax.random.fold_in(key, 1), (4, 8, 8, 3))
    eps = jax.random.normal(jax.random.fold_in(key, 2), (4, 8, 8, 3))
    ones = jnp.ones((4,), jnp.float32)
    alpha_t, sigma_t = broadcast_rates(schedule, ones * t, x0)
    alpha_s, sigma_s = broadcast_rates(schedule, ones * s, x0)

    noise = jax.random.normal(key, x0.shape)
    std = jnp.sqrt(sigma_s**2 / sigma_t**2 * (1 - alpha_t**2 / alpha_s**2))
    expected = alpha_s * x0 + alpha_t * sigma_s**2 / (alpha_s * sigma_t) * eps + std * noise

    actual, _ = DDPM().step(alpha_t * x0 + sigma_t * eps, ones * t, ones * s, x0, eps, (),
                            key, process, None)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_ddim_eta_converges():
    process, model = vp_process()
    assert_gaussian_stats(generate(process, model, DDIM(eta=0.5)))


def test_rk4_sampler_runs():
    process, model = karras_process()
    samples = generate(process, model, RK4(), steps=20)
    assert jnp.all(jnp.isfinite(samples))


def test_a_key_reproduces_a_trajectory():
    """Nothing lives on a solver between calls: the same key gives the same
    samples, which is what a multistep history kept on the object broke."""
    process, model = karras_process()
    first = generate(process, model, MultiStepDPM())
    second = generate(process, model, MultiStepDPM())
    assert jnp.allclose(first, second, atol=1e-5)


@pytest.mark.parametrize("solver", [MultiStepDPM(), RK4()], ids=lambda s: type(s).__name__)
def test_sigma_integrators_reject_a_vp_schedule(solver):
    """Both integrate dx/dsigma = eps, which only holds when alpha is 1."""
    process, model = vp_process()
    with pytest.raises(ValueError, match="GeneralizedNoiseScheduler"):
        generate(process, model, solver)


def test_every_solver_is_registered():
    assert {type(s).__name__ for s in (DDPM(), DDIM(), Euler(), SimplifiedEuler(),
                                       EulerAncestral(), Heun(), RK4(), MultiStepDPM())} \
        <= {member.__name__ for member in samplers.values()}
    assert samplers["heun"] is Heun and samplers.Heun is Heun


class ConstantVelocity(nn.Module):
    """A flow model whose velocity is its input, so both Heun evaluations are
    finite and the last interval reaches sigma = 0 exactly."""
    @nn.compact
    def __call__(self, x, temb):
        return x


def test_heun_takes_the_euler_step_where_sigma_reaches_zero():
    """Karras et al. 2022, Algorithm 2: at sigma_next = 0 there is no
    derivative to average with, and the step is the Euler one. On the flow
    path t = 0 is exactly sigma = 0, so the last interval of any grid hits it."""
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    model = ConstantVelocity()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 4)), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 4))
    t = jnp.full((3,), 0.25)
    zero = jnp.zeros((3,))
    x_0, eps = denoise(x, t)

    heun, _ = Heun().step(x, t, zero, x_0, eps, (), jax.random.PRNGKey(0), process, denoise)
    euler, _ = Euler().step(x, t, zero, x_0, eps, (), jax.random.PRNGKey(0), process, denoise)
    assert jnp.all(jnp.isfinite(heun))
    assert jnp.allclose(heun, euler, atol=1e-6)


############################################################################################################
# Interval-limited classifier-free guidance (Kynkaanniemi et al. 2024)
############################################################################################################

class ConditionalVPOracle(nn.Module):
    """VPOracle offset by the label, so the guided output reads back the scale
    that was actually applied."""
    schedule: CosineNoiseScheduler

    @nn.compact
    def __call__(self, x, temb, label):
        alpha, sigma = broadcast_rates(self.schedule, temb, x)
        eps = x * sigma / (alpha**2 * DATA_STD**2 + sigma**2)
        return eps + expand(label, x)


def guided_denoiser(count=4):
    schedule = CosineNoiseScheduler(1000)
    process = Process(schedule, EpsilonPredictionTransform())
    model = ConditionalVPOracle(schedule=schedule)
    params = model.init(
        jax.random.PRNGKey(1), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)), jnp.ones((1, 1)))
    labels = jnp.full((count, 1), 0.7)
    return process, process.denoiser(model, params, {"label": labels},
                                     unconditional={"label": jnp.zeros((1, 1))})


@pytest.mark.parametrize("progress,inside", [(0.1, False), (0.5, True), (0.9, False)])
def test_interval_cfg_applies_only_inside_the_interval(progress, inside):
    _, denoise = guided_denoiser()
    full = CFG(3.0)(denoise)
    interval = CFG(3.0, interval=(0.4, 0.6))(denoise)

    x_t = jax.random.normal(jax.random.PRNGKey(3), (4, 8, 8, 3))
    t = jnp.full((4,), (1.0 - progress) * 1000)

    matches_full = bool(jnp.allclose(interval(x_t, t)[1], full(x_t, t)[1], atol=1e-5))
    matches_unguided = bool(jnp.allclose(interval(x_t, t)[1], denoise(x_t, t)[1], atol=1e-5))
    assert matches_full is inside
    assert matches_unguided is not inside


def test_cfg_scales_the_conditional_offset():
    """uncond + scale (cond - uncond): the oracle's label offset is exactly
    what the guided epsilon carries, times the scale."""
    _, denoise = guided_denoiser()
    x_t = jax.random.normal(jax.random.PRNGKey(3), (4, 8, 8, 3))
    t = jnp.full((4,), 500.0)
    guided = CFG(3.0)(denoise)(x_t, t)[1]
    unguided = denoise(x_t, t)[1]
    # the unconditional label is 0, the conditional one 0.7
    assert jnp.allclose(guided - unguided, 2 * 0.7, atol=1e-5)


def test_interval_cfg_defaults_to_the_full_range():
    _, denoise = guided_denoiser()
    x_t = jax.random.normal(jax.random.PRNGKey(3), (4, 8, 8, 3))
    t = jnp.full((4,), 500.0)
    assert jnp.allclose(CFG(3.0)(denoise)(x_t, t)[1],
                        CFG(3.0, interval=(0.0, 1.0))(denoise)(x_t, t)[1], atol=1e-6)


def test_empty_guidance_interval_generates_the_unguided_samples():
    process, denoise = guided_denoiser(count=16)
    x_T = process.noise(jax.random.PRNGKey(2), (16, 8, 8, 3))

    def run(guidance):
        return sample(denoise, x_T, 25, solver=DDIM(), guidance=guidance,
                      key=jax.random.PRNGKey(4))

    assert jnp.allclose(run(CFG(3.0, interval=(0.9, 0.1))), run(None), atol=1e-5)


def test_guidance_needs_the_unconditional_branch():
    process, model = vp_process()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    x_T = process.noise(jax.random.PRNGKey(2), (2, 8, 8, 3))
    with pytest.raises(ValueError, match="unconditional"):
        sample(denoise, x_T, 5, solver=DDIM(), guidance=CFG(2.0), key=jax.random.PRNGKey(0))
