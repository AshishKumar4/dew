"""Sampler tests against an analytic oracle denoiser.

For gaussian data x0 ~ N(0, s^2 I), the optimal denoiser has a closed form:
E[x0 | x_t] = alpha * s^2 / (alpha^2 s^2 + sigma^2) * x_t. A solver
integrating the reverse process with this oracle must produce samples with
mean 0 and std s, and on a variance exploding schedule the probability flow
ODE it integrates, dx/dsigma = x sigma / (s^2 + sigma^2), has the closed
form x(sigma) = x(sigma_max) sqrt(s^2 + sigma^2) / sqrt(s^2 + sigma_max^2).
Each solver's order of accuracy is measured against that closed form.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.diffusion import (
    CosineNoiseScheduler, EpsilonPredictionTransform, FlowMatchingScheduler,
    FlowMatchPredictionTransform, KarrasPredictionTransform, KarrasVENoiseScheduler,
    LinearNoiseScheduler, Process, broadcast_rates, expand,
)
from dew.sampling import (
    CFG, DDIM, DDPM, DPMSolverPP, Euler, EulerAncestral, Heun, MultiStepDPM, RK4, sample,
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


@pytest.mark.parametrize("solver", [Euler(), DDIM(), DDPM(), DPMSolverPP()],
                         ids=lambda s: type(s).__name__)
def test_vp_sampler_converges(solver):
    process, model = vp_process()
    assert_gaussian_stats(generate(process, model, solver))


@pytest.mark.parametrize(
    "solver", [Euler(), EulerAncestral(), DDIM(), Heun(), MultiStepDPM(), DDPM(), RK4(),
               DPMSolverPP()],
    ids=lambda s: type(s).__name__)
def test_karras_sampler_converges(solver):
    process, model = karras_process()
    assert_gaussian_stats(generate(process, model, solver))


def integrate(process, solver, x_T, steps):
    """`sample`'s walk over the grid without the final denoise, so what comes
    back is the solver's x at the grid's last sigma."""
    _, model = karras_process()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 4)), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    times = process.times(steps)
    x, state = x_T, solver.init(x_T)
    for i in range(steps - 1):
        t = jnp.full((x.shape[0],), times[i])
        t_next = jnp.full((x.shape[0],), times[i + 1])
        denoised, eps = denoise(x, t)
        x, state = solver.step(x, t, t_next, denoised, eps, state,
                               jax.random.fold_in(jax.random.PRNGKey(0), i), process, denoise)
    return x, process.schedule.sigmas(times[-1])


def test_solvers_integrate_the_flow_ode_at_their_order():
    """Twenty rho-spaced steps from sigma 80 down, against the ODE's closed
    form: RK4 (fourth order) lands within 2e-3, Heun (second) within 8e-2 and
    Euler (first) within 2e-1 of the solution's scale, and each is closer than
    the next; the multistep integrator beats Euler. Observed 8.3e-4, 5.1e-2,
    1.5e-1 and 6.9e-2. A dropped stage weight or a halved average moves an
    integrator out of its bracket."""
    process, _ = karras_process()
    x_T = jax.random.normal(jax.random.PRNGKey(0), (64, 4)) * 80.0

    def error(solver):
        x, sigma = integrate(process, solver, x_T, steps=20)
        exact = x_T * jnp.sqrt(DATA_STD**2 + sigma**2) / jnp.sqrt(DATA_STD**2 + 80.0**2)
        return float(jnp.max(jnp.abs(x - exact)) / jnp.max(jnp.abs(exact)))

    euler, heun, rk4, multistep = error(Euler()), error(Heun()), error(RK4()), error(MultiStepDPM())
    assert rk4 < 2e-3 and heun < 8e-2 and euler < 2e-1, (euler, heun, rk4)
    assert rk4 < heun < euler
    assert multistep < euler


def test_euler_ancestral_is_k_diffusions_ancestral_step():
    """k-diffusion's `get_ancestral_step` at eta 1: sigma_up^2 = sigma_s^2
    (sigma_t^2 - sigma_s^2) / sigma_t^2, sigma_down^2 = sigma_s^2 - sigma_up^2,
    and the update is x + (x - x_0) / sigma_t (sigma_down - sigma_t) plus
    sigma_up of the key's noise."""
    process, _ = karras_process()
    key = jax.random.PRNGKey(5)
    x = jax.random.normal(jax.random.fold_in(key, 1), (4, 8, 8, 3)) * 10.0
    x_0 = jax.random.normal(jax.random.fold_in(key, 2), (4, 8, 8, 3))
    t, t_next = jnp.full((4,), 0.6), jnp.full((4,), 0.4)
    (_, sigma_t), (_, sigma_s) = (broadcast_rates(process.schedule, when, x) for when in (t, t_next))
    eps = (x - x_0) / sigma_t

    sigma_up = jnp.sqrt(sigma_s**2 * (sigma_t**2 - sigma_s**2) / sigma_t**2)
    sigma_down = jnp.sqrt(sigma_s**2 - sigma_up**2)
    expected = x + eps * (sigma_down - sigma_t) + jax.random.normal(key, x.shape) * sigma_up

    stepped, _ = EulerAncestral().step(x, t, t_next, x_0, eps, (), key, process, None)
    assert jnp.allclose(stepped, expected, atol=1e-5)


def test_sampling_starts_from_sigma_max():
    """x_T is drawn at the top of the schedule, sigma_max on the Karras grid,
    so a trajectory that started lower would come out too narrow."""
    process, _ = karras_process()
    x_T = process.noise(jax.random.PRNGKey(0), (4096, 4))
    assert abs(float(jnp.std(x_T)) - 80.0) < 1.5


@pytest.mark.parametrize("solver", [DDIM(), Euler()], ids=lambda s: type(s).__name__)
def test_video_samples_converge(solver):
    """The frame axis is one more sample axis: the rates broadcast over it and
    the statistics come out the same as for images."""
    process, model = karras_process()
    assert_gaussian_stats(generate(process, model, solver, steps=50, count=64, shape=(3, 8, 8, 3)))


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
    # The closed form and the step group the same products differently; an
    # RTX 4080 puts them 2.6e-6 apart at the adjacent step, a CPU 1e-7.
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_ddim_eta_converges():
    process, model = vp_process()
    assert_gaussian_stats(generate(process, model, DDIM(eta=0.5)))


def test_a_key_reproduces_a_trajectory():
    """Nothing lives on a solver between calls: a multistep solver's history
    travels in its state, so the same key gives the same samples twice."""
    process, model = karras_process()
    first = generate(process, model, MultiStepDPM())
    second = generate(process, model, MultiStepDPM())
    assert jnp.allclose(first, second, atol=1e-5)


@pytest.mark.parametrize("solver", [MultiStepDPM(), RK4(), EulerAncestral()],
                         ids=lambda s: type(s).__name__)
def test_sigma_integrators_reject_a_vp_schedule(solver):
    """Each integrates dx/dsigma = eps, which only holds when alpha is 1. On a
    VP schedule the step would run and drift the samples narrow (0.275 std for
    a 0.3 oracle on the cosine schedule, inside the convergence tolerance), so
    the schedule is refused by type."""
    process, model = vp_process()
    with pytest.raises(ValueError, match="GeneralizedNoiseScheduler"):
        generate(process, model, solver)


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
# DPM-Solver++ (2M) against Diffusers 0.34.0
############################################################################################################

DPM_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dpm_solver"


class LinearVPOracle(nn.Module):
    """The epsilon oracle on the linear beta table, the model
    tools/dpm_solver_reference.py runs under Diffusers."""
    schedule: LinearNoiseScheduler

    @nn.compact
    def __call__(self, x, temb):
        alpha, sigma = broadcast_rates(self.schedule, temb, x)
        return x * sigma / (alpha**2 * DATA_STD**2 + sigma**2)


def dpm_walk(solver, x_T, steps=21):
    """Every latent after each solver step on the linear VP table, the walk
    the reference tool records."""
    schedule = LinearNoiseScheduler(1000)
    process = Process(schedule, EpsilonPredictionTransform())
    model = LinearVPOracle(schedule=schedule)
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 3, 4, 4)), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    times = process.times(steps)

    def body(carry, inputs):
        x, state = carry
        t, t_next = inputs
        t = jnp.full((x.shape[0],), t)
        t_next = jnp.full((x.shape[0],), t_next)
        denoised, eps = denoise(x, t)
        x, state = solver.step(x, t, t_next, denoised, eps, state,
                               jax.random.PRNGKey(0), process, denoise)
        return (x, state), x

    _, latents = jax.lax.scan(body, (x_T, solver.init(x_T)), (times[:-1], times[1:]))
    return latents


def test_dpmpp_2m_matches_diffusers_latents_and_gradient():
    """Twenty second-order steps on the linear VP table against Diffusers
    0.34.0's DPMSolverMultistepScheduler (dpmsolver++, order 2, midpoint,
    sigma_min final) on the same integer grid and the same closed-form
    epsilon oracle: every latent within 1e-4 (observed max 1.4e-06 at latents
    up to 3.1), and the vector-Jacobian product of the final latent against
    the fixture's cotangent through the whole trajectory within 1e-4 (observed
    1.0e-06 at gradients up to 1.0). Dropping the D1 term, which turns every
    step first order, misses the latents by more than 1e-2."""
    fixture = np.load(DPM_FIXTURES / "linear_vp_2m.npz")
    x_T = jnp.asarray(fixture["x_T"])
    latents = dpm_walk(DPMSolverPP(), x_T)
    assert latents.shape == fixture["latents"].shape
    assert float(jnp.max(jnp.abs(latents - fixture["latents"]))) < 1e-4
    _, vjp = jax.vjp(lambda x: dpm_walk(DPMSolverPP(), x)[-1], x_T)
    (grad,) = vjp(jnp.asarray(fixture["cotangent"]))
    assert float(jnp.max(jnp.abs(grad - fixture["grad"]))) < 1e-4

    class FirstOrderOnly:
        """The same update with its history discarded every step."""
        inner = DPMSolverPP()

        def init(self, x):
            return self.inner.init(x)

        def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
            stepped, _ = self.inner.step(x, t, t_next, denoised, eps, self.inner.init(x),
                                         key, process, denoise)
            return stepped, state

    first_order = dpm_walk(FirstOrderOnly(), x_T)
    assert float(jnp.max(jnp.abs(first_order - fixture["latents"]))) > 1e-2


def test_dpmpp_2m_lands_on_the_clean_prediction_at_sigma_zero():
    """The step onto sigma 0 is the h -> infinity limit of the first order
    update, x_t = x0, with no log of zero in the arithmetic. The flow path
    reaches sigma 0 at t = 0, so the last interval of any grid hits it; the
    check holds with and without history."""
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    model = ConstantVelocity()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 4)), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 4))
    t = jnp.full((3,), 0.25)
    zero = jnp.zeros((3,))
    x_0, eps = denoise(x, t)
    solver = DPMSolverPP()
    fresh, state = solver.step(x, t, zero, x_0, eps, solver.init(x), jax.random.PRNGKey(0),
                               process, denoise)
    assert jnp.all(jnp.isfinite(fresh)) and jnp.allclose(fresh, x_0, atol=1e-6)
    with_history, _ = solver.step(x, t, zero, x_0, eps, state, jax.random.PRNGKey(0),
                                  process, denoise)
    assert jnp.all(jnp.isfinite(with_history)) and jnp.allclose(with_history, x_0, atol=1e-6)


def test_dpmpp_2m_is_second_order_on_the_karras_ode():
    """Twenty rho-spaced steps from sigma 80 down against the closed form,
    the bracket test_solvers_integrate_the_flow_ode_at_their_order sets:
    DPM-Solver++ (2M) lands within 8e-2 like Heun and closer than Euler.
    Observed 5.7e-2 beside Heun's 5.1e-2, against Euler's 1.5e-1."""
    process, _ = karras_process()
    x_T = jax.random.normal(jax.random.PRNGKey(0), (64, 4)) * 80.0

    def error(solver):
        x, sigma = integrate(process, solver, x_T, steps=20)
        exact = x_T * jnp.sqrt(DATA_STD**2 + sigma**2) / jnp.sqrt(DATA_STD**2 + 80.0**2)
        return float(jnp.max(jnp.abs(x - exact)) / jnp.max(jnp.abs(exact)))

    dpmpp, euler = error(DPMSolverPP()), error(Euler())
    assert dpmpp < 8e-2 and dpmpp < euler, (dpmpp, euler)


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
