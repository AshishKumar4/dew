"""Sampler tests against an analytic oracle denoiser.

For gaussian data x0 ~ N(0, s^2 I), the optimal denoiser has a closed form:
E[x0 | x_t] = alpha * s^2 / (alpha^2 s^2 + sigma^2) * x_t. A solver
integrating the reverse process with this oracle must produce samples with
mean 0 and std s, and on a variance exploding schedule the probability flow
ODE it integrates, dx/dsigma = x sigma / (s^2 + sigma^2), has the closed
form x(sigma) = x(sigma_max) sqrt(s^2 + sigma^2) / sqrt(s^2 + sigma_max^2).
Each solver's order of accuracy is measured against that closed form.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.diffusion import (
    ConsistencyBoundary, CosineNoiseScheduler, DirectPredictionTransform, EDMNoiseScheduler,
    EpsilonPredictionTransform, FlowMatchingScheduler, FlowMatchPredictionTransform, KarrasPredictionTransform,
    KarrasVENoiseScheduler, LinearNoiseScheduler, Process, broadcast_rates, expand,
)
from dew.sampling import (
    CFG, DDIM, DDPM, DEIS, LMS, PNDM, TCD, Consistency, DPMSolverMultistep, DPMSolverSinglestep,
    Euler, EulerAncestral, Heun, KDPM2, MultiStepDPM, RK4, UniPC, sample,
)

DATA_STD = 0.3


def solver_id(solver) -> str:
    return repr(solver).replace(" ", "")


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


@pytest.mark.parametrize(
    "solver", [Euler(), DDIM(), DDPM(), DPMSolverMultistep(), DPMSolverMultistep(3, "dpmsolver"),
               DPMSolverMultistep(2, "sde-dpmsolver++"), DPMSolverSinglestep(3, lower_order_final=True),
               DEIS(), UniPC(), TCD(eta=0.0), TCD(eta=0.3)],
    ids=solver_id)
def test_vp_sampler_converges(solver):
    process, model = vp_process()
    assert_gaussian_stats(generate(process, model, solver))


class EndpointOracle(nn.Module):
    """The probability flow ODE's endpoint from x_t, x_0 = x_t s / sqrt(alpha^2
    s^2 + sigma^2), what a consistency model is distilled to output; the
    posterior mean VPOracle returns shrinks instead, so re-noising it narrows
    the samples (std 0.21 for TCD at eta 1 over a hundred steps)."""
    schedule: CosineNoiseScheduler

    @nn.compact
    def __call__(self, x, temb):
        alpha, sigma = broadcast_rates(self.schedule, temb, x)
        return x * DATA_STD / jnp.sqrt(alpha**2 * DATA_STD**2 + sigma**2)


@pytest.mark.parametrize("solver", [Consistency(), TCD(eta=1.0)], ids=solver_id)
def test_distilled_samplers_converge_with_an_endpoint_model(solver):
    """Eight steps of re-noising the endpoint reproduce the data statistics;
    TCD at eta 1 goes through the schedule's clean end and is the same
    walk up to that end's residual noise."""
    schedule = CosineNoiseScheduler(1000)
    process = Process(schedule, DirectPredictionTransform())
    assert_gaussian_stats(generate(process, EndpointOracle(schedule=schedule), solver, steps=8))


@pytest.mark.parametrize("solver", [PNDM(), PNDM(skip_prk_steps=True)], ids=solver_id)
def test_pndm_converges_on_the_linear_table(solver):
    """PNDM's transfer scales x by alpha_s / alpha_t, which is 316 over the
    first of a hundred intervals on the cosine table from its capped end
    (alpha 4.9e-5 at index 999, against 2.1 from the index 990 Diffusers'
    leading grid starts at) and turns the pseudo Runge-Kutta stages'
    noise-level differences into a 23-fold blow-up. The linear table's
    ratio there is 1.1 and both warmups converge, observed std 0.2986."""
    schedule = LinearNoiseScheduler(1000)
    process, model = Process(schedule, EpsilonPredictionTransform()), LinearVPOracle(schedule=schedule)
    assert_gaussian_stats(generate(process, model, solver))


@pytest.mark.parametrize(
    "solver", [Euler(), EulerAncestral(), DDIM(), Heun(), MultiStepDPM(), DDPM(), RK4(),
               KDPM2(), KDPM2(ancestral=True), LMS(), DPMSolverMultistep(), UniPC(), DEIS()],
    ids=solver_id)
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
    x, state = x_T, solver.init(x_T, times)
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


@pytest.mark.parametrize("solver", [MultiStepDPM(), RK4(), EulerAncestral(), KDPM2(), LMS()],
                         ids=solver_id)
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
# The Diffusers 0.34.0 schedulers, from the fixtures tools/diffusers_reference.py records
############################################################################################################

DIFFUSERS_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "diffusers"
REFERENCE = json.loads((DIFFUSERS_FIXTURES / "schedulers.json").read_text())
ARRAYS = np.load(DIFFUSERS_FIXTURES / "schedulers.npz")

# Each fixture case's Dew solver. The scheduler, its config and the model
# convention a case was recorded under are in schedulers.json.
DIFFUSERS_CASES = {
    "dpm_multistep.pp_2m": DPMSolverMultistep(2, "dpmsolver++", "midpoint"),
    "dpm_multistep.pp_2h": DPMSolverMultistep(2, "dpmsolver++", "heun"),
    "dpm_multistep.pp_3": DPMSolverMultistep(3, "dpmsolver++"),
    "dpm_multistep.dpm_2m": DPMSolverMultistep(2, "dpmsolver", "midpoint"),
    "dpm_multistep.dpm_2h": DPMSolverMultistep(2, "dpmsolver", "heun"),
    "dpm_multistep.dpm_3": DPMSolverMultistep(3, "dpmsolver"),
    "dpm_multistep.sde_pp_2m": DPMSolverMultistep(2, "sde-dpmsolver++", "midpoint"),
    "dpm_multistep.sde_pp_2h": DPMSolverMultistep(2, "sde-dpmsolver++", "heun"),
    "dpm_multistep.sde_pp_3": DPMSolverMultistep(3, "sde-dpmsolver++"),
    "dpm_multistep.sde_2m": DPMSolverMultistep(2, "sde-dpmsolver", "midpoint"),
    "dpm_multistep.sde_2h": DPMSolverMultistep(2, "sde-dpmsolver", "heun"),
    "dpm_multistep.pp_3_short": DPMSolverMultistep(3, "dpmsolver++"),
    "dpm_multistep.pp_2_euler_final": DPMSolverMultistep(2, "dpmsolver++", euler_at_final=True),
    "dpm_singlestep.pp_2": DPMSolverSinglestep(2, "dpmsolver++"),
    "dpm_singlestep.pp_3h_final": DPMSolverSinglestep(3, "dpmsolver++", "heun", lower_order_final=True),
    "dpm_singlestep.pp_3m": DPMSolverSinglestep(3, "dpmsolver++", "midpoint"),
    "dpm_singlestep.dpm_3h_final": DPMSolverSinglestep(3, "dpmsolver", "heun", lower_order_final=True),
    "dpm_singlestep.sde_pp_2": DPMSolverSinglestep(2, "sde-dpmsolver++"),
    "dpm_singlestep.sde_pp_3h_final": DPMSolverSinglestep(3, "sde-dpmsolver++", "heun",
                                                          lower_order_final=True),
    "deis.2": DEIS(2),
    "deis.3": DEIS(3),
    "deis.3_short": DEIS(3),
    "unipc.bh2_2": UniPC(2, "bh2"),
    "unipc.bh1_3": UniPC(3, "bh1"),
    "unipc.eps_bh2_2": UniPC(2, "bh2", predict_x0=False),
    "unipc.bh2_3_short_nocorr": UniPC(3, "bh2", disable_corrector=(2, 5)),
    "pndm.plms": PNDM(skip_prk_steps=True),
    "pndm.prk": PNDM(),
    "lcm.4": Consistency(),
    "tcd.eta_0": TCD(0.0),
    "tcd.eta_03": TCD(0.3),
    "tcd.eta_1": TCD(1.0),
    "kdpm2.plain": KDPM2(),
    "kdpm2.ancestral": KDPM2(ancestral=True),
    "lms.4": LMS(4),
    "lms.2": LMS(2),
    "edm_dpm.pp_2m": DPMSolverMultistep(2, "dpmsolver++", "midpoint"),
    "edm_dpm.sde_pp_2m": DPMSolverMultistep(2, "sde-dpmsolver++", "midpoint"),
}
assert set(DIFFUSERS_CASES) == set(REFERENCE["cases"])


class LinearVPOracle(nn.Module):
    """The epsilon oracle on the linear beta table, the model the reference
    tool runs under Diffusers' VP-table schedulers."""
    schedule: LinearNoiseScheduler

    @nn.compact
    def __call__(self, x, temb):
        alpha, sigma = broadcast_rates(self.schedule, temb, x)
        return x * sigma / (alpha**2 * DATA_STD**2 + sigma**2)


class VEOracle(nn.Module):
    """The epsilon oracle on a variance-exploding latent x = x_0 + sigma eps,
    read at c_noise = log(sigma) / 4, the model under the k-diffusion
    schedulers."""

    @nn.compact
    def __call__(self, x, temb):
        sigma = expand(jnp.exp(4.0 * temb), x)
        return x * sigma / (DATA_STD**2 + sigma**2)


def reference_process(name: str) -> tuple[Process, nn.Module]:
    """The Dew process and oracle of a fixture case's model convention:
    the linear VP table with epsilon (under the consistency boundary for
    LCM), the Karras grid between that table's sigma extremes in
    variance-exploding form, or EDM preconditioning on Karras' own grid."""
    convention = REFERENCE["cases"][name]["convention"]
    if convention == "vp":
        schedule = LinearNoiseScheduler(REFERENCE["train_steps"])
        prediction = EpsilonPredictionTransform()
        if name.startswith("lcm"):
            prediction = ConsistencyBoundary(prediction)
        return Process(schedule, prediction), LinearVPOracle(schedule=schedule)
    if convention == "ve":
        karras = REFERENCE["karras"]
        return Process(KarrasVENoiseScheduler(karras["sigma_min"], karras["sigma_max"], karras["rho"]),
                       EpsilonPredictionTransform()), VEOracle()
    edm = REFERENCE["edm"]
    process = Process(EDMNoiseScheduler(sigma_data=edm["sigma_data"]),
                      KarrasPredictionTransform(edm["sigma_data"]),
                      sampling=KarrasVENoiseScheduler(edm["sigma_min"], edm["sigma_max"], edm["rho"],
                                                      edm["sigma_data"]))
    return process, KarrasOracle(sigma_data=edm["sigma_data"])


def walk(solver, process, model, x_T, times, key=jax.random.PRNGKey(0)):
    """Every latent after each interval of `times`, the walk `sample` takes
    (its state, its per-step keys) without the final denoise; the reference
    tool records the same latents and draws the same per-step noise."""
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *x_T.shape[1:])), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    times = jnp.asarray(times, jnp.float32)

    def body(carry, inputs):
        x, state = carry
        t, t_next, index = inputs
        t = jnp.full((x.shape[0],), t)
        t_next = jnp.full((x.shape[0],), t_next)
        denoised, eps = denoise(x, t)
        x, state = solver.step(x, t, t_next, denoised, eps, state, jax.random.fold_in(key, index),
                               process, denoise)
        return (x, state), x

    _, latents = jax.lax.scan(body, (x_T, solver.init(x_T, times)),
                              (times[:-1], times[1:], jnp.arange(times.shape[0] - 1)))
    return latents


def relative_gap(actual, expected) -> float:
    """The largest entry of |actual - expected| over the larger of 1 and the
    largest entry of |expected|, per leading index."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    return max(np.abs(a - e).max() / max(1.0, np.abs(e).max()) for a, e in zip(actual, expected))


@pytest.mark.parametrize("name", sorted(DIFFUSERS_CASES))
def test_solver_matches_diffusers_latents_and_gradient(name):
    """Every latent of the walk, within 1e-4 of that latent's own scale
    (1 for the VP cases, up to 392 at the top of the k-diffusion ones), and
    the vector-Jacobian product of the final latent against the fixture's
    cotangent through the whole trajectory, within 1e-4 of its scale,
    against Diffusers 0.34.0 on the same grid, oracle and per-step noise.
    Observed at most 3.0e-5 on the Karras grid, whose float32 sigmas differ
    from the reference's float64 ones, 1.5e-5 for DEIS at third order, whose
    coefficients cancel logarithms, and 8.7e-6 elsewhere for the latents;
    1.3e-5 for the gradients."""
    solver = DIFFUSERS_CASES[name]
    process, model = reference_process(name)
    x_T = jnp.asarray(ARRAYS[f"{name}.x_T"])
    grid = ARRAYS[f"{name}.grid"]
    latents = walk(solver, process, model, x_T, grid)
    expected = ARRAYS[f"{name}.latents"]
    assert latents.shape == expected.shape
    assert relative_gap(latents, expected) < 1e-4
    _, vjp = jax.vjp(lambda x: walk(solver, process, model, x, grid)[-1], x_T)
    (grad,) = vjp(jnp.asarray(ARRAYS[f"{name}.cotangent"]))
    assert relative_gap(grad[None], ARRAYS[f"{name}.grad"][None]) < 1e-4


class Forgetful:
    """A multistep solver whose state is rebuilt before every step, so each
    step is the one it takes with no history."""

    def __init__(self, inner):
        self.inner = inner

    def init(self, x, times):
        return self.inner.init(x, times)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        stepped, _ = self.inner.step(x, t, t_next, denoised, eps, self.inner.init(x, jnp.stack([t[0], t_next[0]])),
                                     key, process, denoise)
        return stepped, state


@pytest.mark.parametrize("name", ["dpm_multistep.pp_3", "dpm_multistep.sde_2h", "dpm_singlestep.pp_3h_final",
                                  "deis.3", "unipc.bh1_3", "pndm.plms", "lms.4"])
def test_diffusers_fixtures_see_the_history(name):
    """The same solvers with their history discarded every step (first order
    throughout, UniPC without a corrector, PNDM's predictor-corrector pair
    at every step) miss the fixture's final latent by more than 1e-2 of its
    scale, a hundred times the match tolerance, so a dropped higher-order
    term cannot pass the match. Observed 4.8e-2 to 0.64."""
    solver = DIFFUSERS_CASES[name]
    process, model = reference_process(name)
    x_T = jnp.asarray(ARRAYS[f"{name}.x_T"])
    grid = ARRAYS[f"{name}.grid"]
    expected = ARRAYS[f"{name}.latents"]
    latents = walk(Forgetful(solver), process, model, x_T, grid)
    assert relative_gap(latents[-1:], expected[-1:]) > 1e-2


def test_singlestep_groups_need_a_step_count_the_order_divides():
    """Diffusers' order list without `lower_order_final` is `[1, 2, 3]`
    repeated, and a walk its length does not divide runs out of orders; Dew
    refuses at `init` where Diffusers fails at the last step, and the
    lowered list takes any count."""
    x = jnp.zeros((1, 4))
    with pytest.raises(ValueError, match="groups of 3 steps"):
        DPMSolverSinglestep(3).init(x, jnp.linspace(1.0, 0.0, 21))
    assert DPMSolverSinglestep(3, lower_order_final=True).init(x, jnp.linspace(1.0, 0.0, 21)).orders.tolist() \
        == [1, 2, 3] * 6 + [1, 2]
    assert DPMSolverSinglestep(3).init(x, jnp.linspace(1.0, 0.0, 22)).orders.tolist() == [1, 2, 3] * 7


LAMBDA_SOLVERS = [DPMSolverMultistep(3, "dpmsolver++", "heun"), DPMSolverMultistep(3, "dpmsolver"),
                  DPMSolverMultistep(2, "sde-dpmsolver++"), DPMSolverMultistep(2, "sde-dpmsolver"),
                  DPMSolverSinglestep(3, lower_order_final=True), DEIS(3), UniPC(3)]


@pytest.mark.parametrize("solver", LAMBDA_SOLVERS, ids=solver_id)
def test_lambda_solvers_land_on_the_clean_prediction_at_sigma_zero(solver):
    """A step onto sigma 0 is the h -> infinity limit of every first order
    update, x_t = alpha_t x_0, with no log of zero in the arithmetic and a
    finite gradient through the step. The flow path reaches sigma 0 at
    t = 0, so the last interval of any grid hits it; the check holds for
    the first step of a walk and for the second, with the first in its
    history."""
    process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
    model = ConstantVelocity()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 4)), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 4))
    key = jax.random.PRNGKey(0)

    def step(x, t, t_next, state):
        t, t_next = jnp.full((3,), t), jnp.full((3,), t_next)
        x_0, eps = denoise(x, t)
        return solver.step(x, t, t_next, x_0, eps, state, key, process, denoise)

    fresh, _ = step(x, 0.25, 0.0, solver.init(x, jnp.asarray([0.25, 0.0])))
    assert jnp.all(jnp.isfinite(fresh)) and jnp.allclose(fresh, denoise(x, jnp.full((3,), 0.25))[0], atol=1e-6)

    def two_steps(x):
        x, state = step(x, 0.5, 0.25, solver.init(x, jnp.asarray([0.5, 0.25, 0.0])))
        return step(x, 0.25, 0.0, state)[0], denoise(x, jnp.full((3,), 0.25))[0]

    with_history, x_0 = two_steps(x)
    assert jnp.all(jnp.isfinite(with_history)) and jnp.allclose(with_history, x_0, atol=1e-6)
    assert jnp.all(jnp.isfinite(jax.grad(lambda x: jnp.sum(two_steps(x)[0]))(x)))


def test_consistency_sampling_in_one_step_is_the_consistency_function():
    """LCM's one-step generation: with one interval the solver adds no noise
    and the boundary at t = 0 returns its input, so `sample` is f(x_T, T),
    c_skip x_T + c_out x_0 with x_0 the model's prediction at T."""
    schedule = CosineNoiseScheduler(1000)
    process = Process(schedule, ConsistencyBoundary(DirectPredictionTransform()))
    model = EndpointOracle(schedule=schedule)
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 4)), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    x_T = process.noise(jax.random.PRNGKey(2), (3, 4))
    top = jnp.full((3,), 1000.0)
    x_0 = model.apply(params, x_T, top)
    scaled = 1000.0 * 10.0
    expected = 0.25 / (scaled**2 + 0.25) * x_T + scaled / jnp.sqrt(scaled**2 + 0.25) * x_0
    assert jnp.allclose(denoise(x_T, top)[0], expected, atol=1e-6)
    generated = sample(denoise, x_T, 2, solver=Consistency(), key=jax.random.PRNGKey(3))
    assert jnp.allclose(generated, expected, atol=1e-6)


@pytest.mark.parametrize("solver", [DPMSolverMultistep(), UniPC(), DEIS(), KDPM2(), LMS()], ids=solver_id)
def test_diffusers_solvers_are_second_order_on_the_karras_ode(solver):
    """Twenty rho-spaced steps from sigma 80 down against the closed form,
    the bracket test_solvers_integrate_the_flow_ode_at_their_order sets:
    each lands within 8e-2 like Heun and closer than Euler. Observed 5.7e-2
    for DPM-Solver++ (2M), 4.2e-2 for UniPC, 2.6e-2 for DEIS, 2.9e-2 for
    KDPM2 and 4.2e-2 for LMS, against Euler's 1.5e-1 and Heun's 5.1e-2."""
    process, _ = karras_process()
    x_T = jax.random.normal(jax.random.PRNGKey(0), (64, 4)) * 80.0

    def error(solver):
        x, sigma = integrate(process, solver, x_T, steps=20)
        exact = x_T * jnp.sqrt(DATA_STD**2 + sigma**2) / jnp.sqrt(DATA_STD**2 + 80.0**2)
        return float(jnp.max(jnp.abs(x - exact)) / jnp.max(jnp.abs(exact)))

    measured, euler = error(solver), error(Euler())
    assert measured < 8e-2 and measured < euler, (measured, euler)


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
