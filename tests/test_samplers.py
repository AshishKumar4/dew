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
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffusion_support import walk
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference, assert_as_exact_over_orders, distance

from dew.diffusion import (
    ConsistencyBoundary,
    CosineNoiseScheduler,
    DirectPredictionTransform,
    EDMNoiseScheduler,
    EpsilonPredictionTransform,
    FlowMatchingScheduler,
    FlowMatchPredictionTransform,
    KarrasPredictionTransform,
    KarrasVENoiseScheduler,
    LinearNoiseScheduler,
    Process,
    broadcast_rates,
    expand,
)
from dew.diffusion.discrete import DiscreteProcess, LogLinear
from dew.registry import solvers
from dew.sampling import (
    CFG,
    DDIM,
    DDPM,
    DEIS,
    KDPM2,
    LMS,
    PNDM,
    RK4,
    TCD,
    Consistency,
    DPMSolverMultistep,
    DPMSolverSDE,
    DPMSolverSinglestep,
    Euler,
    Heun,
    MultiStepDPM,
    UniPC,
    sample,
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


def endpoint_process():
    schedule = CosineNoiseScheduler(1000)
    return Process(schedule, DirectPredictionTransform()), EndpointOracle(schedule=schedule)


class VEEndpointOracle(nn.Module):
    """EndpointOracle on the variance-exploding path, x_0 = x s / sqrt(s^2 +
    sigma^2), read at c_noise = log(sigma) / 4."""

    @nn.compact
    def __call__(self, x, temb):
        sigma = expand(jnp.exp(4.0 * temb), x)
        return x * DATA_STD / jnp.sqrt(DATA_STD**2 + sigma**2)


def ve_endpoint_process():
    schedule = KarrasVENoiseScheduler(sigma_max=80, rho=7, sigma_data=0.5)
    return Process(schedule, DirectPredictionTransform()), VEEndpointOracle()


def linear_process():
    schedule = LinearNoiseScheduler(1000)
    return Process(schedule, EpsilonPredictionTransform()), LinearVPOracle(schedule=schedule)


class FlowOracle(nn.Module):
    """The velocity E[eps - x_0 | x_t] for x_0 ~ N(0, DATA_STD^2) on the linear
    path x_t = (1 - t) x_0 + t eps, read at the model time 1000 t."""

    @nn.compact
    def __call__(self, x, temb):
        t = expand(temb / 1000, x)
        return x * (t - (1 - t) * DATA_STD**2) / ((1 - t) ** 2 * DATA_STD**2 + t**2)


def flow_process():
    return Process(FlowMatchingScheduler(), FlowMatchPredictionTransform()), FlowOracle()


TOKEN_PROBS = np.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.3])
"""The masked case's data: every token drawn from these alone, the id past them the mask."""


class TokenOracle(nn.Module):
    """A masked token's posterior when every token is drawn from TOKEN_PROBS
    alone: those probabilities, whatever the context. The time is the
    Gaussian oracles' signature, and unread."""

    @nn.compact
    def __call__(self, tokens, time=None):
        logits = jnp.log(jnp.append(jnp.asarray(TOKEN_PROBS, jnp.float32), 1.0))
        return jnp.broadcast_to(logits, (*tokens.shape, logits.shape[0]))


def masked_process():
    return DiscreteProcess(LogLinear(), mask_id=len(TOKEN_PROBS)), TokenOracle()


def assert_token_frequencies(samples, tol=0.03):
    counts = np.bincount(np.asarray(samples).ravel(), minlength=len(TOKEN_PROBS) + 1)
    assert counts[-1] == 0, "a sample kept the mask"
    np.testing.assert_allclose(counts[:-1] / counts.sum(), TOKEN_PROBS, atol=tol)


IMAGE = (8, 8, 3)
PROCESS_CASES = {
    "vp": (vp_process, IMAGE, assert_gaussian_stats),
    "linear": (linear_process, IMAGE, assert_gaussian_stats),
    "endpoint": (endpoint_process, IMAGE, assert_gaussian_stats),
    "karras": (karras_process, IMAGE, assert_gaussian_stats),
    "ve_endpoint": (ve_endpoint_process, IMAGE, assert_gaussian_stats),
    "flow": (flow_process, IMAGE, assert_gaussian_stats),
    "masked": (masked_process, (16,), assert_token_frequencies),
}
"""Each process family over an oracle of known data: the process and its
model, the sample shape, and the check a walk's samples pass."""

GAUSSIAN = ("vp", "karras")
"""The cases a solver that integrates any Gaussian process samples."""

FAMILIES = ("vp", "karras", "flow")
"""One case of each continuous process family: variance preserving,
variance exploding, rectified flow."""

INTEGRATES = {
    # Each integrates dx/dsigma = eps, which only holds when alpha is 1. On a
    # VP schedule the step would run and drift the samples narrow (0.275 std
    # for the 0.3 oracle on the cosine schedule, inside the tolerance), so
    # every other schedule is refused by type.
    **dict.fromkeys(("dpmsolver_sde", "euler_ancestral", "kdpm2", "lms", "multistep_dpm", "rk4"),
                    ("karras",)),
    "flow_sde": ("flow",),
    "unmask": ("masked",),
}
"""The cases a solver that integrates one process family samples; it refuses
every other family's (FAMILIES)."""

CONVERGES_ON = {
    # A consistency model is distilled to the endpoint (EndpointOracle).
    "consistency": ("endpoint", "ve_endpoint"),
    # From a posterior mean, TCD's intermediate point at (1 - eta) t_next
    # reaches the data on the VP table at eta 0.3; on the Karras grid that
    # time is far lower in sigma, and the samples narrow to std 0.24.
    "tcd": ("vp",),
    # PNDM's transfer scales x by alpha_s / alpha_t, which is 316 over the
    # first of a hundred intervals on the cosine table from its capped end
    # (alpha 4.9e-5 at index 999, against 2.1 from the index 990 Diffusers'
    # leading grid starts at) and turns the pseudo Runge-Kutta stages'
    # noise-level differences into a 23-fold blow-up. The linear table's
    # ratio there is 1.1, observed std 0.2986.
    "pndm": ("linear",),
}
"""The cases a solver that integrates any Gaussian process reaches the data
on, where that takes its own model or table."""

REFUSES = {
    # PNDM's transfer is DDIM's for a variance-preserving process: with alpha 1
    # its eps term vanishes, and with flow's rates it is another process's.
    "pndm": ("karras", "flow"),
}
"""The families a solver that reaches its data by `CONVERGES_ON` refuses."""


def assert_reaches_its_data(case, solver, steps=100):
    build, shape, reaches = PROCESS_CASES[case]
    reaches(generate(*build(), solver, steps, shape=shape))


@pytest.mark.parametrize("name", sorted(solvers))
def test_every_registered_solver_samples_its_processes_and_refuses_the_rest(name):
    """Each solver `dew.registry.solvers` names, at its defaults, walks every
    process case it integrates from the prior to the oracle's data, and a
    solver of one process family refuses every other continuous case."""
    solver = solvers.build(name)
    integrated = INTEGRATES.get(name)
    for case in integrated or CONVERGES_ON.get(name, GAUSSIAN):
        assert_reaches_its_data(case, solver)
    refused = REFUSES.get(name, ()) if integrated is None else sorted(set(FAMILIES) - set(integrated))
    for case in refused:
        build, shape, _ = PROCESS_CASES[case]
        with pytest.raises(ValueError):
            generate(*build(), solver, shape=shape)


@pytest.mark.parametrize("case, solver, steps", [
    ("vp", DDPM(), 1000),
    ("vp", DDIM(eta=0.5), 100),
    ("vp", DPMSolverMultistep(lower_order_final=False), 100),
    ("vp", DPMSolverMultistep(3, "dpmsolver"), 100),
    ("vp", DPMSolverMultistep(2, "sde-dpmsolver++"), 100),
    ("vp", DPMSolverSinglestep(3, lower_order_final=True), 100),
    ("vp", TCD(eta=0.0), 100),
    ("endpoint", TCD(eta=1.0), 8),
    ("ve_endpoint", TCD(eta=1.0), 8),
    ("linear", PNDM(skip_prk_steps=True), 100),
    ("karras", KDPM2(ancestral=True), 100),
    ("karras", DPMSolverMultistep(lower_order_final=False), 100),
], ids=lambda value: solver_id(value) if not isinstance(value, (str, int)) else str(value))
def test_a_solvers_settings_reach_the_data(case, solver, steps):
    """Settings past a solver's defaults, on the case their update reads:
    DDPM over its whole table, the other DPM-Solver algorithms and orders,
    TCD at its two ends, PLMS without the Runge-Kutta warmup, KDPM2's
    ancestral step. At eta 1 TCD goes through the schedule's clean end and
    is the consistency walk up to that end's residual noise, on a variance
    exploding grid too, where re-noising by 1 - ratio^2 added none (TCD at
    eta 0.3 drew std 0.0 against 0.3 on the Karras grid)."""
    assert_reaches_its_data(case, solver, steps)


FLAXDIFF = dict(np.load(Path(__file__).resolve().parent / "fixtures" / "flaxdiff_solvers" / "solvers.npz"))
FLAXDIFF_ROUNDING = np.load(
    Path(__file__).resolve().parent / "fixtures" / "flaxdiff_solvers" / "rounding.npz")


@dataclass(frozen=True)
class StandIn:
    """The FlaxDiff fixture's stand-in denoiser, the Gaussian-data optimum
    bent by a tanh, read through the process's own sigma."""

    process: Process

    def __call__(self, x, t):
        _, sigma = self.process.rates(t, like=x)
        bent = 0.25 * x + sigma * 0.5 * jnp.tanh(x / 0.5)
        x_0 = bent / (0.25 + sigma**2)
        return x_0, (x - x_0) / sigma


def flaxdiff_walk(solver, x_T):
    """Every interval of the fixture's grid from `x_T`, with the reference's
    stand-in denoiser read through the Karras schedule's own sigma."""
    process, _ = karras_process()
    times = jnp.asarray(FLAXDIFF["times"])
    denoise = StandIn(process)

    x, state = x_T, solver.init(x_T, times, process, key=jax.random.PRNGKey(0))
    latents = []
    for i in range(times.shape[0] - 1):
        t, t_next = jnp.full((x.shape[0],), times[i]), jnp.full((x.shape[0],), times[i + 1])
        denoised, eps = denoise(x, t)
        x, state = solver.step(x, t, t_next, denoised, eps, state, jax.random.PRNGKey(i), process, denoise)
        latents.append(x)
    return jnp.stack(latents)


@pytest.mark.parametrize("name,solver", [("rk4", RK4()), ("multistep", MultiStepDPM())],
                         ids=["rk4", "multistep"])
def test_the_closing_denoise_is_flaxdiffs_generate_samples(name, solver):
    """`sample` with its default `final_denoise` against FlaxDiff's own
    `generate_samples` (AshishKumar4/FlaxDiff@15c55b0, run as published by
    tools/flaxdiff_solver_reference.py) from the same x_T over the same
    twelve-point grid: eleven solver intervals, then the clean prediction
    at the grid's last point. The output is held to FlaxDiff's float64 run
    by the float64 rule, and, through the clip to [-1, 1] that FlaxDiff's
    `post_process` applies as the pipeline's output step, to that run's
    clipped images: the stand-in's clean predictions reach |x| = 3, so the
    clip moves most entries and the unclipped comparison is the one that
    reads the walk. RK4's clipped comparison averages 8192 independent
    initial entries, since its pointwise stages give identical rounding
    under pixel permutations. In a 64-copy pilot, every disjoint four-copy
    subset stayed below ratio 0.98 on default, AVX and Ice Lake CPU codegen;
    the chosen four copies reach 0.92. Its oracle widens FlaxDiff's timestep
    casts too, so sigma arithmetic runs in float64."""
    process, _ = karras_process()
    closed = sample(StandIn(process), jnp.asarray(FLAXDIFF["x_T"]), 12, solver=solver, key=jax.random.key(0))
    assert_as_exact_as_the_reference(closed, FLAXDIFF[f"{name}/closed"], FLAXDIFF[f"{name}/closed_f64"],
                                     f"{name} closed")
    assert float(np.abs(FLAXDIFF[f"{name}/closed_f64"]).max()) > 2
    if name == "rk4":
        def clipped(value):
            return jnp.clip(sample(StandIn(process), value, 12, solver=solver, key=jax.random.key(0)), -1, 1)
        actual = jax.jit(clipped)(jnp.asarray(FLAXDIFF_ROUNDING["x_T"]))
        assert_as_exact_as_the_reference(actual, FLAXDIFF_ROUNDING["clipped"],
                                         FLAXDIFF_ROUNDING["clipped_f64"], "rk4 clipped")
    else:
        assert_as_exact_as_the_reference(jnp.clip(closed, -1, 1), FLAXDIFF[f"{name}/clipped"],
                                         FLAXDIFF[f"{name}/clipped_f64"], f"{name} clipped")


@pytest.mark.parametrize("name,solver", [("rk4", RK4()), ("multistep", MultiStepDPM())],
                         ids=["rk4", "multistep"])
def test_rk4_and_the_multistep_integrator_are_flaxdiffs(name, solver):
    """`RK4` and `MultiStepDPM` are FlaxDiff's `RK4Sampler` and
    `MultiStepDPM` (AshishKumar4/FlaxDiff@15c55b0, run as published by
    tools/flaxdiff_solver_reference.py): from the same x_T over FlaxDiff's
    twelve-point grid on a rho-7 Karras schedule, with a denoiser nonlinear
    in x, the latent after every interval, so every RK4 stage and each of the
    multistep's first, second and third order updates, and the gradient of
    the last one through the walk, held to FlaxDiff's float64 run by the
    float64 rule. The grid is Dew's own `times(12)`."""
    process, _ = karras_process()
    np.testing.assert_array_equal(process.times(12), FLAXDIFF["times"])
    x_T = jnp.asarray(FLAXDIFF["x_T"])
    latents = flaxdiff_walk(solver, x_T)
    for step in range(latents.shape[0]):
        assert_as_exact_as_the_reference(latents[step], FLAXDIFF[f"{name}/latents"][step],
                                         FLAXDIFF[f"{name}/latents_f64"][step],
                                         f"{name} after interval {step}")
    (gradient,) = jax.vjp(lambda x: flaxdiff_walk(solver, x)[-1], x_T)[1](jnp.asarray(FLAXDIFF["cotangent"]))
    assert_as_exact_as_the_reference(gradient, FLAXDIFF[f"{name}/grad"], FLAXDIFF[f"{name}/grad_f64"],
                                     f"{name} gradient")


@pytest.mark.parametrize("solver", [DDIM(), Euler()], ids=lambda s: type(s).__name__)
def test_video_samples_converge(solver):
    """The frame axis is one more sample axis: the rates broadcast over it and
    the statistics come out the same as for images."""
    process, model = karras_process()
    assert_gaussian_stats(generate(process, model, solver, steps=50, count=64, shape=(3, 8, 8, 3)))


def test_a_key_reproduces_a_trajectory():
    """Nothing lives on a solver between calls: a multistep solver's history
    travels in its state, so the same key gives the same samples twice."""
    process, model = karras_process()
    first = generate(process, model, MultiStepDPM())
    second = generate(process, model, MultiStepDPM())
    assert jnp.allclose(first, second, atol=1e-5)


############################################################################################################
# The Diffusers 0.34.0 schedulers, from the fixtures tools/diffusers_reference.py records
############################################################################################################

DIFFUSERS_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "diffusers"
REFERENCE = json.loads((DIFFUSERS_FIXTURES / "schedulers.json").read_text())
ARRAYS = np.load(DIFFUSERS_FIXTURES / "schedulers.npz")

# Each fixture case's Dew solver. The scheduler, its config and the model
# convention a case was recorded under are in schedulers.json.
DIFFUSERS_CASES = {
    "dpm_multistep.pp_2m": DPMSolverMultistep(2, "dpmsolver++", "midpoint", lower_order_final=False),
    "dpm_multistep.pp_2m_short_no_taper": DPMSolverMultistep(lower_order_final=False),
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
    "dpm_singlestep.pp_2_uneven": DPMSolverSinglestep(2),
    "dpm_singlestep.pp_3_uneven": DPMSolverSinglestep(3, solver_type="heun"),
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


def relative_gap(actual, expected) -> float:
    """The largest entry of |actual - expected| over the larger of 1 and the
    largest entry of |expected|, per leading index."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    return max(np.abs(a - e).max() / max(1.0, np.abs(e).max()) for a, e in zip(actual, expected, strict=True))


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


LIMIT_CASES = {
    "source.multi_pp3": DPMSolverMultistep(3, lower_order_final=False),
    "source.multi_eps3": DPMSolverMultistep(3, "dpmsolver", lower_order_final=False),
    "source.multi_sdepp3": DPMSolverMultistep(3, "sde-dpmsolver++", lower_order_final=False),
    "source.single_pp2m": DPMSolverSinglestep(2),
    "source.single_pp2h": DPMSolverSinglestep(2, solver_type="heun"),
    "source.single_pp3m": DPMSolverSinglestep(3),
    "source.single_pp3h": DPMSolverSinglestep(3, solver_type="heun"),
    "source.single_sdepp2h": DPMSolverSinglestep(2, "sde-dpmsolver++", "heun"),
    "source.single_sdepp3m": DPMSolverSinglestep(3, "sde-dpmsolver++"),
    "source.single_sdepp3h_final_zero": DPMSolverSinglestep(3, "sde-dpmsolver++", "heun"),
    "source.deis2": DEIS(2, lower_order_final=False),
    "source.deis3": DEIS(3, lower_order_final=False),
    "source.unipc_x0_bh2": UniPC(3, lower_order_final=False),
    "source.unipc_x0_bh1_disabled": UniPC(3, "bh1", lower_order_final=False, disable_corrector=(0,)),
    "source.unipc_eps_bh2_disabled": UniPC(
        3, predict_x0=False, lower_order_final=False, disable_corrector=(0,)
    ),
    "source.unipc_eps_bh1_disabled": UniPC(
        3, "bh1", predict_x0=False, lower_order_final=False, disable_corrector=(0,)
    ),
    "target.deis2": DEIS(2, lower_order_final=False),
    "target.deis3": DEIS(3, lower_order_final=False),
    "target.unipc_eps3": UniPC(3, predict_x0=False),
    "target.unipc_x03": UniPC(3),
    "target.sde_eps1": DPMSolverMultistep(1, "sde-dpmsolver"),
    "target.sde_eps2_lowered": DPMSolverMultistep(2, "sde-dpmsolver"),
    "target.kdpm2_false": KDPM2(),
    "target.kdpm2_true": KDPM2(ancestral=True),
}


class NonlinearVelocity(nn.Module):
    @nn.compact
    def __call__(self, x, temb):
        return 0.7 * jnp.tanh(x) + 0.15 * expand(temb / 1000, x)


class NonlinearEpsilon(nn.Module):
    @nn.compact
    def __call__(self, x, temb):
        return 0.2 * jnp.tanh(x)


@pytest.mark.parametrize("name", LIMIT_CASES)
def test_solver_endpoint_latents_and_vjp_match_reference_limits(name):
    """Nonlinear endpoint trajectories, including corrected UniPC epsilon
    and DEIS history terms at sigma=0. tools/diffusers_limits_reference.py
    records actual zero-endpoint calls where defined, otherwise converged
    reference-equation limits rather than assuming the output is clean.
    """
    with np.load(DIFFUSERS_FIXTURES / "limits.npz") as fixture:
        x = jnp.asarray(fixture[f"{name}.x_T"])
        times = fixture[f"{name}.grid"]
        expected, cotangent, expected_grad = (
            fixture[f"{name}.{field}"] for field in ("latents", "cotangent", "grad")
        )
    if name.startswith("target.kdpm2"):
        config = json.loads((DIFFUSERS_FIXTURES / "limits.json").read_text())["cases"][name]
        process = Process(KarrasVENoiseScheduler(sigma_min=0, sigma_max=config["sigma"], rho=1),
                          EpsilonPredictionTransform())
        model = NonlinearEpsilon()
    else:
        process = Process(FlowMatchingScheduler(), FlowMatchPredictionTransform())
        model = NonlinearVelocity()
    run = jax.jit(lambda initial: walk(LIMIT_CASES[name], process, model, initial, times))
    result = run(x)
    _, vjp = jax.vjp(lambda initial: run(initial)[-1], x)
    gradient, = vjp(jnp.asarray(cotangent))
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(gradient, expected_grad, rtol=3e-6, atol=3e-6)


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


def test_empty_guidance_interval_generates_the_unguided_samples():
    process, denoise = guided_denoiser(count=16)
    x_T = process.noise(jax.random.PRNGKey(2), (16, 8, 8, 3))

    def run(guidance):
        return sample(denoise, x_T, 25, solver=DDIM(), guidance=guidance,
                      key=jax.random.PRNGKey(4))

    assert jnp.allclose(run(CFG(3.0, interval=(0.9, 0.1))), run(None), atol=1e-5)


############################################################################################################
# Published scheduler files, reconstructed by SourceSchedule from their own configs
############################################################################################################

SOURCE = json.loads((DIFFUSERS_FIXTURES / "source_schedulers.json").read_text())
SOURCE_ARRAYS = np.load(DIFFUSERS_FIXTURES / "source_schedulers.npz")
SOURCE_ROUNDING = np.load(DIFFUSERS_FIXTURES / "source_rounding.npz")


class SourceOracle(nn.Module):
    """The model `tools/diffusers_source_reference.py` runs on both sides: a
    bounded nonlinear function of the scaled input and the model time, so a
    trajectory and its Jacobian depend on every scheduler decision the config
    carries and on nothing else."""

    @nn.compact
    def __call__(self, x, temb):
        return jnp.sin(x) * 0.07 + expand(temb, x) * 0.001


class ConditionalSourceOracle(nn.Module):
    """The same model offset by its condition, so a guided walk reads back the
    combination that was applied."""

    @nn.compact
    def __call__(self, x, temb, label):
        return jnp.sin(x) * 0.07 + expand(temb, x) * 0.001 + expand(label, x)


def source_case(name: str):
    """The reconstructed process, grid and solver of a fixture case, with the
    initial latent the reference walked from."""
    from dew.diffusion.schedules.source import SourceSchedule

    config = json.loads(str(SOURCE_ARRAYS[f"{name}.config"]))
    schedule = SourceSchedule.from_config(config)
    process, times = schedule.sampling(SOURCE["cases"][name]["steps"])
    return schedule, process, times, jnp.asarray(SOURCE_ARRAYS[f"{name}.x_T"])


def source_gradient_over_orders(final, initial, cotangent) -> None:
    """Pixel permutations move CFG's variance reductions alone.

    The pointwise oracle and the per-sample quantile commute with the same
    permutation of input and cotangent. The generating tool checks the
    float64 VJP's invariance and records each reference distance. A single
    draw concentrates on the few unclipped entries; 52 orders average the
    reference's reduction roundings before the factor-two comparison. The
    measured RMS ratios are 0.90/0.60/0.90 on default/AVX/Ice Lake CPU codegen.
    """
    gradient = jax.jit(lambda value, cot: jax.vjp(final, value)[1](cot)[0])
    truth = SOURCE_ARRAYS["cfg.epsilon.grad"]
    distances = []
    for order in SOURCE_ROUNDING["orders"]:
        def moved(value, order=order):
            return np.asarray(value).reshape(initial.shape[0], -1)[:, order].reshape(initial.shape)
        actual = gradient(jnp.asarray(moved(initial)), jnp.asarray(moved(cotangent)))
        distances.append(distance(actual, moved(truth)))
    assert_as_exact_over_orders(distances, SOURCE_ROUNDING["cfg.epsilon.distances"], "cfg.epsilon")


@pytest.mark.parametrize("name", sorted(name for name in SOURCE["cases"] if not name.startswith("dpm_sde.")))
def test_source_config_rebuilds_its_scheduler_trajectory_and_gradient(name):
    """Every published scheduler file the reference tool saved, rebuilt from
    that file alone: its beta table, the standard deviation its
    `init_noise_sigma` draws, the model time at each grid point, the latent
    after every interval and the vector-Jacobian product of the last one
    through the whole trajectory.

    The reference calls the actual scheduler class, saves its config before
    `set_timesteps` and walks every call a pipeline makes, so a spacing,
    sigma transformation, terminal sigma, clipping order or stage placement
    that this reconstruction read differently would move the trajectory.
    A stochastic class integrates the exact draws `sample` folds per step, and
    `DPMSolverSDEScheduler`'s walk integrates the draws its own `torchsde`
    tree answered, whose interval the reconstruction is checked to prepare.
    Every native walk and VJP runs in float32.

    The VJP is held to twice the source's own RMS distance from its
    float64 evaluation, using reference_error's common rounding rule.
    The source fixture widens every scheduler table and retains that
    gradient in float64, so its oracle shares no float32 arithmetic with
    the reference whose rounding it measures. Five pointwise cases average
    32 independent copies (768 gradient entries) in source_rounding.npz:
    pixel permutations reproduce the same draw on both sides. A 256-copy
    pilot's disjoint original-size subsets reached ratio 3.52, while every
    32-copy subset stayed below 1.40 on default/AVX/Ice Lake CPU codegen.
    The chosen new draws reach 1.43. Thresholding's maximum is 1 in these
    cases, so the quantile is clamped to a constant and the limit is a
    pointwise clip. CFG epsilon instead averages 52 pixel orders of its
    per-sample variance reductions.
    """
    schedule, process, times, x_T = source_case(name)
    rescale = SOURCE["cases"][name]["guidance"]
    if f"{name}.betas" in SOURCE_ARRAYS:
        np.testing.assert_allclose(schedule.betas, SOURCE_ARRAYS[f"{name}.betas"],
                                   atol=1e-5, rtol=1e-5)
    expected_times = SOURCE_ARRAYS[f"{name}.grid_times"]
    model_times = np.asarray(process.sampler_schedule.model_time(times[:-1]))
    assert model_times.shape == expected_times.shape
    np.testing.assert_allclose(model_times, expected_times, atol=1e-3, rtol=1e-6)
    prior = float(process.sampler_schedule.prior_scale())
    assert abs(prior - float(SOURCE_ARRAYS[f"{name}.prior"])) < 1e-5 * max(prior, 1.0)
    expected = SOURCE_ARRAYS[f"{name}.latents"]
    cotangent = jnp.asarray(SOURCE_ARRAYS[f"{name}.cotangent"])
    if rescale is None:
        model = SourceOracle()
        params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *x_T.shape[1:])), jnp.ones((1,)))
        denoise = process.denoiser(model, params, {})
        solver = schedule.solver
        def run(value):
            return walk(solver, process, model, value, times)
        latents = run(x_T)
        assert latents.shape == expected.shape
        assert relative_gap(latents, expected) < 1e-4
        def final(value):
            return run(value)[-1]
    else:
        model = ConditionalSourceOracle()
        rows = x_T.shape[0]
        params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *x_T.shape[1:])),
                            jnp.ones((1,)), jnp.ones((1,)))
        denoise = process.denoiser(model, params, {"label": jnp.full((rows,), SOURCE["label"])},
                                   {"label": jnp.zeros((rows,))})
        guidance = CFG(SOURCE["guidance_scale"], rescale=rescale)

        def final(value):
            return sample(denoise, value, solver=schedule.solver, guidance=guidance,
                          key=jax.random.PRNGKey(0), times=times, final_denoise=False)

        assert relative_gap(final(x_T)[None], expected[-1:]) < 1e-4
    if name == "cfg.epsilon":
        source_gradient_over_orders(final, x_T, cotangent)
    elif f"{name}.x_T" in SOURCE_ROUNDING:
        gradient = jax.jit(lambda value, cot: jax.vjp(final, value)[1](cot)[0])(
            jnp.asarray(SOURCE_ROUNDING[f"{name}.x_T"]), jnp.asarray(SOURCE_ROUNDING[f"{name}.cotangent"]))
        assert_as_exact_as_the_reference(
            gradient, SOURCE_ROUNDING[f"{name}.grad_float32"], SOURCE_ROUNDING[f"{name}.grad"], name)
    else:
        (gradient,) = jax.vjp(final, x_T)[1](cotangent)
        assert_as_exact_as_the_reference(
            gradient, SOURCE_ARRAYS[f"{name}.grad_float32"], SOURCE_ARRAYS[f"{name}.grad"], name)


@pytest.mark.parametrize("config", [
    {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 3.0},
    "euler.default", "kdpm2.default", "dpm_multi.default"],
    ids=lambda config: config if isinstance(config, str) else "flow.shift3")
def test_one_row_walks_each_source_grid_as_it_walks_in_a_batch(config):
    """A prompt alone walks where its row of a batch walks, on the flow, sigma,
    stage and VP grids.

    One row is the walk XLA compiles differently: each grid read becomes a
    dynamic slice rather than a gather. On a GPU (the cuda lane) XLA's
    DynamicSliceAnnotator failed that compile while the lookups searched
    the grid with jnp.interp's bisection loop inside the scan, which is how
    every published flow-matching checkpoint first failed on one prompt.
    """
    from dew.diffusion.schedules.source import SourceSchedule

    if isinstance(config, str):
        config = json.loads(str(SOURCE_ARRAYS[f"{config}.config"]))
    schedule = SourceSchedule.from_config(config)
    process, times = schedule.sampling(6)
    model = SourceOracle()
    x_T = jax.random.normal(jax.random.PRNGKey(3), (2, 3, 4))
    denoise = process.denoiser(model, model.init(jax.random.PRNGKey(1), x_T[:1], jnp.ones((1,))), {})
    run = jax.jit(lambda value: sample(denoise, value, solver=schedule.solver, key=jax.random.PRNGKey(0),
                                       times=times, final_denoise=False))
    batch = run(x_T)
    for row in range(2):
        np.testing.assert_allclose(run(x_T[row:row + 1])[0], batch[row], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kind, extra", [
    ("EulerDiscrete", {"sigma_min": 30000, "sigma_max": 80000}),
    ("KDPM2Discrete", {}),
    ("DPMSolverMultistep", {"solver_order": 1}),
])
def test_source_grids_walk_repeated_initial_model_times(kind, extra):
    """Repeated model times keep their own sigma rows, as an explicitly indexed source walk does:
    each walked latent is as near the source's float64 walk as its own float32 walk is. The float64
    walk builds its tables in float64 (`float64_scheduler`), at the float32 walk's integer times."""
    import contextlib

    import torch
    from diffusers.schedulers import (
        scheduling_dpmsolver_multistep,
        scheduling_euler_discrete,
        scheduling_k_dpm_2_discrete,
    )
    from tool_support import load

    from dew.diffusion.schedules.source import SourceSchedule

    modules = {"EulerDiscrete": scheduling_euler_discrete,
               "KDPM2Discrete": scheduling_k_dpm_2_discrete,
               "DPMSolverMultistep": scheduling_dpmsolver_multistep}
    module = modules[kind]
    config = {"_class_name": kind + "Scheduler", "beta_schedule": "squaredcos_cap_v2",
              "use_karras_sigmas": True, **extra}
    schedule = SourceSchedule.from_config(config)
    process, times = schedule.sampling(6)
    model_times = np.asarray(process.sampler_schedule.model_time(times[:-1]))
    assert np.count_nonzero(model_times == model_times[0]) > 1
    sigmas = process.sampler_schedule.sigmas(times)
    assert np.all(np.diff(sigmas) < 0)
    initial = np.random.default_rng(7).normal(size=(2, 3, 64)).astype(np.float32)
    model = SourceOracle()
    denoise = process.denoiser(model, {}, {})
    _, walked = sample(denoise, jnp.asarray(initial), solver=schedule.solver, key=0,
                       times=times, final_denoise=False, record=lambda x, _: x)

    precision = load("diffusers_wan_reference")
    reference = []
    published = None
    for dtype in (torch.float32, torch.float64):
        context = precision.float64_scheduler(module) if dtype == torch.float64 else contextlib.nullcontext()
        with context:
            source = getattr(module, kind + "Scheduler").from_config(config)
            source.set_timesteps(6)
            # A pipeline can choose its first sigma by index, without a model-time lookup.
            source.set_begin_index(0)
            if published is None:
                published = source.timesteps.clone()
            elif not published.is_floating_point():
                # Integer conditioning times are the published ones, not a float64 rounding of them.
                source.timesteps = published
            x = torch.tensor(initial, dtype=dtype)
            latents = []
            for index, time in enumerate(source.timesteps):
                scaled = source.scale_model_input(x, time)
                prediction = torch.sin(scaled) * 0.07 + time * 0.001
                x = source.step(prediction, time, x).prev_sample
                if kind != "KDPM2Discrete" or index % 2 == 1 or index == len(source.timesteps) - 1:
                    latents.append(x.numpy().copy())
        reference.append(np.stack(latents))
    for index, actual in enumerate(walked):
        assert_as_exact_as_the_reference(actual, reference[0][index], reference[1][index],
                                         f"{kind} repeated model time, step {index}")


@pytest.mark.parametrize("kind", ["UniPCMultistep", "DPMSolverMultistep"])
def test_a_spacing_too_fine_for_its_table_is_refused_by_the_grid(kind):
    """A multistep source's four leading-spaced steps over a four-entry table give two of them one
    sigma, an interval of no length that its coefficients divide by: the grid refuses it, where the
    walk would otherwise end in NaN. Three steps over the same table walk distinct sigmas."""
    from dew.diffusion.schedules.source import SourceSchedule

    config = {"_class_name": kind + "Scheduler", "num_train_timesteps": 4, "timestep_spacing": "leading"}
    with pytest.raises(ValueError):
        SourceSchedule.from_config(config).sampling(4)
    schedule = SourceSchedule.from_config(config)
    process, times = schedule.sampling(3)
    model = SourceOracle()
    x_T = jax.random.normal(jax.random.PRNGKey(3), (2, 3, 4))
    denoise = process.denoiser(model, model.init(jax.random.PRNGKey(1), x_T[:1], jnp.ones((1,))), {})
    walked = sample(denoise, x_T, solver=schedule.solver, key=jax.random.PRNGKey(0), times=times,
                    final_denoise=False)
    assert np.all(np.isfinite(np.asarray(walked)))


def test_a_source_noise_sampler_seed_pins_the_brownian_path():
    """`noise_sampler_seed` makes the tree's entropy the checkpoint's, so two
    walks with different keys integrate one path; without it the walk's own
    key decides, as the source's unseeded tree draws its entropy per run."""

    _, process, times, x_T = source_case("dpm_sde.default")
    model = SourceOracle()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *x_T.shape[1:])), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})

    def run(solver, seed):
        return sample(denoise, x_T, solver=solver, key=jax.random.PRNGKey(seed), times=times,
                      final_denoise=False)

    pinned = DPMSolverSDE(seed=7)
    np.testing.assert_array_equal(run(pinned, 0), run(pinned, 1))
    free = DPMSolverSDE()
    assert not np.allclose(run(free, 0), run(free, 1), atol=1e-3)


############################################################################################################
# EDM's own stochastic sampler
############################################################################################################

EDM = np.load(Path(__file__).resolve().parent / "fixtures" / "edm" / "heun.npz")


class EDMOracle(nn.Module):
    """The closed-form D(x; sigma) `tools/edm_reference.py` runs NVlabs'
    `edm_sampler` over; the model time on the test grid is sigma itself."""

    @nn.compact
    def __call__(self, x, sigma):
        sigma = expand(sigma, x)
        return jnp.tanh(x / jnp.sqrt(1 + sigma ** 2)) * 0.5 + 0.1 * jnp.sin(sigma)


def edm_grid(steps: int) -> Process:
    """NVlabs' rho-7 grid from 80 to 0.002 with its appended zero, as a
    paired table whose model time is sigma."""
    from dew.diffusion.schedules.source_grids import SigmaGrid

    ramp = np.arange(steps, dtype=np.float64) / (steps - 1)
    sigmas = np.append((80.0 ** (1 / 7) + ramp * (0.002 ** (1 / 7) - 80.0 ** (1 / 7))) ** 7, 0.0)
    return Process(SigmaGrid(sigmas, sigmas, 80.0), DirectPredictionTransform())


@pytest.mark.parametrize("name", ["deterministic", "churn", "churn_to_the_end"])
def test_heun_walks_edms_algorithm_2_with_its_churn(name):
    """Heun with S_churn, S_tmin, S_tmax and S_noise against the published
    sampler on its own rho-7 grid and the same per-step draws, as exact as
    the published sampler's own float32 walk."""
    arguments = json.loads(str(EDM[f"{name}.arguments"]))
    steps = arguments["num_steps"]
    solver = Heun(s_churn=arguments.get("S_churn", 0.0), s_tmin=arguments.get("S_min", 0.0),
                  s_tmax=arguments.get("S_max", float("inf")), s_noise=arguments.get("S_noise", 1.0))
    x_T = jnp.asarray(EDM["latents"] * np.float32(80.0))
    walked = sample(edm_grid(steps).denoiser(EDMOracle(), {}, {}), x_T, solver=solver,
                    key=jax.random.PRNGKey(7), times=np.arange(steps, -1, -1, dtype=np.float32),
                    final_denoise=False)
    assert_as_exact_as_the_reference(walked, EDM[f"{name}.result32"], EDM[f"{name}.result"], name)


@pytest.mark.parametrize("name", ["deterministic", "churn"])
def test_the_closing_denoise_is_edms_last_euler_step_onto_sigma_zero(name):
    """NVlabs' sampler ends with an Euler step from its last sigma onto the
    zero it appends, which lands on the model's clean prediction at that
    sigma. `sample` stopped at the last sigma closes with that prediction
    (`final_denoise`, its default) and matches the published sampler as
    exactly as its own float32 walk; the churn case's last sigma, 0.002, is
    below its S_min, so nothing is churned there."""
    arguments = json.loads(str(EDM[f"{name}.arguments"]))
    steps = arguments["num_steps"]
    solver = Heun(s_churn=arguments.get("S_churn", 0.0), s_tmin=arguments.get("S_min", 0.0),
                  s_tmax=arguments.get("S_max", float("inf")), s_noise=arguments.get("S_noise", 1.0))
    x_T = jnp.asarray(EDM["latents"] * np.float32(80.0))
    closed = sample(edm_grid(steps).denoiser(EDMOracle(), {}, {}), x_T, solver=solver,
                    key=jax.random.PRNGKey(7), times=np.arange(steps, 0, -1, dtype=np.float32))
    assert_as_exact_as_the_reference(closed, EDM[f"{name}.result32"], EDM[f"{name}.result"], name)
