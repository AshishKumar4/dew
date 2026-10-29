"""Sampler tests against an analytic oracle denoiser.

For gaussian data x0 ~ N(0, s^2 I), the optimal denoiser has a closed form:
E[x0 | x_t] = alpha * s^2 / (alpha^2 s^2 + sigma^2) * x_t. A solver
integrating the reverse process with this oracle must produce samples with
mean 0 and std s, and on a variance exploding schedule the probability flow
ODE it integrates, dx/dsigma = x sigma / (s^2 + sigma^2), has the closed
form x(sigma) = x(sigma_max) sqrt(s^2 + sigma^2) / sqrt(s^2 + sigma_max^2).
Each solver's order of accuracy is measured against that closed form.
"""

import dataclasses
import functools
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffusion_support import SourceOracle, relative_gap, walk
from flax import linen as nn
from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording
from tool_support import run_from

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
    schedule: CosineNoiseScheduler | LinearNoiseScheduler

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
    return Process(schedule, EpsilonPredictionTransform()), VPOracle(schedule=schedule)


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

FLOATING = ("vp", "linear", "endpoint", "karras", "ve_endpoint", "flow")
"""The cases whose state is continuous, a float array a run holds in its precision."""

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


CONFIDENCE = {"confident": "masked"}
"""The solvers that reveal by confidence, a release's own heuristic, by the case they walk: each draws a
position's token from its distribution but reveals the likeliest draws first, which favors likely tokens
by design, so at its defaults it walks its case to a full reveal within the oracle's support, not to the
oracle's frequencies (its parity is with its release's walk, tests/test_blockwise_walk.py); with no
threshold every masked position reveals from its first draw, which samples the oracle exactly."""


def assert_reaches_its_data(case, solver, steps=100):
    build, shape, reaches = PROCESS_CASES[case]
    reaches(generate(*build(), solver, steps, shape=shape))


@pytest.mark.parametrize("name", sorted(solvers))
def test_every_registered_solver_samples_its_processes_and_refuses_the_rest(name):
    """Each solver `dew.registry.solvers` names, at its defaults, walks every
    process case it integrates from the prior to the oracle's data, a
    confidence reveal its case to a full reveal (`CONFIDENCE`), and a
    solver of one process family refuses every other continuous case."""
    solver = solvers.build(name)
    if name in CONFIDENCE:
        build, shape, _ = PROCESS_CASES[CONFIDENCE[name]]
        counts = np.bincount(np.asarray(generate(*build(), solver, 100, shape=shape)).ravel(),
                             minlength=len(TOKEN_PROBS) + 1)
        assert counts[-1] == 0 and counts.sum() > 0, counts
        assert_reaches_its_data(CONFIDENCE[name], dataclasses.replace(solver, threshold=0.0))
    integrated = INTEGRATES.get(name, (CONFIDENCE[name],) if name in CONFIDENCE else None)
    for case in () if name in CONFIDENCE else integrated or CONVERGES_ON.get(name, GAUSSIAN):
        assert_reaches_its_data(case, solver)
    refused = REFUSES.get(name, ()) if integrated is None else sorted(set(FAMILIES) - set(integrated))
    for case in refused:
        build, shape, _ = PROCESS_CASES[case]
        with pytest.raises(ValueError):
            generate(*build(), solver, shape=shape)


@pytest.mark.parametrize("name, case", [
    (name, case) for name in sorted(solvers) if name not in CONFIDENCE
    for case in INTEGRATES.get(name, CONVERGES_ON.get(name, GAUSSIAN)) if case in FLOATING])
def test_every_registered_solver_walks_a_float64_run_in_float64(name, case):
    """Each solver `dew.registry.solvers` names walks a float64 state over float64 times through
    `sample`'s scan in float64 on every continuous case it samples, a multistep history included
    (ledger 2063: their histories were float32 and the scan refused them), so its float64 walk is check
    1's truth for its float32 one: from the same prior draw and key, not the float32 walk widened."""
    build, shape, _ = PROCESS_CASES[case]
    process, model = build()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *shape)), jnp.ones((1,)))
    x_T = np.asarray(process.noise(jax.random.PRNGKey(0), (4, *shape)))

    denoise = process.denoiser(model, params, {})

    def walked(dtype):
        return sample(denoise, jnp.asarray(x_T, dtype), solver=solvers.build(name), key=jax.random.PRNGKey(1),
                      times=np.asarray(process.times(8), dtype))

    narrow = walked(np.float32)
    with jax.enable_x64(new_val=True):
        wide = walked(np.float64)
    assert (narrow.dtype, wide.dtype) == (jnp.float32, jnp.float64)
    assert np.all(np.isfinite(wide)) and not np.array_equal(np.asarray(wide), np.asarray(narrow, np.float64))


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


FLAXDIFF = Path(__file__).resolve().parent / "fixtures" / "flaxdiff_solvers" / "solvers.npz"


@functools.cache
def writer(name: str):
    """tools/`name`.py, whose seeded draw a recording's inputs are drawn again by, loaded as a script
    from its directory."""
    with run_from(name, "tools") as module:
        return module


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


def flaxdiff_start(solver, x_T):
    """`solver`'s state before its first step, built on Dew's twelve-point grid on the host, where a
    multistep solver checks its gaps: what `sample` takes beside a traced grid (`initial`). A draw's
    grid is that grid moved within a hundredth of a spacing (the writer's JITTER), which these solvers'
    states start no different on."""
    process, _ = karras_process()
    with jax.ensure_compile_time_eval():
        # The grid in the walk's precision, as `flaxdiff_walk` and `flaxdiff_closed` cast it, so the state
        # starts in the dtype the walk carries it in.
        return solver.init(x_T, np.asarray(process.times(12), x_T.dtype), process, key=jax.random.key(0))


def flaxdiff_walk(solver, x_T, times):
    """Every interval of `times`, a draw's grid, from `x_T` in its precision, with the reference's
    stand-in denoiser read through the Karras schedule's own sigma."""
    process, _ = karras_process()
    times = jnp.asarray(times, x_T.dtype)
    denoise = StandIn(process)

    x, state = x_T, flaxdiff_start(solver, x_T)
    latents = []
    for i in range(times.shape[0] - 1):
        t, t_next = jnp.full((x.shape[0],), times[i]), jnp.full((x.shape[0],), times[i + 1])
        denoised, eps = denoise(x, t)
        x, state = solver.step(x, t, t_next, denoised, eps, state, jax.random.PRNGKey(i), process, denoise)
        latents.append(x)
    return jnp.stack(latents)


def flaxdiff_walked(solver):
    """Dew's arm of a FlaxDiff walk under `solver`: a draw's latent after every interval of its grid, and
    the last one's pullback of its cotangent (tools/flaxdiff_solver_reference.py pairs its sizing with
    it)."""
    def walked(inputs):
        x_T = jnp.asarray(inputs["x_T"])
        times = inputs["times"]
        latents = flaxdiff_walk(solver, x_T, times)
        cotangent = jnp.asarray(inputs["cotangent"])
        (gradient,) = jax.vjp(lambda x: flaxdiff_walk(solver, x, times)[-1], x_T)[1](cotangent)
        return {**{f"after {index}": latent for index, latent in enumerate(latents)}, "gradient": gradient}

    return walked


def flaxdiff_outputs(closed):
    """A closing walk's leaves: its output as it is, and through the clip to [-1, 1] of FlaxDiff's
    `post_process`."""
    return {"closed": closed, "clipped": jnp.clip(closed, -1, 1)}


def flaxdiff_closed(solver):
    """Dew's arm of a FlaxDiff closing walk under `solver`: `sample` over a draw's grid with its default
    closing denoise (tools/flaxdiff_solver_reference.py pairs its sizing with it)."""
    process, _ = karras_process()

    def closing(inputs):
        x_T = jnp.asarray(inputs["x_T"])
        return flaxdiff_outputs(sample(StandIn(process), x_T, solver=solver, key=jax.random.key(0),
                                       times=jnp.asarray(inputs["times"], x_T.dtype),
                                       initial=flaxdiff_start(solver, x_T)))

    return closing


def assert_on_par_with_flaxdiff(site: str, dew, truth_fn=None) -> None:
    """`dew(inputs)`, Dew's leaves on a draw, on par with `site`'s recording of FlaxDiff's own float32
    run over its declared seeds (check 1). The truth is `truth_fn(inputs)`, `dew` unless given, run
    widened, held to the recording's float64 sketch of FlaxDiff's float64 twin (`indexed_recording`)."""
    seeds = declared_seeds(FLAXDIFF)[site]
    draw, recorded, truth = indexed_recording(recording(FLAXDIFF), site, seeds,
                                              inputs_of=writer("flaxdiff_solver_reference").draw,
                                              truth_fn=dew if truth_fn is None else truth_fn)
    compiled = jax.jit(dew)
    assert_as_exact(lambda index, inputs: compiled(inputs), recorded, draw, site, seeds=seeds,
                    truth_fn=truth)


@pytest.mark.parametrize("name,solver", [("rk4", RK4()), ("multistep", MultiStepDPM())],
                         ids=["rk4", "multistep"])
def test_the_closing_denoise_is_flaxdiffs_generate_samples(name, solver):
    """`sample` with its default `final_denoise` against FlaxDiff's own
    `generate_samples` (AshishKumar4/FlaxDiff@15c55b0, run as published by
    tools/flaxdiff_solver_reference.py) from the same x_T over the same
    twelve-point grid, each draw's its own (the writer's JITTER): eleven
    solver intervals, then the clean prediction at the grid's last point,
    over the recorded seeds (check 1). The output
    is held as it is and through the clip to [-1, 1] that FlaxDiff's
    `post_process` applies as the pipeline's output step: the stand-in's
    clean predictions reach |x| = 3, so the clip moves most entries and the
    unclipped comparison is the one that reads the walk. The truth is the
    same `sample` in float64, MultiStepDPM's history included (ledger 2063)."""
    closing = flaxdiff_closed(solver)
    assert_on_par_with_flaxdiff(f"{name} closed", closing)
    assert float(jnp.abs(closing(writer("flaxdiff_solver_reference").draw(0))["closed"]).max()) > 2


@pytest.mark.parametrize("name,solver", [("rk4", RK4()), ("multistep", MultiStepDPM())],
                         ids=["rk4", "multistep"])
def test_rk4_and_the_multistep_integrator_are_flaxdiffs(name, solver):
    """`RK4` and `MultiStepDPM` are FlaxDiff's `RK4Sampler` and
    `MultiStepDPM` (AshishKumar4/FlaxDiff@15c55b0, run as published by
    tools/flaxdiff_solver_reference.py): from the same x_T over FlaxDiff's
    twelve-point grid on a rho-7 Karras schedule, each draw's moved within a
    hundredth of a spacing (the writer's JITTER), with a denoiser nonlinear
    in x, the latent after every interval, so every RK4 stage and each of the
    multistep's first, second and third order updates, and the gradient of
    the last one through the walk, on par with FlaxDiff's float32 run over
    the recorded seeds (check 1). FlaxDiff's grid is Dew's own `times(12)`."""
    process, _ = karras_process()
    np.testing.assert_array_equal(process.times(12), recording(FLAXDIFF)["times"])
    assert_on_par_with_flaxdiff(f"{name} walk", flaxdiff_walked(solver))


@pytest.mark.parametrize("solver", [DDIM(), Euler()], ids=lambda s: type(s).__name__)
def test_video_samples_converge(solver):
    """The frame axis is one more sample axis: the rates broadcast over it and
    the statistics come out the same as for images."""
    process, model = karras_process()
    assert_gaussian_stats(generate(process, model, solver, steps=50, count=64, shape=(3, 8, 8, 3)))


@pytest.mark.parametrize("solver, times, walks", [
    (DPMSolverMultistep(order=2), [0.9, 0.9, 0.1], True),
    (DEIS(order=2), [0.9, 0.9, 0.1], True),
    (DPMSolverSinglestep(order=2), [0.9, 0.5, 0.5, 0.1], True),
    (UniPC(order=1), [0.9, 0.9, 0.5, 0.1], True),
    (DPMSolverMultistep(order=2), [0.9, 0.5, 0.5, 0.1], True),
    (DPMSolverMultistep(order=2, lower_order_final=False), [0.9, 0.9, 0.1], False),
    (DPMSolverMultistep(order=2, solver_type="heun"), [0.9, 0.5, 0.5, 0.1], False),
    (UniPC(order=2), [0.9, 0.9, 0.5, 0.1], False),
], ids=["dpm_multi", "deis", "dpm_single", "unipc.order1", "dpm_multi.midpoint", "dpm_multi.second_order",
        "dpm_multi.heun", "unipc.order2"])
def test_only_an_update_that_divides_by_an_empty_interval_refuses_it(solver, times, walks):
    """A grid with two points at one level walks wherever no update divides by the gap between
    them. A first-order step over it stays where it is: the multistep solvers' first step, which
    has no history, the short walk's tapered last step, a singlestep group's second step, whose
    anchor and target are apart, and a first-order UniPC's predictor and corrector, all walk to
    finite latents, as does DPM-Solver's second-order midpoint step, whose coefficient is 0 over
    it. An update that divides by the interval, a second-order step whose history spans it, a
    Heun step over it, refuses the grid before the walk runs."""
    process, model = flow_process()
    x_T = jax.random.normal(jax.random.PRNGKey(0), (2, 4))
    if walks:
        assert np.isfinite(np.asarray(walk(solver, process, model, x_T, times))).all()
    else:
        with pytest.raises(ValueError):
            walk(solver, process, model, x_T, times)


def test_a_key_reproduces_a_trajectory():
    """Nothing lives on a solver between calls: a multistep solver's history
    travels in its state, so the same key gives the same samples twice."""
    process, model = karras_process()
    first = generate(process, model, MultiStepDPM())
    second = generate(process, model, MultiStepDPM())
    assert np.array_equal(np.asarray(first), np.asarray(second))


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
        return Process(schedule, prediction), VPOracle(schedule=schedule)
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
# EDM's own stochastic sampler
############################################################################################################

EDM = Path(__file__).resolve().parent / "fixtures" / "edm" / "heun.npz"




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


def assert_walks_as_edm(name: str, *, final_denoise: bool) -> None:
    """Heun with the recorded case's S_churn, S_tmin, S_tmax and S_noise from each seed's latent under
    its key, on par with the published sampler's own float32 walk of its rho-7 grid and the same
    per-step draws over the case's declared seeds (check 1). With `final_denoise`, the walk stops at the
    last sigma and closes with the model's clean prediction there; without, it steps onto the zero NVlabs
    appends."""
    stored = recording(EDM)
    arguments = json.loads(str(stored[f"{name}.arguments"]))
    steps = arguments["num_steps"]
    solver = Heun(s_churn=arguments.get("S_churn", 0.0), s_tmin=arguments.get("S_min", 0.0),
                  s_tmax=arguments.get("S_max", float("inf")), s_noise=arguments.get("S_noise", 1.0))
    denoiser = edm_grid(steps).denoiser(EDMOracle(), {}, {})
    times = np.arange(steps, 0 if final_denoise else -1, -1, dtype=np.float32)

    # The closing walk takes `sample`'s default, which is the closing denoise.
    closing = {} if final_denoise else {"final_denoise": False}

    @jax.jit
    def walked(index, inputs):
        x_T = jnp.asarray(inputs["latents"]) * np.float32(80.0)
        return {"result": sample(denoiser, x_T, solver=solver, key=inputs["key"], times=times, **closing)}

    seeds = declared_seeds(EDM)[name]
    draw, recorded, truth = indexed_recording(stored, name, seeds, inputs_of=writer("edm_reference").draw)
    assert_as_exact(walked, recorded, draw, name, seeds=seeds, truth_fn=truth)


@pytest.mark.parametrize("name", ["deterministic", "churn", "churn_to_the_end"])
def test_heun_walks_edms_algorithm_2_with_its_churn(name):
    """Heun with S_churn, S_tmin, S_tmax and S_noise against the published
    sampler on its own rho-7 grid and the same per-step draws, as exact as
    the published sampler's own float32 walk over the recorded seeds."""
    assert_walks_as_edm(name, final_denoise=False)


@pytest.mark.parametrize("name", ["deterministic", "churn"])
def test_the_closing_denoise_is_edms_last_euler_step_onto_sigma_zero(name):
    """NVlabs' sampler ends with an Euler step from its last sigma onto the
    zero it appends, which lands on the model's clean prediction at that
    sigma. `sample` stopped at the last sigma closes with that prediction
    (`final_denoise`, its default) and matches the published sampler as
    exactly as its own float32 walk; the churn case's last sigma, 0.002, is
    below its S_min, so nothing is churned there."""
    assert_walks_as_edm(name, final_denoise=True)


class UnitEpsilon(nn.Module):
    """An epsilon prediction of one everywhere, so a transfer's eps term is its whole coefficient."""

    @nn.compact
    def __call__(self, x, temb):
        return jnp.ones_like(x)


def test_a_pndm_transfer_at_a_small_signal_stays_finite():
    """PNDM's transfer between two levels of a schedule whose signal is small, alpha^2 of 2^-100 and
    2^-80, here a beta table of halves, is DDIM's in float64 to float32's rounding: the root its
    denominator takes of three variances is a product of roots, whose product of the three would
    round to zero and blow the step up by six orders."""
    from dew.diffusion.schedules import DiscreteNoiseScheduler

    process = Process(DiscreteNoiseScheduler(np.full(100, 0.5)), EpsilonPredictionTransform())
    x_T = jnp.ones((1, 3), jnp.float32)
    stepped = np.asarray(walk(PNDM(skip_prk_steps=True), process, UnitEpsilon(), x_T, [99, 79]))[-1]
    alpha2_t, alpha2_s = 0.5 ** 100, 0.5 ** 80
    sigma2_t, sigma2_s = 1 - alpha2_t, 1 - alpha2_s
    denominator = alpha2_t * np.sqrt(sigma2_s) + np.sqrt(alpha2_t * sigma2_t * alpha2_s)
    expected = np.sqrt(alpha2_s / alpha2_t) - (alpha2_s - alpha2_t) / denominator
    np.testing.assert_allclose(stepped, np.full((1, 3), expected), rtol=1e-5)


def test_an_euler_step_near_pure_noise_on_a_variance_preserving_schedule_rounds_as_ddim():
    """An Euler step from pure noise (alpha 0) to a level whose signal is 2^-11.5 and whose noise is
    an ulp below one is DDIM's step, alpha_s x_0 + sigma_s (x - alpha_t x_0) / sigma_t, to float32's
    rounding: the secant coefficient the source's x + derivative dt reads there is near 5800,
    whose products lose the low bits of x that the step keeps."""
    from dew.sampling.solvers.sigma import _euler_step

    x, denoised = jnp.full((1, 1), np.float32(1.0001)), jnp.ones((1, 1), jnp.float32)
    alpha_s, sigma_s = np.float32(np.sqrt(2.0 ** -23)), np.float32(1 - 2.0 ** -24)
    stepped = np.asarray(_euler_step(x, denoised, (jnp.float32(0), jnp.float32(1)),
                                     (jnp.asarray(alpha_s), jnp.asarray(sigma_s)))[0])
    exact = float(alpha_s) + float(sigma_s) * float(np.float32(1.0001))
    assert abs(float(stepped[0, 0]) - exact) <= np.spacing(np.float32(exact))


POSTERIOR_SEEDS = 8
"""The DDIM posterior site's declared draws (check 1), sized by `seeds_for` paired with Dew's step on
armada: the most any process took."""


@pytest.mark.parametrize("case", FAMILIES)
def test_ddim_at_eta_one_steps_from_the_ddpm_posterior_on_every_gaussian_process(case):
    """DDIM at eta 1 draws its step from the forward posterior q(x_s | x_t, x_0) of the process it
    walks, variance preserving, exploding or a rectified flow: its mean alpha_s x_0 +
    alpha_t sigma_s^2 / (alpha_s sigma_t) eps and its standard deviation sigma_s sigma_{t|s} /
    sigma_t, with sigma_{t|s}^2 = sigma_t^2 - (alpha_t / alpha_s)^2 sigma_s^2. DDIM's own form,
    alpha_s x_0 + sqrt(sigma_s^2 - std^2) eps + std z, is that posterior in float64. Each row steps
    over its own interval of the process's grid, as the walk passes a row's times, so the
    coefficients' roundings are many: Dew's step is on par with DDIM's form written plainly in float32
    over the draws (check 1), truth the posterior in float64 on the same denoiser outputs and noise. The
    variance-preserving special case of the deviation drew no noise on a variance-exploding process and
    the wrong amount on a flow."""
    build, shape, _ = PROCESS_CASES[case]
    process, model = build()
    grid = np.asarray(process.times(65))
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *shape)), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    solver = DDIM(eta=1.0)
    schedule = process.sampler_schedule

    def draw(seed):
        rows = np.random.default_rng([6, seed]).integers(0, len(grid) - 2, 512)
        t, t_next = jnp.asarray(grid[rows]), jnp.asarray(grid[rows + 1])
        x_T = jax.random.normal(jax.random.fold_in(jax.random.key(4), seed), (len(rows), *shape),
                                jnp.float32) * 2
        denoised, eps = denoise(x_T, t)
        noise = jax.random.normal(jax.random.fold_in(jax.random.key(5), seed), x_T.shape, jnp.float32)
        return x_T, t, t_next, denoised, eps, noise, np.int32(seed)

    def coefficients(t, t_next):
        """alpha_t, sigma_t, alpha_s, sigma_s and DDIM's standard deviation, each shaped to a row."""
        shaped = (-1,) + (1,) * len(shape)
        target = t - schedule.step_interval(t, t_next)
        # A tabulated schedule's rates are float32 at any time: widened with the time, so the truth's
        # coefficients are float64's.
        a_t, s_t, a_s, s_s = (jnp.reshape(rate, shaped).astype(t.dtype) for time in (t, target)
                              for rate in schedule.rates(time))
        return a_t, s_t, a_s, s_s, s_s / s_t * jnp.sqrt(s_t ** 2 - (a_t / a_s) ** 2 * s_s ** 2)

    def stepped(x_T, t, t_next, denoised, eps, noise, seed):
        key = jax.random.fold_in(jax.random.key(5), seed)
        return solver.step(x_T, t, t_next, denoised, eps, (), key, process, denoise)[0]

    def ddim(x_T, t, t_next, denoised, eps, noise, seed):
        _, _, a_s, s_s, std = coefficients(t, t_next)
        return a_s * denoised + jnp.sqrt(s_s ** 2 - std ** 2) * eps + std * noise

    def posterior(x_T, t, t_next, denoised, eps, noise, seed):
        a_t, s_t, a_s, s_s, std = coefficients(t, t_next)
        return a_s * denoised + a_t * s_s ** 2 / (a_s * s_t) * eps + std * noise

    # The two forms are one in float64, to its rounding of each term, on the first draw's float32 rates.
    _, t, t_next, denoised, eps, noise, _ = draw(0)
    a_t, s_t, a_s, s_s = (np.asarray(rate, np.float64) for rate in coefficients(t, t_next)[:4])
    std = s_s / s_t * np.sqrt(s_t ** 2 - (a_t / a_s) ** 2 * s_s ** 2)
    assert np.all(std > 0)
    denoised, eps, noise = (np.asarray(value, np.float64) for value in (denoised, eps, noise))
    exact = a_s * denoised + a_t * s_s ** 2 / (a_s * s_t) * eps + std * noise
    np.testing.assert_allclose(a_s * denoised + np.sqrt(s_s ** 2 - std ** 2) * eps + std * noise, exact,
                               rtol=1e-12, atol=1e-12 * float(np.abs(exact).max()))
    assert_as_exact(jax.jit(stepped), jax.jit(ddim), draw, f"DDIM at eta 1 on the {case} process",
                    seeds=POSTERIOR_SEEDS, truth_fn=posterior)


@pytest.mark.parametrize("solver", [DDIM(eta=1.0), DDPM("small"), DDPM("large")],
                         ids=["ddim", "ddpm-small", "ddpm-large"])
def test_a_step_between_equal_levels_draws_no_noise(solver):
    """A step between two equal levels of a variance-exploding process, as a grid's plateau holds,
    has a transition of no variance: its draw is multiplied by exactly zero, so two draws give one
    result, compiled as a walk compiles it."""
    process, model = karras_process()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 4)), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    x = jax.random.normal(jax.random.key(2), (8, 4), jnp.float32)
    level = jnp.full((8,), 0.4, jnp.float32)

    # The two levels are separate inputs, equal only in their values, as two positions of a walk's
    # grid are, so the compiled step computes each from its own rates.
    @jax.jit
    def stepped(t, t_next, key):
        denoised, eps = denoise(x, t)
        state = solver.init(x, jnp.stack([t[0], t_next[0]]), process, key=key)
        return solver.step(x, t, t_next, denoised, eps, state, key, process, denoise)[0]

    first, second = (np.asarray(stepped(level, jnp.array(level), jax.random.key(seed))) for seed in (3, 4))
    np.testing.assert_array_equal(first, second)


WALK_SEEDS = 8
"""The DDIM-against-Diffusers walk's declared draws (check 1), sized by `seeds_for` paired with Dew's walk on
armada."""


def test_ddim_at_eta_one_walks_as_diffusers_ddim_scheduler_on_the_same_draws():
    """DDIM at eta 1 over a variance-preserving source grid walks as Diffusers 0.34.0's
    `DDIMScheduler(eta=1)` given the same draws (`variance_noise`): every walked latent on par with the
    source's float32 walk over the draws (check 1), truth the source's float64 walk
    (`float64_scheduler`)."""
    import contextlib

    import torch
    from diffusers.schedulers import scheduling_ddim
    from tool_support import load

    from dew.diffusion.schedules.source import SourceSchedule
    from dew.nn.inputs import request_key

    config = {"_class_name": "DDIMScheduler", "num_train_timesteps": 1000, "beta_start": 0.00085,
              "beta_end": 0.012, "beta_schedule": "scaled_linear", "clip_sample": False,
              "set_alpha_to_one": False, "timestep_spacing": "trailing"}
    steps = 6
    schedule = SourceSchedule.from_config(config)
    process, times = schedule.sampling(steps)
    denoise = process.denoiser(SourceOracle(), {}, {})
    precision = load("diffusers_wan_reference")

    def draw(seed):
        return np.random.default_rng([9, seed]).normal(size=(2, 3, 64)).astype(np.float32), np.int32(seed)

    def walked(initial, seed):
        _, latents = sample(denoise, jnp.asarray(initial), solver=DDIM(eta=1.0), key=int(seed), times=times,
                            final_denoise=False, record=lambda x, state, denoised: x)
        return tuple(latents)

    def source(dtype):
        def run(initial, seed):
            draws = [np.asarray(jax.random.normal(jax.random.fold_in(request_key(int(seed)), index),
                                                  initial.shape, jnp.float32)) for index in range(steps)]
            wide = dtype == torch.float64
            with precision.float64_scheduler(scheduling_ddim) if wide else contextlib.nullcontext():
                scheduler = scheduling_ddim.DDIMScheduler.from_config(config)
                scheduler.set_timesteps(steps)
                x = torch.tensor(np.asarray(initial), dtype=dtype)
                latents = []
                for time, noise in zip(scheduler.timesteps, draws, strict=True):
                    prediction = torch.sin(scheduler.scale_model_input(x, time)) * 0.07 + time * 0.001
                    x = scheduler.step(prediction, time, x, eta=1.0,
                                       variance_noise=torch.tensor(noise, dtype=dtype)).prev_sample
                    latents.append(x.numpy().copy())
            return tuple(latents)
        return run

    assert_as_exact(as_given(walked), as_given(source(torch.float32)), draw,
                    "DDIM at eta 1 against Diffusers", seeds=WALK_SEEDS,
                    truth_fn=as_given(source(torch.float64)))


def test_rk4s_update_divides_by_six_as_ieee_division_does():
    """RK4's update, x + (k1 + 2 k2 + 2 k3 + k4) dt / 6, divides by six as IEEE does
    (`dew.nn.precision.divided`), bit for bit: with every stage's eps the same sixteenths, so
    the stages' sum is exact, the update is the product with dt rounded once, its correctly rounded
    sixth and the sum with x, where XLA:CPU's division by a constant multiplies by the rounded
    reciprocal and parts with IEEE's quotient on about a third of values."""
    process, _ = karras_process()
    rng = np.random.default_rng(6)
    x = rng.standard_normal((4096, 1)).astype(np.float32)
    eps = (rng.integers(-1024, 1024, (4096, 1)) / 16).astype(np.float32)
    times = process.times(12)
    t, t_next = jnp.full((4096,), times[0]), jnp.full((4096,), times[1])

    def denoise(at, time):
        return at, jnp.asarray(eps)

    stepped, _ = jax.jit(lambda x: RK4().step(x, t, t_next, x, jnp.asarray(eps), (), jax.random.key(0),
                                              process, denoise))(jnp.asarray(x))
    _, sigma_t = process.rates(t, like=x)
    _, sigma_s = process.rates(t_next, like=x)
    dt = np.asarray(sigma_s, np.float32) - np.asarray(sigma_t, np.float32)
    expected = x + (6 * eps * dt) / np.float32(6)
    np.testing.assert_array_equal(np.asarray(stepped).view(np.uint32), expected.view(np.uint32))


def test_pndms_runge_kutta_weighs_a_low_precision_stage_in_fp32_as_diffusers_step_prk_does():
    """PNDM's Runge-Kutta warmup weighs each stage as Diffusers' step_prk does, the product with 1/6 or
    1/3 in fp32 and rounded once to the stage's dtype, as torch's `bf16_tensor * python_float` does
    (`dew.nn.precision.scaled`): a bf16 stage of 7 weighs to 1.1640625, where rounding the sixth to
    bf16 first gives 1.171875. With the later stages zero and the state zero, the step is the
    transfer of the first stage's weight alone, `_get_prev_sample`'s, bit for bit."""
    import ml_dtypes

    process, _ = vp_process()
    rng = np.random.default_rng(7)
    eps = rng.standard_normal((4096, 1)).astype(ml_dtypes.bfloat16)
    eps[0, 0] = 7
    x = jnp.zeros((4096, 1), jnp.float32)
    times = process.times(12)
    t, t_next = jnp.full((4096,), times[0]), jnp.full((4096,), times[1])

    def denoise(at, time):
        return at, jnp.zeros((4096, 1), jnp.bfloat16)

    solver = PNDM()
    state = solver.init(x, times, process, key=jax.random.key(0))
    stepped, _ = jax.jit(lambda x, eps: solver.step(x, t, t_next, x, eps, state, jax.random.key(0), process,
                                                    denoise))(x, jnp.asarray(eps))
    weighed = (eps.astype(np.float32) * np.float32(1 / 6)).astype(ml_dtypes.bfloat16)
    assert weighed[0, 0] == 1.1640625

    @jax.jit
    def transfer(x, weighed):
        alpha2_t, sigma2_t = process.variances(t, like=x)
        alpha2_s, sigma2_s = process.variances(t_next, like=x)
        denominator = alpha2_t * sigma2_s ** 0.5 + (alpha2_t * sigma2_t) ** 0.5 * alpha2_s ** 0.5
        return (alpha2_s / alpha2_t) ** 0.5 * x - (alpha2_s - alpha2_t) * weighed / denominator

    expected = np.asarray(transfer(x, jnp.asarray(weighed)))
    np.testing.assert_array_equal(np.asarray(stepped).view(np.uint32), expected.view(np.uint32))


class ZeroEndpoint(nn.Module):
    """A model whose clean estimate is zero everywhere: x walks along its noise level alone."""

    @nn.compact
    def __call__(self, x, temb):
        return jnp.zeros_like(x)


def test_lms_walks_a_variance_exploding_grid_near_float32s_largest():
    """LMS(order=2) on VE sigmas 2**70, 2**69, 2**68 toward a zero clean estimate: dx/dsigma = x /
    sigma, so x walks 1 -> 0.5 -> 0.25, its Lagrange basis integrated on the step's own scale where
    the absolute sigma powers would overflow on its second step (ledger 1874)."""
    process = Process(KarrasVENoiseScheduler(sigma_min=0, sigma_max=2.0 ** 70, rho=1),
                      DirectPredictionTransform())
    latents = np.asarray(walk(LMS(2), process, ZeroEndpoint(), jnp.ones((1, 3), jnp.float32),
                              [1.0, 0.5, 0.25]))
    # Every step's arithmetic is exact in binary on this grid: the walk is the exact one.
    np.testing.assert_array_equal(latents, [np.full((1, 3), 0.5), np.full((1, 3), 0.25)])


def test_a_tiny_lms_step_beside_a_distant_node_keeps_its_coefficient():
    """Nodes 2**-100 and 2**-99 beside 1, a step from 2**-100 to 0: the basis of the latest node
    integrates to -1.5 * 2**-100 (on s in [0, 1], (1 + s)(1 + 2**-100 s) times the step -2**-100), its
    coefficient kept where a scale set by the distant node would round it to zero; and beside nodes 1
    and 2, where a power of the nodes would overflow, to the step's width (ledger 1874)."""
    from dew.sampling.solvers.sigma import _lagrange_integral

    tiny = jnp.float32(2.0 ** -100)
    coefficient = float(_lagrange_integral([tiny, 2 * tiny, jnp.float32(1.0)], 0, tiny, jnp.float32(0.0)))
    # 1.5 plus terms of 2**-100 rounds to 1.5, and the step's width is a power of two: exact in float32.
    assert coefficient == -1.5 * 2.0 ** -100
    # A step to zero from 2**-100 beside nodes 1 and 2, the end of a VE walk at order 3: its factors
    # are 1 plus terms of 2**-100, so the latest node's basis integrates to the step's width.
    nodes = [tiny, jnp.float32(1.0), jnp.float32(2.0)]
    coefficient = float(_lagrange_integral(nodes, 0, tiny, jnp.float32(0.0)))
    assert coefficient == -(2.0 ** -100)


def test_an_empty_lms_step_integrates_to_zero_with_the_basis_as_its_derivative():
    """Over an empty step (a = b) the Lagrange basis integrates to zero, and its derivative in the
    step's end is the basis at that end: 1 for the node it is 1 at, 0 for the other (ledger 1874)."""
    from dew.sampling.solvers.sigma import _lagrange_integral

    for j, at in ((0, 1.0), (1, 0.0)):
        def integral(b, j=j):
            return _lagrange_integral([1.0, 2.0], j, jnp.float32(1.0), b)

        value, slope = jax.value_and_grad(integral)(jnp.float32(1.0))
        assert float(value) == 0.0 and float(slope) == at
