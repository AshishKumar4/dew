"""Diffusers 0.34.0 scheduler trajectories, for tests/fixtures/diffusers.

Every case runs one Diffusers scheduler over a grid Dew's solvers walk, with
the optimal model for x0 ~ N(0, 0.3^2) in closed form, so both sides are
deterministic and nothing is trained. What lands per case: the x_T draw, the
latent after each grid interval, the grid itself (the times Dew's walk takes,
ending where the scheduler's sigma_min terminal lands), and the
vector-Jacobian product of the final latent against a fixed cotangent through
the whole trajectory, in float64 on the torch side. A stochastic scheduler
draws the noise Dew's `sample` draws at that step, `jax.random.normal` under
the step's folded key, so the two sides integrate one Brownian path.

Three model conventions cover the schedulers:

- improved-diffusion's linear VP beta table at 1000 steps (its own
  `get_named_beta_schedule`, in tests/fixtures/schedules/betas.npz),
  which Dew's `LinearNoiseScheduler(1000)` tabulates, with an epsilon
  oracle: DPM-Solver multistep and singlestep in every algorithm, order
  and solver type Diffusers offers, DEIS, UniPC, PNDM, LCM and TCD;
- the Karras rho-7 grid between that table's sigma extremes, in the
  variance-exploding form k-diffusion samplers integrate: KDPM2, KDPM2
  ancestral and LMS;
- EDM preconditioning on Karras' own grid, sigma 0.002 to 80 at sigma_data
  0.5: the EDM DPM-Solver scheduler.

Check 1 holds Dew's walk of each case over seeds (`seeds`): per scheduler
family, each case's float32 walk and its float64 truth (the class's
independent oracle, tools/float64_schedulers.py) from a seed's x_T and
cotangent, the latents and the VJP kept whole (Dew's float64 walk reads its
schedule's float32 tables, so it is no truth of the scheduler's), under
tests/fixtures/diffusers/scheduler_seeds/.

    PYTHONPATH=src python tools/diffusers_reference.py
    PYTHONPATH=src:tools:tests python tools/diffusers_reference.py seeds <family>
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import itertools
import json
import os
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType

# The per-step noise is the test suite's draw, which conftest.py makes on
# the CPU; a GPU draw of the same key differs in the last float32 bit and
# the fixture would then not regenerate byte-identically.
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch
from diffusers_reference_helpers import fed_noise as fed_noise, scheduler_module, step_noise as step_noise
from float64_schedulers import ORACLES as SCHEDULER_ORACLES
from scheduler_cases import Case as Stepping, calls

from dew.diffusion import EpsilonPredictionTransform, LinearNoiseScheduler, Process

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "diffusers"
SEEDS = FIXTURES / "scheduler_seeds"
"""Each scheduler family's seeded walks (`seeds`), one recording a family, `<family>.npz`."""
sys.path.insert(0, str(FIXTURES.parents[1]))
ORACLES = importlib.import_module("diffusion_support")
"""The oracles' closed forms, which Dew's arm reads too (tests/diffusion_support.py)."""
DATA_STD = ORACLES.DATA_STD

TRAIN_STEPS = 1000
STEPS = 21
SHAPE = (2, 2, 4, 4)
# improved-diffusion's own linear table (tools/beta_schedule_reference.py).
BETAS = np.load(FIXTURES.parent / "schedules" / "betas.npz")["improved_diffusion_linear_1000"]
ALPHAS_CUMPROD = torch.cumprod(1 - torch.tensor(BETAS, dtype=torch.float64), dim=0)
# The variance-exploding sigma of the table, sigma / alpha, at its two ends.
SIGMA_VE = ((1 - ALPHAS_CUMPROD) / ALPHAS_CUMPROD).sqrt()
EDM = dict(sigma_min=0.002, sigma_max=80.0, sigma_data=0.5, rho=7.0)


def dew_grid(steps: int) -> np.ndarray:
    """The integer indices `Process.times(steps)` reaches on the tabulated
    schedule, read from the process itself: jax's float32 linspace lands
    549.99994 where numpy lands 550, and the index truncates."""
    process = Process(LinearNoiseScheduler(TRAIN_STEPS), EpsilonPredictionTransform())
    times = np.asarray(process.times(steps))
    return np.clip(times.astype(np.int32), 0, TRAIN_STEPS - 1)


def published_grid(scheduler: str, config: dict, steps: int) -> np.ndarray:
    """The timesteps `scheduler` under `config` sets for `steps - 1` steps (its own `set_timesteps`, a
    `timestep_spacing` its config names), then the table's index 0 where its sigma_min terminal lands."""
    published = getattr(scheduler_module(scheduler), scheduler)(trained_betas=BETAS, **config)
    published.set_timesteps(steps - 1)
    return np.concatenate([published.timesteps.numpy(), [0]])


def handed_exactly(x: torch.Tensor, oracle: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """`oracle` of `x` in float64, its handed inputs read in their own precision, each entry rounded once
    to `x`'s dtype, its gradient alike: as a model's output is an array whose entries are each rounded once,
    and as Dew's oracles read theirs (tests/test_samplers.py's `handed_exactly`). An oracle that rounds its
    own coefficient scales every entry by one error, which a walk's conversion of eps to x_0 amplifies past
    any solver's rounding (row 13662)."""
    return oracle(x.double()).to(x.dtype)


def vp_epsilon(x: torch.Tensor, index: int) -> torch.Tensor:
    """E[eps | x_t] for x0 ~ N(0, DATA_STD^2) at the table's index:
    x sigma / (alpha^2 s^2 + sigma^2), the table's rates rounded to the
    latent's precision as Dew's model reads its own table, the rest exact
    (`handed_exactly`; tests/test_samplers.py's `VPOracle`)."""
    alpha_cumprod = ALPHAS_CUMPROD[index]
    alpha, sigma = (rate.to(x.dtype).double() for rate in (alpha_cumprod.sqrt(), (1 - alpha_cumprod).sqrt()))
    return handed_exactly(x, lambda wide: ORACLES.vp_epsilon(wide, alpha, sigma))


def read_sigma(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """The sigma a model reads at its noise level `sigma`: the model time c_noise = log(sigma) / 4 it is
    handed in the latent's precision, read back exactly, as Dew's oracles read it (tests/test_samplers.py's
    `VEOracle` and `KarrasOracle`)."""
    return torch.exp(4.0 * (torch.log(sigma.to(x.dtype)) / 4.0).double())


def ve_epsilon(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """The same oracle on the variance-exploding latent x = x0 + sigma eps
    the k-diffusion schedulers carry: x sigma / (s^2 + sigma^2), at the sigma
    the model reads (`read_sigma`), exact (`handed_exactly`)."""
    sigma = read_sigma(x, sigma)
    return handed_exactly(x, lambda wide: ORACLES.ve_epsilon(wide, sigma))


def edm_raw(scaled: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """The raw F whose EDM preconditioning c_skip x + c_out F is the optimal
    x0 = x s^2 / (s^2 + sigma^2), from the input the scheduler scales for its
    model (`scale_model_input`, c_in x) at the sigma the model reads
    (`read_sigma`), as Dew's `KarrasOracle` computes it."""
    sigma = read_sigma(scaled, sigma)
    return handed_exactly(scaled, lambda wide: ORACLES.karras_raw(wide, sigma, EDM["sigma_data"]))


@dataclass(frozen=True)
class Case:
    """One scheduler run: its class and config, the Dew times the walk
    takes, and the model convention (`vp`, `ve` or `edm`) the oracle and
    Dew's process follow."""

    scheduler: str
    config: Mapping[str, object]
    grid: np.ndarray
    convention: str = "vp"
    step_kwargs: dict[str, object] = field(default_factory=dict)
    timesteps: np.ndarray | None = None
    """A seed's timesteps (`seeded`), where they are not the scheduler's own."""
    betas: np.ndarray = field(default_factory=lambda: BETAS)
    """The trained table the scheduler is built on, a seed's (`seeded`)."""


def made(module: ModuleType, case: Case, noises: list[np.ndarray], *, rederived: bool
         ) -> tuple[Callable[..., object], Callable[[], AbstractContextManager[object]]]:
    """What builds the scheduler `case` walks from its config, and the context its steps' draws are fed in:
    the unmodified library class, its module's draws the walk's `noises` (`fed_noise`); where `rederived`
    (a seed's float64 truth) the class's independent oracle (tools/float64_schedulers.py), every table its
    own, the same `noises` its own tape, no module touched."""
    if not rederived:
        return getattr(module, case.scheduler), functools.partial(fed_noise, module, noises)

    def oracle(**config: object) -> object:
        built = SCHEDULER_ORACLES[case.scheduler](torch.float64, **config)
        built.draws = list(noises)
        return built

    return oracle, contextlib.nullcontext


def stepped(scheduler, case: Case, x: torch.Tensor, model, calls_taken: int | None = None, *,
            reads_latent: bool = False) -> list[torch.Tensor]:
    """The latents `scheduler`'s first `calls_taken` steps (every one where None) reach from `x` under
    `model`, told each step's time and place (`scheduler_cases.calls`, the one walk every recorder takes:
    its input scaled as the class's pipeline scales it, or the latent where the model `reads_latent`;
    PNDM's time an integer; `case`'s step options)."""
    walk = calls(scheduler, Stepping(case.scheduler, step_with=case.step_kwargs), x, model,
                 reads_latent=reads_latent)
    return [call.after for call in itertools.islice(walk, calls_taken)]


def lambda_walk(build, fed, case: Case, x: torch.Tensor) -> list[torch.Tensor]:
    """The VP-table schedulers whose model runs once per grid point at an
    integer timestep: DPM-Solver multistep and singlestep, DEIS, UniPC, LCM
    and TCD."""
    scheduler = build(trained_betas=case.betas, **case.config)
    timesteps = case.grid[:-1].tolist()
    if "timestep_spacing" in case.config:
        # The grid is the scheduler's own (`published_grid`).
        scheduler.set_timesteps(len(timesteps))
    else:
        scheduler.set_timesteps(timesteps=timesteps if case.timesteps is None else case.timesteps.tolist())
    with fed():
        return stepped(scheduler, case, x, lambda read, time, at: vp_epsilon(read, int(time)))


def pndm_walk(build, fed, case: Case, x: torch.Tensor) -> list[torch.Tensor]:
    """PNDM runs its own timestep list: the first interval twice (predictor,
    then corrector) under `skip_prk_steps`, otherwise three Runge-Kutta
    intervals of four evaluations, then one evaluation per interval, and a
    last step below index 0 that `set_alpha_to_one=False` makes the
    identity. The latents at the grid points are kept."""
    scheduler = build(trained_betas=BETAS, **case.config)
    points = len(case.grid)
    scheduler.set_timesteps(points)
    stride = TRAIN_STEPS // points
    assert set(case.grid.tolist()) <= set(scheduler.timesteps.tolist())
    assert all(case.grid[:-1] - case.grid[1:] == stride), (case.grid, stride)
    outputs = stepped(scheduler, case, x, lambda read, time, at: vp_epsilon(read, int(time)))
    if case.config["skip_prk_steps"]:
        latents = outputs[1:-1]
    else:
        latents = [outputs[3], outputs[7], outputs[11], *outputs[12:-1]]
    assert len(latents) == points - 1
    return latents


def karras_config() -> dict[str, float]:
    """The Karras grid the k-diffusion schedulers place their sigmas on
    under `use_karras_sigmas`: rho 7 between the table's sigma extremes."""
    return dict(sigma_min=float(SIGMA_VE[0]), sigma_max=float(SIGMA_VE[-1]), rho=7.0)


def kdpm2_walk(build, fed, case: Case, x: torch.Tensor) -> list[torch.Tensor]:
    """Two scheduler calls per interval, the first at the grid sigma and the
    second at the interpolated one; the final Euler step onto sigma 0, past
    Dew's grid, is not taken."""
    scheduler = build(trained_betas=case.betas, **case.config)
    points = len(case.grid)
    scheduler.set_timesteps(points)
    ancestral = case.scheduler == "KDPM2AncestralDiscreteScheduler"

    def sigma(at: int) -> torch.Tensor:
        if at % 2 == 0:
            return scheduler.sigmas[at]
        return scheduler.sigmas_interpol[at - 1 if ancestral else at]

    with fed():
        latents = stepped(scheduler, case, x, lambda read, time, at: ve_epsilon(read, sigma(at)),
                          2 * (points - 1), reads_latent=True)
    # The latents at the grid points, each interval's second call's.
    return latents[1::2]


def lms_walk(build, fed, case: Case, x: torch.Tensor) -> list[torch.Tensor]:
    """One call per interval on the same Karras grid; `scale_model_input`
    is called as a pipeline would, and the final step onto sigma 0 is not
    taken."""
    scheduler = build(trained_betas=case.betas, **case.config)
    points = len(case.grid)
    scheduler.set_timesteps(points)
    return stepped(scheduler, case, x, lambda read, time, at: ve_epsilon(read, scheduler.sigmas[at]),
                   points - 1, reads_latent=True)


def edm_walk(build, fed, case: Case, x: torch.Tensor) -> list[torch.Tensor]:
    """EDM's Karras grid already ends at sigma_min, and the `sigma_min`
    terminal repeats it, so the scheduler's last step is the identity and
    is not taken."""
    scheduler = build(**case.config)
    points = len(case.grid)
    scheduler.set_timesteps(points)
    with fed():
        return stepped(scheduler, case, x, lambda scaled, time, at: edm_raw(scaled, scheduler.sigmas[at]),
                       points - 1)


WALKS: dict[str, Callable[..., list[torch.Tensor]]] = {
    "DPMSolverMultistepScheduler": lambda_walk,
    "DPMSolverSinglestepScheduler": lambda_walk,
    "DEISMultistepScheduler": lambda_walk,
    "UniPCMultistepScheduler": lambda_walk,
    "LCMScheduler": lambda_walk,
    "TCDScheduler": lambda_walk,
    "PNDMScheduler": pndm_walk,
    "KDPM2DiscreteScheduler": kdpm2_walk,
    "KDPM2AncestralDiscreteScheduler": kdpm2_walk,
    "LMSDiscreteScheduler": lms_walk,
    "EDMDPMSolverMultistepScheduler": edm_walk,
}
"""Each scheduler's walk, the tool's own; its module is the one that defines it (`scheduler_module`)."""


def solver(scheduler: str, grid, **defaults):
    """A factory of `scheduler`'s cases, each an epsilon prediction with `defaults` beside the
    case's own config, on `grid(scheduler, config, steps)` of its `steps` (STEPS unless it names them)."""
    def case(**config) -> Case:
        steps = config.pop("steps", STEPS)
        config = dict(prediction_type="epsilon", **defaults, **config)
        return Case(scheduler, config, grid(scheduler, config, steps))

    return case


def dew_steps(scheduler: str, config: dict, steps: int) -> np.ndarray:
    """Dew's own grid of `steps` (`dew_grid`), whatever the scheduler."""
    return dew_grid(steps)


multistep = solver("DPMSolverMultistepScheduler", dew_steps, final_sigmas_type="sigma_min",
                   thresholding=False)
singlestep = solver("DPMSolverSinglestepScheduler", dew_steps, final_sigmas_type="sigma_min")
deis = solver("DEISMultistepScheduler", published_grid, timestep_spacing="trailing")
unipc = solver("UniPCMultistepScheduler", published_grid, timestep_spacing="trailing",
               final_sigmas_type="sigma_min")


def distilled(scheduler: str, **step_kwargs) -> Case:
    """LCM and TCD on their own four-step schedule out of the fifty-step
    distillation grid, then the table's index 0 where the last step lands
    when the scheduler runs out of timesteps."""
    config = dict(prediction_type="epsilon", original_inference_steps=50, timestep_spacing="leading")
    probe = getattr(scheduler_module(scheduler), scheduler)(
        trained_betas=BETAS, **config)
    probe.set_timesteps(4)
    grid = np.concatenate([probe.timesteps.numpy(), [0]])
    return Case(scheduler, config, grid, step_kwargs=step_kwargs)


VE_POINTS = STEPS
VE_GRID = np.linspace(1.0, 0.0, VE_POINTS, dtype=np.float32)

CASES: dict[str, Case] = {
    # DPM-Solver multistep: the four algorithms, the three orders, both
    # second-order forms, the short-run taper and the Euler final step.
    "dpm_multistep.pp_2m": multistep(algorithm_type="dpmsolver++", solver_order=2, solver_type="midpoint",
                                    lower_order_final=False),
    "dpm_multistep.pp_2m_short_no_taper": multistep(algorithm_type="dpmsolver++", solver_order=2,
                                                   lower_order_final=False, steps=11),
    "dpm_multistep.pp_2h": multistep(algorithm_type="dpmsolver++", solver_order=2, solver_type="heun"),
    "dpm_multistep.pp_3": multistep(algorithm_type="dpmsolver++", solver_order=3),
    "dpm_multistep.dpm_2m": multistep(algorithm_type="dpmsolver", solver_order=2, solver_type="midpoint"),
    "dpm_multistep.dpm_2h": multistep(algorithm_type="dpmsolver", solver_order=2, solver_type="heun"),
    "dpm_multistep.dpm_3": multistep(algorithm_type="dpmsolver", solver_order=3),
    "dpm_multistep.sde_pp_2m": multistep(algorithm_type="sde-dpmsolver++", solver_order=2, solver_type="midpoint"),
    "dpm_multistep.sde_pp_2h": multistep(algorithm_type="sde-dpmsolver++", solver_order=2, solver_type="heun"),
    "dpm_multistep.sde_pp_3": multistep(algorithm_type="sde-dpmsolver++", solver_order=3),
    "dpm_multistep.sde_2m": multistep(algorithm_type="sde-dpmsolver", solver_order=2, solver_type="midpoint"),
    "dpm_multistep.sde_2h": multistep(algorithm_type="sde-dpmsolver", solver_order=2, solver_type="heun"),
    "dpm_multistep.pp_3_short": multistep(algorithm_type="dpmsolver++", solver_order=3, steps=11),
    "dpm_multistep.pp_2_euler_final": multistep(algorithm_type="dpmsolver++", solver_order=2, euler_at_final=True),
    # DPM-Solver singlestep: the order list with and without the final
    # lowering, the midpoint third order that keeps one difference, the
    # noise-prediction algorithm and the stochastic one.
    "dpm_singlestep.pp_2": singlestep(algorithm_type="dpmsolver++", solver_order=2, lower_order_final=False),
    "dpm_singlestep.pp_2_uneven": singlestep(algorithm_type="dpmsolver++", solver_order=2, steps=20),
    "dpm_singlestep.pp_3_uneven": singlestep(algorithm_type="dpmsolver++", solver_order=3, solver_type="heun"),
    "dpm_singlestep.pp_3h_final": singlestep(algorithm_type="dpmsolver++", solver_order=3, solver_type="heun",
                                             lower_order_final=True),
    "dpm_singlestep.pp_3m": singlestep(algorithm_type="dpmsolver++", solver_order=3, solver_type="midpoint",
                                       lower_order_final=False, steps=22),
    "dpm_singlestep.dpm_3h_final": singlestep(algorithm_type="dpmsolver", solver_order=3, solver_type="heun",
                                              lower_order_final=True),
    "dpm_singlestep.sde_pp_2": singlestep(algorithm_type="sde-dpmsolver++", solver_order=2, lower_order_final=False),
    "dpm_singlestep.sde_pp_3h_final": singlestep(algorithm_type="sde-dpmsolver++", solver_order=3, solver_type="heun",
                                                 lower_order_final=True),
    # DEIS: both higher orders and the short-run taper.
    "deis.2": deis(solver_order=2),
    "deis.3": deis(solver_order=3),
    "deis.3_short": deis(solver_order=3, steps=11),
    # UniPC: both B(h) forms, both prediction spaces, the third order and
    # steps with the corrector switched off.
    "unipc.bh2_2": unipc(solver_order=2, solver_type="bh2"),
    "unipc.bh1_3": unipc(solver_order=3, solver_type="bh1"),
    "unipc.eps_bh2_2": unipc(solver_order=2, solver_type="bh2", predict_x0=False),
    "unipc.bh2_3_short_nocorr": unipc(solver_order=3, solver_type="bh2", disable_corrector=[2, 5], steps=11),
    # PNDM: the PLMS form Stable Diffusion runs and the paper's Runge-Kutta warmup.
    "pndm.plms": Case("PNDMScheduler", dict(skip_prk_steps=True, set_alpha_to_one=False,
                                             timestep_spacing="leading", steps_offset=0),
                      np.arange(950, -1, -50)),
    "pndm.prk": Case("PNDMScheduler", dict(skip_prk_steps=False, set_alpha_to_one=False,
                                            timestep_spacing="leading", steps_offset=0),
                     np.arange(950, -1, -50)),
    # Distilled-model schedulers.
    "lcm.4": distilled("LCMScheduler"),
    "tcd.eta_0": distilled("TCDScheduler", eta=0.0),
    "tcd.eta_03": distilled("TCDScheduler", eta=0.3),
    "tcd.eta_1": distilled("TCDScheduler", eta=1.0),
    # k-diffusion samplers on the Karras grid.
    "kdpm2.plain": Case("KDPM2DiscreteScheduler", dict(use_karras_sigmas=True, prediction_type="epsilon"),
                        VE_GRID, convention="ve"),
    "kdpm2.ancestral": Case("KDPM2AncestralDiscreteScheduler",
                            dict(use_karras_sigmas=True, prediction_type="epsilon"), VE_GRID, convention="ve"),
    "lms.4": Case("LMSDiscreteScheduler", dict(use_karras_sigmas=True, prediction_type="epsilon"), VE_GRID,
                  convention="ve", step_kwargs=dict(order=4)),
    "lms.2": Case("LMSDiscreteScheduler", dict(use_karras_sigmas=True, prediction_type="epsilon"), VE_GRID,
                  convention="ve", step_kwargs=dict(order=2)),
    # EDM: the same DPM-Solver++ update over EDM preconditioning on Karras' grid.
    "edm_dpm.pp_2m": Case("EDMDPMSolverMultistepScheduler",
                          dict(EDM, solver_order=2, solver_type="midpoint", algorithm_type="dpmsolver++",
                               final_sigmas_type="sigma_min"), VE_GRID, convention="edm"),
    "edm_dpm.sde_pp_2m": Case("EDMDPMSolverMultistepScheduler",
                              dict(EDM, solver_order=2, solver_type="midpoint", algorithm_type="sde-dpmsolver++",
                                   final_sigmas_type="sigma_min"), VE_GRID, convention="edm"),
}


def initial_scale(case: Case) -> float:
    """The marginal standard deviation of unit data at the top of the walk,
    what `Process.noise` draws x_T with."""
    if case.convention == "vp":
        return 1.0
    if case.convention == "ve":
        return float((SIGMA_VE[-1] ** 2 + 1).sqrt())
    return (EDM["sigma_max"] ** 2 + 1) ** 0.5


def run(name: str, case: Case) -> dict[str, np.ndarray]:
    generator = torch.Generator().manual_seed(0)
    x_T = torch.randn(SHAPE, generator=generator, dtype=torch.float64) * initial_scale(case)
    cotangent = torch.randn(SHAPE, generator=generator, dtype=torch.float64)
    x_T.requires_grad_(True)
    module, walk = scheduler_module(case.scheduler), WALKS[case.scheduler]
    noises = step_noise(SHAPE, len(case.grid) - 1, np.float32)
    # Diffusers' own walk, its float64 input's (the open cases' one-draw check, `OPEN_CASES`).
    latents = walk(*made(module, case, noises, rederived=False), case, x_T)
    assert len(latents) == len(case.grid) - 1, (name, len(latents), len(case.grid))
    (grad,) = torch.autograd.grad((latents[-1] * cotangent).sum(), x_T)
    return {
        "x_T": x_T.detach().numpy().astype(np.float32),
        "latents": np.stack([latent.detach().numpy() for latent in latents]).astype(np.float32),
        "cotangent": cotangent.numpy().astype(np.float32),
        "grad": grad.numpy().astype(np.float32),
        "grid": np.asarray(case.grid, np.float32),
    }


def seeded(case: Case, inputs: Mapping[str, np.ndarray]) -> Case:
    """`case` on its draw's own schedule (`diffusion_support.seeded_schedule`, which Dew's walk reads too):
    its timesteps, its table's tail or its config's ends moved, each an input of its public setup."""
    sys.path.insert(0, str(FIXTURES.parents[1]))
    from diffusion_support import seeded_schedule

    schedule = seeded_schedule(case.scheduler, case.convention, case.grid, np.asarray(inputs["jitter"]),
                               betas=BETAS, karras=karras_config(), edm=EDM)
    offset = schedule.config.get("steps_offset", 0) - case.config.get("steps_offset", 0)
    return replace(case, config={**case.config, **schedule.config}, timesteps=schedule.timesteps,
                   betas=BETAS if schedule.betas is None else schedule.betas, grid=case.grid + offset)


def trajectory(case: Case, inputs: Mapping[str, np.ndarray], dtype: torch.dtype,
               noises: list[np.ndarray]) -> dict[str, np.ndarray]:
    """`case`'s walk in `dtype` from a draw's x_T on its own schedule (`seeded`): its latent after each grid
    interval (`after <i>`) and the VJP of the last against the draw's cotangent (`grad`), the per-step draws
    `noises`."""
    case = seeded(case, inputs)
    module, walk = scheduler_module(case.scheduler), WALKS[case.scheduler]
    x_T = torch.from_numpy(inputs["x_T"]).to(dtype).requires_grad_()
    latents = walk(*made(module, case, noises, rederived=dtype == torch.float64), case, x_T)
    assert len(latents) == len(case.grid) - 1, (case.scheduler, len(latents), len(case.grid))
    (grad,) = torch.autograd.grad((latents[-1] * torch.from_numpy(inputs["cotangent"]).to(dtype)).sum(), x_T)
    return {**{f"after {index}": latent.detach().numpy() for index, latent in enumerate(latents)},
            "grad": grad.numpy()}


def seeds(family: str) -> Path:
    """Family `family`'s cases (the name before the dot) over the seeds check 1 draws (`record_case`): a
    seed's x_T at the case's top marginal deviation and its cotangent (tests/diffusion_support.py's
    `scheduler_draw`), the float32 walk from it and its float64 twin, each fed the draws Dew's solver
    makes at its key 0, the count confirmed paired with Dew's own walk (tests/test_samplers.py's
    `diffusers_walked`); with each case's grid (`<case>.grid`) and the deviation x_T was drawn at
    (`<case>.scale`). Returns the file written."""
    import functools

    import diffusers
    import jax

    sys.path.insert(0, str(FIXTURES.parents[1]))
    from diffusion_support import SCHEDULER_SHAPE, scheduler_draw
    from recorded_references import provenance, record_case, save
    from reference_releases import require
    from test_samplers import diffusers_walked

    require("diffusers-0.34")
    torch.set_num_threads(2)
    named = {name: case for name, case in CASES.items() if name.split(".")[0] == family}
    if not named:
        raise SystemExit(f"no cases in a family named {family!r}")
    arrays: dict[str, np.ndarray] = {}
    for name, case in named.items():
        grid, scale = np.asarray(case.grid, np.float32), np.float32(initial_scale(case))
        noises = step_noise(SCHEDULER_SHAPE, len(case.grid) - 1, np.float64)

        def evaluate(inputs, case=case, noises=noises):
            single = trajectory(case, inputs, torch.float32, noises)
            wide = trajectory(case, inputs, torch.float64, noises)
            return single, wide

        walked = diffusers_walked(name, grid)

        def dew(inputs, walked=walked):
            return jax.tree.map(np.asarray, walked(inputs))

        # Sized paired, as check 1's verdict pairs Dew with the reference.
        arrays.update(record_case(name, functools.partial(scheduler_draw, scale=scale), evaluate,
                                  store_inputs=False, dew=dew))
        arrays[f"{name}.grid"], arrays[f"{name}.scale"] = grid, np.asarray(scale)
        print(f"{name}: {int(arrays[f'{name}.seeds'])} seeds")
    version = diffusers.__version__
    command = f"PYTHONPATH=src:tools:tests python tools/diffusers_reference.py seeds {family}"
    return save(SEEDS / f"{family}.npz", {**arrays, "provenance": provenance(
        "diffusers", version, "cpu", command, torch=torch.__version__)},
        reference=f"huggingface/diffusers {version}", license="Apache-2.0")


def main() -> None:
    from reference_releases import require

    require("diffusers")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    cases: dict[str, dict[str, object]] = {}
    record: dict[str, object] = {
        "diffusers": __import__("diffusers").__version__,
        "train_steps": TRAIN_STEPS, "data_std": DATA_STD, "shape": list(SHAPE),
        "karras": karras_config(), "edm": EDM, "cases": cases,
    }
    for name, case in CASES.items():
        result = run(name, case)
        for key, value in result.items():
            arrays[f"{name}.{key}"] = value
        cases[name] = {
            "scheduler": case.scheduler, "config": case.config, "convention": case.convention,
            "step_kwargs": case.step_kwargs, "intervals": len(case.grid) - 1,
            "latent_scale": float(np.abs(result["latents"]).max()),
        }
        print(f"{name}: {len(case.grid) - 1} intervals, |latent| <= {float(np.abs(result['latents']).max()):.3g}")
    np.savez(FIXTURES / "schedulers.npz", allow_pickle=False, **arrays)
    (FIXTURES / "schedulers.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in FIXTURES.iterdir())
    print(f"{FIXTURES}: {size / 1e3:.0f} kB, {sorted(p.name for p in FIXTURES.iterdir())}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["seeds"]:
        print(seeds(sys.argv[2]))
    else:
        main()
