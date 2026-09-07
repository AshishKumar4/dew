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

- the linear VP beta table Dew's `LinearNoiseScheduler(1000)` tabulates with
  an epsilon oracle: DPM-Solver multistep and singlestep in every algorithm,
  order and solver type Diffusers offers, DEIS, UniPC, PNDM, LCM and TCD;
- the Karras rho-7 grid between that table's sigma extremes, in the
  variance-exploding form k-diffusion samplers integrate: KDPM2, KDPM2
  ancestral and LMS;
- EDM preconditioning on Karras' own grid, sigma 0.002 to 80 at sigma_data
  0.5: the EDM DPM-Solver scheduler.

    PYTHONPATH=src python tools/diffusers_reference.py
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

# The per-step noise is the test suite's draw, which conftest.py makes on
# the CPU; a GPU draw of the same key differs in the last float32 bit and
# the fixture would then not regenerate byte-identically.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import torch
from diffusers.schedulers import (
    scheduling_deis_multistep, scheduling_dpmsolver_multistep, scheduling_dpmsolver_singlestep,
    scheduling_edm_dpmsolver_multistep, scheduling_k_dpm_2_ancestral_discrete,
    scheduling_k_dpm_2_discrete, scheduling_lcm, scheduling_lms_discrete, scheduling_pndm,
    scheduling_tcd, scheduling_unipc_multistep,
)

from dew.diffusion import EpsilonPredictionTransform, LinearNoiseScheduler, Process
from dew.diffusion.schedules.linear import linear_beta_schedule

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "diffusers"
DATA_STD = 0.3
TRAIN_STEPS = 1000
STEPS = 21
SHAPE = (2, 2, 4, 4)
BETAS = linear_beta_schedule(TRAIN_STEPS)
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


def trailing_grid(steps: int) -> np.ndarray:
    """Diffusers' `timestep_spacing="trailing"` for `steps - 1` scheduler
    steps, then the table's index 0 where its sigma_min terminal lands."""
    timesteps = np.arange(TRAIN_STEPS, 0, -TRAIN_STEPS / (steps - 1)).round().astype(np.int64) - 1
    return np.concatenate([timesteps, [0]])


def step_noise(steps: int) -> list[np.ndarray]:
    """The noise `sample` draws at each interval: one standard normal per
    step under the walk's key folded with the step index."""
    key = jax.random.PRNGKey(0)
    return [np.asarray(jax.random.normal(jax.random.fold_in(key, i), SHAPE, dtype=jnp.float32))
            for i in range(steps - 1)]


@contextlib.contextmanager
def fed_noise(module: ModuleType, noises: list[np.ndarray]) -> Iterator[None]:
    """The scheduler module's `randn_tensor` handing out `noises` in order,
    in the dtype the scheduler asks for, so its arithmetic runs on the draws
    Dew's solver makes. A module that never draws has no `randn_tensor`."""
    queue = list(noises)

    def draw(shape, generator=None, device=None, dtype=None, layout=None):
        noise = torch.tensor(queue.pop(0), dtype=dtype)
        assert tuple(noise.shape) == tuple(shape), (noise.shape, shape)
        return noise

    original = getattr(module, "randn_tensor", None)
    if original is None:
        yield
        return
    with patch.object(module, "randn_tensor", draw):
        yield


def vp_epsilon(x: torch.Tensor, index: int) -> torch.Tensor:
    """E[eps | x_t] for x0 ~ N(0, DATA_STD^2) at the table's index:
    x sigma / (alpha^2 s^2 + sigma^2)."""
    alpha_cumprod = ALPHAS_CUMPROD[index]
    alpha, sigma = alpha_cumprod.sqrt(), (1 - alpha_cumprod).sqrt()
    return x * sigma / (alpha ** 2 * DATA_STD ** 2 + sigma ** 2)


def ve_epsilon(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """The same oracle on the variance-exploding latent x = x0 + sigma eps
    the k-diffusion schedulers carry: x sigma / (s^2 + sigma^2)."""
    sigma = sigma.to(x.dtype)
    return x * sigma / (DATA_STD ** 2 + sigma ** 2)


def edm_raw(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """The raw F whose EDM preconditioning c_skip x + c_out F is the optimal
    x0 = x s^2 / (s^2 + sigma^2)."""
    sigma = sigma.to(x.dtype)
    sigma_data = EDM["sigma_data"]
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    x0 = x * DATA_STD ** 2 / (DATA_STD ** 2 + sigma ** 2)
    return (x0 - c_skip * x) / c_out


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


def lambda_walk(module: ModuleType, case: Case, x: torch.Tensor,
                noises: list[np.ndarray]) -> list[torch.Tensor]:
    """The VP-table schedulers whose model runs once per grid point at an
    integer timestep: DPM-Solver multistep and singlestep, DEIS, UniPC, LCM
    and TCD."""
    scheduler = getattr(module, case.scheduler)(trained_betas=BETAS, **case.config)
    timesteps = case.grid[:-1].tolist()
    if "timestep_spacing" in case.config:
        scheduler.set_timesteps(len(timesteps))
        assert scheduler.timesteps.tolist() == timesteps, (scheduler.timesteps, timesteps)
    else:
        scheduler.set_timesteps(timesteps=timesteps)
    latents = []
    with fed_noise(module, noises):
        for timestep in scheduler.timesteps:
            eps = vp_epsilon(x, int(timestep))
            x = scheduler.step(eps, timestep, x, **case.step_kwargs).prev_sample
            latents.append(x)
    return latents


def pndm_walk(module: ModuleType, case: Case, x: torch.Tensor,
              noises: list[np.ndarray]) -> list[torch.Tensor]:
    """PNDM runs its own timestep list: the first interval twice (predictor,
    then corrector) under `skip_prk_steps`, otherwise three Runge-Kutta
    intervals of four evaluations, then one evaluation per interval, and a
    last step below index 0 that `set_alpha_to_one=False` makes the
    identity. The latents at the grid points are kept."""
    scheduler = module.PNDMScheduler(trained_betas=BETAS, **case.config)
    points = len(case.grid)
    scheduler.set_timesteps(points)
    stride = TRAIN_STEPS // points
    assert set(case.grid.tolist()) <= set(scheduler.timesteps.tolist())
    assert all(case.grid[:-1] - case.grid[1:] == stride), (case.grid, stride)
    outputs = []
    for timestep in scheduler.timesteps:
        eps = vp_epsilon(x, int(timestep))
        x = scheduler.step(eps, int(timestep), x).prev_sample
        outputs.append(x)
    if case.config["skip_prk_steps"]:
        latents = outputs[1:-1]
    else:
        latents = [outputs[3], outputs[7], outputs[11]] + outputs[12:-1]
    assert len(latents) == points - 1
    return latents


def karras_config() -> dict[str, float]:
    """The Karras grid the k-diffusion schedulers place their sigmas on
    under `use_karras_sigmas`: rho 7 between the table's sigma extremes."""
    return dict(sigma_min=float(SIGMA_VE[0]), sigma_max=float(SIGMA_VE[-1]), rho=7.0)


def kdpm2_walk(module: ModuleType, case: Case, x: torch.Tensor,
               noises: list[np.ndarray]) -> list[torch.Tensor]:
    """Two scheduler calls per interval, the first at the grid sigma and the
    second at the interpolated one; the final Euler step onto sigma 0, past
    Dew's grid, is not taken."""
    scheduler = getattr(module, case.scheduler)(trained_betas=BETAS, **case.config)
    points = len(case.grid)
    scheduler.set_timesteps(points)
    ancestral = case.scheduler == "KDPM2AncestralDiscreteScheduler"
    latents = []
    with fed_noise(module, noises):
        for call in range(2 * (points - 1)):
            first = call % 2 == 0
            if first:
                sigma = scheduler.sigmas[call]
            else:
                sigma = scheduler.sigmas_interpol[call - 1 if ancestral else call]
            eps = ve_epsilon(x, sigma)
            x = scheduler.step(eps, scheduler.timesteps[call], x).prev_sample
            if not first:
                latents.append(x)
    return latents


def lms_walk(module: ModuleType, case: Case, x: torch.Tensor,
             noises: list[np.ndarray]) -> list[torch.Tensor]:
    """One call per interval on the same Karras grid; `scale_model_input`
    is called as a pipeline would, and the final step onto sigma 0 is not
    taken."""
    scheduler = module.LMSDiscreteScheduler(trained_betas=BETAS, **case.config)
    points = len(case.grid)
    scheduler.set_timesteps(points)
    latents = []
    for call in range(points - 1):
        timestep = scheduler.timesteps[call]
        scheduler.scale_model_input(x, timestep)
        eps = ve_epsilon(x, scheduler.sigmas[call])
        x = scheduler.step(eps, timestep, x, **case.step_kwargs).prev_sample
        latents.append(x)
    return latents


def edm_walk(module: ModuleType, case: Case, x: torch.Tensor,
             noises: list[np.ndarray]) -> list[torch.Tensor]:
    """EDM's Karras grid already ends at sigma_min, and the `sigma_min`
    terminal repeats it, so the scheduler's last step is the identity and
    is not taken."""
    scheduler = module.EDMDPMSolverMultistepScheduler(**case.config)
    points = len(case.grid)
    scheduler.set_timesteps(points)
    latents = []
    with fed_noise(module, noises):
        for call in range(points - 1):
            sigma = scheduler.sigmas[call]
            raw = edm_raw(x, sigma)
            x = scheduler.step(raw, scheduler.timesteps[call], x).prev_sample
            latents.append(x)
    return latents


WALKS: dict[str, tuple[ModuleType, Callable[..., list[torch.Tensor]]]] = {
    "DPMSolverMultistepScheduler": (scheduling_dpmsolver_multistep, lambda_walk),
    "DPMSolverSinglestepScheduler": (scheduling_dpmsolver_singlestep, lambda_walk),
    "DEISMultistepScheduler": (scheduling_deis_multistep, lambda_walk),
    "UniPCMultistepScheduler": (scheduling_unipc_multistep, lambda_walk),
    "LCMScheduler": (scheduling_lcm, lambda_walk),
    "TCDScheduler": (scheduling_tcd, lambda_walk),
    "PNDMScheduler": (scheduling_pndm, pndm_walk),
    "KDPM2DiscreteScheduler": (scheduling_k_dpm_2_discrete, kdpm2_walk),
    "KDPM2AncestralDiscreteScheduler": (scheduling_k_dpm_2_ancestral_discrete, kdpm2_walk),
    "LMSDiscreteScheduler": (scheduling_lms_discrete, lms_walk),
    "EDMDPMSolverMultistepScheduler": (scheduling_edm_dpmsolver_multistep, edm_walk),
}


def multistep(**config) -> Case:
    steps = config.pop("steps", STEPS)
    return Case("DPMSolverMultistepScheduler",
                dict(prediction_type="epsilon", final_sigmas_type="sigma_min",
                     thresholding=False, **config), dew_grid(steps))


def singlestep(**config) -> Case:
    steps = config.pop("steps", STEPS)
    return Case("DPMSolverSinglestepScheduler",
                dict(prediction_type="epsilon", final_sigmas_type="sigma_min", **config),
                dew_grid(steps))


def deis(**config) -> Case:
    steps = config.pop("steps", STEPS)
    return Case("DEISMultistepScheduler",
                dict(prediction_type="epsilon", timestep_spacing="trailing", **config),
                trailing_grid(steps))


def unipc(**config) -> Case:
    steps = config.pop("steps", STEPS)
    return Case("UniPCMultistepScheduler",
                dict(prediction_type="epsilon", timestep_spacing="trailing",
                     final_sigmas_type="sigma_min", **config), trailing_grid(steps))


def distilled(scheduler: str, **step_kwargs) -> Case:
    """LCM and TCD on their own four-step schedule out of the fifty-step
    distillation grid, then the table's index 0 where the last step lands
    when the scheduler runs out of timesteps."""
    config = dict(prediction_type="epsilon", original_inference_steps=50, timestep_spacing="leading")
    probe = getattr(scheduling_lcm if scheduler == "LCMScheduler" else scheduling_tcd, scheduler)(
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
    module, walk = WALKS[case.scheduler]
    latents = walk(module, case, x_T, step_noise(len(case.grid)))
    assert len(latents) == len(case.grid) - 1, (name, len(latents), len(case.grid))
    (grad,) = torch.autograd.grad((latents[-1] * cotangent).sum(), x_T)
    return {
        "x_T": x_T.detach().numpy().astype(np.float32),
        "latents": np.stack([latent.detach().numpy() for latent in latents]).astype(np.float32),
        "cotangent": cotangent.numpy().astype(np.float32),
        "grad": grad.numpy().astype(np.float32),
        "grid": np.asarray(case.grid, np.float32),
    }


def main() -> None:
    version = __import__("diffusers").__version__
    if version != "0.34.0":
        raise RuntimeError("Requires diffusers==0.34.0")
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
    main()
