"""Diffusers 0.34.0 scheduler files reconstructed from their own configs.

`tools/diffusers_reference.py` proves the solver updates over grids Dew's own
processes supply. This tool proves the other half: that
`SourceSchedule.from_config` rebuilds what a published `scheduler_config.json`
means. Every case constructs the actual scheduler class, saves its authentic
config with `save_pretrained` *before* `set_timesteps` (DPM singlestep rewrites
its own `lower_order_final` there, and a config saved afterwards would hide
that policy), calls the actual `set_timesteps`, and then walks every scheduler
call a pipeline would make: `scale_model_input`, the model, `step`, including
the repeated stage rows the two-evaluation classes take.

What lands per case: the saved config, the beta table, the source timesteps and
sigmas, its `init_noise_sigma`, the latent after every grid interval and the
vector-Jacobian product of the last one against a fixed cotangent through the
whole trajectory, in float64 on the torch side.

The model is the same closed form on both sides, a function of the scaled model
input and the model time, so the comparison is of scheduler policy alone. A
stochastic class is fed the exact draws Dew's solver makes at that step;
`DPMSolverSDEScheduler`'s Brownian sampler is fed Dew's own bridge over the
interval the source builds its tree on, and its own `torchsde` tree is
recorded separately so the bridge's identities are checked against real ones.

Run in the isolated reference environment on CPU:

    PYTHONPATH=src:/tmp/dew-sched-ref/libs python tools/diffusers_source_reference.py
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch
import transformers.utils as transformers_utils

# Diffusers 0.34.0's pipeline modules import two names Transformers dropped
# after 4.x. The guidance-rescale helper under test lives beside them, so the
# names are restored rather than the helper copied.
for _name, _value in (("FLAX_WEIGHTS_NAME", "flax_model.msgpack"),
                      ("WEIGHTS_INDEX_NAME", "pytorch_model.bin.index.json")):
    if not hasattr(transformers_utils, _name):
        setattr(transformers_utils, _name, _value)

import jax
import jax.numpy as jnp
from diffusers.schedulers import (
    scheduling_ddim, scheduling_ddpm, scheduling_deis_multistep,
    scheduling_dpmsolver_multistep, scheduling_dpmsolver_sde,
    scheduling_dpmsolver_singlestep, scheduling_edm_dpmsolver_multistep,
    scheduling_euler_ancestral_discrete, scheduling_euler_discrete,
    scheduling_heun_discrete, scheduling_k_dpm_2_ancestral_discrete,
    scheduling_k_dpm_2_discrete, scheduling_lcm, scheduling_lms_discrete, scheduling_pndm,
    scheduling_tcd, scheduling_unipc_multistep,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg

from dew.diffusion.schedules.source import SourceSchedule
from dew.sampling.solvers import MAX_BROWNIAN_DEPTH, _Brownian, _brownian_noise

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "diffusers"
SHAPE = (2, 3, 4)
STEPS = 5
DISTILLED_STEPS = 4
SEED = 11
BROWNIAN_DEPTH = MAX_BROWNIAN_DEPTH
GUIDANCE_SCALE = 3.0
LABEL = 0.7

MODULES: Mapping[str, ModuleType] = {
    "DDIMScheduler": scheduling_ddim,
    "PNDMScheduler": scheduling_pndm,
    "DDPMScheduler": scheduling_ddpm,
    "LMSDiscreteScheduler": scheduling_lms_discrete,
    "EulerDiscreteScheduler": scheduling_euler_discrete,
    "EulerAncestralDiscreteScheduler": scheduling_euler_ancestral_discrete,
    "HeunDiscreteScheduler": scheduling_heun_discrete,
    "KDPM2DiscreteScheduler": scheduling_k_dpm_2_discrete,
    "KDPM2AncestralDiscreteScheduler": scheduling_k_dpm_2_ancestral_discrete,
    "DPMSolverSDEScheduler": scheduling_dpmsolver_sde,
    "DPMSolverMultistepScheduler": scheduling_dpmsolver_multistep,
    "DPMSolverSinglestepScheduler": scheduling_dpmsolver_singlestep,
    "DEISMultistepScheduler": scheduling_deis_multistep,
    "UniPCMultistepScheduler": scheduling_unipc_multistep,
    "EDMDPMSolverMultistepScheduler": scheduling_edm_dpmsolver_multistep,
    "LCMScheduler": scheduling_lcm,
    "TCDScheduler": scheduling_tcd,
}
TWO_STAGE = ("HeunDiscreteScheduler", "KDPM2DiscreteScheduler",
             "KDPM2AncestralDiscreteScheduler", "DPMSolverSDEScheduler")


@dataclass(frozen=True)
class Case:
    """One published scheduler file: the class, the config it was written with,
    and the number of steps the walk asks for."""

    scheduler: str
    config: Mapping[str, object] = field(default_factory=dict)
    steps: int = STEPS
    guidance: float | None = None


def oracle(model_input: torch.Tensor, time: torch.Tensor, label: float = 0.0) -> torch.Tensor:
    """The model both sides run: a bounded nonlinear function of the scaled
    input and the model time, so the trajectory and its Jacobian depend on
    every scheduler decision and on nothing else."""
    scalar = torch.as_tensor(time, dtype=torch.float64)
    return torch.sin(model_input) * 0.07 + scalar * 0.001 + label


@contextlib.contextmanager
def fed_noise(module: ModuleType, noises: list[np.ndarray]) -> Iterator[list[int]]:
    """The module's `randn_tensor` handing out `noises` in order, so a
    stochastic class integrates the draws Dew's solver makes. The yielded list
    counts the draws the walk actually took."""
    taken: list[int] = []

    def draw(shape, generator=None, device=None, dtype=None, layout=None):
        noise = torch.tensor(noises[len(taken)], dtype=dtype or torch.float64)
        assert tuple(noise.shape) == tuple(shape), (noise.shape, shape)
        taken.append(1)
        return noise

    if getattr(module, "randn_tensor", None) is None:
        yield taken
        return
    with patch.object(module, "randn_tensor", draw):
        yield taken


def step_noise(count: int) -> list[np.ndarray]:
    """One standard normal per step under the walk's key folded with the step
    index, which is what `sample` hands its solver."""
    key = jax.random.PRNGKey(0)
    return [np.asarray(jax.random.normal(jax.random.fold_in(key, index), SHAPE, jnp.float32),
                       np.float64) for index in range(count)]


class FedBrownian:
    """`BrownianTreeNoiseSampler` replaced by Dew's own bridge, so the source's
    update algebra runs on the path Dew's solver integrates.

    A Brownian path is rough: its slope inside the finest cell is about
    1 / sqrt(cell width), so a query moved by one float32 step in sigma moves
    the increment by far more than the trajectory tolerance. The queries are
    therefore prepared from the native grid's own float32 levels, in the order
    the source asks for them, and each request is checked against the level it
    was prepared for; that pins the placement of every query while both sides
    read one path.
    """

    calls: list[tuple[float, float]] = []
    expected: list[tuple[float, float]] = []
    queue: list[np.ndarray] = []
    bounds: tuple[float, float] = (0.0, 0.0)

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda value: value):
        assert transform(torch.as_tensor(2.0)).item() == 2.0, "the source transform is the identity"
        assert seed is None, "a seeded source tree is Torch's own stream, not a coupled path"
        FedBrownian.bounds = (float(sigma_min), float(sigma_max))
        FedBrownian.calls = []
        self.shape = tuple(x.shape)

    def __call__(self, sigma, sigma_next):
        first, second = float(sigma), float(sigma_next)
        index = len(FedBrownian.calls)
        FedBrownian.calls.append((first, second))
        for asked, prepared in zip((first, second), FedBrownian.expected[index]):
            assert abs(asked - prepared) <= 1e-5 * max(abs(prepared), 1e-3), (
                f"query {index} asks for {(first, second)}, prepared {FedBrownian.expected[index]}")
        return torch.tensor(FedBrownian.queue[index], dtype=torch.float64)


def brownian_queue(config: Mapping[str, object], steps: int, key) -> tuple[
        list[np.ndarray], list[tuple[float, float]], tuple[float, float]]:
    """The increments Dew's solver draws over a prepared grid, in the order the
    source's two stages ask for them, with the levels each was drawn over.

    Both the levels and the geometric midpoint are computed the way the solver
    computes them, in float32 through the same operations, so the path is the
    solver's own rather than a re-derivation of it."""
    schedule = SourceSchedule.from_config(config)
    process, times = schedule.sampling(steps)
    grid = process.sampler_schedule.sigmas(times)
    state = _Brownian(key, jnp.asarray(process.sampler_schedule.sigma_min, jnp.float32),
                      jnp.asarray(process.sampler_schedule.sigma_max, jnp.float32))
    queue, expected = [], []
    for index in range(len(grid) - 1):
        first, second = grid[index], grid[index + 1]
        if float(second) == 0.0:  # the source's terminal step is deterministic
            continue
        middle = jnp.exp(0.5 * (jnp.log(first) + jnp.log(second)))
        for target in (middle, second):
            queue.append(np.asarray(_brownian_noise(state, first, target, SHAPE, BROWNIAN_DEPTH),
                                    np.float64))
            expected.append((float(first), float(target)))
    return queue, expected, (float(state.low), float(state.high))


def walk(scheduler, module: ModuleType, case: Case, x: torch.Tensor,
         noises: list[np.ndarray]) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    """Every scheduler call the walk makes, keeping the latent at each grid
    interval's end: one call per interval, two for the classes that evaluate
    the model at an interpolated stage, and PNDM's own warmup timetable."""
    times = scheduler.timesteps
    last = len(times) - 1
    latents, inputs = [], []
    with fed_noise(module, noises):
        for index, time in enumerate(times):
            model_input = scheduler.scale_model_input(x, time)
            inputs.append(model_input.detach().numpy().astype(np.float32))
            output = oracle(model_input, time)
            if case.scheduler == "PNDMScheduler":
                x = scheduler.step(output, int(time), x).prev_sample
            else:
                x = scheduler.step(output, time, x).prev_sample
            if case.scheduler in TWO_STAGE:
                if index % 2 == 1 or index == last:
                    latents.append(x)
            else:
                latents.append(x)
    if case.scheduler == "PNDMScheduler":
        latents = (latents[1:] if scheduler.config.skip_prk_steps
                   else [latents[3], latents[7], latents[11]] + latents[12:])
    return latents, inputs


def guided_walk(scheduler, module: ModuleType, case: Case, x: torch.Tensor,
                noises: list[np.ndarray]) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    """The same walk under the pipeline's classifier-free guidance: both raw
    predictions, the guided combination, the source's own `rescale_noise_cfg`
    and only then the scheduler's conversion."""
    latents, inputs = [], []
    with fed_noise(module, noises):
        for time in scheduler.timesteps:
            model_input = scheduler.scale_model_input(x, time)
            inputs.append(model_input.detach().numpy().astype(np.float32))
            conditional = oracle(model_input, time, LABEL)
            unconditional = oracle(model_input, time)
            output = unconditional + GUIDANCE_SCALE * (conditional - unconditional)
            assert case.guidance is not None
            output = rescale_noise_cfg(output, conditional, guidance_rescale=case.guidance)
            x = scheduler.step(output, time, x).prev_sample
            latents.append(x)
    return latents, inputs


VP = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
          beta_schedule="scaled_linear")
EDM = dict(sigma_min=0.002, sigma_max=80.0, sigma_data=0.5, num_train_timesteps=1000)

CASES: Mapping[str, Case] = {
    # DDIM and PNDM: rounded model times over a fixed training stride, the
    # clip the default already applies, thresholding, and the trailing
    # zero-terminal-SNR velocity pairing those checkpoints ship.
    "ddim.default": Case("DDIMScheduler", VP),
    "ddim.trailing_zero_snr_v": Case("DDIMScheduler", dict(
        VP, timestep_spacing="trailing", rescale_betas_zero_snr=True,
        prediction_type="v_prediction", clip_sample=False, set_alpha_to_one=False)),
    "ddim.threshold": Case("DDIMScheduler", dict(
        VP, clip_sample=False, thresholding=True, dynamic_thresholding_ratio=0.9,
        sample_max_value=1.5, timestep_spacing="linspace")),
    "pndm.default": Case("PNDMScheduler", VP),
    "pndm.plms_linspace": Case("PNDMScheduler", dict(
        VP, skip_prk_steps=True, timestep_spacing="linspace", steps_offset=1)),
    # DDPM: the grid's own previous-timestep policy, both fixed variances and
    # the two x_0 limits, with epsilon recomputed from the limited x_0.
    "ddpm.default": Case("DDPMScheduler", VP),
    "ddpm.large_trailing": Case("DDPMScheduler", dict(
        VP, variance_type="fixed_large", timestep_spacing="trailing", clip_sample=False)),
    "ddpm.threshold_v": Case("DDPMScheduler", dict(
        VP, thresholding=True, prediction_type="v_prediction", timestep_spacing="linspace",
        clip_sample=False)),
    "ddpm.small_log": Case("DDPMScheduler", dict(VP, variance_type="fixed_small_log")),
    # LMS, Euler, Euler ancestral and Heun: paired VE grids, the three sigma
    # transformations over the interpolated subset, and Heun's clipped stages.
    "lms.default": Case("LMSDiscreteScheduler", VP),
    "lms.karras_leading": Case("LMSDiscreteScheduler", dict(
        VP, use_karras_sigmas=True, timestep_spacing="leading")),
    "lms.exponential_v": Case("LMSDiscreteScheduler", dict(
        VP, use_exponential_sigmas=True, prediction_type="v_prediction")),
    "euler.default": Case("EulerDiscreteScheduler", VP),
    "euler.beta_trailing_min": Case("EulerDiscreteScheduler", dict(
        VP, use_beta_sigmas=True, timestep_spacing="trailing", final_sigmas_type="sigma_min")),
    "euler.zero_snr_v": Case("EulerDiscreteScheduler", dict(
        VP, rescale_betas_zero_snr=True, prediction_type="v_prediction",
        timestep_spacing="trailing")),
    "euler_ancestral.default": Case("EulerAncestralDiscreteScheduler", VP),
    "euler_ancestral.leading_zero_snr_v": Case("EulerAncestralDiscreteScheduler", dict(
        VP, timestep_spacing="leading", rescale_betas_zero_snr=True,
        prediction_type="v_prediction")),
    "heun.default": Case("HeunDiscreteScheduler", VP),
    "heun.karras_clip": Case("HeunDiscreteScheduler", dict(
        VP, use_karras_sigmas=True, clip_sample=True, clip_sample_range=0.8)),
    # KDPM2 and DPMSolverSDE: the stage rows between grid points.
    "kdpm2.default": Case("KDPM2DiscreteScheduler", VP),
    "kdpm2.karras_leading_v": Case("KDPM2DiscreteScheduler", dict(
        VP, use_karras_sigmas=True, timestep_spacing="leading", prediction_type="v_prediction")),
    "kdpm2_ancestral.default": Case("KDPM2AncestralDiscreteScheduler", VP),
    "kdpm2_ancestral.exponential": Case("KDPM2AncestralDiscreteScheduler", dict(
        VP, use_exponential_sigmas=True)),
    "dpm_sde.default": Case("DPMSolverSDEScheduler", VP),
    "dpm_sde.karras_v": Case("DPMSolverSDEScheduler", dict(
        VP, use_karras_sigmas=True, prediction_type="v_prediction")),
    # The log-SNR classes: spacing variants, lambda clipping, every sigma
    # transformation, both terminal sigmas and the epsilon-domain algorithms.
    "dpm_multi.default": Case("DPMSolverMultistepScheduler", VP),
    "dpm_multi.karras": Case("DPMSolverMultistepScheduler", dict(VP, use_karras_sigmas=True)),
    "dpm_multi.cosine_clipped": Case("DPMSolverMultistepScheduler", dict(
        VP, beta_schedule="squaredcos_cap_v2", lambda_min_clipped=-5.1)),
    # The cosine table's terminal alpha is about 2e-9, so its largest sigma is
    # about 2e4 and a float32 accumulation of that product differs between
    # Torch and numpy by enough to move a truncated model time. A sigma
    # transformation over that table is therefore not comparable, and a Karras
    # one there also repeats its first model time; the clipped case above
    # covers the cosine table over the range a source actually walks.
    "dpm_multi.exponential_trailing": Case("DPMSolverMultistepScheduler", dict(
        VP, use_exponential_sigmas=True, timestep_spacing="trailing", solver_order=3)),
    "dpm_multi.eps_threshold": Case("DPMSolverMultistepScheduler", dict(
        VP, algorithm_type="dpmsolver", final_sigmas_type="sigma_min", thresholding=True,
        solver_type="heun", timestep_spacing="trailing")),
    "dpm_multi.beta_leading": Case("DPMSolverMultistepScheduler", dict(
        VP, use_beta_sigmas=True, timestep_spacing="leading", euler_at_final=True)),
    "dpm_single.default": Case("DPMSolverSinglestepScheduler", VP),
    "dpm_single.karras_order3": Case("DPMSolverSinglestepScheduler", dict(
        VP, solver_order=3, use_karras_sigmas=True, final_sigmas_type="sigma_min",
        lower_order_final=False)),
    "dpm_single.exponential_zero": Case("DPMSolverSinglestepScheduler", dict(
        VP, use_exponential_sigmas=True, solver_type="heun", lambda_min_clipped=-5.1)),
    "deis.default": Case("DEISMultistepScheduler", VP),
    "deis.karras_trailing": Case("DEISMultistepScheduler", dict(
        VP, use_karras_sigmas=True, timestep_spacing="trailing", solver_order=3)),
    "deis.threshold_v": Case("DEISMultistepScheduler", dict(
        VP, thresholding=True, prediction_type="v_prediction")),
    "unipc.default": Case("UniPCMultistepScheduler", VP),
    "unipc.beta_eps_bh1": Case("UniPCMultistepScheduler", dict(
        VP, use_beta_sigmas=True, predict_x0=False, solver_type="bh1",
        final_sigmas_type="sigma_min", solver_order=3)),
    "unipc.zero_snr_v_leading": Case("UniPCMultistepScheduler", dict(
        VP, rescale_betas_zero_snr=True, prediction_type="v_prediction",
        timestep_spacing="leading", disable_corrector=[0])),
    # EDM: its own sigma convention, both schedules and the signed c_out.
    "edm.default": Case("EDMDPMSolverMultistepScheduler", EDM),
    "edm.exponential_v_threshold": Case("EDMDPMSolverMultistepScheduler", dict(
        EDM, sigma_schedule="exponential", prediction_type="v_prediction", thresholding=True,
        final_sigmas_type="sigma_min", algorithm_type="sde-dpmsolver++")),
    # The distilled schedules: the original grid, the selection out of it, and
    # the clip that runs before the consistency boundary.
    "lcm.default": Case("LCMScheduler", VP, steps=DISTILLED_STEPS),
    # Zero-terminal-SNR leaves alpha exactly zero at the table's last index,
    # where the distilled grid starts, so epsilon has no finite clean
    # prediction there and the checkpoints that ship it predict velocity.
    "lcm.clip_original_25": Case("LCMScheduler", dict(
        VP, clip_sample=True, clip_sample_range=0.9, original_inference_steps=25,
        rescale_betas_zero_snr=True, prediction_type="v_prediction"), steps=DISTILLED_STEPS),
    "tcd.default": Case("TCDScheduler", VP, steps=DISTILLED_STEPS),
    "tcd.original_20_v": Case("TCDScheduler", dict(
        VP, original_inference_steps=20, prediction_type="v_prediction"),
        steps=DISTILLED_STEPS),
    # Guidance rescaling through the pipeline order, over a thresholding
    # scheduler so the conversion order is observable.
    "cfg.epsilon": Case("DPMSolverMultistepScheduler", dict(
        VP, thresholding=True, dynamic_thresholding_ratio=0.9), guidance=0.7),
    "cfg.velocity": Case("DPMSolverMultistepScheduler", dict(
        VP, prediction_type="v_prediction", algorithm_type="dpmsolver",
        final_sigmas_type="sigma_min", thresholding=True), guidance=0.5),
    "cfg.zero_rescale": Case("DPMSolverMultistepScheduler", dict(VP), guidance=0.0),
}


def grid_times(scheduler, case: Case) -> np.ndarray:
    """The model times at the grid points the outer walk visits, out of the
    call list the source runs: PNDM keeps its own ascending grid beside the
    warmup timetable, Heun repeats each point for its corrector, and the
    stage classes interleave a stage row between points."""
    times = scheduler.timesteps.numpy().astype(np.float64)
    if case.scheduler == "PNDMScheduler":
        return np.asarray(scheduler._timesteps, np.float64)[::-1].copy()
    if case.scheduler == "HeunDiscreteScheduler":
        return np.concatenate([times[:1], times[1::2]])
    if case.scheduler in TWO_STAGE:
        return np.concatenate([times[:1], times[2::2]])
    return times


def run(name: str, case: Case) -> dict[str, np.ndarray]:
    module = MODULES[case.scheduler]
    scheduler = getattr(module, case.scheduler)(**case.config)
    with tempfile.TemporaryDirectory(prefix="dew-source-scheduler-") as saved:
        scheduler.save_pretrained(saved)
        config = json.loads((Path(saved) / "scheduler_config.json").read_text())
    scheduler.set_timesteps(case.steps)
    times = scheduler.timesteps.tolist()
    # The source finds its starting step index by matching the first model
    # time, and takes the second match when that time repeats elsewhere in the
    # grid, which shifts its whole walk by one. A grid that repeats its first
    # model time is not a policy this reconstruction models, so a case must
    # not produce one; the deliberate repeats of PNDM's warmup and of the
    # two-evaluation classes are counted from there and are fine.
    assert times.count(times[0]) == 1, f"{name}: the first model time repeats in {times}"
    generator = torch.Generator().manual_seed(SEED)
    prior = float(scheduler.init_noise_sigma)
    x_T = torch.randn(SHAPE, generator=generator, dtype=torch.float64) * prior
    cotangent = torch.randn(SHAPE, generator=generator, dtype=torch.float64)
    x_T.requires_grad_(True)
    arrays: dict[str, np.ndarray] = {}
    if case.scheduler == "DPMSolverSDEScheduler":
        queue, expected, (low, high) = brownian_queue(config, case.steps, jax.random.PRNGKey(0))
        FedBrownian.queue, FedBrownian.expected = queue, expected
        with patch.object(module, "BrownianTreeNoiseSampler", FedBrownian):
            latents, inputs = walk(scheduler, module, case, x_T, [])
        assert len(FedBrownian.calls) == len(queue), (name, len(FedBrownian.calls), len(queue))
        assert abs(FedBrownian.bounds[0] - low) < 1e-6 and abs(FedBrownian.bounds[1] - high) < 1e-4, (
            f"{name}: the source tree spans {FedBrownian.bounds}, Dew prepares {(low, high)}")
        arrays["intervals"] = np.asarray(FedBrownian.calls, np.float64)
        arrays["bounds"] = np.asarray(FedBrownian.bounds, np.float64)
    else:
        noises = step_noise(len(scheduler.timesteps))
        runner = guided_walk if case.guidance is not None else walk
        latents, inputs = runner(scheduler, module, case, x_T, noises)
    assert len(latents) == case.steps, (name, len(latents), case.steps)
    (gradient,) = torch.autograd.grad((latents[-1] * cotangent).sum(), x_T)
    arrays.update({
        "x_T": x_T.detach().numpy().astype(np.float32),
        "latents": np.stack([latent.detach().numpy() for latent in latents]).astype(np.float32),
        "cotangent": cotangent.numpy().astype(np.float32),
        "grad": gradient.numpy().astype(np.float32),
        "times": scheduler.timesteps.numpy().astype(np.float64),
        "prior": np.asarray(prior, np.float64),
        "inputs": np.stack(inputs),
        "grid_times": grid_times(scheduler, case),
        "config": np.asarray(json.dumps(config)),
    })
    if hasattr(scheduler, "betas"):
        arrays["betas"] = scheduler.betas.numpy().astype(np.float32)
    if hasattr(scheduler, "sigmas"):
        arrays["sigmas"] = np.asarray(scheduler.sigmas, np.float64)
    return arrays


def brownian_record() -> dict[str, np.ndarray]:
    """The actual `torchsde` tree the source builds, over the interval a
    published DPMSolverSDE grid prepares.

    Endpoint values, the increments of a nested query set and the same set
    queried in the reverse order land here, so the identities Dew's bridge is
    built on are checked against a real one rather than assumed.
    """
    case = CASES["dpm_sde.default"]
    scheduler = scheduling_dpmsolver_sde.DPMSolverSDEScheduler(**case.config)
    with tempfile.TemporaryDirectory(prefix="dew-source-brownian-") as saved:
        scheduler.save_pretrained(saved)
        config = json.loads((Path(saved) / "scheduler_config.json").read_text())
    scheduler.set_timesteps(case.steps)
    levels = np.asarray(scheduler.sigmas, np.float64)
    low, high = float(levels[levels > 0].min()), float(levels.max())
    sampler = scheduling_dpmsolver_sde.BrownianTreeNoiseSampler
    # Interior points: a query that starts exactly on the tree's own end warns
    # about its float boundary and says nothing extra about the identities.
    points = np.linspace(low, high, 7)[1:-1]
    queries = [(points[0], points[2]), (points[2], points[4]), (points[0], points[4]),
               (points[1], points[3]), (points[0], points[1])]

    def normalized(order):
        tree = sampler(torch.zeros(SHAPE, dtype=torch.float64), low, high, seed=SEED)
        values = {}
        for index in order:
            first, second = queries[index]
            values[index] = tree(torch.as_tensor(first), torch.as_tensor(second)).numpy()
        return np.stack([values[index] for index in range(len(queries))])

    forward = normalized(range(len(queries)))
    reversed_order = normalized(list(reversed(range(len(queries)))))
    tree = sampler(torch.zeros(SHAPE, dtype=torch.float64), low, high, seed=SEED)
    swapped = np.stack([tree(torch.as_tensor(second), torch.as_tensor(first)).numpy()
                        for first, second in queries])
    widths = np.asarray([abs(second - first) for first, second in queries], np.float64)
    return {
        "config": np.asarray(json.dumps(config)),
        "bounds": np.asarray([low, high], np.float64),
        "queries": np.asarray(queries, np.float64),
        "widths": widths,
        "normalized": forward.astype(np.float64),
        "reordered": reversed_order.astype(np.float64),
        "swapped": swapped.astype(np.float64),
    }


def main() -> None:
    import diffusers

    if diffusers.__version__ != "0.34.0":
        raise RuntimeError("Requires diffusers==0.34.0")
    torch.set_num_threads(2)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    record: dict[str, object] = {"diffusers": diffusers.__version__, "shape": list(SHAPE),
                                 "brownian_depth": BROWNIAN_DEPTH, "guidance_scale": GUIDANCE_SCALE,
                                 "label": LABEL, "cases": {}}
    cases: dict[str, dict[str, object]] = record["cases"]  # type: ignore[assignment]
    for name, case in CASES.items():
        result = run(name, case)
        for key, value in result.items():
            arrays[f"{name}.{key}"] = value
        cases[name] = {"scheduler": case.scheduler, "steps": case.steps,
                       "guidance": case.guidance,
                       "latent_scale": float(np.abs(result["latents"]).max())}
        print(f"{name}: {case.steps} intervals, |latent| <= {cases[name]['latent_scale']:.4g}")
    for key, value in brownian_record().items():
        arrays[f"brownian.{key}"] = value
    print(f"brownian: torchsde tree over {arrays['brownian.bounds']}")
    np.savez_compressed(FIXTURES / "source_schedulers.npz", allow_pickle=False, **arrays)
    (FIXTURES / "source_schedulers.json").write_text(json.dumps(record, indent=1) + "\n")
    size = (FIXTURES / "source_schedulers.npz").stat().st_size
    print(f"{FIXTURES / 'source_schedulers.npz'}: {size / 1e3:.0f} kB, {len(CASES)} cases")


if __name__ == "__main__":
    main()
