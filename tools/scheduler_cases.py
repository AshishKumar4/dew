"""The published scheduler configs the Diffusers references walk (`CASES`) and the closed-form model each
walk runs (`oracle`): import-safe, so a test reads them without a reference's environment
(tools/diffusers_source_reference.py records them; tests/test_float64_schedulers.py holds the oracles to the
library over them)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import torch

STEPS = 5
DISTILLED_STEPS = 4
SEED = 11


@dataclass(frozen=True)
class Case:
    """One published scheduler file: the class, the config it was written with,
    and the number of steps the walk asks for."""

    scheduler: str
    config: Mapping[str, Any] = field(default_factory=dict)
    steps: int = STEPS
    guidance: float | None = None
    set_with: Mapping[str, Any] = field(default_factory=dict)
    """What its `set_timesteps` is given past the step count: a dynamically shifted flow grid's `mu`, the
    sigmas a pipeline gives, the timesteps a custom grid walks (in place of the count)."""
    step_with: Mapping[str, Any] = field(default_factory=dict)
    """The options each `step` is given (LMS's `order`, TCD's `eta`)."""


def oracle(model_input: torch.Tensor, time: torch.Tensor, label: float = 0.0) -> torch.Tensor:
    """The model both sides run: a bounded nonlinear function of the scaled
    input and the model time, so the trajectory and its Jacobian depend on
    every scheduler decision and on nothing else."""
    # The model time in the input's precision, as Dew's `SourceOracle` reads it.
    scalar = torch.as_tensor(time).to(model_input.dtype)
    return torch.sin(model_input) * 0.07 + scalar * 0.001 + label


Model = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
"""What a walk asks the model of: its scaled input, its time and the step's place in the walk to its output
(`unguided`, `guided`, `flow_model`)."""


def unguided(model_input: torch.Tensor, time: torch.Tensor, at: int) -> torch.Tensor:
    """`oracle` as a walk's model, its place in the walk not read."""
    del at
    return oracle(model_input, time)

GUIDANCE_SCALE = 3.0
LABEL = 0.7
"""Classifier-free guidance's scale and the conditional label the guided cases' model reads (`guided`)."""


def guided(case: Case, rescaling: Callable[[], Callable[..., torch.Tensor]]) -> Model:
    """The model a guided case's pipeline hands its scheduler (`case.guidance` its rescale): both raw
    predictions of `oracle`, the guided combination, then the rescale `rescaling()` loads (Diffusers' own
    `rescale_noise_cfg`, which the caller imports from its pipelines), loaded for a guided case alone;
    `unguided` for an unguided case."""
    if case.guidance is None:
        return unguided
    rescale, rescaled = case.guidance, rescaling()

    def model(model_input: torch.Tensor, time: torch.Tensor, at: int) -> torch.Tensor:
        del at
        conditional, unconditional = oracle(model_input, time, LABEL), oracle(model_input, time)
        output = unconditional + GUIDANCE_SCALE * (conditional - unconditional)
        return rescaled(output, conditional, guidance_rescale=rescale)

    return model


UNSCALED = frozenset({"FlowMatchEulerDiscreteScheduler"})
"""The classes whose pipelines give the model the latent as it is: they state no `scale_model_input`."""


class Called(NamedTuple):
    """One call of a scheduler's step in a walk (`calls`): its place in the walk, what the model read, and
    the latent it stepped to."""

    at: int
    model_input: torch.Tensor
    after: torch.Tensor


def calls(scheduler, case: Case, x: torch.Tensor, model: Model = unguided, *, reads_latent: bool = False,
          drawing: Mapping[str, object] | None = None) -> Iterator[Called]:
    """Each call of `scheduler`'s step (a library class or its oracle) over the grid it holds from `x`, the
    model `model` (`unguided`, a guided case's `guided`, a flow's `flow_model`) told the step's place: its
    input scaled where the class scales it (`UNSCALED`), or the latent as it is where the model `reads_latent`
    (a variance-exploding oracle's E[eps | x], the class's input scaled as its pipeline scales it all the
    same), PNDM's time an integer as its pipeline hands it, each step given `drawing` (a library's
    generator) and `case`'s step options. The one walk the oracles' check (`walked`), the source and the
    fixtures' recorders (tools/diffusers_source_reference.py, tools/diffusers_reference.py), the flow walk
    (`flow_walk`) and the native edges' recorder (tools/native_diffusion_edges_reference.py) take, guided or
    not."""
    for index, time in enumerate(scheduler.timesteps):
        scaled = x if case.scheduler in UNSCALED else scheduler.scale_model_input(x, time)
        model_input = x if reads_latent else scaled
        output = model(model_input, time, index)
        handed = int(time) if case.scheduler == "PNDMScheduler" else time
        after = scheduler.step(output, handed, x, **(drawing or {}), **case.step_with).prev_sample
        yield Called(index, model_input, after)
        x = after


def walked(scheduler, case: Case, x: torch.Tensor, model: Model = unguided,
           **drawing: object) -> dict[str, np.ndarray]:
    """The walk of `scheduler` (a library class or its oracle) over `case`'s grid from `x` (`calls`): its
    latent after each step."""
    # A grid of given timesteps states no step count: the library refuses both.
    scheduler.set_timesteps(None if "timesteps" in case.set_with else case.steps, **case.set_with)
    return {f"after {call.at}": call.after.detach().numpy()
            for call in calls(scheduler, case, x, model, drawing=drawing)}


VP = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear"}
EDM = {"sigma_min": 0.002, "sigma_max": 80.0, "sigma_data": 0.5, "num_train_timesteps": 1000}

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
    # A leading grid ends at index 0, where the source adds no noise at all.
    "ddpm.large_leading": Case("DDPMScheduler", dict(
        VP, variance_type="fixed_large", timestep_spacing="leading", clip_sample=False)),
    # LMS, Euler, Euler ancestral and Heun: paired VE grids, the three sigma
    # transformations over the interpolated subset, and Heun's clipped stages.
    "lms.default": Case("LMSDiscreteScheduler", VP),
    "lms.karras_leading": Case("LMSDiscreteScheduler", dict(
        VP, use_karras_sigmas=True, timestep_spacing="leading")),
    "lms.exponential_v": Case("LMSDiscreteScheduler", dict(
        VP, use_exponential_sigmas=True, prediction_type="v_prediction")),
    "euler.default": Case("EulerDiscreteScheduler", VP),
    # A linspace grid ends at the table's first sigma, which sigma_min appends
    # again: the last interval has no length.
    "euler.linspace_min": Case("EulerDiscreteScheduler", dict(VP, final_sigmas_type="sigma_min")),
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
    # The tree's entropy is the config's, so the recorded path is reproducible.
    "dpm_sde.default": Case("DPMSolverSDEScheduler", dict(VP, noise_sampler_seed=SEED)),
    "dpm_sde.karras_v": Case("DPMSolverSDEScheduler", dict(
        VP, use_karras_sigmas=True, prediction_type="v_prediction",
        noise_sampler_seed=SEED + 1)),
    # The log-SNR classes: spacing variants, lambda clipping, every sigma
    # transformation, both terminal sigmas and the epsilon-domain algorithms.
    "dpm_multi.default": Case("DPMSolverMultistepScheduler", VP),
    "dpm_multi.karras": Case("DPMSolverMultistepScheduler", dict(VP, use_karras_sigmas=True)),
    "dpm_multi.cosine_clipped": Case("DPMSolverMultistepScheduler", dict(
        VP, beta_schedule="squaredcos_cap_v2", lambda_min_clipped=-5.1)),
    "dpm_multi.cosine_exponential": Case("DPMSolverMultistepScheduler", dict(
        VP, beta_schedule="squaredcos_cap_v2", use_exponential_sigmas=True)),
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
    # The same classes on flow sigmas, as SANA ships DPM-Solver++ and Wan
    # UniPC: the shifted flow path, velocity read as a clean prediction, and
    # each class's own terminal sigma.
    "dpm_multi.flow": Case("DPMSolverMultistepScheduler", dict(
        VP, use_flow_sigmas=True, flow_shift=3.0, prediction_type="flow_prediction")),
    "dpm_multi.flow_sde_order3_sigma_min": Case("DPMSolverMultistepScheduler", dict(
        VP, use_flow_sigmas=True, flow_shift=5.0, prediction_type="flow_prediction",
        algorithm_type="sde-dpmsolver++", solver_order=3, final_sigmas_type="sigma_min")),
    "unipc.flow": Case("UniPCMultistepScheduler", dict(
        VP, use_flow_sigmas=True, flow_shift=5.0, prediction_type="flow_prediction")),
    "unipc.flow_order3_sigma_min": Case("UniPCMultistepScheduler", dict(
        VP, use_flow_sigmas=True, flow_shift=3.0, prediction_type="flow_prediction",
        solver_order=3, final_sigmas_type="sigma_min")),
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


FLOW_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    # The files SD3 and SD3.5 ship, plus the controls the class declares that
    # change where its sigmas land.
    "flow.plain": {"num_train_timesteps": 1000},
    "flow.shift3": {"num_train_timesteps": 1000, "shift": 3.0},
    "flow.dynamic": {"num_train_timesteps": 1000, "use_dynamic_shifting": True, "base_shift": 0.5,
                     "max_shift": 1.15, "base_image_seq_len": 256, "max_image_seq_len": 4096},
    "flow.linear_dynamic": {"num_train_timesteps": 1000, "use_dynamic_shifting": True,
                            "time_shift_type": "linear"},
    "flow.terminal": {"num_train_timesteps": 1000, "shift": 3.0, "shift_terminal": 0.1},
    "flow.karras": {"num_train_timesteps": 1000, "shift": 3.0, "use_karras_sigmas": True},
    "flow.exponential": {"num_train_timesteps": 1000, "use_exponential_sigmas": True},
}
"""`FlowMatchEulerDiscreteScheduler`'s configs (tools/diffusers_sd3_reference.py records them)."""

FLOW_GRID = 7
MU = 0.75
"""A dynamically shifted grid's `mu`: what a pipeline's `calculate_shift` gives a 1024-token image is 0.63, a
4096-token one's 1.15."""

FLOW_CASES: Mapping[str, Case] = {
    **{f"{name}.scheduler": Case("FlowMatchEulerDiscreteScheduler", config, FLOW_GRID,
                                 set_with={"mu": MU} if config.get("use_dynamic_shifting") else {})
       for name, config in FLOW_CONFIGS.items()},
    # The sigmas Flux's pipelines give (`np.linspace(1, 1 / steps, steps)`), in float32 as the library
    # takes them, shifted by `mu`.
    **{f"{name}.linspace": Case("FlowMatchEulerDiscreteScheduler", config, FLOW_GRID, set_with={
        "sigmas": np.linspace(1.0, 1 / FLOW_GRID, FLOW_GRID).tolist(),
        **({"mu": MU} if config.get("use_dynamic_shifting") else {})})
       for name, config in FLOW_CONFIGS.items()},
    "flow.inverted": Case("FlowMatchEulerDiscreteScheduler", {"shift": 3.0, "invert_sigmas": True},
                          FLOW_GRID),
    "flow.stochastic": Case("FlowMatchEulerDiscreteScheduler", {"shift": 3.0, "stochastic_sampling": True},
                            FLOW_GRID),
}
"""The flow-matching walks the oracles are held to: each config over its own grid and over a pipeline's
sigmas, inverted, and stochastic."""

GIVEN = (999, 857, 714, 571, 428, 285, 142, 20)
"""A grid of given training timesteps, falling, its last short of the table's start."""

ORACLE_CASES: Mapping[str, Case] = {
    "flow.beta": Case("FlowMatchEulerDiscreteScheduler", {"shift": 3.0, "use_beta_sigmas": True}, FLOW_GRID),
    # Four steps tapered (orders 1, 2, 2, 1): an SDE walk of order 3 that never takes the third order.
    "edm.sde_order3_short": Case("EDMDPMSolverMultistepScheduler", {
        **EDM, "algorithm_type": "sde-dpmsolver++", "solver_order": 3}, 4),
    # The DPM-Solver classes on a grid of given training timesteps (tools/diffusers_reference.py's walks).
    "dpm_multi.given": Case("DPMSolverMultistepScheduler", {
        **VP, "solver_order": 3, "final_sigmas_type": "sigma_min"}, set_with={"timesteps": GIVEN}),
    "dpm_single.given": Case("DPMSolverSinglestepScheduler", {
        **VP, "solver_order": 2, "final_sigmas_type": "sigma_min"}, set_with={"timesteps": GIVEN}),
    # The step's own options.
    "lms.order2": Case("LMSDiscreteScheduler", VP, step_with={"order": 2}),
    "ddim.eta1": Case("DDIMScheduler", {**VP, "clip_sample": False}, step_with={"eta": 1.0}),
    "tcd.eta": Case("TCDScheduler", {}, DISTILLED_STEPS, step_with={"eta": 0.6}),
    "tcd.eta0": Case("TCDScheduler", {}, DISTILLED_STEPS, step_with={"eta": 0.0}),
}
"""Walks only the oracles are held to (tests/test_float64_schedulers.py), no reference records."""


def flow_arguments(config: Mapping[str, Any], steps: int, tokens: int, linspace: bool) -> dict[str, Any]:
    """What a pipeline hands a `FlowMatchEulerDiscreteScheduler`'s `set_timesteps` beside its step count:
    the `mu` its `calculate_shift` gives `tokens` where the file shifts dynamically (all SD3's pipeline
    hands it), and a Flux pipeline's `linspace(1, 1/N, N)` sigmas where `linspace`."""
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    arguments: dict[str, Any] = {}
    if config.get("use_dynamic_shifting"):
        arguments["mu"] = calculate_shift(tokens, config.get("base_image_seq_len", 256),
                                          config.get("max_image_seq_len", 4096),
                                          config.get("base_shift", 0.5), config.get("max_shift", 1.15))
    if linspace:
        arguments["sigmas"] = np.linspace(1.0, 1 / steps, steps)
    return arguments


FLOW = Case("FlowMatchEulerDiscreteScheduler")
"""The flow walk's case (`flow_walk`): its scheduler's class, the latent handed its model as it is."""


def flow_model(scheduler, level: Callable[[int, torch.Tensor], torch.Tensor] | None, *, std: float) -> Model:
    """The velocity of the optimal transport path for x0 ~ N(0, std^2) at the level `level(at, time)` reads
    (the scheduler's own sigma at the step where none)."""
    def model(latent: torch.Tensor, time: torch.Tensor, at: int) -> torch.Tensor:
        sigma = scheduler.sigmas[at] if level is None else level(at, time)
        # E[eps - x0 | x_t] for a Gaussian data law under x = (1-s) x0 + s eps
        variance = (1 - sigma) ** 2 * std ** 2 + sigma ** 2
        x0 = latent * (1 - sigma) * std ** 2 / variance
        eps = latent * sigma / variance
        return eps - x0

    return model


def flow_walk(scheduler, latent, level=None, *, std: float):
    """The Euler walk the source takes from `latent` (`calls`): one model call per interval, the velocity of
    `flow_model`, the scheduler the library's or its oracle (tools/float64_schedulers.py)."""
    return [call.after for call in calls(scheduler, FLOW, latent, flow_model(scheduler, level, std=std))]
