"""The independent float64 oracles of Diffusers' discrete-table schedulers (0.39.0): DDIM, DDPM, PNDM,
LCM and TCD, each its class's tables, grid and update re-derived from its public config (`oracle_of`,
tools/float64_schedulers.py)."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import ClassVar, Literal, overload

import numpy as np
import torch
from float64_oracle import Drawn, Grid, Table, predicted


class DDIM(Table, Drawn):
    """`DDIMScheduler` (`scheduling_ddim.py`), its tables and grid in the oracle's precision, at the step's
    eta (0 its default), its noise the step's `variance_noise` or else its draw."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
        "trained_betas": None, "clip_sample": True, "set_alpha_to_one": True, "steps_offset": 0,
        "prediction_type": "epsilon", "thresholding": False, "dynamic_thresholding_ratio": 0.995,
        "clip_sample_range": 1.0, "sample_max_value": 1.0, "timestep_spacing": "leading",
        "rescale_betas_zero_snr": False}

    @overload
    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None, return_dict: Literal[True] = True, *,
             eta: float = 0.0, use_clipped_model_output: bool = False,
             variance_noise: torch.Tensor | None = None) -> SimpleNamespace: ...

    @overload
    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None, *, return_dict: Literal[False],
             eta: float = 0.0, use_clipped_model_output: bool = False,
             variance_noise: torch.Tensor | None = None) -> tuple[torch.Tensor]: ...

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None, return_dict: bool = True, *,
             eta: float = 0.0, use_clipped_model_output: bool = False,
             variance_noise: torch.Tensor | None = None) -> SimpleNamespace | tuple[torch.Tensor]:
        """The library's `step`, its own options named as the library's are."""
        return self.answered(model_output, timestep, sample, generator, return_dict, {
            "eta": eta, "use_clipped_model_output": use_clipped_model_output,
            "variance_noise": variance_noise})

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor, eta: float = 0.0,
                use_clipped_model_output: bool = False,
                variance_noise: torch.Tensor | None = None) -> torch.Tensor:
        """`step` (`scheduling_ddim.py:384-480`)."""
        timestep = int(timestep)
        before = timestep - self.config.num_train_timesteps // self.steps
        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod[before] if before >= 0 else self.final_alpha_cumprod
        beta_t = 1 - alpha_t
        clean, noise = predicted(self.config.prediction_type, sample, output, alpha_t ** 0.5, beta_t ** 0.5)
        clean = self.limited(clean)
        # The step's deviation (`_get_variance`): zero at eta 0, where the walk is deterministic.
        deviation = eta * (((1 - alpha_prev) / beta_t) * (1 - alpha_t / alpha_prev)) ** 0.5
        if use_clipped_model_output:
            noise = (sample - alpha_t ** 0.5 * clean) / beta_t ** 0.5
        stepped = alpha_prev ** 0.5 * clean + (1 - alpha_prev - deviation ** 2) ** 0.5 * noise
        if eta > 0:
            stepped = stepped + deviation * (self.drawn(output) if variance_noise is None else variance_noise)
        return stepped


class DDPM(Table, Drawn):
    """`DDPMScheduler` (`scheduling_ddpm.py`), its fixed variances, in float64."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
        "trained_betas": None, "variance_type": "fixed_small", "clip_sample": True,
        "prediction_type": "epsilon", "thresholding": False, "dynamic_thresholding_ratio": 0.995,
        "clip_sample_range": 1.0, "sample_max_value": 1.0, "timestep_spacing": "leading",
        "steps_offset": 0, "rescale_betas_zero_snr": False}

    def __init__(self, dtype: torch.dtype = torch.float64, **config: object):
        super().__init__(dtype, **config)
        self.one = torch.tensor(1.0, dtype=dtype)

    def previous(self, timestep: int) -> int:
        """`previous_timestep` (`scheduling_ddpm.py:648-659`): the grid's next timestep, -1 past its last."""
        grid = [int(each) for each in self.timesteps]
        index = grid.index(timestep)
        return -1 if index == len(grid) - 1 else grid[index + 1]

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        """`step` (`scheduling_ddpm.py:461-540`) with its fixed variances (`_get_variance`, `:348-387`)."""
        timestep = int(timestep)
        before = self.previous(timestep)
        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod[before] if before >= 0 else self.one
        beta_t, beta_prev = 1 - alpha_t, 1 - alpha_prev
        current_alpha = alpha_t / alpha_prev
        current_beta = 1 - current_alpha
        clean, _ = predicted(self.config.prediction_type, sample, output, alpha_t ** 0.5, beta_t ** 0.5)
        clean = self.limited(clean)
        stepped = (alpha_prev ** 0.5 * current_beta / beta_t) * clean + (
            current_alpha ** 0.5 * beta_prev / beta_t) * sample
        if timestep > 0:
            noise = self.drawn(output)
            variance = torch.clamp((1 - alpha_prev) / (1 - alpha_t) * current_beta, min=1e-20)
            kind = self.config.variance_type
            if kind == "fixed_small_log":
                stepped = stepped + torch.exp(0.5 * torch.log(variance)) * noise
            elif kind == "fixed_small":
                stepped = stepped + variance ** 0.5 * noise
            elif kind == "fixed_large":
                stepped = stepped + current_beta ** 0.5 * noise
            else:
                raise NotImplementedError(kind)
        return stepped


class PNDM(Table):
    """`PNDMScheduler` (`scheduling_pndm.py`): Runge-Kutta warmup steps (unless `skip_prk_steps`), then
    linear multistep ones, each a pseudo-numerical step between two training timesteps; its timetable
    re-derived from the config, the library's private grid read by no one."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
        "trained_betas": None, "skip_prk_steps": False, "set_alpha_to_one": False,
        "prediction_type": "epsilon", "timestep_spacing": "leading", "steps_offset": 0}

    def laid(self, asked: Grid) -> None:
        steps = asked.steps_alone()
        train, spacing = self.config.num_train_timesteps, self.config.timestep_spacing
        self.steps = steps
        if spacing == "linspace":
            grid = np.linspace(0, train - 1, steps).round().astype(np.int64)
        elif spacing == "leading":
            grid = (np.arange(0, steps) * (train // steps)).round() + self.config.steps_offset
        elif spacing == "trailing":
            grid = np.round(np.arange(train, 0, -train / steps))[::-1].astype(np.int64) - 1
        else:
            raise ValueError(spacing)
        # The grid's points, ascending (the library's `_timesteps`), beside the timetable its warm-up walks.
        self.grid_points = np.asarray(grid, np.int64)
        if self.config.skip_prk_steps:
            self.prk = np.array([], np.int64)
            plms = np.concatenate([grid[:-1], grid[-2:-1], grid[-1:]])[::-1].copy()
        else:
            warm = np.array(grid[-4:]).repeat(2) + np.tile(np.array([0, train // steps // 2]), 4)
            self.prk = warm[:-1].repeat(2)[1:-1][::-1].copy()
            plms = grid[:-3][::-1].copy()
        self.timesteps = torch.from_numpy(np.concatenate([self.prk, plms]).astype(np.int64))
        self.outputs: list[torch.Tensor] = []
        self.counter, self.summed, self.started = 0, 0, None

    def previous(self, sample: torch.Tensor, timestep: int, before: int,
                 output: torch.Tensor) -> torch.Tensor:
        """`_get_prev_sample`: the pseudo-numerical step from `timestep` to `before`."""
        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod[before] if before >= 0 else self.final_alpha_cumprod
        beta_t, beta_prev = 1 - alpha_t, 1 - alpha_prev
        if self.config.prediction_type == "v_prediction":
            output = alpha_t ** 0.5 * output + beta_t ** 0.5 * sample
        denominator = alpha_t * beta_prev ** 0.5 + (alpha_t * beta_t * alpha_prev) ** 0.5
        return (alpha_prev / alpha_t) ** 0.5 * sample - (alpha_prev - alpha_t) * output / denominator

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        timestep, gap = int(timestep), self.config.num_train_timesteps // self.steps
        if self.counter < len(self.prk) and not self.config.skip_prk_steps:
            before = timestep - (0 if self.counter % 2 else gap // 2)
            timestep = int(self.prk[self.counter // 4 * 4])
            phase = self.counter % 4
            if phase == 0:
                self.summed = self.summed + 1 / 6 * output
                self.outputs.append(output)
                self.started = sample
            elif phase in (1, 2):
                self.summed = self.summed + 1 / 3 * output
            else:
                output, self.summed = self.summed + 1 / 6 * output, 0
            stepped = self.previous(self.started if self.started is not None else sample, timestep, before,
                                    output)
        else:
            before = timestep - gap
            if self.counter != 1:
                self.outputs = [*self.outputs[-3:], output]
            else:
                before, timestep = timestep, timestep + gap
            held = self.outputs
            if len(held) == 1 and self.counter == 0:
                self.started = sample
            elif len(held) == 1 and self.counter == 1:
                assert self.started is not None
                output, sample, self.started = (output + held[-1]) / 2, self.started, None
            elif len(held) == 2:
                output = (3 * held[-1] - held[-2]) / 2
            elif len(held) == 3:
                output = (23 * held[-1] - 16 * held[-2] + 5 * held[-3]) / 12
            else:
                output = (1 / 24) * (55 * held[-1] - 59 * held[-2] + 37 * held[-3] - 9 * held[-4])
            stepped = self.previous(sample, timestep, before, output)
        self.counter += 1
        return stepped


class LCM(Table, Drawn):
    """`LCMScheduler` (`scheduling_lcm.py`): each step the consistency function's denoised sample (its
    boundary scalings at the scaled timestep), noised again to the next timestep but at the last."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012,
        "beta_schedule": "scaled_linear", "trained_betas": None, "original_inference_steps": 50,
        "clip_sample": False, "clip_sample_range": 1.0, "set_alpha_to_one": True, "steps_offset": 0,
        "prediction_type": "epsilon", "thresholding": False, "dynamic_thresholding_ratio": 0.995,
        "sample_max_value": 1.0, "timestep_spacing": "leading", "timestep_scaling": 10.0,
        "rescale_betas_zero_snr": False}

    def laid(self, asked: Grid) -> None:
        steps = asked.steps_alone()
        original = self.config.original_inference_steps
        every = self.config.num_train_timesteps // original
        origin = np.asarray(list(range(1, original + 1))) * every - 1
        origin = origin[::-1].copy()
        chosen = np.floor(np.linspace(0, len(origin), num=steps, endpoint=False)).astype(np.int64)
        self.timesteps = torch.from_numpy(origin[chosen]).to(torch.long)
        self.steps, self.index = steps, 0

    def following(self) -> int:
        """The grid's next timestep, its own at the last (`step`'s `prev_timestep`)."""
        after = self.index + 1
        return int(self.timesteps[after]) if after < len(self.timesteps) else int(self.timesteps[self.index])

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        timestep = int(timestep)
        before = self.following()
        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod[before] if before >= 0 else self.final_alpha_cumprod
        # The library's tensor timestep scaled in its float width
        # (`get_scalings_for_boundary_condition_discrete`).
        scaled = torch.tensor(timestep, dtype=self.dtype) * self.config.timestep_scaling
        skip = 0.5 ** 2 / (scaled ** 2 + 0.5 ** 2)
        out = scaled / (scaled ** 2 + 0.5 ** 2) ** 0.5
        prediction = self.config.prediction_type
        if prediction == "epsilon":
            clean = (sample - (1 - alpha_t).sqrt() * output) / alpha_t.sqrt()
        elif prediction == "sample":
            clean = output
        else:
            clean = alpha_t.sqrt() * sample - (1 - alpha_t).sqrt() * output
        denoised = out * self.limited(clean) + skip * sample
        if self.index != self.steps - 1:
            denoised = alpha_prev.sqrt() * denoised + (1 - alpha_prev).sqrt() * self.drawn(output)
        self.index += 1
        return denoised


class TCD(LCM):
    """`TCDScheduler` (`scheduling_tcd.py`) at the step's gamma (`eta`, 0.3 its default): each step's clean
    prediction and noise taken to an intermediate timestep, then noised again to the next but at the last."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012,
        "beta_schedule": "scaled_linear", "trained_betas": None, "original_inference_steps": 50,
        "clip_sample": False, "clip_sample_range": 1.0, "set_alpha_to_one": True, "steps_offset": 0,
        "prediction_type": "epsilon", "thresholding": False, "dynamic_thresholding_ratio": 0.995,
        "sample_max_value": 1.0, "timestep_spacing": "leading", "timestep_scaling": 10.0,
        "rescale_betas_zero_snr": False}

    @overload
    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None, return_dict: Literal[True] = True, *,
             eta: float = 0.3) -> SimpleNamespace: ...

    @overload
    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None, *, return_dict: Literal[False],
             eta: float = 0.3) -> tuple[torch.Tensor]: ...

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None, return_dict: bool = True, *,
             eta: float = 0.3) -> SimpleNamespace | tuple[torch.Tensor]:
        """The library's `step`, its own options named as the library's are."""
        return self.answered(model_output, timestep, sample, generator, return_dict, {"eta": eta})

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor,
                eta: float = 0.3) -> torch.Tensor:
        timestep = int(timestep)
        after = self.index + 1
        before = int(self.timesteps[after]) if after < len(self.timesteps) else 0
        # The library's tensor timestep floored in its float width (`timestep_s`).
        between = int(torch.floor((1 - eta) * torch.tensor(before, dtype=self.dtype)))
        alpha_t, alpha_prev, alpha_s = (self.alphas_cumprod[timestep], self.alphas_cumprod[before],
                                        self.alphas_cumprod[between])
        beta_t, beta_s = 1 - alpha_t, 1 - alpha_s
        prediction = self.config.prediction_type
        if prediction == "epsilon":
            clean, noise = (sample - beta_t.sqrt() * output) / alpha_t.sqrt(), output
        elif prediction == "sample":
            clean = output
            noise = (sample - alpha_t ** 0.5 * clean) / beta_t ** 0.5
        else:
            clean = alpha_t ** 0.5 * sample - beta_t ** 0.5 * output
            noise = alpha_t ** 0.5 * output + beta_t ** 0.5 * sample
        noised = alpha_s.sqrt() * clean + beta_s.sqrt() * noise
        if eta > 0 and self.index != self.steps - 1:
            ratio = alpha_prev / alpha_s
            noised = ratio.sqrt() * noised + (1 - ratio).sqrt() * self.drawn(output)
        self.index += 1
        return noised
