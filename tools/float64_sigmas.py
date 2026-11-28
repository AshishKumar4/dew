"""The independent float64 oracles of Diffusers' sigma schedulers (0.39.0): the k-diffusion family
(Euler, Euler ancestral, Heun, LMS, KDPM2 and its ancestral form, DPM-Solver SDE) over a sigma grid, and flow
matching's Euler, each re-derived from its public config (`oracle_of`, tools/float64_schedulers.py)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import ClassVar, Literal, overload

import numpy as np
import torch
from float64_oracle import Drawn, Grid, Oracle, sigma_to_t, table, transformed


class Sigmas(Oracle):
    """The variance-exploding classes' sigma grid (`scheduling_euler_discrete.py`'s `__init__` and
    `set_timesteps`): the table's sigmas, interpolated at the grid's times and transformed as the config
    asks, ending at zero or the table's least. At float32 the library's own sequence: a float32 time grid
    and table, interpolated and transformed in float64, rounded once to float32; at float64, float64
    throughout."""

    def __init__(self, dtype: torch.dtype = torch.float64, **config: object):
        super().__init__(dtype, **config)
        self.betas, self.alphas_cumprod = table(self.config, dtype)
        if getattr(self.config, "rescale_betas_zero_snr", False):
            # A zero last alpha's sigma is infinite: the library holds it at 2^-24.
            self.alphas_cumprod[-1] = 2 ** -24
        self.width = np.float32 if dtype == torch.float32 else np.float64
        self.index = 0

    def grid(self, steps: int) -> tuple[np.ndarray, np.ndarray, bool]:
        """The grid's times, its sigmas, the final one appended, and whether a transform made them."""
        train, spacing = self.config.num_train_timesteps, self.config.timestep_spacing
        if spacing == "linspace":
            times = np.linspace(0, train - 1, steps, dtype=self.width)[::-1].copy()
        elif spacing == "leading":
            times = (np.arange(0, steps) * (train // steps)).round()[::-1].copy().astype(self.width)
            times += self.config.steps_offset
        elif spacing == "trailing":
            times = np.arange(train, 0, -train / steps).round().copy().astype(self.width) - 1
        else:
            raise ValueError(spacing)
        cumulative = self.alphas_cumprod.numpy()
        table = np.array(((1 - cumulative) / cumulative) ** 0.5)
        log_table = np.log(table)
        sigmas = np.interp(times, np.arange(0, len(table)), table)
        config = self.config
        stated_least, stated_most = getattr(config, "sigma_min", None), getattr(config, "sigma_max", None)
        least = float(sigmas[-1]) if stated_least is None else stated_least
        most = float(sigmas[0]) if stated_most is None else stated_most
        moved = transformed(config, least, most, steps)
        if moved is not None:
            sigmas = moved
            times = np.array([sigma_to_t(sigma, log_table) for sigma in sigmas])
        table_least = ((1 - cumulative[0]) / cumulative[0]) ** 0.5
        last = table_least if getattr(config, "final_sigmas_type", "zero") == "sigma_min" else 0
        return times, np.concatenate([sigmas, [last]]).astype(self.width), moved is not None

    def laid(self, asked: Grid) -> None:
        steps = asked.steps_alone()
        times, sigmas, _ = self.grid(steps)
        self.steps, self.index = steps, 0
        # The library's times rounded to float32 whatever grid made them (`EulerDiscreteScheduler`'s).
        self.timesteps, self.sigmas = torch.from_numpy(times.astype(self.width)), torch.from_numpy(sigmas)

    @property
    def init_noise_sigma(self) -> float | torch.Tensor:
        """The prior's scale: the grid's largest sigma, or its total scale where the grid leads."""
        most = self.sigmas.max()
        return most if self.config.timestep_spacing in ("linspace", "trailing") else (most ** 2 + 1) ** 0.5

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        return sample / ((self.sigmas[self.index] ** 2 + 1) ** 0.5)

    def clean(self, output: torch.Tensor, sample: torch.Tensor, sigma: torch.Tensor,
              sigma_hat: torch.Tensor) -> torch.Tensor:
        """The clean sample `output` predicts (`scheduling_euler_discrete.py`'s `step`)."""
        prediction = self.config.prediction_type
        if prediction in ("original_sample", "sample"):
            return output
        if prediction == "epsilon":
            return sample - sigma_hat * output
        if prediction == "v_prediction":
            return output * (-sigma / (sigma ** 2 + 1) ** 0.5) + sample / (sigma ** 2 + 1)
        raise ValueError(prediction)


class Euler(Sigmas):
    """`EulerDiscreteScheduler` without churn."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
        "trained_betas": None, "prediction_type": "epsilon", "interpolation_type": "linear",
        "use_karras_sigmas": False, "use_exponential_sigmas": False, "use_beta_sigmas": False,
        "sigma_min": None, "sigma_max": None, "timestep_spacing": "linspace", "timestep_type": "discrete",
        "steps_offset": 0, "rescale_betas_zero_snr": False, "final_sigmas_type": "zero"}
    refused: ClassVar[Mapping[str, object]] = {"interpolation_type": "linear", "timestep_type": "discrete"}

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        sigma = self.sigmas[self.index]
        derivative = (sample - self.clean(output, sample, sigma, sigma)) / sigma
        stepped = sample + derivative * (self.sigmas[self.index + 1] - sigma)
        self.index += 1
        return stepped


class EulerAncestral(Sigmas, Drawn):
    """`EulerAncestralDiscreteScheduler`: Euler to sigma_down, then its draw at sigma_up."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
        "trained_betas": None, "prediction_type": "epsilon", "timestep_spacing": "linspace",
        "steps_offset": 0, "rescale_betas_zero_snr": False}

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        sigma, after = self.sigmas[self.index], self.sigmas[self.index + 1]
        up = (after ** 2 * (sigma ** 2 - after ** 2) / sigma ** 2) ** 0.5
        down = (after ** 2 - up ** 2) ** 0.5
        derivative = (sample - self.clean(output, sample, sigma, sigma)) / sigma
        stepped = sample + derivative * (down - sigma) + self.drawn(output) * up
        self.index += 1
        return stepped


class Heun(Sigmas):
    """`HeunDiscreteScheduler`: each interval an Euler stage and a trapezoidal correction, its grid's
    sigmas and times each held for both (`scheduling_heun_discrete.py`)."""

    order = 2
    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "linear",
        "trained_betas": None, "prediction_type": "epsilon", "use_karras_sigmas": False,
        "use_exponential_sigmas": False, "use_beta_sigmas": False, "clip_sample": False,
        "clip_sample_range": 1.0, "timestep_spacing": "linspace", "steps_offset": 0}

    def laid(self, asked: Grid) -> None:
        super().laid(asked)
        self.sigmas = torch.cat([self.sigmas[:1], self.sigmas[1:-1].repeat_interleave(2), self.sigmas[-1:]])
        self.timesteps = torch.cat([self.timesteps[:1], self.timesteps[1:].repeat_interleave(2)])
        self.held: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        first = self.held is None
        sigma, after = ((self.sigmas[self.index], self.sigmas[self.index + 1]) if first
                        else (self.sigmas[self.index - 1], self.sigmas[self.index]))
        read = sigma if first else after
        clean = self.clean(output, sample, read, read)
        if self.config.clip_sample:
            clean = clean.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)
        if first:
            derivative, dt = (sample - clean) / sigma, after - sigma
            self.held = (derivative, dt, sample)
        else:
            assert self.held is not None
            earlier, dt, started = self.held
            derivative = (earlier + (sample - clean) / after) / 2
            sample, self.held = started, None
        self.index += 1
        return sample + derivative * dt


class LMS(Sigmas):
    """`LMSDiscreteScheduler`: a linear multistep step over the last four derivatives, each weighted by the
    integral of its Lagrange basis over the interval (`get_lms_coefficient`), which the library takes by
    quadrature of a polynomial of degree at most three, and this oracle in closed form."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
        "trained_betas": None, "use_karras_sigmas": False, "use_exponential_sigmas": False,
        "use_beta_sigmas": False, "prediction_type": "epsilon", "timestep_spacing": "linspace",
        "steps_offset": 0}

    def laid(self, asked: Grid) -> None:
        super().laid(asked)
        self.derivatives: list[torch.Tensor] = []

    def coefficient(self, order: int, at: int, current: int) -> float:
        """The integral over [sigma_at, sigma_at+1] of the basis polynomial of the history's `current`."""
        sigmas = [float(each) for each in self.sigmas]
        basis = np.polynomial.Polynomial([1.0])
        for k in range(order):
            if k != current:
                basis = basis * np.polynomial.Polynomial([-sigmas[at - k], 1.0]) / (
                    sigmas[at - current] - sigmas[at - k])
        integral = basis.integ()
        return float(integral(sigmas[at + 1]) - integral(sigmas[at]))

    @overload
    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None, return_dict: Literal[True] = True, *,
             order: int = 4) -> SimpleNamespace: ...

    @overload
    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None, *, return_dict: Literal[False],
             order: int = 4) -> tuple[torch.Tensor]: ...

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None, return_dict: bool = True, *,
             order: int = 4) -> SimpleNamespace | tuple[torch.Tensor]:
        """The library's `step`, its own options named as the library's are."""
        return self.answered(model_output, timestep, sample, generator, return_dict, {"order": order})

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor,
                order: int = 4) -> torch.Tensor:
        sigma = self.sigmas[self.index]
        self.derivatives.append((sample - self.clean(output, sample, sigma, sigma)) / sigma)
        if len(self.derivatives) > order:
            self.derivatives.pop(0)
        order = min(self.index + 1, order)
        stepped = sample + sum(self.coefficient(order, self.index, current) * derivative
                               for current, derivative in zip(range(order), reversed(self.derivatives),
                                                              strict=False))
        self.index += 1
        return stepped


class KDPM2(Sigmas):
    """`KDPM2DiscreteScheduler`: each interval an Euler stage to the geometric midpoint of its sigmas and a
    midpoint correction, its grid's sigmas held for both, its midpoint times interleaved
    (`scheduling_k_dpm_2_discrete.py`)."""

    order = 2
    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "linear",
        "trained_betas": None, "use_karras_sigmas": False, "use_exponential_sigmas": False,
        "use_beta_sigmas": False, "prediction_type": "epsilon", "timestep_spacing": "linspace",
        "steps_offset": 0}

    def laid(self, asked: Grid) -> None:
        steps = asked.steps_alone()
        times, sigmas = self.kdpm2_grid(steps)
        interpol = self.interpolated(sigmas)
        self.timesteps = self.interleaved(times, self.midpoint_times(interpol, times))
        self.sigmas, self.sigmas_interpol = self.held_twice(sigmas), self.held_twice(interpol)
        self.held: torch.Tensor | None = None

    def kdpm2_grid(self, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The grid's times and sigmas as the KDPM2 classes hold them: the times at the dtype the grid made
        them (float32 at the library's width; float64 where a transform found them from sigmas, Karras's
        rounded), the sigmas at the arm's width."""
        times, sigmas, moved = self.grid(steps)
        if moved and self.config.use_karras_sigmas:
            times = times.round()
        self.steps, self.index = steps, 0
        return torch.from_numpy(np.ascontiguousarray(times)), torch.from_numpy(sigmas)

    def midpoint_times(self, interpol: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Each midpoint sigma's time in the table, at the grid times' dtype."""
        cumulative = self.alphas_cumprod.numpy()
        log_table = np.log(np.array(((1 - cumulative) / cumulative) ** 0.5))
        found = np.array([sigma_to_t(np.asarray(sigma), log_table) for sigma in interpol.numpy()])
        return torch.from_numpy(found).to(times.dtype)

    @staticmethod
    def interpolated(sigmas: torch.Tensor) -> torch.Tensor:
        """Each sigma's geometric mean with the one before it (`lerp` in log space)."""
        return sigmas.log().lerp(sigmas.roll(1).log(), 0.5).exp()

    @staticmethod
    def held_twice(levels: torch.Tensor) -> torch.Tensor:
        """A grid's levels past its first each held for an interval's two stages."""
        return torch.cat([levels[:1], levels[1:].repeat_interleave(2), levels[-1:]])

    @staticmethod
    def interleaved(times: torch.Tensor, midpoints: torch.Tensor) -> torch.Tensor:
        """The grid's times with each interval's midpoint time before its end."""
        return torch.cat([times[:1], torch.stack((midpoints[1:-1, None], times[1:, None]), dim=-1).flatten()])

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        sigma = self.sigmas[self.index] if self.held is None else self.sigmas_interpol[self.index]
        return sample / ((sigma ** 2 + 1) ** 0.5)

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        first = self.held is None
        if first:
            sigma, middle, after = (self.sigmas[self.index], self.sigmas_interpol[self.index + 1],
                                    self.sigmas[self.index + 1])
        else:
            sigma, middle, after = (self.sigmas[self.index - 1], self.sigmas_interpol[self.index],
                                    self.sigmas[self.index])
        read = sigma if first else middle
        clean = self.clean(output, sample, read, read)
        if first:
            derivative, dt, self.held = (sample - clean) / sigma, middle - sigma, sample
        else:
            derivative, dt = (sample - clean) / middle, after - sigma
            assert self.held is not None
            sample, self.held = self.held, None
        self.index += 1
        return sample + derivative * dt


class KDPM2Ancestral(KDPM2, Drawn):
    """`KDPM2AncestralDiscreteScheduler`: KDPM2's two stages toward each interval's sigma_down, its
    midpoint between the sigma and sigma_down, then its draw at sigma_up
    (`scheduling_k_dpm_2_ancestral_discrete.py`)."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "linear",
        "trained_betas": None, "use_karras_sigmas": False, "use_exponential_sigmas": False,
        "use_beta_sigmas": False, "prediction_type": "epsilon", "timestep_spacing": "linspace",
        "steps_offset": 0}

    def laid(self, asked: Grid) -> None:
        steps = asked.steps_alone()
        times, sigmas = self.kdpm2_grid(steps)
        after = sigmas.roll(-1)
        after[-1] = 0.0
        up = (after ** 2 * (sigmas ** 2 - after ** 2) / sigmas ** 2) ** 0.5
        down = (after ** 2 - up ** 2) ** 0.5
        down[-1] = 0.0
        interpol = sigmas.log().lerp(down.log(), 0.5).exp()
        interpol[-2:] = 0.0
        midpoint_times = self.midpoint_times(interpol, times)
        paired = torch.stack((midpoint_times[:-2, None], times[1:, None]), dim=-1).flatten()
        self.timesteps = torch.cat([times[:1], paired])
        self.sigmas, self.sigmas_interpol = self.held_twice(sigmas), self.held_twice(interpol)
        self.sigmas_up, self.sigmas_down = self.held_twice(up), self.held_twice(down)
        self.held = None

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        sigma = self.sigmas[self.index] if self.held is None else self.sigmas_interpol[self.index - 1]
        return sample / ((sigma ** 2 + 1) ** 0.5)

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        first = self.held is None
        at = self.index if first else self.index - 1
        sigma, middle = self.sigmas[at], self.sigmas_interpol[at]
        read = sigma if first else middle
        clean = self.clean(output, sample, read, read)
        if first:
            stepped = sample + (sample - clean) / sigma * (middle - sigma)
            self.held = sample
        else:
            assert self.held is not None
            derivative = (sample - clean) / middle
            stepped = (self.held + derivative * (self.sigmas_down[at] - sigma)
                       + self.drawn(output) * self.sigmas_up[at])
            self.held = None
        self.index += 1
        return stepped


class DPMSDE(Sigmas, Drawn):
    """`DPMSolverSDEScheduler` (`scheduling_dpmsolver_sde.py`): each interval an ancestral stage to the
    midpoint of its log sigmas and one from the interval's start to its end, each down to its sigma_down
    and noised back up by sigma_up with the Brownian tree's answer between its ends (the walk's draws:
    the tree's answers recorded, so `noise_sampler_seed` is theirs). Its grid's times stay float64 and its
    sigmas take the arm's width, as the library's."""

    order = 2
    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "linear",
        "trained_betas": None, "prediction_type": "epsilon", "use_karras_sigmas": False,
        "use_exponential_sigmas": False, "use_beta_sigmas": False, "noise_sampler_seed": None,
        "timestep_spacing": "linspace", "steps_offset": 0}

    def laid(self, asked: Grid) -> None:
        steps = asked.steps_alone()
        config, train = self.config, self.config.num_train_timesteps
        if config.timestep_spacing == "linspace":
            times = np.linspace(0, train - 1, steps, dtype=float)[::-1].copy()
        elif config.timestep_spacing == "leading":
            times = (np.arange(0, steps) * (train // steps)).round()[::-1].copy().astype(float)
            times += config.steps_offset
        elif config.timestep_spacing == "trailing":
            times = np.arange(train, 0, -train / steps).round().copy().astype(float) - 1
        else:
            raise ValueError(config.timestep_spacing)
        cumulative = self.alphas_cumprod.numpy()
        table = np.array(((1 - cumulative) / cumulative) ** 0.5)
        log_table = np.log(table)
        sigmas = np.interp(times, np.arange(0, len(table)), table)
        moved = transformed(config, float(sigmas[-1]), float(sigmas[0]), steps)
        if moved is not None:
            sigmas = moved
            times = np.array([sigma_to_t(np.asarray(sigma), log_table) for sigma in sigmas])
        # Each interval's second stage at the midpoint of its log sigmas (`_second_order_timesteps`).
        logs = -np.log(sigmas)
        midpoints = np.exp(-(logs[:-1] + np.diff(logs) * 0.5))
        stages = np.array([sigma_to_t(np.asarray(sigma), log_table) for sigma in midpoints])
        found = torch.from_numpy(np.concatenate([sigmas, [0.0]]).astype(self.width))
        self.sigmas = torch.cat([found[:1], found[1:-1].repeat_interleave(2), found[-1:]])
        held = torch.from_numpy(np.asarray(times, np.float64))
        held = torch.cat([held[:1], held[1:].repeat_interleave(2)])
        held[1::2] = torch.from_numpy(stages)
        self.timesteps, self.steps, self.index = held, steps, 0
        self.held: torch.Tensor | None = None
        self.midpoint: torch.Tensor | None = None

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        sigma = self.sigmas[self.index] if self.midpoint is None else self.midpoint
        return sample / ((sigma ** 2 + 1) ** 0.5)

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        first = self.held is None
        sigma, after = ((self.sigmas[self.index], self.sigmas[self.index + 1]) if first
                        else (self.sigmas[self.index - 1], self.sigmas[self.index]))
        t, t_next = -sigma.log(), -after.log()
        proposed = t + (t_next - t) * 0.5
        read = sigma if first else (-proposed).exp()
        if self.config.prediction_type == "epsilon":
            clean = sample - read * output
        elif self.config.prediction_type == "v_prediction":
            clean = output * (-read / (read ** 2 + 1) ** 0.5) + (sample / (read ** 2 + 1))
        else:
            raise NotImplementedError(self.config.prediction_type)
        if after == 0:
            stepped = sample + ((sample - clean) / sigma) * (after - sigma)
        else:
            if first:
                t_next = proposed
            else:
                assert self.held is not None
                sample = self.held
            start, end = (-t).exp(), (-t_next).exp()
            up = torch.minimum(end, (end ** 2 * (start ** 2 - end ** 2) / start ** 2) ** 0.5)
            ancestral = -((end ** 2 - up ** 2) ** 0.5).log()
            stepped = ((-ancestral).exp() / (-t).exp()) * sample - (t - ancestral).expm1() * clean
            stepped = stepped + self.drawn(stepped) * up
            self.held, self.midpoint = (sample, end) if first else (None, None)
        self.index += 1
        return stepped


class FlowMatch(Oracle, Drawn):
    """`FlowMatchEulerDiscreteScheduler` (`scheduling_flow_match_euler_discrete.py`): Euler's step along the
    velocity, or stochastically the clean prediction noised again to the next sigma, over a grid between
    its shifted training table's ends (or the sigmas a pipeline gives), shifted by `shift` or a pipeline's
    `mu`, stretched to `shift_terminal` and moved by a sigma transform where the config asks. At float32
    the library's own sequence: its table shifted in float32, a grid it computes in float64 and one given
    in float32, its sigmas and every step float32. `base_shift`, `max_shift` and the image sequence lengths
    are the pipelines' (`calculate_shift`), which the scheduler never reads."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "shift": 1.0, "use_dynamic_shifting": False, "base_shift": 0.5,
        "max_shift": 1.15, "base_image_seq_len": 256, "max_image_seq_len": 4096, "invert_sigmas": False,
        "shift_terminal": None, "use_karras_sigmas": False, "use_exponential_sigmas": False,
        "use_beta_sigmas": False, "time_shift_type": "exponential", "stochastic_sampling": False}

    def __init__(self, dtype: torch.dtype = torch.float64, **config: object):
        super().__init__(dtype, **config)
        stated = self.config
        if stated.use_karras_sigmas + stated.use_exponential_sigmas + stated.use_beta_sigmas > 1:
            raise ValueError("FlowMatch takes one sigma transform")
        if stated.time_shift_type not in ("exponential", "linear"):
            raise ValueError(stated.time_shift_type)
        train = stated.num_train_timesteps
        width = np.float32 if dtype == torch.float32 else np.float64
        table = torch.from_numpy(np.linspace(1, train, train, dtype=width)[::-1].copy()) / train
        if not stated.use_dynamic_shifting:
            table = stated.shift * table / (1 + (stated.shift - 1) * table)
        self.width, self.ends = width, (table[0].item(), table[-1].item())

    def laid(self, asked: Grid) -> None:
        config, train = self.config, self.config.num_train_timesteps
        num_inference_steps, sigmas, mu = asked.steps, asked.sigmas, asked.mu
        if asked.timesteps is not None:
            raise ValueError("a flow grid takes no given timesteps")
        if config.use_dynamic_shifting and mu is None:
            raise ValueError("a dynamically shifted grid takes its pipeline's mu")
        if sigmas is None:
            if num_inference_steps is None:
                raise ValueError("a grid takes its step count or its sigmas")
            steps, most, least = num_inference_steps, *self.ends
            grid = np.linspace(most * train, least * train, steps) / train
        else:
            # A pipeline's own sigmas (`retrieve_timesteps` gives them alone), as many as any count it states.
            grid = np.asarray(sigmas).astype(self.width)
            steps = len(grid)
            if num_inference_steps not in (None, steps):
                raise ValueError(f"{steps} sigmas for {num_inference_steps} steps")
        if config.use_dynamic_shifting:
            assert mu is not None
            shifted = math.exp(mu) if config.time_shift_type == "exponential" else mu
            grid = shifted / (shifted + (1 / grid - 1) ** 1.0)
        else:
            grid = config.shift * grid / (1 + (config.shift - 1) * grid)
        if config.shift_terminal:
            rest = 1 - grid
            grid = 1 - (rest / (rest[-1] / (1 - config.shift_terminal)))
        moved = transformed(config, grid[-1].item(), grid[0].item(), steps)
        if moved is not None:
            grid = moved
        found = torch.from_numpy(np.asarray(grid)).to(self.dtype)
        end = torch.ones(1, dtype=self.dtype) if config.invert_sigmas else torch.zeros(1, dtype=self.dtype)
        if config.invert_sigmas:
            found = 1.0 - found
        self.timesteps, self.sigmas = found * train, torch.cat([found, end])
        self.steps, self.index = len(found), 0

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        sigma, after = self.sigmas[self.index], self.sigmas[self.index + 1]
        sample = sample.to(self.dtype)
        if self.config.stochastic_sampling:
            stepped = (1.0 - after) * (sample - sigma * output) + after * self.drawn(sample)
        else:
            stepped = sample + (after - sigma) * output
        self.index += 1
        return stepped.to(output.dtype)
