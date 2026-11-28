"""The independent float64 oracles of Diffusers' multistep solvers (0.39.0): DPM-Solver multistep and
singlestep, DEIS, UniPC and EDM's DPM-Solver, each its log-SNR grid and history update re-derived from its
public config (`oracle_of`, tools/float64_schedulers.py)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import torch
from float64_oracle import Drawn, Grid, Oracle, predicted, sigma_to_t, stated_ends, table, transformed


class Multistep(Oracle):
    """What the multistep solvers share whatever their grid (`scheduling_dpmsolver_multistep.py`'s
    `_sigma_to_alpha_sigma_t`, `convert_model_output`): their rates at each sigma, and the model's output as
    the clean prediction (the ++ algorithms) or the noise (the rest)."""

    index: int
    """The step the walk is at."""
    steps: int
    """The grid's step count."""

    def __init__(self, dtype: torch.dtype = torch.float64, **config: object):
        super().__init__(dtype, **config)
        self.width = np.float32 if dtype == torch.float32 else np.float64

    def rates(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`_sigma_to_alpha_sigma_t`: the signal and noise rates at `sigma`."""
        if getattr(self.config, "use_flow_sigmas", False):
            return 1 - sigma, sigma
        alpha = 1 / ((sigma ** 2 + 1) ** 0.5)
        return alpha, sigma * alpha

    def converted(self, output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """`convert_model_output`: the clean prediction (the ++ algorithms), else the noise."""
        sigma = self.sigmas[self.index]
        alpha, noise_rate = self.rates(sigma)
        prediction = self.config.prediction_type
        if self.config.algorithm_type in ("dpmsolver++", "sde-dpmsolver++"):
            clean = (sample - sigma * output if prediction == "flow_prediction"
                     else predicted(prediction, sample, output, alpha, noise_rate)[0])
            return self.limited(clean)
        _, noise = predicted(prediction, sample, output, alpha, noise_rate)
        if self.config.thresholding:
            clean = self.limited((sample - noise_rate * noise) / alpha)
            noise = (sample - alpha * clean) / noise_rate
        return noise

    def lambdas(self, *offsets: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Each sigma `index + offset`'s rates and lambda."""
        found = []
        for offset in offsets:
            alpha, sigma = self.rates(self.sigmas[self.index + offset])
            found.append((alpha, sigma, torch.log(alpha) - torch.log(sigma)))
        return found


class LogSNR(Multistep):
    """The grid of the multistep classes over a training table (`scheduling_dpmsolver_multistep.py`'s
    `set_timesteps`): its training timesteps, clipped where lambda falls below `lambda_min_clipped`, its
    sigmas the table's or the ones the config's transform asks, or a shifted flow's, ending at zero or the
    table's least."""

    def __init__(self, dtype: torch.dtype = torch.float64, **config: object):
        super().__init__(dtype, **config)
        self.betas, self.alphas_cumprod = table(self.config, dtype)
        if getattr(self.config, "rescale_betas_zero_snr", False):
            self.alphas_cumprod[-1] = 2 ** -24

    def laid(self, asked: Grid) -> None:
        """The grid of its step count by the config's spacing, or of the training timesteps given (the
        DPM-Solver classes' custom grid, which takes no sigma transform), one of them."""
        steps, timesteps = asked.steps, asked.timesteps
        if (asked.sigmas, asked.mu) != (None, None):
            raise ValueError(f"a log-SNR grid takes no sigmas or mu, not {asked}")
        # DPMSinglestep states no `timestep_spacing`: its grid is always linspace's.
        train, spacing = self.config.num_train_timesteps, getattr(self.config, "timestep_spacing", "linspace")
        cumulative = self.alphas_cumprod
        lambdas = torch.log(cumulative ** 0.5) - torch.log((1 - cumulative) ** 0.5)
        clipped = int(torch.searchsorted(torch.flip(lambdas, [0]),
                                         torch.tensor(getattr(self.config, "lambda_min_clipped", -math.inf),
                                                      dtype=lambdas.dtype)))
        last = train - clipped
        if timesteps is not None:
            if steps is not None or any(getattr(self.config, f"use_{name}_sigmas", False)
                                        for name in ("karras", "exponential", "beta")):
                raise ValueError("a grid of given timesteps takes no step count or sigma transform")
            times = np.array(timesteps).astype(np.int64)
            steps = len(times)
        elif steps is None:
            raise ValueError("a grid takes its step count or its timesteps")
        elif spacing == "linspace":
            times = np.linspace(0, last - 1, steps + 1).round()[::-1][:-1].copy().astype(np.int64)
        elif spacing == "leading":
            ratio = last // (steps + 1)
            times = (np.arange(0, steps + 1) * ratio).round()[::-1][:-1].copy().astype(np.int64)
            times += self.config.steps_offset
        elif spacing == "trailing":
            times = np.arange(last, 0, -train / steps).round().copy().astype(np.int64) - 1
        else:
            raise ValueError(spacing)
        table = np.array(((1 - cumulative.numpy()) / cumulative.numpy()) ** 0.5)
        log_table = np.log(table)
        moved = transformed(self.config, *stated_ends(self.config, table), steps)
        if moved is not None:
            sigmas = moved
            times = np.array([sigma_to_t(np.asarray(sigma), log_table) for sigma in sigmas])
            if self.config.use_karras_sigmas and self.rounds_karras():
                times = times.round()
        elif self.config.use_flow_sigmas:
            shift = self.config.flow_shift
            raw = 1.0 - np.linspace(1, 1 / train, steps + 1)
            sigmas = np.flip(shift * raw / (1 + (shift - 1) * raw))[:-1].copy()
            times = sigmas * train
        else:
            sigmas = np.interp(times, np.arange(0, len(table)), table)
        least = self.last(sigmas, float(((1 - cumulative[0]) / cumulative[0]) ** 0.5),
                          moved is not None or self.config.use_flow_sigmas)
        self.sigmas = torch.from_numpy(np.concatenate([sigmas, [least]]).astype(self.width))
        self.timesteps = torch.from_numpy(np.asarray(times)).to(torch.int64)
        self.steps, self.index, self.outputs, self.lower = len(self.timesteps), 0, [], 0

    def rounds_karras(self) -> bool:
        """Whether a Karras grid's times are rounded: DPM-Solver's but a cosine schedule's."""
        return self.config.beta_schedule != "squaredcos_cap_v2"

    def last(self, sigmas: np.ndarray, least: float, moved: bool) -> float:
        """The sigma the grid ends at: the table's least or zero, as `final_sigmas_type` states."""
        return least if self.config.final_sigmas_type == "sigma_min" else 0.0


class DPMUpdates(Multistep, Drawn):
    """DPM-Solver's first, second and third order multistep updates for each algorithm
    (`scheduling_dpmsolver_multistep.py`'s `*_update`), over whatever grid and rates the class holds."""

    algorithms: ClassVar[tuple[str, ...]] = ("dpmsolver++", "dpmsolver", "sde-dpmsolver++", "sde-dpmsolver")
    """The algorithms the library class walks; it refuses the rest."""
    third_orders: ClassVar[tuple[str, ...]] = ("dpmsolver++", "dpmsolver", "sde-dpmsolver++")
    """The algorithms the library's third order update has a branch for: a walk that selects it for another
    is no walk the library takes (a short one that never selects it is)."""

    def __init__(self, dtype: torch.dtype = torch.float64, **config: object):
        super().__init__(dtype, **config)
        kind = self.config.algorithm_type
        if kind not in self.algorithms or self.config.solver_type not in ("midpoint", "heun"):
            raise NotImplementedError(f"{type(self).__name__} walks no {kind} {self.config.solver_type}")
        if kind not in ("dpmsolver++", "sde-dpmsolver++") and self.config.final_sigmas_type == "zero":
            raise ValueError(f"{type(self).__name__} ends no {kind} grid at zero")

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        final = self.index == self.steps - 1 and (
            self.config.euler_at_final or (self.config.lower_order_final and self.steps < 15)
            or self.config.final_sigmas_type == "zero")
        second = self.index == self.steps - 2 and self.config.lower_order_final and self.steps < 15
        converted = self.converted(output, sample)
        self.outputs = [*self.outputs[-(self.config.solver_order - 1):], converted] if (
            self.config.solver_order > 1) else [converted]
        noise = self.drawn(output) if self.config.algorithm_type.startswith("sde") else None
        order = self.config.solver_order
        if order == 1 or self.lower < 1 or final:
            stepped = self.first(converted, sample, noise)
        elif order == 2 or self.lower < 2 or second:
            stepped = self.second(sample, noise)
        else:
            stepped = self.third(sample, noise)
        self.lower = min(self.lower + 1, order)
        self.index += 1
        return stepped

    def first(self, m: torch.Tensor, sample: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
        (alpha_t, sigma_t, lambda_t), (alpha_s, sigma_s, lambda_s) = self.lambdas(1, 0)
        h = lambda_t - lambda_s
        kind = self.config.algorithm_type
        if kind == "dpmsolver++":
            return (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * m
        if kind == "dpmsolver":
            return (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * m
        assert noise is not None
        if kind == "sde-dpmsolver++":
            return ((sigma_t / sigma_s * torch.exp(-h)) * sample + (alpha_t * (1 - torch.exp(-2.0 * h))) * m
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise)
        return ((alpha_t / alpha_s) * sample - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * m
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise)

    def second(self, sample: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
        found = self.lambdas(1, 0, -1)
        (alpha_t, sigma_t, lambda_t), (alpha_s0, sigma_s0, lambda_s0) = found[0], found[1]
        lambda_s1 = found[2][2]
        m0, m1 = self.outputs[-1], self.outputs[-2]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        return self.second_order(sample, (alpha_t, sigma_t), (alpha_s0, sigma_s0), h,
                                 m0, (1.0 / (h_0 / h)) * (m0 - m1), noise)

    def second_order(self, sample: torch.Tensor, to: tuple[torch.Tensor, torch.Tensor],
                     start: tuple[torch.Tensor, torch.Tensor], h: torch.Tensor, d0: torch.Tensor,
                     d1: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
        """The second order update from `sample` at rates `start` to rates `to` over log-SNR step `h`, from
        the output `d0` and its slope `d1`: the multistep and singlestep classes' shared formula."""
        (alpha_t, sigma_t), (alpha_s, sigma_s) = to, start
        kind, heun = self.config.algorithm_type, self.config.solver_type == "heun"
        if kind == "dpmsolver++":
            tail = (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * d1 if heun else (
                -0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * d1)
            return (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * d0 + tail
        if kind == "dpmsolver":
            tail = (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * d1 if heun else (
                0.5 * (sigma_t * (torch.exp(h) - 1.0)) * d1)
            return (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * d0 - tail
        assert noise is not None
        if kind == "sde-dpmsolver++":
            tail = (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * d1 if heun else (
                0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * d1)
            return ((sigma_t / sigma_s * torch.exp(-h)) * sample + (alpha_t * (1 - torch.exp(-2.0 * h))) * d0
                    + tail + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise)
        tail = 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * d1 if heun else (
            (sigma_t * (torch.exp(h) - 1.0)) * d1)
        return ((alpha_t / alpha_s) * sample - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * d0 - tail
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise)

    def third(self, sample: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
        if self.config.algorithm_type not in self.third_orders:
            # The library's update has no branch for it: a walk that reaches it is no walk it takes.
            kind = self.config.algorithm_type
            raise NotImplementedError(f"{type(self).__name__} has no third order {kind}")
        found = self.lambdas(1, 0, -1, -2)
        (alpha_t, sigma_t, lambda_t), (alpha_s0, sigma_s0, lambda_s0) = found[0], found[1]
        lambda_s1, lambda_s2 = found[2][2], found[3][2]
        m0, m1, m2 = self.outputs[-1], self.outputs[-2], self.outputs[-3]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        d1_0, d1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1)
        d2 = (1.0 / (r0 + r1)) * (d1_0 - d1_1)
        kind = self.config.algorithm_type
        if kind == "dpmsolver++":
            return ((sigma_t / sigma_s0) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * m0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * d1
                    - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h ** 2 - 0.5)) * d2)
        if kind == "dpmsolver":
            return ((alpha_t / alpha_s0) * sample - (sigma_t * (torch.exp(h) - 1.0)) * m0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * d1
                    - (sigma_t * ((torch.exp(h) - 1.0 - h) / h ** 2 - 0.5)) * d2)
        assert noise is not None
        return ((sigma_t / sigma_s0 * torch.exp(-h)) * sample + (alpha_t * (1.0 - torch.exp(-2.0 * h))) * m0
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * d1
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h) - 2.0 * h) / (2.0 * h) ** 2 - 0.5)) * d2
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise)


class DPMMultistep(LogSNR, DPMUpdates):
    """`DPMSolverMultistepScheduler`: DPM-Solver's updates over the log-SNR grid."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
        "trained_betas": None, "solver_order": 2, "prediction_type": "epsilon", "thresholding": False,
        "dynamic_thresholding_ratio": 0.995, "sample_max_value": 1.0, "algorithm_type": "dpmsolver++",
        "solver_type": "midpoint", "lower_order_final": True, "euler_at_final": False,
        "use_karras_sigmas": False, "use_exponential_sigmas": False, "use_beta_sigmas": False,
        "use_lu_lambdas": False, "use_flow_sigmas": False, "flow_shift": 1.0, "final_sigmas_type": "zero",
        "lambda_min_clipped": -math.inf, "variance_type": None, "timestep_spacing": "linspace",
        "steps_offset": 0, "rescale_betas_zero_snr": False, "use_dynamic_shifting": False,
        "time_shift_type": "exponential"}
    refused: ClassVar[Mapping[str, object]] = {
        "use_lu_lambdas": False, "variance_type": None, "use_dynamic_shifting": False,
        "time_shift_type": "exponential"}


class DEIS(LogSNR):
    """`DEISMultistepScheduler` (`scheduling_deis_multistep.py`): the noise read through its clean prediction,
    stepped by the exponential integrator's coefficients over log rho."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
        "trained_betas": None, "solver_order": 2, "prediction_type": "epsilon", "thresholding": False,
        "dynamic_thresholding_ratio": 0.995, "sample_max_value": 1.0, "algorithm_type": "deis",
        "solver_type": "logrho", "lower_order_final": True, "use_karras_sigmas": False,
        "use_exponential_sigmas": False, "use_beta_sigmas": False, "use_flow_sigmas": False,
        "flow_shift": 1.0, "timestep_spacing": "linspace", "steps_offset": 0,
        "use_dynamic_shifting": False, "time_shift_type": "exponential"}
    refused: ClassVar[Mapping[str, object]] = {
        "algorithm_type": "deis", "solver_type": "logrho", "use_dynamic_shifting": False,
        "time_shift_type": "exponential"}

    def rounds_karras(self) -> bool:
        return True

    def last(self, sigmas: np.ndarray, least: float, moved: bool) -> float:
        """A transformed or flow grid's last sigma held again; else the table's least."""
        return float(sigmas[-1]) if moved else least

    def converted(self, output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        sigma = self.sigmas[self.index]
        alpha, noise_rate = self.rates(sigma)
        prediction = self.config.prediction_type
        clean = (sample - sigma * output if prediction == "flow_prediction"
                 else predicted(prediction, sample, output, alpha, noise_rate)[0])
        clean = self.limited(clean)
        return (sample - alpha * clean) / noise_rate

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        late = self.config.lower_order_final and self.steps < 15
        final, second = late and self.index == self.steps - 1, late and self.index == self.steps - 2
        converted = self.converted(output, sample)
        order = self.config.solver_order
        self.outputs = [*self.outputs[-(order - 1):], converted] if order > 1 else [converted]
        if order == 1 or self.lower < 1 or final:
            (alpha_t, sigma_t, lambda_t), (alpha_s, _, lambda_s) = self.lambdas(1, 0)
            rise = torch.exp(lambda_t - lambda_s) - 1.0
            stepped = (alpha_t / alpha_s) * sample - (sigma_t * rise) * converted
        elif order == 2 or self.lower < 2 or second:
            stepped = self.second(sample)
        else:
            stepped = self.third(sample)
        self.lower = min(self.lower + 1, order)
        self.index += 1
        return stepped

    def rhos(self, *offsets: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Each sigma `index + offset`'s signal rate and rho, its noise rate over its signal rate."""
        found = []
        for offset in offsets:
            alpha, sigma = self.rates(self.sigmas[self.index + offset])
            found.append((alpha, sigma / alpha))
        return found

    def second(self, sample: torch.Tensor) -> torch.Tensor:
        (alpha_t, rho_t), (alpha_s0, rho_s0), (_, rho_s1) = self.rhos(1, 0, -1)
        log = torch.log

        def integral(t, b, c):
            return t * (-log(c) + log(t) - 1) / (log(b) - log(c))

        first = integral(rho_t, rho_s0, rho_s1) - integral(rho_s0, rho_s0, rho_s1)
        second = integral(rho_t, rho_s1, rho_s0) - integral(rho_s0, rho_s1, rho_s0)
        return alpha_t * (sample / alpha_s0 + first * self.outputs[-1] + second * self.outputs[-2])

    def third(self, sample: torch.Tensor) -> torch.Tensor:
        (alpha_t, rho_t), (alpha_s0, rho_s0), (_, rho_s1), (_, rho_s2) = self.rhos(1, 0, -1, -2)
        log = torch.log

        def integral(t, b, c, d):
            numerator = t * (log(c) * (log(d) - log(t) + 1) - log(d) * log(t) + log(d) + log(t) ** 2
                             - 2 * log(t) + 2)
            return numerator / ((log(b) - log(c)) * (log(b) - log(d)))

        first = integral(rho_t, rho_s0, rho_s1, rho_s2) - integral(rho_s0, rho_s0, rho_s1, rho_s2)
        second = integral(rho_t, rho_s1, rho_s2, rho_s0) - integral(rho_s0, rho_s1, rho_s2, rho_s0)
        third = integral(rho_t, rho_s2, rho_s0, rho_s1) - integral(rho_s0, rho_s2, rho_s0, rho_s1)
        return alpha_t * (sample / alpha_s0 + first * self.outputs[-1] + second * self.outputs[-2]
                          + third * self.outputs[-3])


class UniPC(LogSNR):
    """`UniPCMultistepScheduler` (`scheduling_unipc_multistep.py`): each step a UniC correction of the last
    one's sample by the output read at it (where `disable_corrector` leaves it), then a UniP prediction,
    their coefficients solved from the B(h) family (`solver_type`) over the history's lambdas."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
        "trained_betas": None, "solver_order": 2, "prediction_type": "epsilon", "thresholding": False,
        "dynamic_thresholding_ratio": 0.995, "sample_max_value": 1.0, "predict_x0": True,
        "solver_type": "bh2", "lower_order_final": True, "disable_corrector": (), "solver_p": None,
        "use_karras_sigmas": False, "use_exponential_sigmas": False, "use_beta_sigmas": False,
        "use_flow_sigmas": False, "flow_shift": 1.0, "timestep_spacing": "linspace", "steps_offset": 0,
        "final_sigmas_type": "zero", "rescale_betas_zero_snr": False, "use_dynamic_shifting": False,
        "time_shift_type": "exponential", "sigma_min": None, "sigma_max": None, "shift_terminal": None}
    refused: ClassVar[Mapping[str, object]] = {
        "solver_p": None, "shift_terminal": None, "use_dynamic_shifting": False,
        "time_shift_type": "exponential"}

    def laid(self, asked: Grid) -> None:
        steps = asked.steps_alone()
        train, spacing = self.config.num_train_timesteps, self.config.timestep_spacing
        if spacing == "linspace":
            times = np.linspace(0, train - 1, steps + 1).round()[::-1][:-1].copy().astype(np.int64)
        elif spacing == "leading":
            grid = (np.arange(0, steps + 1) * (train // (steps + 1))).round()[::-1][:-1].copy()
            times = grid.astype(np.int64) + self.config.steps_offset
        elif spacing == "trailing":
            times = np.arange(train, 0, -train / steps).round().copy().astype(np.int64) - 1
        else:
            raise ValueError(spacing)
        cumulative = self.alphas_cumprod.numpy()
        table = np.array(((1 - cumulative) / cumulative) ** 0.5)
        log_table, zero = np.log(table), self.config.final_sigmas_type == "zero"
        moved = transformed(self.config, *stated_ends(self.config, table), steps)
        if moved is not None:
            sigmas = moved
            if self.config.use_karras_sigmas and self.config.use_flow_sigmas:
                sigmas = sigmas / (sigmas + 1)
                times = sigmas * train
            else:
                times = np.array([sigma_to_t(np.asarray(sigma), log_table) for sigma in sigmas])
                if self.config.use_karras_sigmas:
                    times = times.round()
            last = 0.0 if zero else float(sigmas[-1])
        elif self.config.use_flow_sigmas:
            raw = np.linspace(1, 1 / train, steps + 1)[:-1]
            shift = self.config.flow_shift
            sigmas = shift * raw / (1 + (shift - 1) * raw)
            # The library's nudge off 1, where its unshifted grid starts it, as every flow shift keeps it.
            sigmas[0] -= 1e-6
            times = sigmas * train
            last = 0.0 if zero else float(sigmas[-1])
        else:
            sigmas = np.interp(times, np.arange(0, len(table)), table)
            last = 0.0 if zero else float(((1 - cumulative[0]) / cumulative[0]) ** 0.5)
        self.sigmas = torch.from_numpy(np.concatenate([sigmas, [last]]).astype(self.width))
        self.timesteps = torch.from_numpy(np.asarray(times)).to(torch.int64)
        self.steps, self.index, self.outputs, self.lower = len(self.timesteps), 0, [], 0
        self.last_sample: torch.Tensor | None = None
        self.this_order = 1

    def converted(self, output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        sigma = self.sigmas[self.index]
        alpha, noise_rate = self.rates(sigma)
        prediction = self.config.prediction_type
        if self.config.predict_x0:
            clean = (sample - sigma * output if prediction == "flow_prediction"
                     else predicted(prediction, sample, output, alpha, noise_rate)[0])
            return self.limited(clean)
        return predicted(prediction, sample, output, alpha, noise_rate)[1]

    def solved(self, at: int, before: int, order: int, offset: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                          list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """What a UniP (`offset` 0) or UniC (`offset` 1) update from sigma `before` to sigma `at` reads: the
        target's rates, the source's noise rate, h's phi, B(h), the system R and b, and the history's scaled
        differences."""
        alpha_t, sigma_t = self.rates(self.sigmas[at])
        alpha_s0, sigma_s0 = self.rates(self.sigmas[before])
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = torch.log(alpha_t) - torch.log(sigma_t) - lambda_s0
        m0 = self.outputs[-1]
        rks, differences = [], []
        for k in range(1, order):
            alpha_k, sigma_k = self.rates(self.sigmas[self.index - k - offset])
            rk = (torch.log(alpha_k) - torch.log(sigma_k) - lambda_s0) / h
            rks.append(rk)
            differences.append((self.outputs[-(k + 1)] - m0) / rk)
        rks = torch.stack([*rks, torch.ones((), dtype=h.dtype)])
        hh = -h if self.config.predict_x0 else h
        phi_1 = torch.expm1(hh)
        phi_k, factorial = phi_1 / hh - 1, 1
        b_h = hh if self.config.solver_type == "bh1" else torch.expm1(hh)
        powers, rights = [], []
        for i in range(1, order + 1):
            powers.append(torch.pow(rks, i - 1))
            rights.append(phi_k * factorial / b_h)
            factorial *= i + 1
            phi_k = phi_k / hh - 1 / factorial
        return (alpha_t, sigma_t, alpha_s0, sigma_s0, phi_1, differences, b_h, torch.stack(powers),
                torch.stack(rights))

    def moved(self, x: torch.Tensor, alpha_t, sigma_t, alpha_s0, sigma_s0, phi_1, b_h,
              residual) -> torch.Tensor:
        """The exponential integrator's step from `x` less B(h) times `residual`, in the predicted
        quantity's frame."""
        if self.config.predict_x0:
            return sigma_t / sigma_s0 * x - alpha_t * phi_1 * self.outputs[-1] - alpha_t * b_h * residual
        return alpha_t / alpha_s0 * x - sigma_t * phi_1 * self.outputs[-1] - sigma_t * b_h * residual

    def predictor(self, x: torch.Tensor, order: int) -> torch.Tensor:
        alpha_t, sigma_t, alpha_s0, sigma_s0, phi_1, differences, b_h, powers, rights = self.solved(
            self.index + 1, self.index, order, 0)
        residual: torch.Tensor | float = 0
        if differences:
            weights = (torch.full((1,), 0.5, dtype=x.dtype) if order == 2
                       else torch.linalg.solve(powers[:-1, :-1], rights[:-1]))
            residual = sum(weight * difference
                           for weight, difference in zip(weights, differences, strict=True))
        return self.moved(x, alpha_t, sigma_t, alpha_s0, sigma_s0, phi_1, b_h, residual)

    def corrector(self, model_t: torch.Tensor, x: torch.Tensor, order: int) -> torch.Tensor:
        alpha_t, sigma_t, alpha_s0, sigma_s0, phi_1, differences, b_h, powers, rights = self.solved(
            self.index, self.index - 1, order, 1)
        weights = (torch.full((1,), 0.5, dtype=x.dtype) if order == 1
                   else torch.linalg.solve(powers, rights))
        residual = sum(weight * difference
                       for weight, difference in zip(weights[:-1], differences, strict=True))
        return self.moved(x, alpha_t, sigma_t, alpha_s0, sigma_s0, phi_1, b_h,
                          residual + weights[-1] * (model_t - self.outputs[-1]))

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        converted = self.converted(output, sample)
        corrects = self.index > 0 and self.index - 1 not in self.config.disable_corrector
        if corrects and self.last_sample is not None:
            sample = self.corrector(converted, self.last_sample, self.this_order)
        order = self.config.solver_order
        self.outputs = [*self.outputs[-(order - 1):], converted] if order > 1 else [converted]
        this = min(order, self.steps - self.index) if self.config.lower_order_final else order
        self.this_order = min(this, self.lower + 1)
        self.last_sample = sample
        stepped = self.predictor(sample, self.this_order)
        self.lower = min(self.lower + 1, order)
        self.index += 1
        return stepped


class DPMSinglestep(DPMMultistep):
    """`DPMSolverSinglestepScheduler` (`scheduling_dpmsolver_singlestep.py`): steps in groups of its orders
    (`get_order_list`), each group's updates taken from the sample it began at."""

    defaults: ClassVar[Mapping[str, object]] = {
        "num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear",
        "trained_betas": None, "solver_order": 2, "prediction_type": "epsilon", "thresholding": False,
        "dynamic_thresholding_ratio": 0.995, "sample_max_value": 1.0, "algorithm_type": "dpmsolver++",
        "solver_type": "midpoint", "lower_order_final": False, "use_karras_sigmas": False,
        "use_exponential_sigmas": False, "use_beta_sigmas": False, "use_flow_sigmas": False,
        "flow_shift": 1.0, "final_sigmas_type": "zero", "lambda_min_clipped": -math.inf,
        "variance_type": None, "use_dynamic_shifting": False, "time_shift_type": "exponential"}
    refused: ClassVar[Mapping[str, object]] = {
        "variance_type": None, "use_dynamic_shifting": False, "time_shift_type": "exponential"}
    algorithms = ("dpmsolver++", "dpmsolver", "sde-dpmsolver++")

    def rounds_karras(self) -> bool:
        return True

    def laid(self, asked: Grid) -> None:
        super().laid(asked)
        steps, order = len(self.timesteps), self.config.solver_order
        if steps % order != 0 or self.config.final_sigmas_type == "zero":
            # The library registers it into its config (`set_timesteps`): every later grid tapers too.
            self.config.lower_order_final = True
        if self.config.lower_order_final:
            orders = ([1, 2, 3] * (steps // 3 - 1) + [1, 2] + [1] if order == 3 and steps % 3 == 0
                      else [1, 2, 3] * (steps // 3) + [1] if order == 3 and steps % 3 == 1
                      else [1, 2, 3] * (steps // 3) + [1, 2] if order == 3
                      else [1, 2] * (steps // 2 - 1) + [1, 1] if order == 2 and steps % 2 == 0
                      else [1, 2] * (steps // 2) + [1] if order == 2 else [1] * steps)
        else:
            orders = ([1, 2, 3] * (steps // 3) if order == 3 else [1, 2] * (steps // 2) if order == 2
                      else [1] * steps)
        if self.config.final_sigmas_type == "zero":
            orders[-1] = 1
        self.orders, self.group = orders, None
        self.history: list[torch.Tensor] = []

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        converted = self.converted(output, sample)
        self.history = [*self.history[-(self.config.solver_order - 1):], converted] if (
            self.config.solver_order > 1) else [converted]
        noise = self.drawn(output) if self.config.algorithm_type == "sde-dpmsolver++" else None
        order = self.orders[self.index]
        order = min(order, len(self.history))
        if order == 1:
            self.group = sample
        assert self.group is not None
        if order == 1:
            stepped = self.first(converted, self.group, noise)
        elif order == 2:
            stepped = self.second_single(self.group, noise)
        else:
            stepped = self.third_single(self.group, noise)
        self.index += 1
        return stepped

    def second_single(self, sample: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
        found = self.lambdas(1, 0, -1)
        (alpha_t, sigma_t, lambda_t), (alpha_s1, sigma_s1, lambda_s1) = found[0], found[2]
        lambda_s0 = found[1][2]
        m0, m1 = self.history[-1], self.history[-2]
        h = lambda_t - lambda_s1
        return self.second_order(sample, (alpha_t, sigma_t), (alpha_s1, sigma_s1), h,
                                 m1, (1.0 / ((lambda_s0 - lambda_s1) / h)) * (m0 - m1), noise)

    def third_single(self, sample: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
        found = self.lambdas(1, 0, -1, -2)
        (alpha_t, sigma_t, lambda_t), (alpha_s2, sigma_s2, lambda_s2) = found[0], found[3]
        lambda_s0, lambda_s1 = found[1][2], found[2][2]
        m0, m1, m2 = self.history[-1], self.history[-2], self.history[-3]
        h = lambda_t - lambda_s2
        r0, r1 = (lambda_s0 - lambda_s2) / h, (lambda_s1 - lambda_s2) / h
        d1_0, d1_1 = (1.0 / r1) * (m1 - m2), (1.0 / r0) * (m0 - m2)
        d1 = (r0 * d1_0 - r1 * d1_1) / (r0 - r1)
        d2 = 2.0 * (d1_1 - d1_0) / (r0 - r1)
        kind, heun = self.config.algorithm_type, self.config.solver_type == "heun"
        if kind == "dpmsolver++":
            base = (sigma_t / sigma_s2) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * m2
            if heun:
                return (base + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * d1
                        - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h ** 2 - 0.5)) * d2)
            return base + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * d1_1
        if kind == "dpmsolver":
            base = (alpha_t / alpha_s2) * sample - (sigma_t * (torch.exp(h) - 1.0)) * m2
            if heun:
                return (base - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * d1
                        - (sigma_t * ((torch.exp(h) - 1.0 - h) / h ** 2 - 0.5)) * d2)
            return base - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * d1_1
        assert noise is not None
        base = ((sigma_t / sigma_s2 * torch.exp(-h)) * sample + (alpha_t * (1.0 - torch.exp(-2.0 * h))) * m2
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise)
        if heun:
            return (base + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * d1
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h) + (-2.0 * h)) / (-2.0 * h) ** 2 - 0.5)) * d2)
        return base + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * d1_1


class EDM(DPMUpdates):
    """`EDMDPMSolverMultistepScheduler` (`scheduling_edm_dpmsolver_multistep.py`): DPM-Solver's updates over
    EDM's sigmas at a signal rate of one, its model's input and output preconditioned (`precondition_*`);
    no training table (`num_train_timesteps` sizes only the library's training ramp)."""

    algorithms = ("dpmsolver++", "sde-dpmsolver++")
    third_orders = ("dpmsolver++",)

    defaults: ClassVar[Mapping[str, object]] = {
        "sigma_min": 0.002, "sigma_max": 80.0, "sigma_data": 0.5, "sigma_schedule": "karras",
        "num_train_timesteps": 1000, "prediction_type": "epsilon", "rho": 7.0, "solver_order": 2,
        "thresholding": False, "dynamic_thresholding_ratio": 0.995, "sample_max_value": 1.0,
        "algorithm_type": "dpmsolver++", "solver_type": "midpoint", "lower_order_final": True,
        "euler_at_final": False, "final_sigmas_type": "zero"}

    def laid(self, asked: Grid) -> None:
        steps = asked.steps_alone()
        low, high = self.config.sigma_min, self.config.sigma_max
        if self.config.sigma_schedule == "karras":
            ramp = torch.linspace(0, 1, steps, dtype=self.dtype)
            rho = self.config.rho
            sigmas = (high ** (1 / rho) + ramp * (low ** (1 / rho) - high ** (1 / rho))) ** rho
        else:
            sigmas = torch.linspace(math.log(low), math.log(high), steps, dtype=self.dtype).exp().flip(0)
        sigmas = sigmas.to(self.dtype)
        self.timesteps = 0.25 * torch.log(sigmas)
        last = low if self.config.final_sigmas_type == "sigma_min" else 0.0
        self.sigmas = torch.cat([sigmas, torch.tensor([last], dtype=self.dtype)])
        self.steps, self.index, self.outputs, self.lower = steps, 0, [], 0

    @property
    def init_noise_sigma(self) -> float | torch.Tensor:
        return (self.config.sigma_max ** 2 + 1) ** 0.5

    def rates(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones((), dtype=self.dtype), sigma

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        sigma = self.sigmas[self.index]
        return sample * (1 / ((sigma ** 2 + self.config.sigma_data ** 2) ** 0.5))

    def converted(self, output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        return self.limited(self.precondition_outputs(sample, output, self.sigmas[self.index]))

    def precondition_outputs(self, sample: torch.Tensor, model_output: torch.Tensor,
                             sigma: torch.Tensor) -> torch.Tensor:
        """The clean sample the model's output at `sigma` predicts, before thresholding (the library's
        `precondition_outputs`)."""
        data = self.config.sigma_data
        skip = data ** 2 / (sigma ** 2 + data ** 2)
        out = sigma * data / (sigma ** 2 + data ** 2) ** 0.5
        if self.config.prediction_type == "v_prediction":
            out = -out
        elif self.config.prediction_type != "epsilon":
            # The library's `precondition_outputs` refuses it ("sample" too, which its signature names).
            raise ValueError(f"EDM's step takes no {self.config.prediction_type} prediction")
        return skip * sample + out * model_output
