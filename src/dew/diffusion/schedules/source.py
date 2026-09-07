"""Native sampling grids for published discrete variance-preserving checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from dew.diffusion.process import Process
from dew.diffusion.schedules.common import GeneralizedNoiseScheduler
from dew.diffusion.schedules.discrete import DiscreteNoiseScheduler
from dew.diffusion.transforms import EpsilonPredictionTransform, PredictionTransform, VPredictionTransform
from dew.sampling.solvers import DDIM, DPMSolverMultistep, Euler, LMS, PNDM


def published_betas(config: Mapping) -> np.ndarray:
    count = int(config["num_train_timesteps"])
    start, end = float(config["beta_start"]), float(config["beta_end"])
    kind = config.get("beta_schedule", "linear")
    trained = config.get("trained_betas")
    if trained is not None:
        return np.asarray(trained, np.float32)
    if kind == "linear":
        return np.linspace(start, end, count, dtype=np.float32)
    if kind == "scaled_linear":
        return np.linspace(start ** 0.5, end ** 0.5, count, dtype=np.float32) ** 2
    if kind == "squaredcos_cap_v2":
        steps = np.arange(count + 1, dtype=np.float64) / count
        bars = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        return np.minimum(1 - bars[1:] / bars[:-1], 0.999).astype(np.float32)
    raise ValueError(f"Unsupported beta schedule: {kind}")


class TabulatedVP(DiscreteNoiseScheduler):
    """A published beta table, indexed on grid values that may be terminal."""

    def __init__(self, betas: np.ndarray, *, final_alpha_cumprod: float):
        super().__init__(betas, p2_loss_weight_gamma=0)
        self.final_alpha_cumprod = jnp.asarray(final_alpha_cumprod, jnp.float32)

    def rates(self, t):
        t = jnp.asarray(t, jnp.float32)
        index = jnp.clip(t.astype(jnp.int32), 0, self.T - 1)
        alpha_cumprod = jnp.where(t < 0, self.final_alpha_cumprod, self.alpha_cumprod[index])
        return jnp.sqrt(alpha_cumprod), jnp.sqrt(1 - alpha_cumprod)

    def model_time(self, t):
        return jnp.maximum(jnp.asarray(t, jnp.float32), 0.0)

    def half_interval(self, t, t_next):
        """Published PRK stages use half the integer index stride, rounded down."""
        return jnp.floor((jnp.asarray(t, jnp.float32) - jnp.asarray(t_next, jnp.float32)) / 2)



class SigmaGrid(GeneralizedNoiseScheduler):
    """Paired sigma and model-time tables; t is the grid position."""

    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        super().__init__(sigma_min=float(sigmas[-2]), sigma_max=float(sigmas[0]))
        self.table = jnp.asarray(sigmas, jnp.float32)
        self.times = jnp.asarray(model_times, jnp.float32)
        self.prior = jnp.asarray(prior, jnp.float32)
        self.T = float(len(sigmas) - 1)

    def sigmas(self, t):
        position = self.T - jnp.asarray(t, jnp.float32)
        return jnp.interp(position, jnp.arange(len(self.table)), self.table)

    def t_of_sigma(self, sigma):
        position = jnp.interp(jnp.asarray(sigma), self.table[::-1], jnp.arange(len(self.table))[::-1])
        return self.T - position

    def model_time(self, t):
        position = self.T - jnp.asarray(t, jnp.float32)
        return jnp.interp(position, jnp.arange(len(self.times)), self.times)

    def prior_scale(self):
        return self.prior


@dataclass(frozen=True, eq=False)
class SourceSchedule:
    """A checkpoint scheduler config as native training and sampling policy."""
    config: Mapping
    betas: np.ndarray
    prediction: type[PredictionTransform]

    @classmethod
    def from_config(cls, config: Mapping) -> "SourceSchedule":
        prediction = {"epsilon": EpsilonPredictionTransform, "v_prediction": VPredictionTransform}
        kind = config.get("prediction_type", "epsilon")
        if kind not in prediction:
            raise ValueError(f"Unsupported prediction type: {kind}")
        name = config["_class_name"].removeprefix("Flax")
        if name not in ("DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler",
                        "EulerDiscreteScheduler", "DPMSolverMultistepScheduler"):
            raise ValueError(f"Unsupported scheduler: {name}")
        return cls(dict(config), published_betas(config), prediction[kind])

    @property
    def kind(self) -> str:
        return self.config["_class_name"].removeprefix("Flax").removesuffix("Scheduler")

    def training_process(self) -> Process:
        return Process(DiscreteNoiseScheduler(self.betas, p2_loss_weight_gamma=0), self.prediction())

    def solver(self):
        if self.kind == "DDIM":
            return DDIM()
        if self.kind == "PNDM":
            return PNDM(skip_prk_steps=bool(self.config.get("skip_prk_steps", False)))
        if self.kind == "LMSDiscrete":
            return LMS(order=4)
        if self.kind == "EulerDiscrete":
            return Euler()
        return DPMSolverMultistep(order=int(self.config.get("solver_order", 2)),
                                  algorithm=self.config.get("algorithm_type", "dpmsolver++"),
                                  solver_type=self.config.get("solver_type", "midpoint"),
                                  lower_order_final=bool(self.config.get("lower_order_final", True)),
                                  euler_at_final=bool(self.config.get("euler_at_final", False)))

    def _timesteps(self, steps: int) -> np.ndarray:
        count = int(self.config["num_train_timesteps"])
        default = "linspace" if self.kind in ("LMSDiscrete", "EulerDiscrete", "DPMSolverMultistep") else "leading"
        spacing = self.config.get("timestep_spacing", default)
        offset = int(self.config.get("steps_offset", 0))
        extra = 1 if self.kind == "DPMSolverMultistep" else 0
        if spacing == "linspace":
            times = np.linspace(0, count - 1, steps + extra, dtype=np.float32)[::-1].copy()
            return times.round()[:-1] if extra else times
        if spacing == "leading":
            ratio = count // (steps + extra)
            times = (np.arange(steps + extra) * ratio).round()[::-1].astype(np.float32) + offset
            return times[:-1] if extra else times
        if spacing == "trailing":
            ratio = count / steps
            return (np.arange(count, 0, -ratio).round().astype(np.float32) - 1)
        raise ValueError(f"Unsupported timestep spacing: {spacing}")

    def _sigma_table(self, steps: int):
        alphas = np.cumprod(1 - self.betas.astype(np.float64))
        base = np.sqrt((1 - alphas) / alphas)
        count = len(alphas)
        if self.config.get("use_karras_sigmas", False):
            rho = 7.0
            ramp = np.linspace(0, 1, steps)
            low, high = base.min() ** (1 / rho), base.max() ** (1 / rho)
            sigmas = (high + ramp * (low - high)) ** rho
            log_base = np.log(base)
            dists = np.log(sigmas)[:, None] - log_base[None]
            low_index = np.cumsum(dists >= 0, axis=-1).argmax(axis=-1).clip(0, count - 2)
            high_index = low_index + 1
            weight = ((log_base[low_index] - np.log(sigmas)) / (log_base[low_index] - log_base[high_index])).clip(0, 1)
            times = (1 - weight) * low_index + weight * high_index
        else:
            times = self._timesteps(steps).astype(np.float64)
            sigmas = np.interp(times, np.arange(count), base)
        prior = float(sigmas.max()) if self.config.get("timestep_spacing", "linspace") in ("linspace", "trailing") \
            else float(np.sqrt(sigmas.max() ** 2 + 1))
        return np.append(sigmas, 0.0).astype(np.float32), times.astype(np.float32), prior

    @lru_cache(maxsize=32)
    def sampling(self, steps: int) -> tuple[Process, jax.Array]:
        """The paired native process and explicit grid for one step count."""
        if steps < 1:
            raise ValueError("steps must be positive")
        if self.kind not in ("LMSDiscrete", "EulerDiscrete") and steps > len(self.betas):
            raise ValueError("The requested step count exceeds the training time table")
        if self.kind in ("LMSDiscrete", "EulerDiscrete"):
            sigmas, times, prior = self._sigma_table(steps)
            return (Process(SigmaGrid(sigmas, times, prior), self.prediction(normalize_input=True)),
                    jnp.arange(len(sigmas) - 1, -1, -1, dtype=jnp.float32))
        times = self._timesteps(steps)
        stride = int(self.config["num_train_timesteps"]) // steps
        if self.kind == "DPMSolverMultistep":
            final_type = self.config.get("final_sigmas_type", "sigma_min")
            final = 1.0 if final_type == "zero" else float(1 - self.betas[0])
            schedule = TabulatedVP(self.betas, final_alpha_cumprod=final)
            terminal = -1.0 if final_type == "zero" else 0.0
            return Process(schedule, self.prediction()), jnp.asarray(np.append(times, terminal), jnp.float32)
        final = 1.0 if self.config.get("set_alpha_to_one", True) else float(np.cumprod(1 - self.betas)[0])
        schedule = TabulatedVP(self.betas, final_alpha_cumprod=final)
        # The native solvers own the PNDM warmup and the DDIM transfer; the
        # grid holds each published step plus the terminal step past index 0.
        terminal = max(float(times[-1]) - stride, -1.0)
        return Process(schedule, self.prediction()), jnp.asarray(np.append(times, terminal), jnp.float32)
