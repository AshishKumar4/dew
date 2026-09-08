"""Native grids and policies for published VP diffusion checkpoint files."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from dew.diffusion.process import Process
from dew.diffusion.schedules.common import GeneralizedNoiseScheduler, NoiseScheduler
from dew.diffusion.schedules.discrete import DiscreteNoiseScheduler
from dew.diffusion.transforms import EpsilonPredictionTransform, PredictionTransform, VPredictionTransform
from dew.sampling.solvers import Algorithm, DDIM, DPMSolverMultistep, Euler, LMS, PNDM

Kind = Literal["DDIM", "PNDM", "LMSDiscrete", "EulerDiscrete", "DPMSolverMultistep"]
Spacing = Literal["leading", "linspace", "trailing"]


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def published_betas(config: Mapping[str, object]) -> np.ndarray:
    count = _integer(config["num_train_timesteps"], "num_train_timesteps")
    if count < 1:
        raise ValueError("num_train_timesteps must be positive")
    trained = config.get("trained_betas")
    if trained is not None:
        if not isinstance(trained, (list, tuple, np.ndarray)):
            raise ValueError("trained_betas must be a numeric sequence")
        betas = np.asarray(trained, np.float32)
    else:
        start = _number(config["beta_start"], "beta_start")
        end = _number(config["beta_end"], "beta_end")
        kind = config.get("beta_schedule", "linear")
        if kind == "linear":
            betas = np.linspace(start, end, count, dtype=np.float32)
        elif kind == "scaled_linear":
            betas = np.linspace(start ** 0.5, end ** 0.5, count, dtype=np.float32) ** 2
        elif kind == "squaredcos_cap_v2":
            steps = np.arange(count + 1, dtype=np.float64) / count
            bars = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            betas = np.minimum(1 - bars[1:] / bars[:-1], 0.999).astype(np.float32)
        else:
            raise ValueError(f"Unsupported beta schedule: {kind}")
    if betas.shape != (count,) or not np.all(np.isfinite(betas)) or np.any((betas < 0) | (betas > 1)):
        raise ValueError("The beta table must have num_train_timesteps finite entries in [0,1]")
    if _boolean(config.get("rescale_betas_zero_snr", False), "rescale_betas_zero_snr"):
        bars = np.cumprod(1 - betas, dtype=np.float64).astype(np.float32)
        signal = np.sqrt(bars)
        first, last = signal[0].copy(), signal[-1].copy()
        if first <= last:
            raise ValueError("Zero-terminal-SNR rescaling needs a decreasing signal schedule")
        signal = (signal - last) * (first / (first - last))
        bars = signal ** 2
        betas = 1 - np.concatenate([bars[:1], bars[1:] / bars[:-1]])
    return np.asarray(betas, np.float32)


class TabulatedVP(DiscreteNoiseScheduler):
    def __init__(self, betas: np.ndarray, *, final_alpha_cumprod: float, stride: int):
        super().__init__(betas, p2_loss_weight_gamma=0)
        self.final_alpha_cumprod = jnp.asarray(final_alpha_cumprod, jnp.float32)
        self.stride = stride

    def rates(self, t):
        t = jnp.asarray(t, jnp.float32)
        index = jnp.clip(t.astype(jnp.int32), 0, self.T - 1)
        alpha = jnp.where(t < 0, self.final_alpha_cumprod, self.alpha_cumprod[index])
        return jnp.sqrt(alpha), jnp.sqrt(1 - alpha)

    def model_time(self, t):
        return jnp.maximum(jnp.asarray(t, jnp.float32), 0.0)

    def step_interval(self, t, t_next):
        """Published DDIM/PNDM transfer stride, independent of evaluation spacing."""
        return jnp.full_like(jnp.asarray(t, jnp.float32), self.stride)

    def half_interval(self, t, t_next):
        """Published PRK uses the integer transfer stride divided by two."""
        return jnp.floor(self.step_interval(t, t_next) / 2)


class _PairedGrid:
    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        self.table = jnp.asarray(sigmas, jnp.float32)
        self.times = jnp.asarray(model_times, jnp.float32)
        self.prior = jnp.asarray(prior, jnp.float32)
        self.T = float(len(sigmas) - 1)

    def sigmas(self, t):
        return jnp.interp(self.T - jnp.asarray(t, jnp.float32), jnp.arange(len(self.table)), self.table)

    def t_of_sigma(self, sigma):
        return self.T - jnp.interp(jnp.asarray(sigma), self.table[::-1], jnp.arange(len(self.table))[::-1])

    def model_time(self, t):
        return jnp.interp(self.T - jnp.asarray(t, jnp.float32), jnp.arange(len(self.times)), self.times)

    def prior_scale(self):
        return self.prior


class SigmaGrid(_PairedGrid, GeneralizedNoiseScheduler):
    """A VE process with paired continuous sigma and model-time coordinates."""
    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        GeneralizedNoiseScheduler.__init__(self, sigma_min=float(sigmas[-2]), sigma_max=float(sigmas[0]))
        _PairedGrid.__init__(self, sigmas, model_times, prior)


class VPGrid(_PairedGrid, NoiseScheduler):
    """The same paired coordinates in normalized VP latent space."""
    def rates(self, t):
        sigma = self.sigmas(t)
        alpha = jax.lax.rsqrt(1 + sigma ** 2)
        return alpha, sigma * alpha

    def sample_t(self, key, n: int):
        return jax.random.uniform(key, (n,), minval=0, maxval=self.T)

    def weight(self, t):
        return jnp.ones_like(jnp.asarray(t, jnp.float32))


@dataclass(frozen=True)
class _Policy:
    kind: Kind
    spacing: Spacing
    offset: int
    clean_terminal: bool
    terminal_sigma: Literal["zero", "sigma_min"]
    zero_snr: bool
    karras: bool
    clip: float | None
    skip_prk: bool
    order: int
    algorithm: Algorithm
    solver_type: Literal["midpoint", "heun"]
    lower_order_final: bool
    euler_at_final: bool
    sigma_min: float | None
    sigma_max: float | None


@dataclass(frozen=True, eq=False)
class SourceSchedule:
    """Source-file policy interpreted once into native solver and grid fields."""
    config: Mapping[str, object]
    betas: np.ndarray
    prediction: type[PredictionTransform]
    policy: _Policy

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> SourceSchedule:
        name = config.get("_class_name")
        if not isinstance(name, str):
            raise ValueError("The scheduler config requires a class name")
        kind = name.removeprefix("Flax").removesuffix("Scheduler")
        if kind not in ("DDIM", "PNDM", "LMSDiscrete", "EulerDiscrete", "DPMSolverMultistep"):
            raise ValueError(f"Unsupported scheduler: {name}")
        prediction = config.get("prediction_type", "epsilon")
        if prediction not in ("epsilon", "v_prediction"):
            raise ValueError(f"Unsupported prediction type: {prediction}")
        if kind == "PNDM" and prediction == "v_prediction":
            raise ValueError("Published PNDM v-prediction requires velocity-domain history; native PNDM uses epsilon history")
        spacing = config.get("timestep_spacing", "leading" if kind in ("DDIM", "PNDM") else "linspace")
        if spacing not in ("leading", "linspace", "trailing"):
            raise ValueError(f"Unsupported timestep spacing: {spacing}")
        algorithm = config.get("algorithm_type", "dpmsolver++")
        if algorithm not in ("dpmsolver++", "dpmsolver", "sde-dpmsolver++", "sde-dpmsolver"):
            raise ValueError(f"Unsupported DPM algorithm: {algorithm}")
        solver_type = config.get("solver_type", "midpoint")
        if solver_type not in ("midpoint", "heun"):
            raise ValueError(f"Unsupported DPM update: {solver_type}")
        terminal = config.get("final_sigmas_type", "sigma_min" if kind == "DPMSolverMultistep" else "zero")
        if terminal not in ("zero", "sigma_min"):
            raise ValueError(f"Unsupported terminal sigma: {terminal}")
        for key in ("thresholding", "use_exponential_sigmas", "use_beta_sigmas", "use_lu_lambdas", "use_flow_sigmas"):
            if config.get(key, False) not in (False, None):
                raise ValueError(f"Native source scheduling does not implement active {key}")
        if config.get("timestep_type", "discrete") != "discrete" or config.get("interpolation_type", "linear") != "linear":
            raise ValueError("Native source scheduling requires discrete model times and linear sigma interpolation")
        if config.get("lambda_min_clipped", -float("inf")) != -float("inf"):
            raise ValueError("Native source scheduling does not implement lambda_min_clipped")
        if config.get("variance_type") not in (None, "fixed_small", "fixed_small_log", "fixed_large", "fixed_large_log"):
            raise ValueError("Native source scheduling does not implement learned variance")
        clip = None
        if kind == "DDIM" and _boolean(config.get("clip_sample", True), "clip_sample"):
            clip = _number(config.get("clip_sample_range", 1.0), "clip_sample_range")
            if clip < 0:
                raise ValueError("clip_sample_range must be nonnegative")
        karras = _boolean(config.get("use_karras_sigmas", False), "use_karras_sigmas")
        if karras and kind in ("DDIM", "PNDM"):
            raise ValueError(f"{kind} does not define Karras source grids")
        policy = _Policy(kind, spacing, _integer(config.get("steps_offset", 0), "steps_offset"),
            _boolean(config.get("set_alpha_to_one", True), "set_alpha_to_one"), terminal,
            _boolean(config.get("rescale_betas_zero_snr", False), "rescale_betas_zero_snr"), karras, clip,
            _boolean(config.get("skip_prk_steps", False), "skip_prk_steps"),
            _integer(config.get("solver_order", 2), "solver_order"), algorithm, solver_type,
            _boolean(config.get("lower_order_final", True), "lower_order_final"),
            _boolean(config.get("euler_at_final", False), "euler_at_final"),
            None if config.get("sigma_min") is None else _number(config["sigma_min"], "sigma_min"),
            None if config.get("sigma_max") is None else _number(config["sigma_max"], "sigma_max"))
        betas = published_betas(config)
        betas.setflags(write=False)
        return cls(MappingProxyType(dict(config)), betas,
                   EpsilonPredictionTransform if prediction == "epsilon" else VPredictionTransform, policy)

    @property
    def kind(self) -> Kind:
        return self.policy.kind

    def training_process(self) -> Process:
        return Process(DiscreteNoiseScheduler(self.betas, p2_loss_weight_gamma=0), self.prediction())

    def solver(self):
        policy = self.policy
        if policy.kind == "DDIM":
            return DDIM(clip=policy.clip)
        if policy.kind == "PNDM":
            return PNDM(skip_prk_steps=policy.skip_prk)
        if policy.kind == "LMSDiscrete":
            return LMS(order=4)
        if policy.kind == "EulerDiscrete":
            return Euler()
        return DPMSolverMultistep(policy.order, policy.algorithm, policy.solver_type,
                                  policy.lower_order_final, policy.euler_at_final)

    def _timesteps(self, steps: int) -> np.ndarray:
        count, policy = len(self.betas), self.policy
        extra = int(policy.kind == "DPMSolverMultistep")
        if policy.spacing == "linspace":
            times = np.linspace(0, count - 1, steps + extra, dtype=np.float32)[::-1].copy()
            if policy.kind in ("DDIM", "PNDM", "DPMSolverMultistep"):
                times = times.round()
            return times[:-1] if extra else times
        if policy.spacing == "leading":
            times = (np.arange(steps + extra) * (count // (steps + extra)))[::-1].astype(np.float32) + policy.offset
            return times[:-1] if extra else times
        return np.arange(count, 0, -count / steps).round().astype(np.float32) - 1

    def _sigma_table(self, steps: int):
        policy = self.policy
        alphas = np.cumprod(1 - self.betas.astype(np.float64))
        if policy.zero_snr and policy.kind in ("EulerDiscrete", "DPMSolverMultistep"):
            alphas[-1] = 2 ** -24
        base = np.sqrt((1 - alphas) / alphas)
        times = self._timesteps(steps)
        initial = np.interp(times, np.arange(len(base)), base)
        if policy.karras:
            low = (float(initial[-1]) if policy.sigma_min is None else policy.sigma_min) ** (1 / 7)
            high = (float(initial[0]) if policy.sigma_max is None else policy.sigma_max) ** (1 / 7)
            # DPM's Karras grid spans the whole training table, not its initial timestep subset.
            if policy.kind == "DPMSolverMultistep":
                low = (float(base[0]) if policy.sigma_min is None else policy.sigma_min) ** (1 / 7)
                high = (float(base[-1]) if policy.sigma_max is None else policy.sigma_max) ** (1 / 7)
            sigmas = (high + np.linspace(0, 1, steps) * (low - high)) ** 7
            times = np.interp(np.log(sigmas), np.log(base), np.arange(len(base))).astype(np.float32)
            if policy.kind == "DPMSolverMultistep" and self.config.get("beta_schedule") != "squaredcos_cap_v2":
                times = times.round()
        else:
            sigmas = initial
        terminal = 0.0 if policy.terminal_sigma == "zero" else float(base[0])
        prior = float(np.max(sigmas)) if policy.spacing in ("linspace", "trailing") else float(np.sqrt(np.max(sigmas) ** 2 + 1))
        return np.append(sigmas, terminal).astype(np.float32), times.astype(np.float32), prior

    @lru_cache(maxsize=32)
    def sampling(self, steps: int) -> tuple[Process, jax.Array]:
        if steps < 1 or (self.kind not in ("LMSDiscrete", "EulerDiscrete") and steps > len(self.betas)):
            raise ValueError("The sampling count must be positive and fit the training table")
        if self.kind in ("LMSDiscrete", "EulerDiscrete", "DPMSolverMultistep"):
            sigmas, times, prior = self._sigma_table(steps)
            schedule = (VPGrid(sigmas, times, 1.0) if self.kind == "DPMSolverMultistep" else SigmaGrid(sigmas, times, prior))
            prediction = self.prediction(normalize_input=self.kind != "DPMSolverMultistep")
            return Process(schedule, prediction), jnp.arange(len(sigmas) - 1, -1, -1, dtype=jnp.float32)
        times = self._timesteps(steps)
        final = 1.0 if self.policy.clean_terminal else float(1 - self.betas[0])
        schedule = TabulatedVP(self.betas, final_alpha_cumprod=final, stride=len(self.betas) // steps)
        terminal = max(float(times[-1]) - len(self.betas) // steps, -1.0)
        return Process(schedule, self.prediction()), jnp.asarray(np.append(times, terminal), jnp.float32)
