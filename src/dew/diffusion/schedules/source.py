"""Native grids and policies for published diffusion checkpoint scheduler files.

One class of a published `scheduler_config.json` is interpreted once, here,
into the native process, solver and time grid that reproduce its
`set_timesteps` and `step`. Each pinned Diffusers 0.34.0 class declares its own
constructor controls with its own defaults and builds its grid its own way, so
`_SOURCES` names, per class, exactly the keys that class reads and the grid
family it belongs to; a key another class declares is not read here, the way
the source ignores it. Whatever a class does declare and this file does not
reconstruct is refused rather than dropped.

The five families are the shapes those `set_timesteps` take:

- `tabulated`: DDIM, PNDM, DDPM, LCM and TCD step between integer indices of
  the training beta table, so the schedule is that table and the grid is the
  indices. DDIM and PNDM transfer over a fixed training stride whatever their
  evaluation spacing; DDPM, LCM and TCD step to the grid's own next point.
- `lambda`: DPM-Solver multistep and singlestep, DEIS and UniPC integrate in
  log-SNR over paired sigma and model-time tables, normalized so that
  alpha^2 + sigma^2 is 1, and truncate their model times to integers.
- `sigma`: LMS, Euler, Euler ancestral and Heun integrate the
  variance-exploding sigma directly and scale the model input by
  1 / sqrt(sigma^2 + 1).
- `stage`: KDPM2, KDPM2 ancestral and DPMSolverSDE evaluate the model twice
  per interval. Their grids carry the interpolated stage rows the source
  places between grid points, so the solver's own second evaluation reads the
  source's sigma and model time there while the outer walk still visits one
  point per interval.
- `edm`: EDMDPMSolverMultistep is EDM's own convention, sigma_min to sigma_max
  at rho with c_noise = log(sigma) / 4 and a signed c_out. It has no beta
  table and no VP training law, so its training process is EDM's log-normal
  sigma draw over the same preconditioning.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Callable, Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from dew.diffusion.process import Process
from dew.diffusion.schedules.common import GeneralizedNoiseScheduler, NoiseScheduler
from dew.diffusion.schedules.discrete import DiscreteNoiseScheduler
from dew.diffusion.schedules.karras import EDMNoiseScheduler
from dew.diffusion.transforms import (
    ConsistencyBoundary, DirectPredictionTransform, EpsilonPredictionTransform,
    KarrasPredictionTransform, PredictionTransform, SourceLimitedPrediction,
    VPredictionTransform,
)
from dew.sampling.solvers import (
    Algorithm, Consistency, DDIM, DDPM, DEIS, DPMSolverMultistep, DPMSolverSDE,
    DPMSolverSinglestep, Euler, EulerAncestral, Heun, KDPM2, LMS, PNDM, TCD, UniPC,
)

Kind = Literal[
    "DDIM", "PNDM", "DDPM", "LMSDiscrete", "EulerDiscrete", "EulerAncestralDiscrete",
    "HeunDiscrete", "KDPM2Discrete", "KDPM2AncestralDiscrete", "DPMSolverMultistep",
    "DPMSolverSinglestep", "DPMSolverSDE", "DEISMultistep", "UniPCMultistep",
    "EDMDPMSolverMultistep", "LCM", "TCD",
]
Family = Literal["tabulated", "lambda", "sigma", "stage", "edm"]
Spacing = Literal["leading", "linspace", "trailing"]
Transform = Literal["none", "karras", "exponential", "beta"]
Terminal = Literal["zero", "sigma_min"]
Variance = Literal["small", "large"]

_SPACINGS: tuple[Spacing, ...] = ("leading", "linspace", "trailing")
_TERMINALS: tuple[Terminal, ...] = ("zero", "sigma_min")
_SIGMA_SCHEDULES: tuple[Transform, ...] = ("karras", "exponential")
# Each sigma transformation with the control that turns it on.
_TRANSFORM_CONTROLS: tuple[tuple[Transform, str], ...] = (
    ("karras", "use_karras_sigmas"), ("exponential", "use_exponential_sigmas"),
    ("beta", "use_beta_sigmas"))


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


def _choice[ChoiceT: str](value: object, name: str, allowed: tuple[ChoiceT, ...]) -> ChoiceT:
    for choice in allowed:
        if value == choice:
            return choice
    raise ValueError(f"Native source scheduling does not implement {name}={value!r}; "
                     f"this class supports {', '.join(allowed)}")


def published_betas(*, count: object, start: object, end: object, schedule: object,
                    trained: object, zero_snr: bool, schedules: tuple[str, ...]) -> np.ndarray:
    """The class's beta table, rescaled for zero terminal SNR when it asks.

    Every control arrives already resolved against the class's own declared
    default, so a file that omits one gets that class's value and a file that
    carries a control the class does not declare does not reach the table.
    `schedules` are the `beta_schedule` tables the class implements: all of
    them accept the three common ones, DDPM adds GeoDiff's sigmoid and Heun
    the exponential alpha-bar.
    """
    length = _integer(count, "num_train_timesteps")
    if length < 1:
        raise ValueError("num_train_timesteps must be positive")
    if trained is not None:
        if not isinstance(trained, (list, tuple, np.ndarray)):
            raise ValueError("trained_betas must be a numeric sequence")
        betas = np.asarray(trained, np.float32)
    else:
        first = _number(start, "beta_start")
        final = _number(end, "beta_end")
        kind = _choice(schedule, "beta_schedule", schedules)
        if kind == "linear":
            betas = np.linspace(first, final, length, dtype=np.float32)
        elif kind == "scaled_linear":
            betas = np.linspace(first ** 0.5, final ** 0.5, length, dtype=np.float32) ** 2
        elif kind == "sigmoid":
            ramp = np.linspace(-6, 6, length, dtype=np.float32)
            betas = 1 / (1 + np.exp(-ramp)) * (final - first) + first
        else:
            steps = np.arange(length + 1, dtype=np.float64) / length
            bars = (np.exp(steps * -12.0) if kind == "exp"
                    else np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2)
            betas = np.minimum(1 - bars[1:] / bars[:-1], 0.999).astype(np.float32)
    if betas.shape != (length,) or not np.all(np.isfinite(betas)) or np.any((betas < 0) | (betas > 1)):
        raise ValueError("The beta table must have num_train_timesteps finite entries in [0,1]")
    if zero_snr:
        bars = np.cumprod(1 - betas, dtype=np.float64).astype(np.float32)
        signal = np.sqrt(bars)
        head, tail = signal[0].copy(), signal[-1].copy()
        if head <= tail:
            raise ValueError("Zero-terminal-SNR rescaling needs a decreasing signal schedule")
        signal = (signal - tail) * (head / (head - tail))
        bars = signal ** 2
        betas = 1 - np.concatenate([bars[:1], bars[1:] / bars[:-1]])
    return np.asarray(betas, np.float32)


class TabulatedVP(DiscreteNoiseScheduler):
    """The training beta table as the sampling schedule, indexed by t.

    `stride` is the fixed training transfer DDIM and PNDM step over whatever
    their evaluation grid is; None leaves the grid's own interval, which is
    what DDPM's previous-timestep policy and the distilled schedules take.
    A t below zero is the source's "no previous alpha" end.
    """

    def __init__(self, betas: np.ndarray, *, final_alpha_cumprod: float, stride: int | None):
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
        if self.stride is None:
            return super().step_interval(t, t_next)
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
    """A VE process with paired continuous sigma and model-time coordinates.

    `sigma_min` and `sigma_max` are the prepared grid's own positive extremes,
    which is the domain a source noise sampler is built over."""

    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        levels = np.asarray(sigmas, np.float64)
        GeneralizedNoiseScheduler.__init__(self, sigma_min=float(np.min(levels[levels > 0])),
                                           sigma_max=float(np.max(levels)))
        _PairedGrid.__init__(self, sigmas, model_times, prior)


class StageSigmaGrid(SigmaGrid):
    """A source VE grid that carries the stage rows of a two-evaluation solver.

    Even coordinates are the grid points the outer walk visits; the odd one
    between each pair is the source's own interpolated evaluation, at the
    sigma it places there and the model time it reads back for that sigma.
    `t_of_sigma` resolves a sigma to a stage coordinate, which is the only
    inversion these solvers ask of a schedule: KDPM2's midpoint and
    DPMSolverSDE's proposal both land on a stage row.
    """

    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        super().__init__(sigmas, model_times, prior)
        stages = np.asarray(sigmas, np.float64)[1::2]
        self.stages = jnp.asarray(stages[::-1].copy(), jnp.float32)
        self.stage_positions = jnp.asarray(
            np.arange(1, len(sigmas), 2, dtype=np.float32)[::-1].copy())

    def t_of_sigma(self, sigma):
        return self.T - jnp.interp(jnp.asarray(sigma), self.stages, self.stage_positions)


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


def _fields(*groups: Mapping[str, object], **extra: object) -> Mapping[str, object]:
    merged: dict[str, object] = {}
    for group in groups:
        merged.update(group)
    merged.update(extra)
    return MappingProxyType(merged)


def _betas(start: float, end: float, schedule: str = "linear") -> Mapping[str, object]:
    return {"num_train_timesteps": 1000, "beta_start": start, "beta_end": end,
            "beta_schedule": schedule, "trained_betas": None, "prediction_type": "epsilon"}


_VP_BETAS = _betas(0.0001, 0.02)
_KD_BETAS = _betas(0.00085, 0.012)
_LCM_BETAS = _betas(0.00085, 0.012, "scaled_linear")
_THRESHOLD = {"thresholding": False, "dynamic_thresholding_ratio": 0.995, "sample_max_value": 1.0}
_SPACED = {"timestep_spacing": "linspace", "steps_offset": 0}
_LEADING = {"timestep_spacing": "leading", "steps_offset": 0}
_TRANSFORMS = {"use_karras_sigmas": False, "use_exponential_sigmas": False,
               "use_beta_sigmas": False}
_FLOW = {"use_flow_sigmas": False, "flow_shift": 1.0}
_DPM = {"solver_order": 2, "algorithm_type": "dpmsolver++", "solver_type": "midpoint",
        "final_sigmas_type": "zero", "lambda_min_clipped": -float("inf"), "variance_type": None}
_DISTILLED = _fields(_THRESHOLD, {"original_inference_steps": 50, "clip_sample": False,
                                  "clip_sample_range": 1.0, "set_alpha_to_one": True,
                                  "steps_offset": 0, "timestep_spacing": "leading",
                                  "timestep_scaling": 10.0, "rescale_betas_zero_snr": False})


_ALL_PREDICTIONS = ("epsilon", "sample", "v_prediction")
_NO_SAMPLE = ("epsilon", "v_prediction")


@dataclass(frozen=True)
class _Class:
    """One pinned scheduler class: its grid family, the constructor controls it
    declares with that class's own defaults, the `beta_schedule` tables it
    accepts and the `prediction_type` values its own `step` converts."""

    family: Family
    fields: Mapping[str, object]
    schedules: tuple[str, ...] = ("linear", "scaled_linear", "squaredcos_cap_v2")
    predictions: tuple[str, ...] = _ALL_PREDICTIONS


_SOURCES: Mapping[str, _Class] = MappingProxyType({
    "DDIM": _Class("tabulated", _fields(_VP_BETAS, _THRESHOLD, _LEADING, clip_sample=True,
                                        clip_sample_range=1.0, set_alpha_to_one=True,
                                        rescale_betas_zero_snr=False)),
    "PNDM": _Class("tabulated", _fields(_VP_BETAS, _LEADING, skip_prk_steps=False,
                                        set_alpha_to_one=False), predictions=_NO_SAMPLE),
    "DDPM": _Class("tabulated", _fields(_VP_BETAS, _THRESHOLD, _LEADING,
                                        variance_type="fixed_small", clip_sample=True,
                                        clip_sample_range=1.0, rescale_betas_zero_snr=False),
                   ("linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid")),
    "LMSDiscrete": _Class("sigma", _fields(_VP_BETAS, _SPACED, _TRANSFORMS)),
    "EulerDiscrete": _Class("sigma", _fields(_VP_BETAS, _SPACED, _TRANSFORMS,
                                             interpolation_type="linear", sigma_min=None,
                                             sigma_max=None, timestep_type="discrete",
                                             rescale_betas_zero_snr=False,
                                             final_sigmas_type="zero")),
    "EulerAncestralDiscrete": _Class("sigma", _fields(_VP_BETAS, _SPACED,
                                                      rescale_betas_zero_snr=False),
                                     predictions=_NO_SAMPLE),
    "HeunDiscrete": _Class("sigma", _fields(_KD_BETAS, _SPACED, _TRANSFORMS, clip_sample=False,
                                            clip_sample_range=1.0),
                           ("linear", "scaled_linear", "squaredcos_cap_v2", "exp")),
    "KDPM2Discrete": _Class("stage", _fields(_KD_BETAS, _SPACED, _TRANSFORMS),
                            predictions=_NO_SAMPLE),
    "KDPM2AncestralDiscrete": _Class("stage", _fields(_KD_BETAS, _SPACED, _TRANSFORMS),
                                     predictions=_NO_SAMPLE),
    "DPMSolverSDE": _Class("stage", _fields(_KD_BETAS, _SPACED, _TRANSFORMS,
                                            noise_sampler_seed=None), predictions=_NO_SAMPLE),
    "DPMSolverMultistep": _Class("lambda", _fields(_VP_BETAS, _THRESHOLD, _SPACED, _TRANSFORMS,
                                                   _FLOW, _DPM, lower_order_final=True,
                                                   euler_at_final=False, use_lu_lambdas=False,
                                                   rescale_betas_zero_snr=False)),
    "DPMSolverSinglestep": _Class("lambda", _fields(_VP_BETAS, _THRESHOLD, _TRANSFORMS, _FLOW,
                                                    _DPM, lower_order_final=False)),
    "DEISMultistep": _Class("lambda", _fields(_VP_BETAS, _THRESHOLD, _SPACED, _TRANSFORMS, _FLOW,
                                              solver_order=2, algorithm_type="deis",
                                              solver_type="logrho", lower_order_final=True)),
    "UniPCMultistep": _Class("lambda", _fields(_VP_BETAS, _THRESHOLD, _SPACED, _TRANSFORMS, _FLOW,
                                               solver_order=2, predict_x0=True, solver_type="bh2",
                                               lower_order_final=True, disable_corrector=(),
                                               solver_p=None, final_sigmas_type="zero",
                                               rescale_betas_zero_snr=False)),
    "EDMDPMSolverMultistep": _Class("edm", _fields(
        _THRESHOLD, num_train_timesteps=1000, prediction_type="epsilon", sigma_min=0.002,
        sigma_max=80.0, sigma_data=0.5, sigma_schedule="karras", rho=7.0, solver_order=2,
        algorithm_type="dpmsolver++", solver_type="midpoint", lower_order_final=True,
        euler_at_final=False, final_sigmas_type="zero"), (), _NO_SAMPLE),
    "LCM": _Class("tabulated", _fields(_LCM_BETAS, _DISTILLED)),
    "TCD": _Class("tabulated", _fields(_LCM_BETAS, _DISTILLED)),
})

# Controls a class declares whose active meaning this file does not
# reconstruct. Each maps to the only value that leaves the source's own
# behaviour unchanged, so an active one is refused instead of dropped.
_UNIMPLEMENTED: Mapping[str, object] = MappingProxyType({
    "use_lu_lambdas": False, "use_flow_sigmas": False, "solver_p": None,
    "interpolation_type": "linear", "timestep_type": "discrete",
})

# The classes that round a Karras grid's recovered model times before
# truncating them. The rest keep the log-linear inverse as it comes, and no
# class rounds an exponential or beta grid's.
_KARRAS_ROUNDS = ("DPMSolverSinglestep", "DEISMultistep", "UniPCMultistep",
                  "KDPM2Discrete", "KDPM2AncestralDiscrete")


@dataclass(frozen=True)
class _Policy:
    """One source class's controls, resolved into the numbers its grid and its
    solver need."""

    kind: str
    family: Family
    train_steps: int
    spacing: Spacing
    offset: int
    transform: Transform
    karras_round: bool
    terminal: Terminal
    grid_terminal: bool
    zero_snr_tail: bool
    lambda_clipped: int
    sigma_min: float | None
    sigma_max: float | None
    sigma_data: float
    rho: float
    stride: bool
    clean_terminal: bool
    variance: Variance
    clip: float | None
    threshold: tuple[float, float] | None
    recompute_epsilon: bool
    order: int
    algorithm: Algorithm
    solver_type: str
    lower_order_final: bool
    euler_at_final: bool
    predict_x0: bool
    disable_corrector: tuple[int, ...]
    skip_prk: bool
    distilled: bool
    original_steps: int
    timestep_scaling: float
    seed: int | None


def _spaced_times(spacing: str, *, train_steps: int, steps: int, offset: int, last: int,
                  extra: int, rounded: bool) -> np.ndarray:
    """A source `set_timesteps` evaluation grid: Table 2 of Lin et al. 2023 in
    the three forms the pinned classes write it.

    `extra` is the additional point the log-SNR classes lay out and drop,
    `last` the end of the table left after lambda clipping, and `rounded`
    whether the class rounds its linspace before using it.
    """
    if spacing == "linspace":
        times = np.linspace(0, last - 1, steps + extra, dtype=np.float64)
        if rounded:
            times = times.round()
        return times[::-1][:steps].copy()
    if spacing == "leading":
        ratio = last // (steps + extra)
        times = (np.arange(steps + extra, dtype=np.float64) * ratio).round()
        return times[::-1][:steps].copy() + offset
    ratio = train_steps / steps
    return np.arange(last, 0, -ratio, dtype=np.float64).round() - 1


def _sigma_to_time(sigmas, log_sigmas: np.ndarray) -> np.ndarray:
    """The source's `_sigma_to_t`: each sigma's log-linear position in the
    training table, clamped to the table's ends the way its weight is."""
    values = np.log(np.maximum(np.asarray(sigmas, np.float64), 1e-10))
    return np.interp(values, log_sigmas, np.arange(len(log_sigmas), dtype=np.float64))


def _transformed_sigmas(transform: str, low: float, high: float, steps: int,
                        rho: float) -> np.ndarray:
    """The sigma grid one of the source's sigma transformations lays between
    `low` and `high`."""
    ramp = np.linspace(0, 1, steps, dtype=np.float64)
    if transform == "karras":
        low_inv, high_inv = low ** (1 / rho), high ** (1 / rho)
        return (high_inv + ramp * (low_inv - high_inv)) ** rho
    if transform == "exponential":
        return np.exp(np.linspace(np.log(high), np.log(low), steps, dtype=np.float64))
    try:
        from scipy.stats import beta as beta_distribution
    except ImportError as missing:  # the source gates its own beta sigmas on scipy
        raise ValueError("Beta sigmas need scipy, as the source scheduler does") from missing
    quantiles = np.asarray(beta_distribution.ppf(1 - ramp, 0.6, 0.6), np.float64)
    return low + quantiles * (high - low)


def _distilled_times(train_steps: int, original_steps: int, steps: int) -> np.ndarray:
    """The distilled schedule LCM and TCD select from: every `k`-th index of
    the training table, reversed, sampled at evenly floored indices."""
    skip = train_steps // original_steps
    distilled = (np.arange(1, original_steps + 1, dtype=np.int64) * skip - 1)[::-1]
    if len(distilled) // steps < 1:
        raise ValueError("The distilled schedule is shorter than the requested step count")
    indices = np.floor(np.linspace(0, len(distilled), num=steps, endpoint=False)).astype(np.int64)
    return distilled[indices].astype(np.float64)


@dataclass(frozen=True, eq=False)
class SourceSchedule:
    """Source-file policy interpreted once into native solver and grid fields."""

    config: Mapping[str, object]
    betas: np.ndarray
    prediction: PredictionTransform
    policy: _Policy

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> SourceSchedule:
        name = config.get("_class_name")
        if not isinstance(name, str):
            raise ValueError("The scheduler config requires a class name")
        kind = name.removeprefix("Flax").removesuffix("Scheduler")
        source = _SOURCES.get(kind)
        if source is None:
            raise ValueError(f"Unsupported scheduler: {name}")
        declared = source.fields

        def value(key: str, absent: object = None) -> object:
            """The class's value for a control it declares, or `absent` when
            this class declares no such control."""
            return config.get(key, declared[key]) if key in declared else absent

        for key, inactive in _UNIMPLEMENTED.items():
            if key in declared and value(key) != inactive:
                raise ValueError(f"Native source scheduling does not implement active {key}")
        prediction = _choice(value("prediction_type"), "prediction_type", source.predictions)
        if kind == "PNDM" and prediction == "v_prediction":
            raise ValueError("Published PNDM v-prediction requires velocity-domain history; "
                             "native PNDM uses epsilon history")
        betas = (np.zeros((0,), np.float32) if source.family == "edm" else published_betas(
            count=value("num_train_timesteps"), start=value("beta_start"),
            end=value("beta_end"), schedule=value("beta_schedule"),
            trained=value("trained_betas"),
            zero_snr=_boolean(value("rescale_betas_zero_snr", False), "rescale_betas_zero_snr"),
            schedules=source.schedules))
        betas.setflags(write=False)
        policy = _resolve(kind, source, value, betas)
        return cls(MappingProxyType(dict(config)), betas,
                   _prediction_transform(policy, prediction), policy)

    @property
    def kind(self) -> str:
        return self.policy.kind

    @property
    def train_steps(self) -> int:
        """The training step count the class declares, which is the beta
        table's length wherever the class tabulates one."""
        return self.policy.train_steps

    def training_process(self) -> Process:
        """The process the checkpoint was trained under.

        Every class but the EDM one tabulates a VP beta table and trains on
        it. EDM's convention has no beta table and no VP law, so its training
        process is EDM's own log-normal sigma draw over the same
        preconditioning the sampler reads.
        """
        if self.policy.family == "edm":
            schedule = EDMNoiseScheduler(sigma_min=self.policy.sigma_min or 0.002,
                                         sigma_max=self.policy.sigma_max or 80.0,
                                         sigma_data=self.policy.sigma_data)
            return Process(schedule, self.prediction)
        return Process(DiscreteNoiseScheduler(self.betas, p2_loss_weight_gamma=0), self.prediction)

    def solver(self):
        policy = self.policy
        kind = policy.kind
        if kind == "DDIM":
            return DDIM()
        if kind == "PNDM":
            return PNDM(skip_prk_steps=policy.skip_prk)
        if kind == "DDPM":
            return DDPM(policy.variance)
        if kind == "LMSDiscrete":
            return LMS(order=4)
        if kind == "EulerDiscrete":
            return Euler()
        if kind == "EulerAncestralDiscrete":
            return EulerAncestral()
        if kind == "HeunDiscrete":
            return Heun()
        if kind in ("KDPM2Discrete", "KDPM2AncestralDiscrete"):
            return KDPM2(ancestral=kind == "KDPM2AncestralDiscrete")
        if kind == "DPMSolverSDE":
            return DPMSolverSDE(seed=policy.seed)
        if kind == "DPMSolverSinglestep":
            return DPMSolverSinglestep(policy.order, policy.algorithm, policy.solver_type,
                                       policy.lower_order_final)
        if kind == "DEISMultistep":
            return DEIS(policy.order, policy.lower_order_final)
        if kind == "UniPCMultistep":
            return UniPC(policy.order, policy.solver_type, policy.predict_x0,
                         policy.lower_order_final, policy.disable_corrector)
        if kind == "LCM":
            return Consistency()
        if kind == "TCD":
            # The source takes eta as a step argument, not a checkpoint field,
            # so this is the step signature's own default.
            return TCD()
        return DPMSolverMultistep(policy.order, policy.algorithm, policy.solver_type,
                                  policy.lower_order_final, policy.euler_at_final)

    def _training_sigmas(self) -> tuple[np.ndarray, np.ndarray]:
        """The training sigma table sigma/alpha and its logarithm, with the
        near-zero terminal alpha the zero-SNR classes substitute.

        The source accumulates alpha and takes this ratio in float32, and its
        model times come from a log-linear search of the result, which lands
        on an integer boundary now and then; a float64 table would truncate to
        the other side of one.
        """
        alphas = np.cumprod(1 - self.betas, dtype=np.float32)
        if self.policy.zero_snr_tail:
            alphas[-1] = np.float32(2.0 ** -24)
        base = np.sqrt((1 - alphas) / alphas, dtype=np.float32)
        return base, np.log(base, dtype=np.float32)

    def _lambda_grid(self, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
        """The paired sigma and model-time tables of a log-SNR class, with the
        terminal sigma its `final_sigmas_type` appends. Model times end as
        integers: the source stores them as int64."""
        policy = self.policy
        base, log_base = self._training_sigmas()
        last = policy.train_steps - policy.lambda_clipped
        times = _spaced_times(policy.spacing, train_steps=policy.train_steps, steps=steps,
                              offset=policy.offset, last=last, extra=1, rounded=True)
        if policy.transform == "none":
            sigmas = np.interp(times, np.arange(len(base)), base)
        else:
            low = float(base[0]) if policy.sigma_min is None else policy.sigma_min
            high = float(base[-1]) if policy.sigma_max is None else policy.sigma_max
            sigmas = _transformed_sigmas(policy.transform, low, high, steps, policy.rho)
            times = _sigma_to_time(sigmas, log_base)
            if policy.transform == "karras" and policy.karras_round:
                times = times.round()
        if policy.terminal == "zero":
            terminal = 0.0
        elif policy.grid_terminal and policy.transform != "none":
            terminal = float(sigmas[-1])
        else:
            terminal = float(base[0])
        return np.append(sigmas, terminal), np.trunc(times), 1.0

    def _sigma_grid(self, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
        """The paired tables of a variance-exploding class: its spacing
        interpolated out of the training table, then whatever sigma
        transformation it applies to that interpolated subset."""
        policy = self.policy
        base, log_base = self._training_sigmas()
        times = _spaced_times(policy.spacing, train_steps=policy.train_steps, steps=steps,
                              offset=policy.offset, last=policy.train_steps, extra=0,
                              rounded=False)
        sigmas = np.interp(times, np.arange(len(base)), base)
        if policy.transform != "none":
            low = float(sigmas[-1]) if policy.sigma_min is None else policy.sigma_min
            high = float(sigmas[0]) if policy.sigma_max is None else policy.sigma_max
            sigmas = _transformed_sigmas(policy.transform, low, high, steps, policy.rho)
            times = _sigma_to_time(sigmas, log_base)
            if policy.transform == "karras" and policy.karras_round:
                times = times.round()
        terminal = float(base[0]) if policy.terminal == "sigma_min" else 0.0
        prior = float(np.max(sigmas))
        if policy.spacing == "leading":
            prior = float(np.sqrt(prior ** 2 + 1))
        return np.append(sigmas, terminal), times, prior

    def _stage_grid(self, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
        """The same tables refined with the stage row each interval evaluates.

        KDPM2 reads the geometric mean of the interval's ends, its ancestral
        form the geometric mean of the start and k-diffusion's `sigma_down`,
        and DPMSolverSDE the midpoint in -log(sigma), which is that same
        geometric mean. Every stage model time is the source's log-linear
        inverse of the stage sigma in the training table; the interval that
        lands on zero has no stage evaluation and carries zero.
        """
        policy = self.policy
        _, log_base = self._training_sigmas()
        sigmas, times, prior = self._sigma_grid(steps)
        levels = np.asarray(sigmas, np.float64)
        following = levels[1:]
        if policy.kind == "KDPM2AncestralDiscrete":
            up = np.sqrt(following ** 2 * (levels[:-1] ** 2 - following ** 2) / levels[:-1] ** 2)
            targets = np.sqrt(np.maximum(following ** 2 - up ** 2, 0.0))
        else:
            targets = following
        with np.errstate(divide="ignore"):
            stages = np.where(targets > 0,
                              np.exp(0.5 * (np.log(levels[:-1]) + np.log(np.maximum(targets, 1e-300)))),
                              0.0)
        refined = np.empty(2 * steps + 1, np.float64)
        refined[0::2], refined[1::2] = levels, stages
        refined_times = np.empty(2 * steps + 1, np.float64)
        refined_times[0::2] = np.append(times, times[-1])
        refined_times[1::2] = np.where(stages > 0, _sigma_to_time(stages, log_base), 0.0)
        return refined, refined_times, prior

    def _edm_grid(self, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
        """EDM's own grid: rho spacing or an exponential one between sigma_min
        and sigma_max, with c_noise = log(sigma) / 4 as the model time and the
        prior of unit data at sigma_max."""
        policy = self.policy
        low = 0.002 if policy.sigma_min is None else policy.sigma_min
        high = 80.0 if policy.sigma_max is None else policy.sigma_max
        sigmas = _transformed_sigmas(policy.transform, low, high, steps, policy.rho)
        terminal = low if policy.terminal == "sigma_min" else 0.0
        times = 0.25 * np.log(sigmas)
        return (np.append(sigmas, terminal), np.append(times, times[-1]),
                float(np.sqrt(high ** 2 + 1)))

    @lru_cache(maxsize=32)
    def sampling(self, steps: int) -> tuple[Process, jax.Array]:
        """The process and the explicit descending grid a `steps` walk takes."""
        policy = self.policy
        if type(steps) is not int or steps < 1:
            raise ValueError("The sampling count must be a positive integer")
        if policy.family in ("tabulated", "lambda") and steps > policy.train_steps:
            raise ValueError("The sampling count must fit the training table")
        if policy.family == "tabulated":
            return self._tabulated(steps)
        if policy.family == "edm":
            sigmas, times, prior = self._edm_grid(steps)
            schedule: NoiseScheduler = SigmaGrid(sigmas, times, prior)
        elif policy.family == "sigma":
            sigmas, times, prior = self._sigma_grid(steps)
            schedule = SigmaGrid(sigmas, np.append(times, times[-1]), prior)
        elif policy.family == "stage":
            sigmas, times, prior = self._stage_grid(steps)
            schedule = StageSigmaGrid(sigmas, times, prior)
            return (Process(schedule, self.prediction),
                    jnp.arange(len(sigmas) - 1, -1, -2, dtype=jnp.float32))
        else:
            sigmas, times, prior = self._lambda_grid(steps)
            schedule = VPGrid(sigmas, np.append(times, times[-1]), prior)
        return (Process(schedule, self.prediction),
                jnp.arange(len(sigmas) - 1, -1, -1, dtype=jnp.float32))

    def _tabulated(self, steps: int) -> tuple[Process, jax.Array]:
        """A class that steps between integer indices of the training table."""
        policy = self.policy
        if policy.distilled:
            times = _distilled_times(policy.train_steps, policy.original_steps, steps)
            terminal = 0.0
        else:
            times = _spaced_times(policy.spacing, train_steps=policy.train_steps, steps=steps,
                                  offset=policy.offset, last=policy.train_steps, extra=0,
                                  rounded=True)
            terminal = (max(float(times[-1]) - policy.train_steps // steps, -1.0)
                        if policy.stride else -1.0)
        final = 1.0 if policy.clean_terminal else float(1 - self.betas[0])
        schedule = TabulatedVP(self.betas, final_alpha_cumprod=final,
                               stride=policy.train_steps // steps if policy.stride else None)
        return (Process(schedule, self.prediction),
                jnp.asarray(np.append(times, terminal), jnp.float32))


def _algorithm(kind: str, family: str, declared: Mapping[str, object],
               value: Callable[..., object]) -> Algorithm:
    """The class's `algorithm_type` after its own coercions: each pinned class
    rewrites a few foreign names to its own before refusing the rest."""
    if "algorithm_type" not in declared:
        return "dpmsolver++"
    algorithm = str(value("algorithm_type"))
    if kind == "DEISMultistep":
        _choice("deis" if algorithm in ("deis", "dpmsolver", "dpmsolver++") else algorithm,
                "algorithm_type", ("deis",))
        return "dpmsolver++"
    if algorithm == "deis":
        algorithm = "dpmsolver++"
    allowed: tuple[Algorithm, ...] = ("dpmsolver++", "dpmsolver", "sde-dpmsolver++",
                                      "sde-dpmsolver")
    if family == "edm":
        allowed = ("dpmsolver++", "sde-dpmsolver++")
    elif kind == "DPMSolverSinglestep":
        allowed = ("dpmsolver++", "dpmsolver", "sde-dpmsolver++")
    return _choice(algorithm, "algorithm_type", allowed)


def _solver_type(kind: str, declared: Mapping[str, object],
                 value: Callable[..., object]) -> str:
    """The class's `solver_type` after its own coercions."""
    if "solver_type" not in declared:
        return "midpoint"
    solver_type = str(value("solver_type"))
    if kind == "UniPCMultistep":
        return _choice("bh2" if solver_type in ("midpoint", "heun", "logrho") else solver_type,
                       "solver_type", ("bh1", "bh2"))
    if kind == "DEISMultistep":
        _choice("logrho" if solver_type in ("midpoint", "heun", "bh1", "bh2") else solver_type,
                "solver_type", ("logrho",))
        return "midpoint"
    return _choice("midpoint" if solver_type in ("logrho", "bh1", "bh2") else solver_type,
                   "solver_type", ("midpoint", "heun"))


def _x0_limit(kind: str, declared: Mapping[str, object], value: Callable[..., object],
              ) -> tuple[float | None, tuple[float, float] | None]:
    """The clamp or the dynamic thresholding the class's `step` applies to
    x_0, whichever it tests first. A class whose step reads neither is refused
    an active one rather than quietly limited."""
    thresholding = "thresholding" in declared and _boolean(value("thresholding"), "thresholding")
    clipping = "clip_sample" in declared and _boolean(value("clip_sample"), "clip_sample")
    if (thresholding or clipping) and kind == "TCD":
        raise ValueError("The published TCD step reads neither thresholding nor clipping")
    if thresholding:
        if kind == "UniPCMultistep" and not _boolean(value("predict_x0"), "predict_x0"):
            raise ValueError("The published epsilon-prediction UniPC step ignores thresholding")
        ratio = _number(value("dynamic_thresholding_ratio"), "dynamic_thresholding_ratio")
        maximum = _number(value("sample_max_value"), "sample_max_value")
        if not 0 < ratio <= 1 or maximum < 1:
            raise ValueError("Dynamic thresholding needs a ratio in (0, 1] and a maximum above 1")
        return None, (ratio, maximum)
    if clipping:
        clip = _number(value("clip_sample_range"), "clip_sample_range")
        if clip < 0:
            raise ValueError("clip_sample_range must be nonnegative")
        return clip, None
    return None, None


def _resolve(kind: str, source: _Class, value: Callable[..., object],
             betas: np.ndarray) -> _Policy:
    """Every control the class declares, checked and turned into a number."""
    declared, family = source.fields, source.family
    train_steps = _integer(value("num_train_timesteps"), "num_train_timesteps")
    active = [name for name, key in _TRANSFORM_CONTROLS
              if key in declared and _boolean(value(key), key)]
    if len(active) > 1:
        raise ValueError("Only one of the Karras, exponential and beta sigma grids can be used")
    transform: Transform = active[0] if active else "none"
    if family == "edm":
        transform = _choice(value("sigma_schedule"), "sigma_schedule", _SIGMA_SCHEDULES)
    spacing: Spacing = _choice(value("timestep_spacing", "linspace"), "timestep_spacing",
                               _SPACINGS)
    algorithm = _algorithm(kind, family, declared, value)
    solver_type = _solver_type(kind, declared, value)
    terminal: Terminal = "zero" if family in ("sigma", "stage") else "sigma_min"
    if "final_sigmas_type" in declared:
        terminal = _choice(value("final_sigmas_type"), "final_sigmas_type", _TERMINALS)
        if terminal == "zero" and algorithm not in ("dpmsolver++", "sde-dpmsolver++"):
            raise ValueError(f"final_sigmas_type=zero is not supported for algorithm_type "
                             f"{algorithm}, as the source scheduler refuses")
    variance: Variance = "small"
    if kind == "DDPM":
        mode = _choice(value("variance_type"), "variance_type",
                       ("fixed_small", "fixed_small_log", "fixed_large"))
        variance = "large" if mode == "fixed_large" else "small"
    elif value("variance_type") in ("learned", "learned_range"):
        raise ValueError("Native source scheduling does not implement learned variance")
    clip, threshold = _x0_limit(kind, declared, value)
    zero_snr = _boolean(value("rescale_betas_zero_snr", False), "rescale_betas_zero_snr")
    zero_snr_tail = zero_snr and kind in ("DPMSolverMultistep", "UniPCMultistep",
                                          "EulerDiscrete", "EulerAncestralDiscrete")
    lambda_clipped = 0
    limit = value("lambda_min_clipped", -float("inf"))
    if limit != -float("inf"):
        alphas = np.cumprod(1 - betas.astype(np.float64))
        if zero_snr_tail:
            alphas[-1] = 2.0 ** -24
        lambdas = 0.5 * (np.log(alphas) - np.log(1 - alphas))
        lambda_clipped = int(np.searchsorted(np.flip(lambdas),
                                             _number(limit, "lambda_min_clipped")))
    original_steps = _integer(value("original_inference_steps", train_steps),
                              "original_inference_steps")
    if not 0 < original_steps <= train_steps:
        raise ValueError("original_inference_steps must be positive and fit the training table")
    corrector = value("disable_corrector", ())
    if not isinstance(corrector, (list, tuple)) or any(type(index) is not int for index in corrector):
        raise ValueError("disable_corrector must be a sequence of step indices")
    sigma_min, sigma_max = value("sigma_min"), value("sigma_max")
    return _Policy(
        kind=kind, family=family, train_steps=train_steps,
        spacing=spacing,
        offset=_integer(value("steps_offset", 0), "steps_offset"),
        transform=transform,
        karras_round=(kind in _KARRAS_ROUNDS
                      or (kind == "DPMSolverMultistep"
                          and value("beta_schedule") != "squaredcos_cap_v2")),
        terminal=terminal,
        grid_terminal=kind in ("DEISMultistep", "UniPCMultistep"),
        zero_snr_tail=zero_snr_tail, lambda_clipped=lambda_clipped,
        sigma_min=None if sigma_min is None else _number(sigma_min, "sigma_min"),
        sigma_max=None if sigma_max is None else _number(sigma_max, "sigma_max"),
        sigma_data=_number(value("sigma_data", 0.5), "sigma_data"),
        rho=_number(value("rho", 7.0), "rho"),
        stride=kind in ("DDIM", "PNDM"),
        clean_terminal=(kind == "DDPM"
                        or _boolean(value("set_alpha_to_one", True), "set_alpha_to_one")),
        variance=variance,
        clip=clip, threshold=threshold,
        recompute_epsilon=(kind in ("DDPM", "DEISMultistep")
                           or (family == "lambda" and algorithm in ("dpmsolver", "sde-dpmsolver"))),
        order=_integer(value("solver_order", 2), "solver_order"),
        algorithm=algorithm,
        solver_type=solver_type,
        lower_order_final=(_boolean(value("lower_order_final", True), "lower_order_final")
                           or (kind == "DPMSolverSinglestep" and terminal == "zero")),
        euler_at_final=_boolean(value("euler_at_final", False), "euler_at_final"),
        predict_x0=_boolean(value("predict_x0", True), "predict_x0"),
        disable_corrector=tuple(corrector),
        skip_prk=_boolean(value("skip_prk_steps", False), "skip_prk_steps"),
        distilled=kind in ("LCM", "TCD"),
        original_steps=original_steps,
        seed=None if value("noise_sampler_seed") is None
        else _integer(value("noise_sampler_seed"), "noise_sampler_seed"),
        timestep_scaling=_number(value("timestep_scaling", 10.0), "timestep_scaling"),
    )


def _prediction_transform(policy: _Policy, prediction: str) -> PredictionTransform:
    """What the model predicts on this class's grid, with the class's own input
    scaling and its own limit on x_0."""
    if policy.family == "edm":
        inner: PredictionTransform = KarrasPredictionTransform(
            policy.sigma_data, velocity=prediction == "v_prediction")
    else:
        normalize = policy.family in ("sigma", "stage")
        parameterization = {"epsilon": EpsilonPredictionTransform,
                            "sample": DirectPredictionTransform,
                            "v_prediction": VPredictionTransform}[prediction]
        inner = parameterization(normalize_input=normalize)
    if policy.clip is not None or policy.threshold is not None:
        inner = SourceLimitedPrediction(inner, clip=policy.clip, threshold=policy.threshold,
                                        recompute_epsilon=policy.recompute_epsilon)
    if policy.kind == "LCM":
        return ConsistencyBoundary(inner, policy.timestep_scaling)
    return inner


__all__ = ["SourceSchedule", "TabulatedVP", "SigmaGrid", "StageSigmaGrid", "VPGrid",
           "published_betas"]
