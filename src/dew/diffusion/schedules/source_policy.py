"""Published scheduler defaults and the resolved controls of each native grid and solver."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

import numpy as np

from dew import records
from dew.diffusion.schedules.flow import token_mu
from dew.diffusion.schedules.source_grids import named_choice, published_betas
from dew.diffusion.transforms import (
    ConsistencyBoundary,
    DirectPredictionTransform,
    EpsilonPredictionTransform,
    FlowMatchPredictionTransform,
    KarrasPredictionTransform,
    PredictionTransform,
    SourceLimitedPrediction,
    VPredictionTransform,
)
from dew.records import JSON
from dew.sampling.solvers import (
    DDIM,
    DDPM,
    DEIS,
    KDPM2,
    LMS,
    PNDM,
    TCD,
    Algorithm,
    Consistency,
    DPMSolverMultistep,
    DPMSolverSDE,
    DPMSolverSinglestep,
    Euler,
    EulerAncestral,
    Heun,
    Solver,
    UniPC,
)

Family = Literal["tabulated", "lambda", "sigma", "stage", "edm", "flow"]
Origin = Literal["scheduler", "linspace", "empirical"]
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


class Control(Protocol):
    """Reads one class's control by name.

    The answer is the value the file states, the class's own declared default
    when the file omits it, or `absent` when this class declares no such
    control at all.
    """

    def __call__(self, key: str, absent: JSON = None) -> JSON: ...


def _fields(*groups: Mapping[str, JSON], **extra: JSON) -> Mapping[str, JSON]:
    merged: dict[str, JSON] = {}
    for group in groups:
        merged.update(group)
    merged.update(extra)
    return MappingProxyType(merged)


def _betas(start: float, end: float, schedule: str = "linear") -> Mapping[str, JSON]:
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
    """Describes one pinned scheduler class.

    The fields are its grid family, the constructor controls it declares with
    that class's own defaults, the `beta_schedule` tables it accepts, and the
    `prediction_type` values its own `step` converts.
    """

    family: Family
    fields: Mapping[str, JSON]
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
                                               lower_order_final=True, disable_corrector=[],
                                               solver_p=None, final_sigmas_type="zero",
                                               rescale_betas_zero_snr=False)),
    "EDMDPMSolverMultistep": _Class("edm", _fields(
        _THRESHOLD, num_train_timesteps=1000, prediction_type="epsilon", sigma_min=0.002,
        sigma_max=80.0, sigma_data=0.5, sigma_schedule="karras", rho=7.0, solver_order=2,
        algorithm_type="dpmsolver++", solver_type="midpoint", lower_order_final=True,
        euler_at_final=False, final_sigmas_type="zero"), (), _NO_SAMPLE),
    "FlowMatchEulerDiscrete": _Class("flow", _fields(
        _TRANSFORMS, num_train_timesteps=1000, prediction_type="flow_prediction", shift=1.0,
        use_dynamic_shifting=False,
        base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096,
        invert_sigmas=False, shift_terminal=None, time_shift_type="exponential",
        stochastic_sampling=False), (), ("flow_prediction",)),
    "LCM": _Class("tabulated", _fields(_LCM_BETAS, _DISTILLED)),
    "TCD": _Class("tabulated", _fields(_LCM_BETAS, _DISTILLED)),
})

# Controls a class declares whose active meaning this file does not
# reconstruct. Each maps to the only value that leaves the source's own
# behaviour unchanged, so an active one is refused instead of dropped.
_UNIMPLEMENTED: Mapping[str, object] = MappingProxyType({
    "use_lu_lambdas": False, "solver_p": None,
    "invert_sigmas": False, "stochastic_sampling": False,
    "interpolation_type": "linear", "timestep_type": "discrete",
})

# The log-SNR classes whose `use_flow_sigmas` walk is reconstructed: the ones
# the published flow pipelines ship (SANA's DPM-Solver++, Wan's UniPC).
_FLOW_SIGMA_CLASSES = ("DPMSolverMultistep", "UniPCMultistep")

# The classes that round a Karras grid's recovered model times before
# truncating them. The rest keep the log-linear inverse as it comes, and no
# class rounds an exponential or beta grid's.
_KARRAS_ROUNDS = ("DPMSolverSinglestep", "DEISMultistep", "UniPCMultistep",
                  "KDPM2Discrete", "KDPM2AncestralDiscrete")

@dataclass(frozen=True)
class FlowShift:
    """Holds a rectified-flow file's shift controls and the shift they name.

    `shift` alone is the static form. With `dynamic` the shift follows the
    latent's token count through the source pipeline's `calculate_shift`,
    which returns mu itself. `terminal` stretches the result to end where the
    file says.
    """

    shift: float
    dynamic: bool
    base_shift: float
    max_shift: float
    base_tokens: int
    max_tokens: int
    terminal: float | None
    kind: str

    def mu(self, tokens: int) -> float:
        """The source pipeline's `calculate_shift`.

        mu is interpolated linearly in the token count and is not
        exponentiated here.
        """
        return token_mu(tokens, self.base_tokens, self.max_tokens, self.base_shift, self.max_shift)

    def base(self, tokens: int | None, mu: float | None = None) -> float:
        """The shift this file names at `tokens` latent tokens, or at the
        `mu` a pipeline hands the scheduler itself.

        The static and the dynamic forms are the same map with a different
        base. shift s / (1 + (shift - 1) s) is base / (base + 1/s - 1) at
        base = shift, and the dynamic base is exp(mu) or mu itself.
        """
        if not self.dynamic:
            return self.shift
        if mu is None:
            if tokens is None:
                raise ValueError("Dynamic shifting needs the latent token count; bind the "
                                 "geometry through the task's grid")
            mu = self.mu(tokens)
        # math.exp, as Diffusers' `_time_shift_exponential` takes it: numpy's
        # SIMD exp rounds differently on AVX-512 machines, by an ulp here.
        return math.exp(mu) if self.kind == "exponential" else mu

@dataclass(frozen=True)
class SourcePolicy:
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
    clip: float | None
    threshold: tuple[float, float] | None
    recompute_epsilon: bool
    distilled: bool
    original_steps: int
    timestep_scaling: float
    flow: FlowShift | None
    flow_shift: float | None
    """A log-SNR class's `flow_shift` under `use_flow_sigmas`: its grid is then
    the shifted rectified-flow path rather than the beta table's."""


def resolve_config(config: Mapping[str, object]
                   ) -> tuple[np.ndarray, PredictionTransform, SourcePolicy, Solver]:
    """Resolve the published class and its declared controls once, before any grid is built."""
    name = config.get("_class_name")
    if not isinstance(name, str):
        raise ValueError("The scheduler config requires a class name")
    kind = name.removeprefix("Flax").removesuffix("Scheduler")
    source = _SOURCES.get(kind)
    if source is None:
        raise ValueError(f"Unsupported scheduler: {name}")
    declared = source.fields

    def value(key: str, absent: JSON = None) -> JSON:
        """The class's value for a control it declares, or `absent` when
        this class declares no such control."""
        if key not in declared:
            return absent
        return records.json_value(config[key], key) if key in config else declared[key]

    for key, inactive in _UNIMPLEMENTED.items():
        if key in declared and value(key) != inactive:
            raise ValueError(f"Native source scheduling does not implement active {key}")
    # A flow file declares no prediction type: its class fixes the
    # convention, so the family names the transform. A log-SNR class on
    # flow sigmas reads the velocity its flow checkpoint predicts.
    flow_sigmas = records.boolean(value("use_flow_sigmas", absent=False), "use_flow_sigmas")
    if flow_sigmas and kind not in _FLOW_SIGMA_CLASSES:
        raise ValueError(f"Native source scheduling reconstructs use_flow_sigmas for "
                         f"{' and '.join(_FLOW_SIGMA_CLASSES)}, not {kind}")
    prediction = ("flow_prediction" if source.family == "flow"
                  else named_choice(value("prediction_type"), "prediction_type",
                               ("flow_prediction",) if flow_sigmas else source.predictions))
    if kind == "PNDM" and prediction == "v_prediction":
        raise ValueError("Published PNDM v-prediction requires velocity-domain history; "
                         "native PNDM uses epsilon history")
    betas = (np.zeros((0,), np.float32) if source.family in ("edm", "flow")
             else published_betas(
        count=value("num_train_timesteps"), start=value("beta_start"),
        end=value("beta_end"), schedule=value("beta_schedule"),
        trained=value("trained_betas"),
        zero_snr=records.boolean(value("rescale_betas_zero_snr", absent=False), "rescale_betas_zero_snr"),
        schedules=source.schedules))
    betas.setflags(write=False)
    policy, solver = _resolve(kind, source, value, betas)
    return betas, _prediction_transform(policy, prediction), policy, solver


def _algorithm(kind: str, family: str, declared: Mapping[str, JSON],
               value: Control) -> Algorithm:
    """The algorithm this class integrates with, after its own coercions.

    Each pinned class rewrites a few foreign names to its own before refusing
    the rest, and the EDM class refuses the two non-++ ones outright.
    """
    if "algorithm_type" not in declared:
        return "dpmsolver++"
    algorithm = str(value("algorithm_type"))
    if kind == "DEISMultistep":
        named_choice("deis" if algorithm in ("deis", "dpmsolver", "dpmsolver++") else algorithm,
                "algorithm_type", ("deis",))
        return "dpmsolver++"
    if algorithm == "deis":
        algorithm = "dpmsolver++"
    allowed: tuple[Algorithm, ...] = ("dpmsolver++", "dpmsolver", "sde-dpmsolver++",
                                      "sde-dpmsolver")
    if family == "edm":
        allowed = ("dpmsolver++", "sde-dpmsolver++")
    return named_choice(algorithm, "algorithm_type", allowed)


def _solver_type[SolverT: str](kind: str, value: Control,
                               allowed: tuple[SolverT, ...]) -> SolverT:
    """The class's `solver_type` after its own coercions, in the names the
    class this is being built for takes."""
    solver_type = str(value("solver_type"))
    if kind == "UniPCMultistep":
        return named_choice("bh2" if solver_type in ("midpoint", "heun", "logrho") else solver_type,
                       "solver_type", allowed)
    if kind == "DEISMultistep":
        return named_choice("logrho" if solver_type in ("midpoint", "heun", "bh1", "bh2")
                       else solver_type, "solver_type", allowed)
    return named_choice("midpoint" if solver_type in ("logrho", "bh1", "bh2") else solver_type,
                   "solver_type", allowed)


def _x0_limit(kind: str, declared: Mapping[str, JSON], value: Control,
              ) -> tuple[float | None, tuple[float, float] | None]:
    """The clamp or the dynamic thresholding the class's `step` applies to
    x_0, whichever it tests first. A class whose step reads neither is refused
    an active one rather than quietly limited."""
    thresholding = "thresholding" in declared and records.boolean(value("thresholding"), "thresholding")
    clipping = "clip_sample" in declared and records.boolean(value("clip_sample"), "clip_sample")
    if (thresholding or clipping) and kind == "TCD":
        raise ValueError("The published TCD step reads neither thresholding nor clipping")
    if thresholding:
        if kind == "UniPCMultistep" and not records.boolean(value("predict_x0"), "predict_x0"):
            raise ValueError("The published epsilon-prediction UniPC step ignores thresholding")
        ratio = records.number(value("dynamic_thresholding_ratio"), "dynamic_thresholding_ratio")
        maximum = records.number(value("sample_max_value"), "sample_max_value")
        if not 0 < ratio <= 1 or maximum < 1:
            raise ValueError("Dynamic thresholding needs a ratio in (0, 1] and a maximum above 1")
        return None, (ratio, maximum)
    if clipping:
        clip = records.number(value("clip_sample_range"), "clip_sample_range")
        if clip < 0:
            raise ValueError("clip_sample_range must be nonnegative")
        return clip, None
    return None, None


def _flow_controls(value: Control) -> FlowShift:
    """A flow file's shift controls, checked and turned into numbers."""
    terminal = value("shift_terminal")
    return FlowShift(
        shift=records.number(value("shift", 1.0), "shift"),
        dynamic=records.boolean(value("use_dynamic_shifting", absent=False), "use_dynamic_shifting"),
        base_shift=records.number(value("base_shift", 0.5), "base_shift"),
        max_shift=records.number(value("max_shift", 1.15), "max_shift"),
        base_tokens=records.integer(value("base_image_seq_len", 256), "base_image_seq_len"),
        max_tokens=records.integer(value("max_image_seq_len", 4096), "max_image_seq_len"),
        terminal=None if terminal is None else records.number(terminal, "shift_terminal"),
        kind=named_choice(value("time_shift_type", "exponential"), "time_shift_type",
                     ("exponential", "linear")))


def _sigma_transform(family: str, declared: Mapping[str, JSON], value: Control) -> Transform:
    """Which sigma grid the class applies over its spacing, or "none".

    The three boolean controls are mutually exclusive, and an EDM file names
    its grid outright instead.
    """
    active: list[Transform] = [name for name, key in _TRANSFORM_CONTROLS
                               if key in declared and records.boolean(value(key), key)]
    if len(active) > 1:
        raise ValueError("Only one of the Karras, exponential and beta sigma grids can be used")
    if family == "edm":
        return named_choice(value("sigma_schedule"), "sigma_schedule", _SIGMA_SCHEDULES)
    return active[0] if active else "none"


def _final_sigma(family: str, declared: Mapping[str, JSON], value: Control,
                 algorithm: Algorithm) -> Terminal:
    """The sigma the class appends past its last grid point.

    A variance-exploding grid ends at zero; a log-SNR one ends where its own
    `final_sigmas_type` says, and only the clean-prediction algorithms can
    end at zero, as the source scheduler refuses the rest.
    """
    terminal: Terminal = "zero" if family in ("sigma", "stage") else "sigma_min"
    if "final_sigmas_type" not in declared:
        return terminal
    terminal = named_choice(value("final_sigmas_type"), "final_sigmas_type", _TERMINALS)
    if terminal == "zero" and algorithm not in ("dpmsolver++", "sde-dpmsolver++"):
        raise ValueError(f"final_sigmas_type=zero is not supported for algorithm_type "
                         f"{algorithm}, as the source scheduler refuses")
    return terminal


def _variance_type(kind: str, value: Control) -> Variance:
    """Which posterior variance a DDPM file's `step` adds.

    Only DDPM has the control. A learned variance is a second model output
    and is refused rather than approximated.
    """
    if kind == "DDPM":
        mode = named_choice(value("variance_type"), "variance_type",
                       ("fixed_small", "fixed_small_log", "fixed_large"))
        return "large" if mode == "fixed_large" else "small"
    if value("variance_type") in ("learned", "learned_range"):
        raise ValueError("Native source scheduling does not implement learned variance")
    return "small"


def _lambda_clipping(kind: str, value: Control, betas: np.ndarray) -> tuple[bool, int]:
    """`(zero-SNR tail, clipped steps)`: how the class ends its training table.

    The zero-SNR classes substitute a near-zero terminal alpha, and
    `lambda_min_clipped` drops the steps whose half log-SNR falls below it,
    counted off the training betas the way the source counts them.
    """
    zero_snr = records.boolean(value("rescale_betas_zero_snr", absent=False), "rescale_betas_zero_snr")
    zero_snr_tail = zero_snr and kind in ("DPMSolverMultistep", "UniPCMultistep",
                                          "EulerDiscrete", "EulerAncestralDiscrete")
    limit = value("lambda_min_clipped", -float("inf"))
    if limit == -float("inf"):
        return zero_snr_tail, 0
    alphas = np.cumprod(1 - betas.astype(np.float64))
    if zero_snr_tail:
        alphas[-1] = 2.0 ** -24
    lambdas = 0.5 * (np.log(alphas) - np.log(1 - alphas))
    return zero_snr_tail, int(np.searchsorted(np.flip(lambdas),
                                              records.number(limit, "lambda_min_clipped")))


def _flow_shift(kind: str, value: Control, transform: Transform,
                algorithm: Algorithm) -> float | None:
    """`flow_shift` under an active `use_flow_sigmas`, and None without it.

    The source reaches its flow branch only when no sigma transformation is
    on, and converts velocity only to a clean prediction, so a flow file
    with either is refused rather than walked on sigmas it was not trained on.
    """
    if not records.boolean(value("use_flow_sigmas", absent=False), "use_flow_sigmas"):
        return None
    if transform != "none":
        raise ValueError("use_flow_sigmas under a Karras, exponential or beta grid walks the "
                         "beta table's sigmas on the flow path; the source refuses neither")
    clean = (algorithm in ("dpmsolver++", "sde-dpmsolver++") if kind == "DPMSolverMultistep"
             else records.boolean(value("predict_x0"), "predict_x0"))
    if not clean:
        raise ValueError("The source converts flow velocity to a clean prediction only; "
                         "use_flow_sigmas needs a dpmsolver++ algorithm or predict_x0")
    shift = records.number(value("flow_shift"), "flow_shift")
    if not shift > 0:
        raise ValueError("flow_shift must be positive")
    return shift


def _resolve(kind: str, source: _Class, value: Control,
             betas: np.ndarray) -> tuple[SourcePolicy, Solver]:
    """Every control the class declares, checked and turned into a number."""
    declared, family = source.fields, source.family
    train_steps = records.integer(value("num_train_timesteps"), "num_train_timesteps")
    transform = _sigma_transform(family, declared, value)
    algorithm = _algorithm(kind, family, declared, value)
    spacing: Spacing = named_choice(value("timestep_spacing", "linspace"), "timestep_spacing",
                               _SPACINGS)
    terminal = _final_sigma(family, declared, value, algorithm)
    variance = _variance_type(kind, value)
    clip, threshold = _x0_limit(kind, declared, value)
    zero_snr_tail, lambda_clipped = _lambda_clipping(kind, value, betas)
    original_steps = records.integer(value("original_inference_steps", train_steps),
                              "original_inference_steps")
    if not 0 < original_steps <= train_steps:
        raise ValueError("original_inference_steps must be positive and fit the training table")
    corrector = value("disable_corrector", [])
    if not isinstance(corrector, (list, tuple)):
        raise ValueError("disable_corrector must be a sequence of step indices")
    disabled = tuple(records.integer(index, "disable_corrector") for index in corrector)
    sigma_min, sigma_max = value("sigma_min"), value("sigma_max")
    order = records.integer(value("solver_order", 2), "solver_order")
    flow = _flow_controls(value) if family == "flow" else None
    flow_shift = _flow_shift(kind, value, transform, algorithm)
    policy = SourcePolicy(
        kind=kind, family=family, train_steps=train_steps,
        spacing=spacing,
        offset=records.integer(value("steps_offset", 0), "steps_offset"),
        transform=transform,
        karras_round=(kind in _KARRAS_ROUNDS
                      or (kind == "DPMSolverMultistep"
                          and value("beta_schedule") != "squaredcos_cap_v2")),
        terminal=terminal,
        grid_terminal=kind in ("DEISMultistep", "UniPCMultistep"),
        zero_snr_tail=zero_snr_tail, lambda_clipped=lambda_clipped,
        sigma_min=None if sigma_min is None else records.number(sigma_min, "sigma_min"),
        sigma_max=None if sigma_max is None else records.number(sigma_max, "sigma_max"),
        sigma_data=records.number(value("sigma_data", 0.5), "sigma_data"),
        rho=records.number(value("rho", 7.0), "rho"),
        stride=kind in ("DDIM", "PNDM"),
        clean_terminal=(kind == "DDPM"
                        or records.boolean(value("set_alpha_to_one", absent=True), "set_alpha_to_one")),
        clip=clip, threshold=threshold,
        recompute_epsilon=(kind in ("DDPM", "DEISMultistep")
                           or (family == "lambda" and algorithm in ("dpmsolver", "sde-dpmsolver"))),
        distilled=kind in ("LCM", "TCD"),
        original_steps=original_steps,
        timestep_scaling=records.number(value("timestep_scaling", 10.0), "timestep_scaling"),
        flow=flow, flow_shift=flow_shift)
    return policy, _build_solver(kind, value, order, algorithm, terminal,
                                 variance, disabled)


def _build_solver(kind: str, value: Control, order: int, algorithm: Algorithm,
                  terminal: Terminal, variance: Variance,
                  corrector: tuple[int, ...]) -> Solver:
    """The native solver this class and its controls name, built once.

    A solver is a frozen value, so the file's class and controls resolve into
    one here and `SourceSchedule.solver` holds that same value. Nothing
    downstream re-reads a control to rebuild it.
    """
    if kind == "DDIM":
        return DDIM()
    if kind == "PNDM":
        return PNDM(skip_prk_steps=records.boolean(value("skip_prk_steps"), "skip_prk_steps"))
    if kind == "DDPM":
        return DDPM(variance)
    if kind == "LMSDiscrete":
        return LMS(order=4)
    if kind in ("EulerDiscrete", "FlowMatchEulerDiscrete"):
        return Euler()
    if kind == "EulerAncestralDiscrete":
        return EulerAncestral()
    if kind == "HeunDiscrete":
        return Heun()
    if kind in ("KDPM2Discrete", "KDPM2AncestralDiscrete"):
        return KDPM2(ancestral=kind == "KDPM2AncestralDiscrete")
    if kind == "DPMSolverSDE":
        seed = value("noise_sampler_seed")
        return DPMSolverSDE(seed=None if seed is None
                            else records.integer(seed, "noise_sampler_seed"))
    if kind == "DEISMultistep":
        # The class rewrites the two DPM names to its own and refuses the
        # rest; its native integrator carries neither control.
        _solver_type(kind, value, ("logrho",))
        return DEIS(order, records.boolean(value("lower_order_final"), "lower_order_final"))
    if kind == "LCM":
        return Consistency()
    if kind == "TCD":
        # The source takes eta as a step argument, not a checkpoint field, so
        # this is the step signature's own default.
        return TCD()
    lower_order_final = records.boolean(value("lower_order_final"), "lower_order_final")
    if kind == "UniPCMultistep":
        return UniPC(order, _solver_type(kind, value, ("bh1", "bh2")),
                     records.boolean(value("predict_x0"), "predict_x0"), lower_order_final, corrector)
    if kind == "DPMSolverSinglestep":
        # `set_timesteps` rewrites this control for a zero terminal, so the
        # reconstruction reads the value the source would walk with.
        return DPMSolverSinglestep(
            # The class's own `step` has no noise term for the SDE algorithm.
            order, named_choice(algorithm, "algorithm_type",
                           ("dpmsolver++", "dpmsolver", "sde-dpmsolver++")),
            _solver_type(kind, value, ("midpoint", "heun")),
            lower_order_final or terminal == "zero")
    return DPMSolverMultistep(order, algorithm,
                              _solver_type(kind, value, ("midpoint", "heun")),
                              lower_order_final,
                              records.boolean(value("euler_at_final"), "euler_at_final"))


def _prediction_transform(policy: SourcePolicy, prediction: str) -> PredictionTransform:
    """What the model predicts on this class's grid, with the class's own input
    scaling and its own limit on x_0."""
    if policy.family == "flow":
        return FlowMatchPredictionTransform()
    if policy.family == "edm":
        inner: PredictionTransform = KarrasPredictionTransform(
            policy.sigma_data, velocity=prediction == "v_prediction")
    elif policy.flow_shift is not None:
        inner = FlowMatchPredictionTransform()
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
