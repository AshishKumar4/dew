"""Published scheduler defaults and the resolved controls of each native grid and solver."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, NamedTuple, Protocol, get_args

import numpy as np

from dew import _records
from dew._records import JSON
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

_SPACINGS: tuple[Spacing, ...] = get_args(Spacing)
_TERMINALS: tuple[Terminal, ...] = get_args(Terminal)
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
_DPM_ALGORITHMS: tuple[Algorithm, ...] = get_args(Algorithm)
_SOLVER_TYPES = ("midpoint", "heun", "logrho", "bh1", "bh2")


class _Controls(NamedTuple):
    """The resolved controls a class's native solver is built from."""

    value: Control
    order: int
    algorithm: Algorithm
    terminal: Terminal
    corrector: tuple[int, ...]

    def flag(self, key: str) -> bool:
        return _records.boolean(self.value(key), key)


def _never_rounds(value: Control) -> bool:
    return False


def _always_rounds(value: Control) -> bool:
    return True


def _rounds_off_cosine(value: Control) -> bool:
    return value("beta_schedule") != "squaredcos_cap_v2"


@dataclass(frozen=True)
class _Class:
    """Describes one pinned scheduler class.

    The fields are its grid family, the constructor controls it declares with
    that class's own defaults, the native solver its controls resolve into
    (built once, so nothing downstream re-reads a control), the
    `beta_schedule` tables it accepts, the `prediction_type` values its
    own `step` converts, and the `algorithm_type` values it keeps after
    rewriting `deis` to `dpmsolver++` (none: it has no DPM-Solver algorithm).

    The rest is what this class's `set_timesteps` and `step` do that its
    family does not settle. `karras_round`: it rounds a Karras grid's
    recovered model times before truncating them (DPMSolverMultistep except
    on the cosine table); the others keep the log-linear inverse as it comes,
    and no class rounds an exponential or beta grid's. `flow_sigmas`: its
    `use_flow_sigmas` walk is reconstructed, for the classes the published
    flow pipelines ship (SANA's DPM-Solver++, Wan's UniPC). `grid_terminal`:
    a transformed grid ends on its own last sigma. `stride`: it steps
    `train_steps // steps` back. `recompute_epsilon`: it recomputes epsilon
    from a limited x_0 whatever its algorithm. `boundary`: it applies the
    consistency model's boundary condition. `limits`: its `step` reads the
    thresholding and clipping it declares.
    """

    family: Family
    fields: Mapping[str, JSON]
    solver: Callable[[_Controls], Solver]
    schedules: tuple[str, ...] = ("linear", "scaled_linear", "squaredcos_cap_v2")
    predictions: tuple[str, ...] = _ALL_PREDICTIONS
    algorithms: tuple[Algorithm, ...] = ()
    karras_round: Callable[[Control], bool] = _never_rounds
    flow_sigmas: bool = False
    grid_terminal: bool = False
    stride: bool = False
    recompute_epsilon: bool = False
    boundary: bool = False
    limits: bool = True


_SOURCES: Mapping[str, _Class] = MappingProxyType({
    "DDIM": _Class("tabulated", _fields(_VP_BETAS, _THRESHOLD, _LEADING, clip_sample=True,
                                        clip_sample_range=1.0, set_alpha_to_one=True,
                                        rescale_betas_zero_snr=False), lambda c: DDIM(), stride=True),
    # The published PNDM keeps v-prediction history in the velocity domain;
    # the native one keeps epsilon history.
    "PNDM": _Class("tabulated", _fields(_VP_BETAS, _LEADING, skip_prk_steps=False,
                                        set_alpha_to_one=False),
                   lambda c: PNDM(skip_prk_steps=c.flag("skip_prk_steps")), predictions=("epsilon",),
                   stride=True),
    "DDPM": _Class("tabulated", _fields(_VP_BETAS, _THRESHOLD, _LEADING,
                                        variance_type="fixed_small", clip_sample=True,
                                        clip_sample_range=1.0, rescale_betas_zero_snr=False),
                   lambda c: _ddpm(c), ("linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"),
                   recompute_epsilon=True),
    "LMSDiscrete": _Class("sigma", _fields(_VP_BETAS, _SPACED, _TRANSFORMS), lambda c: LMS(order=4)),
    "EulerDiscrete": _Class("sigma", _fields(_VP_BETAS, _SPACED, _TRANSFORMS,
                                             interpolation_type="linear", sigma_min=None,
                                             sigma_max=None, timestep_type="discrete",
                                             rescale_betas_zero_snr=False,
                                             final_sigmas_type="zero"), lambda c: Euler()),
    "EulerAncestralDiscrete": _Class("sigma", _fields(_VP_BETAS, _SPACED,
                                                      rescale_betas_zero_snr=False),
                                     lambda c: EulerAncestral(), predictions=_NO_SAMPLE),
    "HeunDiscrete": _Class("sigma", _fields(_KD_BETAS, _SPACED, _TRANSFORMS, clip_sample=False,
                                            clip_sample_range=1.0),
                           lambda c: Heun(), ("linear", "scaled_linear", "squaredcos_cap_v2", "exp")),
    "KDPM2Discrete": _Class("stage", _fields(_KD_BETAS, _SPACED, _TRANSFORMS),
                            lambda c: KDPM2(ancestral=False), predictions=_NO_SAMPLE,
                            karras_round=_always_rounds),
    "KDPM2AncestralDiscrete": _Class("stage", _fields(_KD_BETAS, _SPACED, _TRANSFORMS),
                                     lambda c: KDPM2(ancestral=True), predictions=_NO_SAMPLE,
                                     karras_round=_always_rounds),
    "DPMSolverSDE": _Class("stage", _fields(_KD_BETAS, _SPACED, _TRANSFORMS,
                                            noise_sampler_seed=None),
                           lambda c: DPMSolverSDE(seed=_optional_seed(c.value)), predictions=_NO_SAMPLE),
    "DPMSolverMultistep": _Class("lambda", _fields(_VP_BETAS, _THRESHOLD, _SPACED, _TRANSFORMS,
                                                   _FLOW, _DPM, lower_order_final=True,
                                                   euler_at_final=False, use_lu_lambdas=False,
                                                   rescale_betas_zero_snr=False),
                                 lambda c: _dpm_multistep(c), algorithms=_DPM_ALGORITHMS,
                                 karras_round=_rounds_off_cosine, flow_sigmas=True),
    "DPMSolverSinglestep": _Class("lambda", _fields(_VP_BETAS, _THRESHOLD, _TRANSFORMS, _FLOW,
                                                    _DPM, lower_order_final=False),
                                  # `set_timesteps` rewrites lower_order_final for a zero
                                  # terminal, and the class's own `step` has no noise term
                                  # for the SDE algorithm.
                                  lambda c: DPMSolverSinglestep(
                                      c.order, named_choice(c.algorithm, "algorithm_type",
                                                            ("dpmsolver++", "dpmsolver", "sde-dpmsolver++")),
                                      _solver_type(c.value, ("midpoint", "heun")),
                                      c.flag("lower_order_final") or c.terminal == "zero"),
                                  algorithms=_DPM_ALGORITHMS, karras_round=_always_rounds),
    "DEISMultistep": _Class("lambda", _fields(_VP_BETAS, _THRESHOLD, _SPACED, _TRANSFORMS, _FLOW,
                                              solver_order=2, algorithm_type="deis",
                                              solver_type="logrho", lower_order_final=True),
                            lambda c: _deis(c), karras_round=_always_rounds, grid_terminal=True,
                            recompute_epsilon=True),
    "UniPCMultistep": _Class("lambda", _fields(_VP_BETAS, _THRESHOLD, _SPACED, _TRANSFORMS, _FLOW,
                                               solver_order=2, predict_x0=True, solver_type="bh2",
                                               lower_order_final=True, disable_corrector=[],
                                               solver_p=None, final_sigmas_type="zero",
                                               rescale_betas_zero_snr=False),
                             lambda c: UniPC(c.order, _solver_type(c.value, ("bh2", "bh1")),
                                             c.flag("predict_x0"), c.flag("lower_order_final"), c.corrector),
                             karras_round=_always_rounds, flow_sigmas=True, grid_terminal=True),
    "EDMDPMSolverMultistep": _Class("edm", _fields(
        _THRESHOLD, num_train_timesteps=1000, prediction_type="epsilon", sigma_min=0.002,
        sigma_max=80.0, sigma_data=0.5, sigma_schedule="karras", rho=7.0, solver_order=2,
        algorithm_type="dpmsolver++", solver_type="midpoint", lower_order_final=True,
        euler_at_final=False, final_sigmas_type="zero"),
        lambda c: _dpm_multistep(c),
        (), _NO_SAMPLE, algorithms=("dpmsolver++", "sde-dpmsolver++")),
    "FlowMatchEulerDiscrete": _Class("flow", _fields(
        _TRANSFORMS, num_train_timesteps=1000, prediction_type="flow_prediction", shift=1.0,
        use_dynamic_shifting=False,
        base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096,
        invert_sigmas=False, shift_terminal=None, time_shift_type="exponential",
        stochastic_sampling=False), lambda c: Euler(), (), ("flow_prediction",)),
    "LCM": _Class("tabulated", _fields(_LCM_BETAS, _DISTILLED), lambda c: Consistency(), boundary=True),
    # The source takes TCD's eta as a step argument, not a checkpoint field, so
    # the solver keeps the step signature's own default.
    "TCD": _Class("tabulated", _fields(_LCM_BETAS, _DISTILLED), lambda c: TCD(), limits=False),
})

# Controls a class declares whose active meaning this file does not
# reconstruct. Each maps to the only value that leaves the source's own
# behaviour unchanged, so an active one is refused instead of dropped.
_UNIMPLEMENTED: Mapping[str, object] = MappingProxyType({
    "use_lu_lambdas": False, "solver_p": None,
    "invert_sigmas": False, "stochastic_sampling": False,
    "interpolation_type": "linear", "timestep_type": "discrete",
})

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
        return _records.json_value(config[key], key) if key in config else declared[key]

    for key, inactive in _UNIMPLEMENTED.items():
        if key in declared and value(key) != inactive:
            raise ValueError(f"Native source scheduling does not implement active {key}")
    # A flow file declares no prediction type: its class fixes the
    # convention, so the family names the transform. A log-SNR class on
    # flow sigmas reads the velocity its flow checkpoint predicts.
    flow_sigmas = _records.boolean(value("use_flow_sigmas", absent=False), "use_flow_sigmas")
    if flow_sigmas and not source.flow_sigmas:
        walked = " and ".join(name for name, other in _SOURCES.items() if other.flow_sigmas)
        raise ValueError(f"Native source scheduling reconstructs use_flow_sigmas for {walked}, not {kind}")
    prediction = ("flow_prediction" if source.family == "flow"
                  else named_choice(value("prediction_type"), "prediction_type",
                               ("flow_prediction",) if flow_sigmas else source.predictions))
    betas = (np.zeros((0,), np.float32) if source.family in ("edm", "flow")
             else published_betas(
        count=value("num_train_timesteps"), start=value("beta_start"),
        end=value("beta_end"), schedule=value("beta_schedule"),
        trained=value("trained_betas"),
        zero_snr=_records.boolean(value("rescale_betas_zero_snr", absent=False), "rescale_betas_zero_snr"),
        schedules=source.schedules))
    betas.setflags(write=False)
    policy, solver = _resolve(kind, source, value, betas)
    return betas, _prediction_transform(policy, prediction, source), policy, solver


def _algorithm(source: _Class, value: Control) -> Algorithm:
    """The DPM-Solver algorithm this class integrates with: the one its file
    names, `deis` read as `dpmsolver++` as each such constructor rewrites it,
    and `dpmsolver++` for a class without one. DEIS reads its own (`_deis`)."""
    if not source.algorithms:
        return "dpmsolver++"
    algorithm = value("algorithm_type")
    return named_choice("dpmsolver++" if algorithm == "deis" else algorithm, "algorithm_type",
                        source.algorithms)


def _solver_type[SolverT: str](value: Control, allowed: tuple[SolverT, ...]) -> SolverT:
    """The class's `solver_type` in the names it takes, `allowed`: each
    constructor rewrites another class's name to its own default, the first
    of `allowed`, and refuses the rest."""
    solver_type = value("solver_type")
    return named_choice(allowed[0] if solver_type in _SOLVER_TYPES and solver_type not in allowed
                        else solver_type, "solver_type", allowed)


def _x0_limit(kind: str, source: _Class, value: Control,
              ) -> tuple[float | None, tuple[float, float] | None]:
    """The clamp or the dynamic thresholding the class's `step` applies to
    x_0, whichever it tests first. A class whose step reads neither is refused
    an active one rather than quietly limited, and so is thresholding where
    the step predicts noise (`predict_x0` off), which its noise branch skips."""
    declared = source.fields
    thresholding = "thresholding" in declared and _records.boolean(value("thresholding"), "thresholding")
    clipping = "clip_sample" in declared and _records.boolean(value("clip_sample"), "clip_sample")
    if (thresholding or clipping) and not source.limits:
        raise ValueError(f"The published {kind} step reads neither thresholding nor clipping")
    if thresholding:
        if not _records.boolean(value("predict_x0", absent=True), "predict_x0"):
            raise ValueError(f"The published epsilon-prediction {kind} step ignores thresholding")
        ratio = _records.number(value("dynamic_thresholding_ratio"), "dynamic_thresholding_ratio")
        maximum = _records.number(value("sample_max_value"), "sample_max_value")
        if not 0 < ratio <= 1 or maximum < 1:
            raise ValueError("Dynamic thresholding needs a ratio in (0, 1] and a maximum above 1")
        return None, (ratio, maximum)
    if clipping:
        clip = _records.number(value("clip_sample_range"), "clip_sample_range")
        if clip < 0:
            raise ValueError("clip_sample_range must be nonnegative")
        return clip, None
    return None, None


def _flow_controls(value: Control) -> FlowShift:
    """A flow file's shift controls, checked and turned into numbers."""
    terminal = value("shift_terminal")
    return FlowShift(
        shift=_records.number(value("shift", 1.0), "shift"),
        dynamic=_records.boolean(value("use_dynamic_shifting", absent=False), "use_dynamic_shifting"),
        base_shift=_records.number(value("base_shift", 0.5), "base_shift"),
        max_shift=_records.number(value("max_shift", 1.15), "max_shift"),
        base_tokens=_records.integer(value("base_image_seq_len", 256), "base_image_seq_len"),
        max_tokens=_records.integer(value("max_image_seq_len", 4096), "max_image_seq_len"),
        terminal=None if terminal is None else _records.number(terminal, "shift_terminal"),
        kind=named_choice(value("time_shift_type", "exponential"), "time_shift_type",
                     ("exponential", "linear")))


def _sigma_transform(family: str, declared: Mapping[str, JSON], value: Control) -> Transform:
    """Which sigma grid the class applies over its spacing, or "none".

    The three boolean controls are mutually exclusive, and an EDM file names
    its grid outright instead.
    """
    active: list[Transform] = [name for name, key in _TRANSFORM_CONTROLS
                               if key in declared and _records.boolean(value(key), key)]
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


def _lambda_clipping(family: Family, value: Control, betas: np.ndarray) -> tuple[bool, int]:
    """`(zero-SNR tail, clipped steps)`: how the class ends its training table.

    The zero-SNR classes that walk sigmas substitute a near-zero terminal
    alpha for the zero one, whose sigma is infinite; a tabulated class walks
    the zero alpha itself. `lambda_min_clipped` drops the steps whose half
    log-SNR falls below it, counted off the training betas the way the
    source counts them.
    """
    zero_snr = _records.boolean(value("rescale_betas_zero_snr", absent=False), "rescale_betas_zero_snr")
    zero_snr_tail = zero_snr and family != "tabulated"
    limit = value("lambda_min_clipped", -float("inf"))
    if limit == -float("inf"):
        return zero_snr_tail, 0
    alphas = np.cumprod(1 - betas.astype(np.float64))
    if zero_snr_tail:
        alphas[-1] = 2.0 ** -24
    lambdas = 0.5 * (np.log(alphas) - np.log(1 - alphas))
    return zero_snr_tail, int(np.searchsorted(np.flip(lambdas),
                                              _records.number(limit, "lambda_min_clipped")))


def _flow_shift(value: Control, transform: Transform, algorithm: Algorithm) -> float | None:
    """`flow_shift` under an active `use_flow_sigmas`, and None without it.

    The source reaches its flow branch only when no sigma transformation is
    on, and converts velocity only to a clean prediction, so a flow file
    with either is refused rather than walked on sigmas it was not trained on.
    """
    if not _records.boolean(value("use_flow_sigmas", absent=False), "use_flow_sigmas"):
        return None
    if transform != "none":
        raise ValueError("use_flow_sigmas under a Karras, exponential or beta grid walks the "
                         "beta table's sigmas on the flow path; the source refuses neither")
    clean = (algorithm in ("dpmsolver++", "sde-dpmsolver++")
             and _records.boolean(value("predict_x0", absent=True), "predict_x0"))
    if not clean:
        raise ValueError("The source converts flow velocity to a clean prediction only; "
                         "use_flow_sigmas needs a dpmsolver++ algorithm or predict_x0")
    shift = _records.number(value("flow_shift"), "flow_shift")
    if not shift > 0:
        raise ValueError("flow_shift must be positive")
    return shift


def _resolve(kind: str, source: _Class, value: Control,
             betas: np.ndarray) -> tuple[SourcePolicy, Solver]:
    """Every control the class declares, checked and turned into a number."""
    declared, family = source.fields, source.family
    train_steps = _records.integer(value("num_train_timesteps"), "num_train_timesteps")
    transform = _sigma_transform(family, declared, value)
    algorithm = _algorithm(source, value)
    spacing: Spacing = named_choice(value("timestep_spacing", "linspace"), "timestep_spacing",
                               _SPACINGS)
    terminal = _final_sigma(family, declared, value, algorithm)
    # A learned variance is a second model output, refused rather than approximated.
    if value("variance_type") in ("learned", "learned_range"):
        raise ValueError("Native source scheduling does not implement learned variance")
    clip, threshold = _x0_limit(kind, source, value)
    zero_snr_tail, lambda_clipped = _lambda_clipping(family, value, betas)
    original_steps = _records.integer(value("original_inference_steps", train_steps),
                              "original_inference_steps")
    if not 0 < original_steps <= train_steps:
        raise ValueError("original_inference_steps must be positive and fit the training table")
    corrector = value("disable_corrector", [])
    if not isinstance(corrector, (list, tuple)):
        raise ValueError("disable_corrector must be a sequence of step indices")
    disabled = tuple(_records.integer(index, "disable_corrector") for index in corrector)
    sigma_min, sigma_max = value("sigma_min"), value("sigma_max")
    order = _records.integer(value("solver_order", 2), "solver_order")
    flow = _flow_controls(value) if family == "flow" else None
    flow_shift = _flow_shift(value, transform, algorithm)
    policy = SourcePolicy(
        kind=kind, family=family, train_steps=train_steps,
        spacing=spacing,
        offset=_records.integer(value("steps_offset", 0), "steps_offset"),
        transform=transform,
        karras_round=source.karras_round(value),
        terminal=terminal,
        grid_terminal=source.grid_terminal,
        zero_snr_tail=zero_snr_tail, lambda_clipped=lambda_clipped,
        sigma_min=None if sigma_min is None else _records.number(sigma_min, "sigma_min"),
        sigma_max=None if sigma_max is None else _records.number(sigma_max, "sigma_max"),
        sigma_data=_records.number(value("sigma_data", 0.5), "sigma_data"),
        rho=_records.number(value("rho", 7.0), "rho"),
        stride=source.stride,
        clean_terminal=_records.boolean(value("set_alpha_to_one", absent=True), "set_alpha_to_one"),
        clip=clip, threshold=threshold,
        recompute_epsilon=source.recompute_epsilon or algorithm in ("dpmsolver", "sde-dpmsolver"),
        distilled="original_inference_steps" in declared,
        original_steps=original_steps,
        timestep_scaling=_records.number(value("timestep_scaling", 10.0), "timestep_scaling"),
        flow=flow, flow_shift=flow_shift)
    return policy, source.solver(_Controls(value, order, algorithm, terminal, disabled))


def _dpm_multistep(controls: _Controls) -> Solver:
    return DPMSolverMultistep(controls.order, controls.algorithm,
                              _solver_type(controls.value, ("midpoint", "heun")),
                              controls.flag("lower_order_final"), controls.flag("euler_at_final"))


def _deis(controls: _Controls) -> Solver:
    # DEIS rewrites the two DPM-Solver names to its own and refuses the rest;
    # its native integrator carries neither control.
    algorithm = controls.value("algorithm_type")
    named_choice("deis" if algorithm in ("dpmsolver", "dpmsolver++") else algorithm, "algorithm_type",
                 ("deis",))
    _solver_type(controls.value, ("logrho",))
    return DEIS(controls.order, controls.flag("lower_order_final"))


def _ddpm(controls: _Controls) -> Solver:
    mode = named_choice(controls.value("variance_type"), "variance_type",
                        ("fixed_small", "fixed_small_log", "fixed_large"))
    return DDPM("large" if mode == "fixed_large" else "small")


def _optional_seed(value: Control) -> int | None:
    seed = value("noise_sampler_seed")
    return None if seed is None else _records.integer(seed, "noise_sampler_seed")


def _prediction_transform(policy: SourcePolicy, prediction: str, source: _Class) -> PredictionTransform:
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
    if source.boundary:
        return ConsistencyBoundary(inner, policy.timestep_scaling)
    return inner
