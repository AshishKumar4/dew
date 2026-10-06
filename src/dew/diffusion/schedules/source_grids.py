"""The schedules that sampling on a published diffusion checkpoint's grid uses.

`source_policy` resolves a scheduler file's controls once. `sampling_grid`
uses those controls to build the numerical tables and `Process` that
`SourceSchedule` caches. Each table is an ordinary Dew schedule, so a solver
reads its rates, model time and prior the same way as any other's.

There are four kinds: the training beta table indexed by t (`TabulatedVP`);
a paired sigma and model-time table, in variance-exploding (`SigmaGrid`) or
normalized-VP (`VPGrid`) coordinates; the variance-exploding table with the
stage rows a two-evaluation solver reads (`StageSigmaGrid`); and a
rectified-flow table whose signal and noise sum to one (`FlowGrid`). A
paired table is read at a coordinate t that counts down from
T = len(sigmas) - 1, so a non-integer t interpolates between two prepared
rows.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from dew import records
from dew.diffusion.process import Process
from dew.diffusion.schedules.common import GeneralizedNoiseScheduler, NoiseScheduler
from dew.diffusion.schedules.discrete import DiscreteNoiseScheduler
from dew.diffusion.transforms import PredictionTransform
from dew.records import JSON

if TYPE_CHECKING:
    from dew.diffusion.schedules.source_policy import Origin, _Flow, _Policy


class TabulatedVP(DiscreteNoiseScheduler):
    """The training beta table used as the sampling schedule, indexed by t.

    `stride` is the fixed training transfer that DDIM and PNDM step over,
    whatever their evaluation grid is. None keeps the grid's own interval,
    which DDPM's previous-timestep policy and the distilled schedules use. A
    t below zero is the source's "no previous alpha" end, where the rates
    come from `final_alpha_cumprod`.
    """

    def __init__(self, betas: np.ndarray, *, final_alpha_cumprod: float, stride: int | None):
        super().__init__(betas, p2_loss_weight_gamma=0)
        self.final_rates = tuple(np.float32(np.sqrt(value))
                                 for value in (final_alpha_cumprod, 1 - final_alpha_cumprod))
        self.stride = stride

    def rates(self, t):
        t = jnp.asarray(t, jnp.float32)
        index = jnp.clip(t.astype(jnp.int32), 0, self.T - 1)
        return (jnp.where(t < 0, self.final_rates[0], self.sqrt_alpha_cumprod[index]),
                jnp.where(t < 0, self.final_rates[1], self.sqrt_one_minus_alpha_cumprod[index]))

    def model_time(self, t):
        return jnp.maximum(jnp.asarray(t, jnp.float32), 0.0)

    def step_interval(self, t, t_next):
        """Return the published DDIM and PNDM transfer stride, independent of the evaluation spacing.

        Without a `stride`, this is the grid's own interval.
        """
        if self.stride is None:
            return super().step_interval(t, t_next)
        return jnp.full_like(jnp.asarray(t, jnp.float32), self.stride)

    def half_interval(self, t, t_next):
        """Return half the integer transfer stride, rounded down, as published PRK uses it."""
        return jnp.floor(self.step_interval(t, t_next) / 2)


def _interpolate(x, xp, fp):
    """`jnp.interp(x, xp, fp)` in its own arithmetic, with the interval found
    by comparing x against every point of xp.

    jnp.interp finds it by bisection, a loop of log2(len(xp)) steps. Inside
    `sample`'s scan with one row, XLA rewrites the table reads into dynamic
    slices whose offsets come out of that loop, and XLA:GPU's
    DynamicSliceAnnotator (jax 0.11.1) evaluates each offset without running
    the loop, then fails the whole compile on the unknown value instead of
    skipping the slice (openxla/xla#49299). A grid holds a few dozen points,
    so comparing against all of them costs nothing and leaves no loop.
    """
    x = jnp.asarray(x, jnp.float32)
    i = jnp.clip(jnp.searchsorted(xp, x, side="right", method="compare_all"), 1, len(xp) - 1)
    df = fp[i] - fp[i - 1]
    dx = xp[i] - xp[i - 1]
    flat = jnp.abs(dx) <= np.spacing(np.finfo(np.float32).eps)
    f = jnp.where(flat, fp[i - 1], fp[i - 1] + (x - xp[i - 1]) / jnp.where(flat, 1, dx) * df)
    return jnp.where(x > xp[-1], fp[-1], jnp.where(x < xp[0], fp[0], f))


class _PairedGrid:
    """Reads a prepared grid of paired sigmas and model times by coordinate.

    Coordinate t counts down from `T = len(sigmas) - 1`, so t and the grid
    index run opposite ways and a non-integer t interpolates between two
    prepared rows. Subclasses say what the sigmas mean as rates.
    """

    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        self.table = jnp.asarray(sigmas, jnp.float32)
        self.times = jnp.asarray(model_times, jnp.float32)
        self.rows = jnp.arange(len(sigmas), dtype=jnp.float32)
        self.prior = jnp.asarray(prior, jnp.float32)
        self.T = float(len(sigmas) - 1)

    def sigmas(self, t):
        return _interpolate(self.T - jnp.asarray(t, jnp.float32), self.rows, self.table)

    def t_of_sigma(self, sigma):
        return self.T - _interpolate(sigma, self.table[::-1], self.rows[::-1])

    def model_time(self, t):
        return _interpolate(self.T - jnp.asarray(t, jnp.float32), self.rows, self.times)

    def prior_scale(self):
        return self.prior


class SigmaGrid(_PairedGrid, GeneralizedNoiseScheduler):
    """A variance-exploding schedule on a prepared grid of paired sigmas and model times.

    `sigma_min` and `sigma_max` are the grid's smallest positive sigma and
    its largest sigma, which is the domain a source's noise sampler is built
    over.
    """

    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        levels = np.asarray(sigmas, np.float64)
        GeneralizedNoiseScheduler.__init__(self, sigma_min=float(np.min(levels[levels > 0])),
                                           sigma_max=float(np.max(levels)))
        _PairedGrid.__init__(self, sigmas, model_times, prior)


class StageSigmaGrid(SigmaGrid):
    """A source's variance-exploding grid that includes the stage rows of a two-evaluation solver.

    Even coordinates are the grid points the outer loop visits. The odd
    coordinate between each pair is the source's own interpolated
    evaluation, at the sigma the source places there and the model time it
    computes for that sigma. `t_of_sigma` maps a sigma to a stage
    coordinate. That is the only inversion these solvers need from a
    schedule, because KDPM2's midpoint and DPMSolverSDE's proposal both land
    on a stage row.
    """

    def __init__(self, sigmas: np.ndarray, model_times: np.ndarray, prior: float):
        super().__init__(sigmas, model_times, prior)
        stages = np.asarray(sigmas, np.float64)[1::2]
        self.stages = jnp.asarray(stages[::-1].copy(), jnp.float32)
        self.stage_positions = jnp.asarray(
            np.arange(1, len(sigmas), 2, dtype=np.float32)[::-1].copy())

    def t_of_sigma(self, sigma):
        return self.T - _interpolate(sigma, self.stages, self.stage_positions)


class _UniformGrid(_PairedGrid, NoiseScheduler):
    """A paired grid whose source trains at uniform times and weights nothing.

    The flow and normalized-VP grids differ only in what their sigmas mean
    as rates; the draw and the loss weight are the same for both.
    """

    def sample_t(self, key, n: int):
        return jax.random.uniform(key, (n,), minval=0, maxval=self.T)

    def weight(self, t):
        return jnp.ones_like(jnp.asarray(t, jnp.float32))


class FlowGrid(_UniformGrid):
    """A rectified-flow grid, where alpha is 1 - sigma.

    The source's forward process is x_t = (1 - sigma) x_0 + sigma eps, so
    signal and noise sum to one; in a normalized VP pair, their squares do.
    Model times are the sigmas times the training step count, which is where
    the source's `timesteps` come from. The prior at sigma 1 is the unit
    Gaussian. Training draws times uniformly with a constant loss weight.
    """

    def rates(self, t):
        sigma = self.sigmas(t)
        return 1.0 - sigma, sigma


class VPGrid(_UniformGrid):
    """A paired sigma and model-time grid in normalized VP coordinates.

    A grid sigma gives alpha = 1 / sqrt(1 + sigma^2) and a noise rate of
    sigma alpha. Training draws times uniformly with a constant loss weight.
    """

    def rates(self, t):
        sigma = self.sigmas(t)
        # The source rounds sqrt before its reciprocal. Fusing to rsqrt moves
        # stiff cosine-grid VJPs beyond the float32 source-parity bound.
        alpha = 1 / jax.lax.optimization_barrier(jnp.sqrt(1 + sigma ** 2))
        return alpha, sigma * alpha


def _choice[ChoiceT: str](value: object, name: str, allowed: tuple[ChoiceT, ...]) -> ChoiceT:
    if not isinstance(value, str):
        raise ValueError(f"{name} must name one of {', '.join(allowed)}, not {value!r}")
    for choice in allowed:
        if value == choice:
            return choice
    raise ValueError(f"Native source scheduling does not implement {name}={value!r}; "
                     f"this class supports {', '.join(allowed)}")


def _published_linspace(start: float, end: float, count: int) -> np.ndarray:
    """The source's float32 endpoints and step, spaced from each end.

    NumPy spaces the original double endpoints. Those different roundings
    change the beta table and accumulate in alpha-bar. Widen the rounded
    operands so the position's multiply-add is rounded only once to float32,
    rather than introducing another rounding at its intermediate product.
    """
    first, last = float(np.float32(start)), float(np.float32(end))
    if count == 1:
        return np.asarray([first], np.float32)
    step = float(np.float32((last - first) / (count - 1)))
    index = np.arange(count, dtype=np.float64)
    return np.where(index < count // 2, first + index * step,
                    last - (count - 1 - index) * step).astype(np.float32)


def published_betas(*, count: JSON, start: JSON, end: JSON, schedule: JSON,
                    trained: JSON | np.ndarray, zero_snr: bool,
                    schedules: tuple[str, ...]) -> np.ndarray:
    """Return a scheduler class's beta table, rescaled for zero terminal SNR when `zero_snr` is set.

    Every control arrives already resolved against the class's own declared
    default. So a file that omits one gets that class's value, and a control
    the class does not declare never reaches the table. `trained`, when
    given, is the table itself. `schedules` lists the `beta_schedule` tables
    the class implements. Every class accepts the three common ones, DDPM
    adds GeoDiff's sigmoid, and Heun adds the exponential alpha-bar. A table
    that does not have `count` finite entries in [0, 1] raises `ValueError`.
    """
    length = records.integer(count, "num_train_timesteps")
    if length < 1:
        raise ValueError("num_train_timesteps must be positive")
    if trained is not None:
        if not isinstance(trained, (list, tuple, np.ndarray)):
            raise ValueError("trained_betas must be a numeric sequence")
        betas = np.asarray(trained, np.float32)
    else:
        first = records.number(start, "beta_start")
        final = records.number(end, "beta_end")
        kind = _choice(schedule, "beta_schedule", schedules)
        if kind == "linear":
            betas = _published_linspace(first, final, length)
        elif kind == "scaled_linear":
            betas = _published_linspace(first ** 0.5, final ** 0.5, length) ** 2
        elif kind == "sigmoid":
            ramp = _published_linspace(-6, 6, length)
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


def empirical_mu(tokens: int, steps: int) -> float:
    """`compute_empirical_mu`, the shift FLUX.2's pipelines fit to the latent
    token count and the step count and hand the scheduler in place of its
    own. Past 4300 tokens it is the 200-step line; below, it interpolates
    linearly in the steps between the 10- and the 200-step lines."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if tokens > 4300:
        return float(a2 * tokens + b2)
    m_200, m_10 = a2 * tokens + b2, a1 * tokens + b1
    slope = (m_200 - m_10) / 190.0
    return float(slope * steps + m_200 - 200.0 * slope)


def _shifted(flow: _Flow, sigmas: np.ndarray, tokens: int | None, mu: float | None = None) -> np.ndarray:
    """The sigmas after this file's shift."""
    base = flow.base(tokens, mu)
    return base * sigmas / (1 + (base - 1) * sigmas)


def _stretched(flow: _Flow, sigmas: np.ndarray) -> np.ndarray:
    """`stretch_shift_to_terminal`, which the source applies once in
    `set_timesteps` and never to the constructor's own seed."""
    if flow.terminal is None:
        return sigmas
    remaining = 1 - sigmas
    return 1 - remaining / (remaining[-1] / (1 - flow.terminal))


def _spaced_times(spacing: str, *, train_steps: int, steps: int, offset: int, last: int,
                  extra: int, rounded: bool) -> np.ndarray:
    """A source `set_timesteps` evaluation grid, Table 2 of Lin et al. 2023.

    The three spacings are the three forms the pinned classes write it in.
    `extra` is the additional point the log-SNR classes lay out and drop,
    `last` is the end of the table left after lambda clipping, and `rounded`
    is whether the class rounds its linspace before using it.
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


def _training_sigmas(betas: np.ndarray, policy: _Policy) -> tuple[np.ndarray, np.ndarray]:
    """The training sigma table sigma/alpha and its logarithm, with the
    near-zero terminal alpha the zero-SNR classes substitute.

    The source accumulates this product in double and keeps the result in
    float32, which is what `torch.cumprod` of a float32 table does, so
    the ratio and its logarithm are float32 too. Both halves matter. A
    float32 accumulation moves the cosine table's last alpha, about
    2.4e-9, by a part in a million and its sigma by 8e-3. A float64 ratio
    moves a model time recovered by log-linear search across the integer
    boundary it is truncated at.
    """
    alphas = np.cumprod(1 - betas, dtype=np.float64).astype(np.float32)
    if policy.zero_snr_tail:
        alphas[-1] = np.float32(2.0 ** -24)
    base = np.sqrt((1 - alphas) / alphas, dtype=np.float32)
    return base, np.log(base, dtype=np.float32)


def _transformed_grid(policy: _Policy, steps: int, log_base: np.ndarray, low: float,
                      high: float) -> tuple[np.ndarray, np.ndarray]:
    """The class's own sigma grid over `[low, high]`, and the model times
    its log-linear inverse recovers for those sigmas.

    A Karras grid rounds those times where the source rounds them.
    """
    sigmas = _transformed_sigmas(policy.transform, low, high, steps, policy.rho)
    times = _sigma_to_time(sigmas, log_base)
    if policy.transform == "karras" and policy.karras_round:
        times = times.round()
    return sigmas, times


def _lambda_grid(policy: _Policy, betas: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    """The paired sigma and model-time tables of a log-SNR class, with the
    terminal sigma its `final_sigmas_type` appends. Model times end as
    integers: the source stores them as int64."""
    base, log_base = _training_sigmas(betas, policy)
    if policy.flow_shift is not None:
        return _flow_sigmas(policy, steps, float(base[0]))
    last = policy.train_steps - policy.lambda_clipped
    times = _spaced_times(policy.spacing, train_steps=policy.train_steps, steps=steps,
                          offset=policy.offset, last=last, extra=1, rounded=True)
    if policy.transform == "none":
        sigmas = np.interp(times, np.arange(len(base)), base)
    else:
        sigmas, times = _transformed_grid(policy,
            steps, log_base,
            float(base[0]) if policy.sigma_min is None else policy.sigma_min,
            float(base[-1]) if policy.sigma_max is None else policy.sigma_max)
    if policy.terminal == "zero":
        terminal = 0.0
    elif policy.grid_terminal and policy.transform != "none":
        terminal = float(sigmas[-1])
    else:
        terminal = float(base[0])
    return np.append(sigmas, terminal), np.trunc(times), 1.0


def _flow_sigmas(policy: _Policy, steps: int, sigma_min: float) -> tuple[np.ndarray, np.ndarray, float]:
    """`use_flow_sigmas`: the shifted flow path from 1 - 1/T down, one point
    dropped at the clean end, whatever the spacing and lambda clipping.

    DPM-Solver's `sigma_min` terminal is still its beta table's first
    sigma, and UniPC's the grid's own last one, as each source appends.
    """
    assert policy.flow_shift is not None
    shift = policy.flow_shift
    rising = 1.0 - np.linspace(1, 1 / policy.train_steps, steps + 1)
    sigmas = np.flip(shift * rising / (1 + (shift - 1) * rising))[:-1].copy()
    terminal = (0.0 if policy.terminal == "zero"
                else float(sigmas[-1]) if policy.grid_terminal else sigma_min)
    return np.append(sigmas, terminal), np.trunc(sigmas * policy.train_steps), 1.0


def _sigma_grid(policy: _Policy, betas: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    """The paired tables of a variance-exploding class: its spacing
    interpolated out of the training table, then whatever sigma
    transformation it applies to that interpolated subset."""
    base, log_base = _training_sigmas(betas, policy)
    times = _spaced_times(policy.spacing, train_steps=policy.train_steps, steps=steps,
                          offset=policy.offset, last=policy.train_steps, extra=0,
                          rounded=False)
    sigmas = np.interp(times, np.arange(len(base)), base)
    if policy.transform != "none":
        sigmas, times = _transformed_grid(policy,
            steps, log_base,
            float(sigmas[-1]) if policy.sigma_min is None else policy.sigma_min,
            float(sigmas[0]) if policy.sigma_max is None else policy.sigma_max)
    terminal = float(base[0]) if policy.terminal == "sigma_min" else 0.0
    prior = float(np.max(sigmas))
    if policy.spacing == "leading":
        prior = float(np.sqrt(prior ** 2 + 1))
    return np.append(sigmas, terminal), times, prior


def _stage_grid(policy: _Policy, betas: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    """The same tables refined with the stage row each interval evaluates.

    KDPM2 reads the geometric mean of the interval's ends, its ancestral
    form the geometric mean of the start and k-diffusion's `sigma_down`,
    and DPMSolverSDE the midpoint in -log(sigma), which is that same
    geometric mean. Every stage model time is the source's log-linear
    inverse of the stage sigma in the training table. The interval that
    lands on zero has no stage evaluation and carries zero.
    """
    _, log_base = _training_sigmas(betas, policy)
    sigmas, times, prior = _sigma_grid(policy, betas, steps)
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


def _check_unique_start(times: np.ndarray) -> None:
    """Refuse a grid whose first model time appears again in it.

    The source looks its starting step up by that time and takes the
    second match, which runs its walk off the end of its sigma table (a
    cosine table's Karras grid at few steps does this). Later repeats are
    the deliberate ones of the two-evaluation classes and PNDM's warmup.
    """
    if len(times) > 1 and float(np.sum(times == times[0])) > 1:
        raise ValueError(
            "The recovered model times repeat their first value "
            f"({times[0]}), which the source scheduler cannot walk: it "
            "looks its starting step up by that value and finds the second")


def _flow_grid(policy: _Policy, flow: _Flow, steps: int, tokens: int | None,
               origin: Origin) -> tuple[np.ndarray, np.ndarray, float]:
    """`FlowMatchEulerDiscreteScheduler.set_timesteps` in its own order.

    `origin` is where the sigmas start, which is a pipeline fact rather
    than a config one. SD3 lets the scheduler lay them out between its own
    sigma extremes, Flux hands it `linspace(1, 1/N, N)`, and FLUX.2 hands
    it the same with its own mu (`empirical_mu`). Then comes
    the file's shift, then whichever sigma conversion it asks for, then
    the appended zero.
    """
    count = policy.train_steps
    if origin in ("linspace", "empirical"):
        sigmas = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float64)
    else:
        # The class's own sigma extremes, which its constructor already
        # shifted statically when it is not shifting dynamically; the
        # shift below then lands on this seed a second time, as the
        # source's own two passes do.
        trained = np.linspace(1, count, count, dtype=np.float32)[::-1].astype(np.float64) / count
        if not flow.dynamic:
            trained = _shifted(flow, trained, tokens)
        times = np.linspace(float(trained[0]) * count, float(trained[-1]) * count, steps)
        sigmas = np.asarray(times, np.float64) / count
    mu = None
    if origin == "empirical":
        if not flow.dynamic or tokens is None:
            raise ValueError("FLUX.2's pipeline hands a dynamically shifting scheduler its own mu, "
                             "which reads the latent token count")
        mu = empirical_mu(tokens, steps)
    sigmas = _stretched(flow, _shifted(flow, sigmas, tokens, mu))
    if policy.transform != "none":
        sigmas = _transformed_sigmas(policy.transform, float(sigmas[-1]), float(sigmas[0]),
                                     steps, policy.rho)
    return np.append(sigmas, 0.0), np.asarray(sigmas, np.float64) * count, 1.0


def _edm_grid(policy: _Policy, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    """EDM's own grid, rho or exponential spacing between the sigma extremes.

    The model time is c_noise = log(sigma) / 4 and the prior is unit data
    at sigma_max.
    """
    low = 0.002 if policy.sigma_min is None else policy.sigma_min
    high = 80.0 if policy.sigma_max is None else policy.sigma_max
    sigmas = _transformed_sigmas(policy.transform, low, high, steps, policy.rho)
    terminal = low if policy.terminal == "sigma_min" else 0.0
    times = 0.25 * np.log(sigmas)
    return (np.append(sigmas, terminal), np.append(times, times[-1]),
            float(np.sqrt(high ** 2 + 1)))


def sampling_grid(policy: _Policy, betas: np.ndarray, prediction: PredictionTransform,
                  steps: int, tokens: int | None, origin: Origin) -> tuple[Process, jax.Array]:
    """Build one family's native tables and descending times from already resolved controls."""
    if type(steps) is not int or steps < 1:
        raise ValueError("The sampling count must be a positive integer")
    if policy.flow is not None:
        sigmas, times, prior = _flow_grid(policy, policy.flow, steps, tokens, origin)
        schedule = FlowGrid(sigmas, np.append(times, 0.0), prior)
        return (Process(schedule, prediction),
                jnp.arange(len(sigmas) - 1, -1, -1, dtype=jnp.float32))
    if policy.family in ("tabulated", "lambda") and steps > policy.train_steps:
        raise ValueError("The sampling count must fit the training table")
    if policy.family == "tabulated":
        return _tabulated(policy, betas, prediction, steps)
    if policy.family == "edm":
        sigmas, times, prior = _edm_grid(policy, steps)
        schedule: NoiseScheduler = SigmaGrid(sigmas, times, prior)
    elif policy.family == "sigma":
        sigmas, times, prior = _sigma_grid(policy, betas, steps)
        _check_unique_start(times)
        schedule = SigmaGrid(sigmas, np.append(times, times[-1]), prior)
    elif policy.family == "stage":
        sigmas, times, prior = _stage_grid(policy, betas, steps)
        # The source matches its starting time against the whole
        # interleaved evaluation list, stage rows included; only the
        # padding row this table carries past the last stage is never
        # evaluated. Repeats after the first entry are the deliberate ones.
        _check_unique_start(times[:2 * steps - 1])
        schedule = StageSigmaGrid(sigmas, times, prior)
        return (Process(schedule, prediction),
                jnp.arange(len(sigmas) - 1, -1, -2, dtype=jnp.float32))
    else:
        sigmas, times, prior = _lambda_grid(policy, betas, steps)
        _check_unique_start(times)
        paired = FlowGrid if policy.flow_shift is not None else VPGrid
        schedule = paired(sigmas, np.append(times, times[-1]), prior)
    return (Process(schedule, prediction),
            jnp.arange(len(sigmas) - 1, -1, -1, dtype=jnp.float32))


def _tabulated(policy: _Policy, betas: np.ndarray, prediction: PredictionTransform,
               steps: int) -> tuple[Process, jax.Array]:
    """A class that steps between integer indices of the training table."""
    if policy.distilled:
        times = _distilled_times(policy.train_steps, policy.original_steps, steps)
        terminal = 0.0
    else:
        times = _spaced_times(policy.spacing, train_steps=policy.train_steps, steps=steps,
                              offset=policy.offset, last=policy.train_steps, extra=0,
                              rounded=True)
        terminal = (max(float(times[-1]) - policy.train_steps // steps, -1.0)
                    if policy.stride else -1.0)
    final = 1.0 if policy.clean_terminal else float(1 - betas[0])
    schedule = TabulatedVP(betas, final_alpha_cumprod=final,
                           stride=policy.train_steps // steps if policy.stride else None)
    return (Process(schedule, prediction),
            jnp.asarray(np.append(times, terminal), jnp.float32))


__all__ = ["FlowGrid", "SigmaGrid", "StageSigmaGrid", "TabulatedVP", "VPGrid", "published_betas"]
