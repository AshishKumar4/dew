"""A run whose looped model runs a count of passes a phase of its data (`PhasePasses`, `Trainer`'s
`phase_passes`): each phase's step compiled with its own count over the one state.

A curriculum over a looped model's depth grows its pass count as its data moves on, one or two passes
while it learns the task's shape, more once it can use them. Each phase's step is the objective's own with
its looped modules run that many passes (`Looping.with_passes`), the same parameters and optimizer state,
compiled once a count; between steps the objective holds the last phase's, the model the run validates and
hands on.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import itertools
import math
from collections.abc import Collection, Hashable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import ClassVar, Protocol

import numpy as np
from flax import linen as nn

from dew._records import JSON
from dew.batch import PHASE, Batch
from dew.nn.protocols import Looped, Looping
from dew.objectives.base import Objective


class Holding[L, E](Protocol):
    """What holds the objective a step is built from: a `Trainer`, and the `StepCompiler` that compiles
    it."""

    objective: Objective[L, E]


def _looped(module: nn.Module) -> bool:
    """Whether `module` runs a block of its layers in passes it can be given (`Looped`, `Looping`)."""
    return isinstance(module, Looped) and isinstance(module, Looping) and module.loop is not None


def _chooses_passes(module: nn.Module) -> bool:
    """Whether `module` loops choosing its passes itself: one cache shared across them (`Loop.shared_cache`)
    or each token handed on at an exit threshold (`Loop.exit_threshold`), which `run_loop` runs at no count
    given per call."""
    loop = module.loop if _looped(module) and isinstance(module, Looped) else None
    return loop is not None and (loop.shared_cache or loop.exit_threshold is not None)


def _takes_counts(module: nn.Module, method: str | None) -> bool:
    """Whether `method`, the forward an objective reads `module` by (`Objective.counted_forward`), names the
    count of its loop's passes a call runs (`loop_passes`) among its own parameters, as Dew's decoders' and
    its multimodal wrapper's do; one that names it not, an adapter's handing its keywords on among them,
    states no count it reads."""
    forward = None if method is None else getattr(type(module), method, None)
    return forward is not None and "loop_passes" in inspect.signature(forward).parameters


def _horizons[L, E](objective: Objective[L, E]) -> tuple[int | None, ...]:
    """The final passes each of `objective`'s modules (`program_key`) trains through
    (`Loop.backprop_steps`), None for one that loops through every pass or loops not at all."""
    return tuple(entry.module.loop.backprop_steps if _looped(entry.module) else None
                 for entry in objective.program_key())


def _counted[L, E](objective: Objective[L, E], passes: int,
                   horizons: Sequence[int | None]) -> list[nn.Module]:
    """`objective`'s modules (`program_key`) with each looped one run `passes` times, each training through
    its run's configured final passes (`horizons`, `_horizons` of the run's own objective) or all of them
    where `passes` is fewer: a module a rung restored at another count keeps the run's horizon, not the one
    that count clamped it to."""
    return [_passed(entry.module, passes, horizon)
            for entry, horizon in zip(objective.program_key(), horizons, strict=True)]


def _passed(module: nn.Module, passes: int, trained: int | None) -> nn.Module:
    """`module` run `passes` times where it loops (`Looping.with_passes`), else as it is, training through
    its final `trained` passes (`Loop.backprop_steps`), or all of them where `passes` is fewer."""
    if not _looped(module):
        return module
    assert isinstance(module, Looping)
    passed = module.with_passes(passes, backprop_steps=None if trained is None else min(trained, passes))
    assert isinstance(passed, nn.Module), "a looped module keeps its loop"
    return passed


@dataclass
class PhasePasses[L, E]:
    """The run's own objective (`base`), a pass count a phase of its data (`counts`, none where every step
    runs the objective's own), the count the step being compiled runs (`current`, None for the
    objective's own), and the objective at each count a step was built at (`objective_for`)."""

    base: Objective[L, E]
    counts: tuple[int, ...] = ()
    current: int | None = None
    built: dict[int, Objective[L, E]] = field(default_factory=dict)
    horizons: tuple[int | None, ...] = ()
    """The final passes each of `base`'s modules trains through (`_horizons`), which every count keeps."""
    sampled: SampledPasses | None = None
    """The law a run draws its pass count from per microbatch, in place of a count a phase (`SampledPasses`):
    its step one program a count from 1 to its cap, its cap the count the objective holds, whatever phase
    its data reads (`count`), with no later phase's step of its own (`counts` none)."""

    @classmethod
    def of(cls, objective: Objective[L, E], stated: Sequence[int] | None,
           sampled: SampledPasses | None = None) -> PhasePasses[L, E]:
        """`stated` checked: each count at least one, the objective's modules looping, the last phase's count
        the one they hold, the model the run hands on. No counts where it states none; a run that draws its
        count per microbatch (`sampled`) holds its cap's, and states no count a phase beside it."""
        if sampled is not None:
            if stated is not None:
                raise ValueError("a run states its loop's pass counts once: one a phase of its data "
                                 "(phase_passes) or one drawn per microbatch (sampled_passes)")
            if not objective.reads_pass_counts:
                raise ValueError(f"{type(objective).__name__} runs its loop at the cap whatever count a "
                                 "microbatch draws, reading none (Objective.reads_pass_counts): "
                                 "sampled_passes trains an objective that does, as LMObjective and "
                                 "DecisionObjective do; phase_passes runs this one at a phase's count")
            if uncounted := [type(entry.module).__name__ for entry in objective.program_key()
                             if _looped(entry.module)
                             and not _takes_counts(entry.module, objective.counted_forward(entry.module))]:
                raise ValueError(f"{', '.join(uncounted)} loops and its forward takes no count of passes a "
                                 "call runs (`loop_passes`), so it would run every pass at the cap whatever "
                                 "count a microbatch draws: sampled_passes needs it; phase_passes runs it at "
                                 "a phase's count")
            if unrunnable := [type(entry.module).__name__ for entry in objective.program_key()
                              if _chooses_passes(entry.module)]:
                raise ValueError(f"{', '.join(unrunnable)} loops sharing one cache across its passes "
                                 "(`Loop.shared_cache`) or handing each token on at an exit threshold "
                                 "(`Loop.exit_threshold`), which the one program at its cap does not run "
                                 "at a count given per call (`run_loop`): sampled_passes refuses it; "
                                 "phase_passes runs it at a phase's count")
            # Its loop holding the cap, checked as a phase's count is; no phase's count beside it.
            return replace(cls.of(objective, (sampled.cap,)), counts=(), current=sampled.cap, sampled=sampled)
        if stated is None:
            return cls(objective)
        counts = tuple(int(count) for count in stated)
        if not counts or min(counts) < 1:
            raise ValueError(f"phase_passes runs each phase at least one pass, got {list(counts)}")
        looped = [entry.module for entry in objective.program_key() if _looped(entry.module)]
        if not looped:
            raise ValueError(f"phase_passes runs a looped model's passes, and {type(objective).__name__} "
                             "runs none (dew.nn.protocols.Looped)")
        held = {module.loop.steps for module in looped
                if isinstance(module, Looped) and module.loop is not None}
        if held != {counts[-1]}:
            raise ValueError(f"the objective's loop runs {sorted(held)} passes and the last phase "
                             f"{counts[-1]}: it holds the count the run hands on")
        return cls(objective, counts, horizons=_horizons(objective))

    def count(self, batch: Batch) -> int | None:
        """The count of the phase `batch` was read in (its host `PHASE`, phase 0 where its data has none);
        None where the run states no counts; the cap where it draws one per microbatch, whatever phase its
        data reads. A phase past the counts is refused, naming both."""
        if self.sampled is not None:
            return self.sampled.cap
        if not self.counts:
            return None
        phase = batch[PHASE].index if PHASE in batch else 0
        if phase >= len(self.counts):
            raise ValueError(f"a batch of the run's phase {phase} meets phase_passes of {len(self.counts)} "
                             "phase(s): one count a phase")
        return self.counts[phase]

    def training_record(self, objective: Objective[L, E]) -> dict[str, JSON]:
        """What a run's steps compute beside its model, as a checkpoint records it and a resume holds it to
        (`dew.training.startup.computation_differs`): `objective`'s settings (`Objective.training_record`),
        and the law a pass count drawn per microbatch follows (`SampledPasses`), whose change would draw
        another schedule of passes from the same key and positions (row 10783)."""
        sampled = {} if self.sampled is None else {"sampled_passes": asdict(self.sampled)}
        return {**objective.training_record(), **sampled}

    def drawn_cap(self) -> int | None:
        """The cap of the pass count a run draws per microbatch, a batch's counted work's width
        (`prepare_counts`); None for any other run."""
        return None if self.sampled is None else self.sampled.cap

    def counted(self) -> list[Objective[L, E]] | None:
        """The run's objective at each count from 1 to its cap where it draws one per microbatch, whose work
        a batch counts at each (`prepare_counts`'s `counts`); None for any other run."""
        return None if self.sampled is None else [self.objective_for(count)
                                                  for count in range(1, self.sampled.cap + 1)]

    def begins(self, compiled: Collection[Hashable]) -> bool:
        """Whether the step being compiled (`current`) begins a phase whose count no step of `compiled`
        (keys of `key`) ran, past the run's first: a step planned afresh, at its own count's split, where
        the run is at an update's end (`Trainer._planned_afresh`)."""
        if self.current is None or not compiled:
            return False
        return all(not isinstance(held, tuple) or len(held) != 2 or held[1] != self.current
                   for held in compiled)

    def at(self, batch: Batch) -> int | None:
        """The count of the phase `batch` was read in (`count`), taken as the step being compiled's
        (`current`)."""
        self.current = self.count(batch)
        return self.current

    def key(self, signature: Hashable) -> Hashable:
        """The compiled step's key for a batch's `signature`: with the count it runs, one program a count."""
        return self.key_at(signature, self.current)

    @staticmethod
    def key_at(signature: Hashable, passes: int | None) -> Hashable:
        """The key of a step at `passes` (None for the objective's own) for a batch's `signature` (`key`), a
        later phase's as the loop keys it (`dew.training.ahead.later_phases`)."""
        return signature if passes is None else (signature, passes)

    def run(self, objective: Objective[L, E], passes: int) -> None:
        """Run `objective`'s looped modules `passes` times, each keeping its other choices, the rest as is."""
        objective.substitute(_counted(objective, passes, self.horizons))

    def restored(self, objective: Objective[L, E]) -> None:
        """`objective`'s looped modules at the step's count again (`current`), where a rung restored the
        modules it was taken with; nothing where the step runs the objective's own."""
        if self.current is not None:
            self.run(objective, self.current)

    @contextlib.contextmanager
    def scoped(self, *holders: Holding[L, E]) -> Iterator[None]:
        """Each of `holders`' objective at the step's count while a step is built (`objective_for`), the one
        the step and its host runner hold; the run's own again after."""
        for holder in holders:
            holder.objective = self.objective_for(self.current)
        try:
            yield
        finally:
            for holder in holders:
                holder.objective = self.base

    def apart(self, passes: int) -> PhasePasses[L, E]:
        """These counts with their step at `passes` over an objective of its own: `base`'s copy at that
        count, kept by no one else (`built` holds none of it), which a compile off the loop builds and
        climbs its step's rungs on (`dew.training.ahead.LaterPlans`) while the loop's objectives stay as
        they are."""
        objective = self.base.substituted(_counted(self.base, passes, self.horizons))
        return PhasePasses(objective, self.counts, passes, dict.fromkeys(self.counts, objective),
                           self.horizons)

    def begun(self, first: int, phases: Sequence[int]) -> list[int]:
        """Those of `phases` whose step is planned at its boundary, after the run's `first` phase: each the
        first of a pass count no phase before it ran (`begins`), one plan a count."""
        seen = {self.counts[first]}
        begun = []
        for phase in phases:
            if self.counts[phase] not in seen:
                seen.add(self.counts[phase])
                begun.append(phase)
        return begun

    def objective_for(self, passes: int | None) -> Objective[L, E]:
        """The run's objective (`base`, at the last phase's count) at `passes` (a phase's `count`, the
        step's `current`): a copy of it whose looped modules run that many passes, kept for every step of
        the count, which a step, its host runner and its batches' counts (`Trainer._counted_batch`, on the
        prefetch worker) hold while validation and the model the run hands on read `base`. Read off
        `base`, never off a holder that holds a count's copy meanwhile; one copy a count, whichever thread
        asks first."""
        if passes is None or passes == (self.counts[-1] if self.sampled is None else self.sampled.cap):
            return self.base
        if passes not in self.built:
            self.built.setdefault(passes, self.base.substituted(_counted(self.base, passes, self.horizons)))
        return self.built[passes]


RATE_SPAN = 39
"""The standard deviations either side of its mode past which a count's integrand over the log rate (a
log-concave one, at least as narrow as its normal density) is 0 beside its peak in float64 (exp(-39² / 2)
underflows past the least subnormal): the span `SampledPasses.cumulative` integrates each count's over."""

MODE_NODES = 64
"""The Gauss-Legendre nodes a unit of count takes in `SampledPasses.cumulative`'s integral of the log rate's
mode over the count, whose integrand is analytic there."""


@dataclass(frozen=True)
class SampledPasses:
    """A looped model's pass count drawn afresh for each microbatch a run steps (`count`), from a named law
    bounded to `cap` (row 7660): `law` 'lognormal-poisson', the count `1 + N`, `N ~ Poisson(Λ)`,
    `log Λ ~ Normal(log(mean - 1) - dispersion² / 2, dispersion²)`, so `mean` is the count's mean before its
    bound, conditioned on 1 to `cap`, as recurrent-depth training draws it (Geiping et al. 2025, arXiv
    2502.05171), by the bounded law's inverse cumulative (`cumulative`), one uniform a draw however little of
    the unbounded law's mass the cap holds. A draw is a function of the run's key and the microbatch's global
    position alone,
    its place among every microbatch the run dispatched: every process draws alike, and a resume, a replan
    or a step of another split draws on as the run would have."""

    mean: float
    dispersion: float
    cap: int
    law: str = "lognormal-poisson"

    LAWS: ClassVar[tuple[str, ...]] = ("lognormal-poisson",)

    def __post_init__(self) -> None:
        if self.law not in self.LAWS:
            raise ValueError(f"a sampled pass count follows one of the laws {list(self.LAWS)}, "
                             f"not {self.law!r}")
        if not (math.isfinite(self.mean) and self.mean > 1):
            raise ValueError("a log-normal Poisson pass count's mean is above 1, its least count, "
                             f"not {self.mean}")
        if not (math.isfinite(self.dispersion) and self.dispersion >= 0):
            raise ValueError("a log-normal Poisson pass count's dispersion is at least 0, "
                             f"not {self.dispersion}")
        if self.cap < 1:
            raise ValueError(f"a sampled pass count's cap is at least one pass, not {self.cap}")

    def count(self, key: Sequence[int], position: int) -> int:
        """The pass count of the run's microbatch at global `position` under the run's `key` (its words,
        `jax.random.key_data`): the bounded law's inverse cumulative (`cumulative`) at one uniform draw."""
        draws = np.random.Generator(np.random.Philox(np.random.SeedSequence([*map(int, key), int(position)])))
        return 1 + int(np.searchsorted(self.cumulative, draws.random(), side="right"))

    @functools.cached_property
    def cumulative(self) -> np.ndarray:
        """The bounded law's cumulative probability at each count 1 to `cap`, its last 1, from each count's
        log probability less a term common to all: the Poisson's of the count less one at the rate
        `exp(location)` without dispersion (its common `-exp(location)` left out, which past a mean of
        2^53 would round every count's alike); else the Poisson's integrated over the log rate's normal
        density, its log integrand `n x - exp(x) - (x - location)² / (2 dispersion²) - log n!` concave in the
        log rate `x`: its peak value, its integral over the count of its mode (`_mode`, Laplace's envelope:
        the peak's derivative in `n` is the mode less `digamma(n + 1)`), so no count's peak is the difference
        of the exponential's large values; and its integral about its own mode scaled by the peak
        (`_about_mode`), never about `location`, from which a cap low beside the mean moves each count's mass
        far. Then conditioned on the cap."""
        below = np.arange(self.cap)
        factorials = np.array([math.lgamma(n + 1) for n in below])
        location = math.log(self.mean - 1) - self.dispersion * self.dispersion / 2
        if location == -math.inf:
            # A dispersion whose square float64 cannot hold: a count past one is drawn with chance
            # exp(-dispersion² / 8) or less (its rate past its least value), 0 in float64.
            return np.ones(self.cap)
        if self.dispersion == 0:
            logs = below * location - factorials
        else:
            nodes, weights = np.polynomial.legendre.leggauss(MODE_NODES)
            counts = below[1:, None] + (nodes[None, :] - 1) / 2
            units = _mode(counts, location, self.dispersion) @ (weights / 2)
            modes = _mode(below.astype(float), location, self.dispersion)
            logs = (np.concatenate([[0.0], np.cumsum(units)]) - factorials
                    + np.array([_about_mode(float(mode), self.dispersion) for mode in modes]))
        probabilities = np.exp(logs - logs.max())
        cumulative = np.cumsum(probabilities / probabilities.sum())
        # The sum's last partial is 1 but for its rounding, which a uniform draw past it must not reach.
        cumulative[-1] = 1.0
        return cumulative


def _mode(counts: np.ndarray, location: float, dispersion: float) -> np.ndarray:
    """The log rate at which each count's log integrand peaks (`SampledPasses.cumulative`), at real counts
    `counts` too: the root of its derivative `dispersion² (count - exp(x)) - (x - location)`, decreasing in
    `x`, bisected between `location` and the count's log (a count of 0 left of `location` by
    `dispersion² exp(location)`, at most a million) to float64's last bit, its sign compared, never
    computed, where the exponential's terms overflow."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        logs = np.log(np.where(counts > 0, counts, 1.0))
        reach = min(dispersion ** 2 * math.exp(min(location, 709.0)), 1e6)
        low = np.where(counts > 0, np.minimum(location, logs), location - reach)
        high = np.where(counts > 0, np.maximum(location, logs), location)
        while True:
            middle = (low + high) / 2
            if not np.any((middle > low) & (middle < high)):
                return low
            rising = dispersion ** 2 * (counts - np.exp(middle)) > middle - location
            low, high = np.where(rising, middle, low), np.where(rising, high, middle)


def _about_mode(mode: float, dispersion: float) -> float:
    """The log of a count's integrand over the log rate (`SampledPasses.cumulative`) about its `mode`, scaled
    by its peak and by the normal density's width: the integral over the standardized log rate `z` (`x = mode
    + dispersion z`) of `exp(-exp(mode) (expm1(dispersion z) - dispersion z) - z² / 2)`, the peak's own terms
    cancelled exactly by the mode's equation, so nothing in it is the difference of large values. In units
    of its peak's width (`1 / sqrt(1 + dispersion² exp(mode))`, held as its log so no dispersion's square
    overflows, `u = z / width`): its log's curvature is at least 1 in `u` above the mode and in `z` below it,
    so it is 0 beside its peak past `RATE_SPAN` units above and `RATE_SPAN` deviations below, or where it
    reaches 0 first; each side in panels doubling from one unit (scipy's adaptive quadrature on each), so no
    panel is wider than the integrand's own scale where it starts."""
    from scipy import integrate

    scale = math.exp(mode)
    # log(1 + dispersion² exp(mode)) as a softplus of its log, finite however large the dispersion.
    curvature = 2 * math.log(dispersion) + mode
    log_width = -(curvature + math.log1p(math.exp(-curvature)) if curvature > 0
                  else math.log1p(math.exp(curvature))) / 2
    width, stride = math.exp(log_width), math.exp(math.log(dispersion) + log_width)

    def integrand(u: float) -> float:
        t = stride * u
        # expm1(t) - t, its series where the difference would cancel.
        lost = t * t / 2 * (1 + t / 3 + t * t / 12) if abs(t) < 1e-4 else math.expm1(min(t, 709.0)) - t
        exponent = scale * lost + (width * u) ** 2 / 2
        return 0.0 if exponent > 745 else math.exp(-exponent)

    def panels(side: float, reach: float) -> list[float]:
        edges = [0.0, side]
        while abs(edges[-1]) < reach and integrand(edges[-1]) > 0:
            edges.append(side * min(2 * abs(edges[-1]), reach))
        return edges

    halves = (panels(1.0, RATE_SPAN), panels(-1.0, RATE_SPAN / width if width > 0 else math.inf)[::-1])
    total = sum(integrate.quad(integrand, start, end, epsabs=0, epsrel=1e-13, limit=200)[0]
                for edges in halves for start, end in itertools.pairwise(edges))
    return log_width + math.log(total)
