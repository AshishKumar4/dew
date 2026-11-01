"""Learning-rate schedules: a rate over a run's steps or its progress (`ScheduleBase`), each one of
Dew's named shapes (cosine, linear, one-cycle, exponential, power, and a power law's tail), and the
functions that compute them (`power_schedule`, `power_law`, `anneal`, `linear_schedule`). An optimizer's
rate is one of these or a constant (`dew.optim.OptimConfig.learning_rate`); `dew.optim.scheduled_solver`
holds one over the run as its state."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import optax

from dew._records import share

__all__ = [
    "Cosine", "Exponential", "Linear", "OneCycle", "Power", "PowerTail", "ScheduleBase", "anneal",
    "linear_schedule", "power_law", "power_schedule",
]


def power_schedule(peak: jax.typing.ArrayLike, warmup_steps: int, a: jax.typing.ArrayLike, b: float,
                   c: float = 1.0,
                   decay_start: int | None = None, decay_end: int | None = None,
                   end_value: jax.typing.ArrayLike = 0.0) -> optax.Schedule:
    """lm-engine's power scheduler (optimization/lr_scheduler/power.py at
    45b6b57b) with an optional linear tail.

    Past the warmup the rate is `min(peak, a * (step * c) ** b)`: the power
    law of the batch size and step, arXiv 2408.13359, capped at `peak` (the
    optimizer's own rate there). The warmup rises linearly from zero to that
    value at `warmup_steps`. `decay_start` set follows the law to that step
    and then decays linearly to `end_value` at `decay_end`, which is how
    Rigel ends its run; lm-engine's scheduler has no tail.
    """
    law = power_law(peak, a, b, c)
    warm = law(warmup_steps) if warmup_steps else peak
    pieces = [optax.linear_schedule(0.0, warm, warmup_steps)] if warmup_steps else []
    pieces.append(lambda count: law(count + warmup_steps))
    boundaries = [warmup_steps] if warmup_steps else []
    if decay_start is not None:
        if decay_end is None or not warmup_steps <= decay_start < decay_end:
            raise ValueError(
                f"the linear tail runs from decay_start past the warmup to a later "
                f"decay_end, got {decay_start} to {decay_end} after {warmup_steps} warmup steps")
        pieces.append(optax.linear_schedule(law(decay_start), end_value,
                                            decay_end - decay_start))
        boundaries.append(decay_start)
    return optax.join_schedules(pieces, boundaries) if boundaries else pieces[0]


def power_law(peak: jax.typing.ArrayLike, a: jax.typing.ArrayLike, b: float,
              c: float = 1.0) -> Callable[[jax.typing.ArrayLike], jax.Array]:
    """The power scheduler's law at a step, `min(peak, a * (step * c) ** b)` with its step in float32
    (`power_schedule`), as a schedule of steps and one over a run's progress read it (`Power`)."""
    def law(step: jax.typing.ArrayLike) -> jax.Array:
        return jnp.minimum(peak, a * (jnp.asarray(step, jnp.float32) * c) ** b)

    return law


def anneal(start: jax.Array | float, stop: jax.Array | float, elapsed: jax.Array,
           span: jax.Array | float) -> jax.Array:
    """A half cosine from `start` to `stop`, `elapsed` of `span` through it (`OneCycle`): each end weighed
    by the cosine's halves, `start * c + stop * (1 - c)`, so it lands on each end exactly and takes any two
    rates, a zero start or ends far apart; torch's form, `stop + (start - stop) / 2 * (cos + 1)`, rounds the
    difference of the ends in float32. The angle is optax's cosine decay's, `pi * elapsed / span`, in that
    order: near a phase's end the cosine's halves cancel, so the angle's rounding is most of the rate's."""
    turned = jnp.cos(jnp.pi * elapsed / span)
    return start * (0.5 * (1 + turned)) + stop * (0.5 * (1 - turned))


def linear_schedule(peak: jax.typing.ArrayLike, warmup_steps: int, decay_start: int | None,
                    decay_end: int, end_value: jax.typing.ArrayLike = 0.0) -> optax.Schedule:
    """lm-engine's linear scheduler (lr_scheduler/linear.py at 45b6b57b): from
    zero to `peak` over the warmup, constant to `decay_start` (None: the
    warmup's end), then linear to `end_value` at `decay_end`."""
    start = warmup_steps if decay_start is None else decay_start
    if not warmup_steps <= start < decay_end:
        raise ValueError(
            f"the linear decay runs from past the warmup to a later step, got "
            f"{start} to {decay_end} after {warmup_steps} warmup steps")
    return optax.join_schedules([
        optax.linear_schedule(0.0, peak, warmup_steps),
        optax.constant_schedule(peak),
        optax.linear_schedule(peak, end_value, decay_end - start),
    ], [warmup_steps, start])


def _after(progress: jax.Array, start: jax.Array) -> jax.Array:
    """How far a run at `progress` stands through the piece from `start` to its end, 0 to 1."""
    return jnp.clip((progress - start) / jnp.maximum(1.0 - start, jnp.finfo(progress.dtype).tiny), 0.0, 1.0)


type Rates = Mapping[str, jax.Array | float]
"""A schedule's rate-valued fields by path (`ScheduleBase.rates`): the configured values, or a solver's
operands for them (`RateState`)."""


@dataclasses.dataclass(frozen=True)
class ScheduleBase:
    """The base of the schedule records, each of which holds its own fields
    and builds an optax schedule.

    A run's record names the schedule's class and holds only the fields that
    schedule reads; `dew.registry.schedules` holds the short aliases. A
    subclass builds its values in `values`; `schedule` reads them. The fields
    it marks `metadata={'rate': True}` are its rates (`rates`), which a solver
    holds as operands (`scheduled_solver`) and `values` reads from `rates`, so
    a run's program does not depend on them; a subclass that marks none
    implements `values(steps)` alone.
    """

    every: int = dataclasses.field(default=1, kw_only=True)
    """The updates each of the schedule's own steps lasts.

    torch recipes step a scheduler once an epoch, and with `every` set to the
    updates of an epoch this one holds each value through an epoch as torch's
    does. A run of `steps` updates is then `steps // every` of the
    schedule's own steps, and every field that counts steps (`warmup_steps`,
    `decay_steps`) counts those.
    """

    def __post_init__(self):
        if self.every < 1:
            raise ValueError(f"a schedule advances once every 1 or more updates, not {self.every}")

    def rates(self) -> dict[str, float]:
        """Each rate-valued field's value (`metadata={'rate': True}`) by its dotted path, nested
        records such as PowerTail's too, a None field's derived default resolved where the schedule
        derives one (OneCycle's) and left out otherwise."""
        def walked(record, prefix):
            for field in dataclasses.fields(record):
                value = getattr(record, field.name)
                if field.metadata.get('rate') and value is not None:
                    yield prefix + field.name, value
                elif dataclasses.is_dataclass(value):
                    yield from walked(value, f"{prefix}{field.name}.")
        return dict(walked(self, ""))

    def schedule(self, steps: int | None, rates: Rates | None = None
                 ) -> optax.Schedule:
        """Return the value at each update of a run of `steps` updates, as an optax schedule, its
        rate-valued fields `rates`' (`rates()` by default): operands in a solver's state, or the
        configured values."""
        held = self.rates() if rates is None else rates

        def built(own):
            return self.values(own, held) if held else self.values(own)

        if self.every == 1:
            return built(steps)
        own = None if steps is None else steps // self.every
        if own is not None and own < 1:
            raise ValueError(f"a run of {steps} updates never advances a schedule that steps once "
                             f"every {self.every}")
        values, every = built(own), self.every
        return lambda count: values(jnp.asarray(count) // every)

    def _horizon(self, steps: int | None, field: str) -> int:
        """Resolve a decay's step horizon, which a token budget cannot supply."""
        if steps is None:
            raise ValueError(f"{type(self).__name__} needs {field} or trainer.steps as its step horizon")
        return steps

    def warmup(self, steps: int | None) -> int | None:
        """Its warmup in its own steps over a run of `steps` updates, its share of the run resolved
        (`_warmup`), as a run's plan records it; None for a schedule with no warmup of steps."""
        return None

    def _warmup(self, written: int | str, steps: int | None, horizon: int | None = None) -> int:
        """A warmup of `written` steps in its own steps over a run of `steps` of them: as written, or a
        share of the run such as `5%` (`dew._records.share`), the whole steps nearest it, of the run's
        steps or, where the run has none, of the schedule's own `horizon`."""
        if not isinstance(written, str):
            return written
        return round(share(written) * self._horizon(steps if steps is not None else horizon, "warmup_steps"))

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        """Return the value at each of the schedule's own steps, over `steps` of them, from its
        rate-valued fields `rates` (`rates()`). A value derived from the rates alone, as a ratio of
        two, is the configured fields', which a solver's operands keep: they differ from them by
        one factor, a param group's multiplier."""
        raise NotImplementedError

    def peak(self) -> float:
        """The highest value the schedule takes: its largest rate-valued field, a warmup's or decay's
        ends, where its formula reaches no higher."""
        return max(self.rates().values(), default=1.0)

    def run_relative(self) -> bool:
        """Whether its horizon is the run's, so a run that counts no updates ahead (one bounded in
        examples) reads it over the run's progress (`progressed`); a schedule whose every horizon is its
        own reads its updates there as anywhere."""
        return False

    def written_warmup(self) -> int | str | float | None:
        """Its warmup as written, in its own steps or as a share of the run, which a run that counts no
        updates ahead records as it is (`OptimConfig.warmups`); None where it has none."""
        return None

    def marks(self) -> dict[str, int]:
        """The update counts its law on a run's progress starts a piece at (`progressed`), by name: the
        end of a warmup counted in updates, say. The run records where its progress stood as its updates
        reached each (`ProgressState.marks`)."""
        return {}

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its value where a run that counts no updates ahead stands: at `progress`, the share of the
        run's bound it has consumed (`dew.state.TrainState.consumed`), after `updates` updates, the
        progress it stood at as it reached each count it `marks`. Each piece written in updates (a
        warmup of 1000) reads the updates, and each that lasts to the run's end reads the progress from
        where it starts, its law kept: a share of the run (`5%`) is that share of the progress, and the
        run's end its whole. A schedule's own steps (`every`) step its updates alone."""
        raise NotImplementedError(f"{type(self).__name__} reads its run's horizon and states no law on the "
                                  "run's progress")

    def _counted(self, updates: jax.Array) -> jax.Array:
        """`updates` in its own steps (`every`)."""
        return jnp.asarray(updates) // self.every

    def _warm(self, written: int | str, progress: jax.Array, updates: jax.Array,
              marks: Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        """A warmup of `written` on a run's progress: how far through it the run stands, 0 to 1 (`in`),
        whether it is over, and the progress it ended at, its law's next piece's start: written in
        updates, of its own steps and its end marked (`marks['warmup']`); a share of the run, of the
        progress."""
        if isinstance(written, str):
            span = share(written)
            through = jnp.clip(progress / span, 0.0, 1.0) if span else jnp.ones_like(progress)
            return through, progress >= span, jnp.asarray(span, progress.dtype)
        count = self._counted(updates)
        through = (jnp.clip(count.astype(progress.dtype) / written, 0.0, 1.0) if written
                   else jnp.ones_like(progress))
        return through, count >= written, marks.get("warmup", jnp.zeros([], progress.dtype))


@dataclasses.dataclass(frozen=True)
class Cosine(ScheduleBase):
    """A linear warmup from `init_value` to `peak_value`, then a cosine decay to `end_value` at
    step `decay_steps`, or at the run's end when that is None.

    It is `optax.warmup_cosine_decay_schedule`.
    `warmup_steps` must be smaller than the decay horizon in schedule steps.
    """

    peak_value: float = dataclasses.field(metadata={'rate': True})
    warmup_steps: int | str
    """Its warmup in its own steps, or a share of the run such as `5%` (`ScheduleBase._warmup`): stated,
    as `optax.warmup_cosine_decay_schedule` takes it, which has no default either."""
    end_value: float = dataclasses.field(default=0.0, metadata={'rate': True})
    init_value: float = dataclasses.field(default=0.0, metadata={'rate': True})
    decay_steps: int | None = None

    def warmup(self, steps: int | None) -> int:
        return self._warmup(self.warmup_steps, steps, self.decay_steps)

    def run_relative(self) -> bool:
        return self.decay_steps is None

    def written_warmup(self) -> int | str | float | None:
        return self.warmup_steps

    def marks(self) -> dict[str, int]:
        warmup = self.warmup_steps
        return {"warmup": warmup * self.every} if isinstance(warmup, int) and warmup else {}

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its warmup, then its cosine decay from where the warmup ended to the run's end."""
        if isinstance(self.warmup_steps, str) and share(self.warmup_steps) >= 1:
            raise ValueError(f"Cosine's warmup of {self.warmup_steps} leaves its run no decay")
        through, over, start = self._warm(self.warmup_steps, progress, updates, marks)
        alpha = 0.0 if self.peak_value == 0.0 else self.end_value / self.peak_value
        # `values`' optax pieces, each read at the fraction of its own span the run stands through.
        warm = optax.linear_schedule(rates['init_value'], rates['peak_value'], 1)(through)
        decayed = optax.cosine_decay_schedule(rates['peak_value'], 1, alpha)(_after(progress, start))
        return jnp.where(over, decayed, warm)

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        decay_steps = self._horizon(steps if self.decay_steps is None else self.decay_steps, "decay_steps")
        warmup = self.warmup(steps)
        if warmup >= decay_steps:
            raise ValueError(f"Cosine needs warmup_steps={self.warmup_steps} smaller than "
                             f"decay_steps={decay_steps}")
        # `optax.warmup_cosine_decay_schedule`'s own pieces, joined as it joins them; it forms the
        # floor's ratio by branching on the peak, so the ratio is the configured fields' (`values`).
        alpha = 0.0 if self.peak_value == 0.0 else self.end_value / self.peak_value
        return optax.join_schedules([
            optax.linear_schedule(rates['init_value'], rates['peak_value'], warmup),
            optax.cosine_decay_schedule(rates['peak_value'], decay_steps - warmup, alpha),
        ], [warmup])


@dataclasses.dataclass(frozen=True)
class PowerTail:
    """The linear tail of a power schedule, from the power law's rate at step
    `start` to `end_value` at step `steps`, or at the run's end when that is None."""

    start: int
    steps: int | None = None
    end_value: float = dataclasses.field(default=0.0, metadata={'rate': True})


@dataclasses.dataclass(frozen=True)
class Power(ScheduleBase):
    """lm-engine's power schedule (`power_schedule`), a power law of the step after a linear warmup.

    After the warmup, the rate is min(peak_value, a * (step * c) ** b). With a
    `tail`, a linear decay follows the power law, as in the last 29% of
    Rigel's run. lm-engine's examples set `a` to 4 * batch size and `c` to
    the tokens per step.
    """

    peak_value: float = dataclasses.field(metadata={'rate': True})
    warmup_steps: int | str
    """Its warmup in its own steps, or a share of the run such as `5%` (`ScheduleBase._warmup`)."""
    a: float = dataclasses.field(metadata={'rate': True})
    b: float = -0.51
    c: float = 1.0
    tail: PowerTail | None = None

    def warmup(self, steps: int | None) -> int:
        return self._warmup(self.warmup_steps, steps, None if self.tail is None else self.tail.steps)

    def peak(self) -> float:
        """`peak_value`, the cap the power law climbs to."""
        return self.peak_value

    def run_relative(self) -> bool:
        return isinstance(self.warmup_steps, str) or (self.tail is not None and self.tail.steps is None)

    def written_warmup(self) -> int | str | float | None:
        return self.warmup_steps

    def marks(self) -> dict[str, int]:
        tail = self.tail
        return {"tail": tail.start * self.every} if tail is not None and tail.steps is None else {}

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its power law of its updates after its warmup, and its tail from `tail.start` to the run's end.
        A warmup written as a share of the run rises over that share of the progress to the law's value
        at the update it reaches, the law scaled by how far through it the run stands."""
        count = self._counted(updates)
        law = power_law(rates['peak_value'], rates['a'], self.b, self.c)
        written = self.warmup_steps
        if isinstance(written, str):
            through, _, _ = self._warm(written, progress, updates, marks)
            value = law(jnp.maximum(count, 1)) * through
        elif written:
            value = jnp.where(count < written, optax.linear_schedule(0.0, law(written), written)(count),
                              law(count))
        else:
            value = law(count)
        tail = self.tail
        if tail is None:
            return value
        start = law(tail.start)
        if tail.steps is None:
            through_tail = _after(progress, marks["tail"])
            decayed = optax.linear_schedule(start, rates['tail.end_value'], 1)(through_tail)
        else:
            decayed = optax.linear_schedule(start, rates['tail.end_value'], tail.steps - tail.start)(
                count - tail.start)
        return jnp.where(count >= tail.start, decayed, value)

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        tail = self.tail
        warmup = self.warmup(steps)
        if tail is None:
            return power_schedule(rates['peak_value'], warmup, rates['a'], self.b, self.c)
        return power_schedule(
            rates['peak_value'], warmup, rates['a'], self.b, self.c, decay_start=tail.start,
            decay_end=self._horizon(steps if tail.steps is None else tail.steps, "tail.steps"),
            end_value=rates['tail.end_value'])


@dataclasses.dataclass(frozen=True)
class Linear(ScheduleBase):
    """lm-engine's linear schedule (`linear_schedule`), with a warmup, a constant rate and a linear decay.

    The rate warms up from zero to `peak_value`, stays constant until
    `decay_start` (the end of the warmup when None), then falls linearly to
    `end_value` at `decay_steps` (the run's end when None).
    """

    peak_value: float = dataclasses.field(metadata={'rate': True})
    warmup_steps: int | str = 0
    """Its warmup in its own steps, or a share of the run such as `5%` (`ScheduleBase._warmup`)."""
    decay_start: int | None = None
    decay_steps: int | None = None
    end_value: float = dataclasses.field(default=0.0, metadata={'rate': True})

    def warmup(self, steps: int | None) -> int:
        return self._warmup(self.warmup_steps, steps, self.decay_steps)

    def run_relative(self) -> bool:
        return self.decay_steps is None

    def written_warmup(self) -> int | str | float | None:
        return self.warmup_steps

    def marks(self) -> dict[str, int]:
        warmup, start = self.warmup_steps, self.decay_start
        return {**({"warmup": warmup * self.every} if isinstance(warmup, int) and warmup else {}),
                **({"decay_start": start * self.every} if start is not None else {})}

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its warmup, its constant rate to `decay_start` (the warmup's end where None), and its linear
        decay from there to the run's end."""
        through, over, start = self._warm(self.warmup_steps, progress, updates, marks)
        decaying = over
        if self.decay_start is not None:
            decaying, start = self._counted(updates) >= self.decay_start, marks["decay_start"]
        peak = rates['peak_value']
        # `linear_schedule`'s optax pieces, each read at the fraction of its own span the run stands through.
        decayed = optax.linear_schedule(peak, rates['end_value'], 1)(_after(progress, start))
        warm = optax.linear_schedule(0.0, peak, 1)(through)
        return jnp.where(decaying, decayed, jnp.where(over, peak, warm))

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        return linear_schedule(rates['peak_value'], self.warmup(steps), self.decay_start,
                               self._horizon(steps if self.decay_steps is None else self.decay_steps,
                                             "decay_steps"),
                               end_value=rates['end_value'])


@dataclasses.dataclass(frozen=True)
class OneCycle(ScheduleBase):
    """torch's `OneCycleLR` with its default two phases and cosine annealing.

    The value follows a half cosine from `init_value` to `peak_value` at step
    `warmup_fraction * steps - 1`, then another to `end_value` at the run's last
    step, `steps - 1`, and holds `end_value` after it, where torch raises. `init_value`
    is torch's `max_lr / div_factor` (`peak_value / 25` when None) and `end_value` its
    `init_value / final_div_factor` (`init_value / 1e4` when None). Adam's momentum cycle
    under `cycle_momentum=True` is `OneCycle(peak_value=0.85, init_value=0.95,
    end_value=0.95)` as a group's `b1`.

    optax's `cosine_onecycle_schedule` places its phase boundaries at
    `pct_start * steps` and `steps`, one step later than torch's
    (torch/optim/lr_scheduler.py, `OneCycleLR.__init__`), so its rate peaks
    one step late and it cannot reproduce a torch run.
    """

    peak_value: float = dataclasses.field(metadata={'rate': True})
    init_value: float | None = dataclasses.field(default=None, metadata={'rate': True})
    end_value: float | None = dataclasses.field(default=None, metadata={'rate': True})
    warmup_fraction: float | str = 0.3
    """The share of the run its rise lasts, torch's `pct_start`: a fraction, or a share such as `30%`."""

    def rates(self) -> dict[str, float]:
        """The peak, and the init and end values with torch's derived defaults resolved."""
        init = self.peak_value / 25 if self.init_value is None else self.init_value
        return {'peak_value': self.peak_value, 'init_value': init,
                'end_value': init / 1e4 if self.end_value is None else self.end_value}

    def _rise(self, steps: int) -> float:
        """The step its rate peaks at over a run of `steps`, torch's phase end, kept a float as torch
        keeps it."""
        fraction = self.warmup_fraction
        return (share(fraction) if isinstance(fraction, str) else fraction) * steps - 1

    def warmup(self, steps: int | None) -> int:
        """The steps of its rise, to the whole step its rate peaks at (`_rise`)."""
        return round(self._rise(self._horizon(steps, "trainer.steps")))

    def run_relative(self) -> bool:
        return True

    def written_warmup(self) -> int | str | float | None:
        return self.warmup_fraction

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its rise over `warmup_fraction` of the run's progress, then its fall to the run's end, each a
        half cosine as over a run of steps; torch's peak a step before the share's end and its last
        value a step before the run's are the share's and the run's ends here."""
        fraction = self.warmup_fraction
        rise = share(fraction) if isinstance(fraction, str) else fraction
        if not 0 < rise < 1:
            raise ValueError(f"a one-cycle schedule with warmup fraction {fraction} has no rise or no fall")

        at = jnp.clip(progress, 0.0, 1.0)
        return jnp.where(at <= rise, anneal(rates['init_value'], rates['peak_value'], at, rise),
                         anneal(rates['peak_value'], rates['end_value'], at - rise, 1 - rise))

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        steps = self._horizon(steps, "trainer.steps")
        rise, last = self._rise(steps), steps - 1
        if not 0 < rise < last:
            raise ValueError(f"a one-cycle schedule over {steps} steps with warmup fraction "
                             f"{self.warmup_fraction} has no rise or no fall; give it more steps")
        # Each phase a half cosine (`anneal`); torch's own form put the run's first rate 25 times further
        # from float64 than optax's cosine decay.
        def value(count) -> jax.Array:
            # The default float: float32 in a run, float64 where a test reads the definition under x64.
            step = jnp.minimum(jnp.asarray(count, float), last)
            return jnp.where(step <= rise, anneal(rates['init_value'], rates['peak_value'], step, rise),
                             anneal(rates['peak_value'], rates['end_value'], step - rise, last - rise))

        return value


@dataclasses.dataclass(frozen=True)
class Exponential(ScheduleBase):
    """A geometric decay from `offset + init_value` to `offset + end_value` over
    `decay_steps` steps (the run's when None), then constant.

    Up to `decay_steps`, the value at step t is
    `offset + init_value * (end_value / init_value) ** (t / decay_steps)`. That is torch's
    `ExponentialLR` with `gamma = (end_value / init_value) ** (1 / decay_steps)`, stepped
    `decay_steps` times and then no more, plus `offset`. It is
    `optax.exponential_decay` with `end_value` as its bound. SNN-delays anneals
    DCLS's kernel width this way, the raw width decaying and a constant
    added to it.
    """

    init_value: float = dataclasses.field(metadata={'rate': True})
    end_value: float = dataclasses.field(metadata={'rate': True})
    decay_steps: int | None = None
    offset: float = dataclasses.field(default=0.0, metadata={'rate': True})

    def peak(self) -> float:
        """Its higher end, `offset` added."""
        return self.offset + max(self.init_value, self.end_value)

    def run_relative(self) -> bool:
        return self.decay_steps is None

    def progressed(self, rates: Rates, progress: jax.Array, updates: jax.Array,
                   marks: Mapping[str, jax.Array]) -> jax.Array:
        """Its geometric decay over the run's progress: `init_value * (end_value / init_value) ** progress`,
        plus `offset`."""
        if self.init_value <= 0 or self.end_value <= 0:
            raise ValueError(f"an exponential decay needs positive init_value and end_value, got "
                             f"{self.init_value} and {self.end_value}")
        ratio = self.end_value / self.init_value
        return rates['offset'] + rates['init_value'] * ratio ** jnp.clip(progress, 0.0, 1.0)

    def values(self, steps: int | None, rates: Rates | None = None
               ) -> optax.Schedule:
        rates = self.rates() if rates is None else rates
        span = self._horizon(steps if self.decay_steps is None else self.decay_steps, "decay_steps")
        if span <= 0 or self.init_value <= 0 or self.end_value <= 0:
            raise ValueError(f"an exponential decay needs positive steps, init_value and end_value, got "
                             f"{span}, {self.init_value} and {self.end_value}")
        # optax branches on the decay rate, so it is the configured fields' ratio (`values`).
        decay = optax.exponential_decay(rates['init_value'], span,
                                        self.end_value / self.init_value,
                                        end_value=rates['end_value'])
        if not self.offset:
            return decay
        return lambda count: rates['offset'] + decay(count)
