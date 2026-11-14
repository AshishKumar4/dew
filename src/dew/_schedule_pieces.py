"""The pieces of a learning-rate schedule as functions of a step count: a warmup's ramp, a cosine, a power
law, a linear decay, joined as Optax joins them (`optax.join_schedules`), each of whose step counts may be a
Python number or an operand, a solver's step horizons (`dew.optim_schedules.RateState.horizons`), so the
program that reads them depends on no run's length. `dew.optim`'s schedules are built from them."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

type Horizon = jax.Array | int | float
"""A step count a schedule reads: a Python number, or an operand
(`dew.optim_schedules.RateState.horizons`)."""


def power_schedule(peak: jax.typing.ArrayLike, warmup_steps: Horizon, a: jax.typing.ArrayLike,
                   b: float, c: float = 1.0, decay_start: Horizon | None = None,
                   decay_end: Horizon | None = None,
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
    # Its steps may be operands (`ScheduleBase.horizons`): every piece is built, each read where it holds.
    warm = jnp.where(jnp.asarray(warmup_steps) > 0, law(jnp.maximum(warmup_steps, 1)), peak)
    pieces = [ramp(0.0, warm, warmup_steps), lambda count: law(count + warmup_steps)]
    boundaries = [warmup_steps]
    if decay_start is not None:
        end = ordered("the linear tail runs from decay_start past the warmup to a later decay_end",
                      warmup_steps, decay_start, decay_end)
        pieces.append(ramp(law(decay_start), end_value, end - decay_start))
        boundaries.append(decay_start)
    return joined(pieces, boundaries)


def power_law(peak: jax.typing.ArrayLike, a: jax.typing.ArrayLike, b: float,
              c: float = 1.0) -> Callable[[Horizon], jax.Array]:
    """The power scheduler's law at a step, `min(peak, a * (step * c) ** b)` with its step in float32
    (`power_schedule`), as a schedule of steps and one over a run's progress read it (`Power`)."""
    def law(step: Horizon) -> jax.Array:
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


def linear_schedule(peak: jax.typing.ArrayLike, warmup_steps: Horizon,
                    decay_start: Horizon | None, decay_end: Horizon,
                    end_value: jax.typing.ArrayLike = 0.0) -> optax.Schedule:
    """lm-engine's linear scheduler (lr_scheduler/linear.py at 45b6b57b): from
    zero to `peak` over the warmup, constant to `decay_start` (None: the
    warmup's end), then linear to `end_value` at `decay_end`."""
    start = warmup_steps if decay_start is None else decay_start
    end = ordered("the linear decay runs from past the warmup to a later step", warmup_steps, start,
                  decay_end)
    return joined([
        ramp(0.0, peak, warmup_steps),
        optax.constant_schedule(peak),
        ramp(peak, end_value, end - start),
    ], [warmup_steps, start])


def ordered(what: str, warmup: Horizon, start: Horizon, end: Horizon | None) -> Horizon:
    """`end`, a decay's, refused where the decay does not run from past the warmup to it, where its steps are
    host numbers; operands (`ScheduleBase.horizons`) were checked as they were resolved."""
    known = all(isinstance(value, int) for value in (warmup, start, end))
    if end is None or (known and not warmup <= start < end):
        raise ValueError(f"{what}, got {start} to {end} after {warmup} warmup steps")
    return end


def ramp(init: jax.typing.ArrayLike, end: jax.typing.ArrayLike,
         span: Horizon) -> optax.Schedule:
    """`optax.linear_schedule(init, end, span)`, its arithmetic Optax's, with `span` a count of steps or an
    operand (`ScheduleBase.horizons`): `init` throughout where it is 0, as Optax's is."""
    first, last = jnp.asarray(init), jnp.asarray(end)

    def schedule(count):
        through = jnp.clip(count, 0, span)
        frac = 1 - through / jnp.maximum(span, 1)
        return jnp.where(jnp.asarray(span) > 0, (first - last) * (frac**1) + last, first)
    return schedule


def cosine(init: jax.typing.ArrayLike, span: Horizon, alpha: jax.typing.ArrayLike) -> optax.Schedule:
    """`optax.cosine_decay_schedule(init, span, alpha)`, its arithmetic Optax's, with `span` a count of
    steps or an operand (`ScheduleBase.horizons`), positive."""
    def schedule(count):
        held = jnp.asarray(span, jnp.result_type(float))
        at = jnp.minimum(jnp.asarray(count, held.dtype), held)
        decay = 0.5 * (1 + jnp.cos(jnp.pi * at / jnp.maximum(held, 1)))
        return init * ((1 - alpha) * decay**1.0 + alpha)
    return schedule


def joined(pieces: list[optax.Schedule], boundaries: list[Horizon]) -> optax.Schedule:
    """`optax.join_schedules(pieces, boundaries)`, its arithmetic Optax's, its boundaries Python counts or
    operands: each piece from its boundary on, counted from it."""
    def schedule(count):
        output = pieces[0](count)
        for boundary, piece in zip(boundaries, pieces[1:], strict=True):
            output = jnp.where(count < boundary, output, piece(count - boundary))
        return output
    return schedule
