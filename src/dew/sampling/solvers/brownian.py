"""The keyed Brownian bridge and the two-stage DPM-Solver SDE that reads it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from dew.registry import solvers

from .common import _sigma_integrator


class _Brownian(NamedTuple):
    """The root interval of a keyed dyadic Brownian bridge, shared by a walk.

    `low` and `high` are the interval the source noise sampler is built over,
    and `key` seeds every node of the bridge. Nothing else is carried: a value
    of the path is a pure function of these three and the point asked for, so
    the state is three scalars whatever the sample's shape.
    """

    key: jax.Array
    low: jax.Array
    high: jax.Array


# A float32 position in [0, 1] carries 24 mantissa bits, so a descent deeper
# than this cannot tell two positions apart and only pretends to refine.
MAX_BROWNIAN_DEPTH = 24


def _brownian_walk(state: _Brownian, point, shape, depth: int) -> jax.Array:
    """W(point) - W(low) of the bridge, by Levy's construction to `depth`.

    W(low) is zero and W(high) is sqrt(high - low) times the root draw; each
    level conditions the midpoint of the half the point falls in, which is the
    Brownian bridge's own N((W(a) + W(b)) / 2, (b - a) / 4). A node's draw comes
    from its level and its dyadic index alone, so deepening the construction
    leaves every coarser node where it was, and the path inside the finest
    cell, of width (high - low) / 2**depth, is read off that cell's two ends.

    The descent reads the binary expansion of the normalized position by
    doubling and subtracting, which is exact in binary floating point, and
    names each node by its integer dyadic index, so no level's bounds are
    accumulated and none collapses however deep the construction runs.

    `point` is a scalar: a source noise sampler is one tree over the whole
    batch tensor, so the interval is the grid's, not a row's.
    """
    span = state.high - state.low
    # A grid whose only interval lands on sigma zero prepares a zero-width
    # interval and never queries it; the walk stays at the path's origin.
    scale = jnp.where(span > 0, span, 1.0)
    position = jnp.clip((point - state.low) / scale, 0.0, 1.0)
    value_low = jnp.zeros(shape, jnp.float32)
    value_high = jnp.sqrt(span) * jax.random.normal(jax.random.fold_in(state.key, 0), shape,
                                                    dtype=jnp.float32)
    index = jnp.zeros((), jnp.int32)
    for level in range(1, depth + 1):
        node = jax.random.fold_in(jax.random.fold_in(state.key, level), 2 * index + 1)
        deviation = 0.5 * jnp.sqrt(span * 2.0 ** (1 - level))
        value_middle = 0.5 * (value_low + value_high) + deviation * jax.random.normal(
            node, shape, dtype=jnp.float32)
        position = position * 2.0
        right = position >= 1.0
        position = position - right.astype(jnp.float32)
        value_low = jnp.where(right, value_middle, value_low)
        value_high = jnp.where(right, value_high, value_middle)
        index = 2 * index + right.astype(jnp.int32)
    return value_low + position * (value_high - value_low)


def _brownian_noise(state: _Brownian, first, second, shape, depth: int) -> jax.Array:
    """The bridge's increment over `[first, second]`, normalized the way
    k-diffusion's `BrownianTreeNoiseSampler` normalizes it: signed with the
    interval's direction and divided by the square root of its width, so the
    result is a standard normal draw carrying the path's correlations."""
    increment = (_brownian_walk(state, second, shape, depth)
                 - _brownian_walk(state, first, shape, depth))
    return increment / jnp.sqrt(jnp.abs(second - first))


def _sde_step(x, denoised, sigma, target):
    """The deterministic part of one `DPMSolverSDEScheduler` step, in the
    exponential form it evaluates: (target / sigma) x - expm1(log target -
    log sigma) x_0, which is the Euler step to `target` regrouped so the
    coefficient stays accurate when the levels are close."""
    return (target / sigma) * x - jnp.expm1(jnp.log(target) - jnp.log(sigma)) * denoised


@solvers("dpmsolver_sde")
@dataclass(frozen=True)
class DPMSolverSDE:
    """Diffusers 0.34.0's `DPMSolverSDEScheduler`, k-diffusion's
    `sample_dpmpp_sde` midpoint solver over a Brownian tree.

    Each interval takes two ancestral first-order steps from its own start:
    one to the geometric midpoint of sigma_t and sigma_s, where the model is
    evaluated, and one to sigma_s with that midpoint's clean prediction. Both
    steps go down to k-diffusion's sigma_down and add sigma_up of noise, and
    both draw that noise from one Brownian path over the trajectory's sigma
    interval: the first over `[sigma_t, sigma_mid]` and the second over
    `[sigma_t, sigma_s]`, so the two are correlated exactly as nested
    increments of one path. The source's sampler transforms sigma with the
    identity even though its own steps integrate -log(sigma), so the interval
    widths are sigma differences.

    `depth` resolves the root interval to `(sigma_max - sigma_min) / 2**depth`:
    on a published VP table's span of about 14.6 the default reaches 8.7e-7,
    at or inside the reference tree's own 1e-6 tolerance, and it is also where
    a float32 position runs out of mantissa, so no deeper descent tells two
    sigmas apart. A zero-sigma target has no ancestral step and lands on the
    clean prediction.

    The root interval is the schedule's positive sigma domain, not the grid's
    extremes, as the source builds its tree over every positive sigma it
    prepared, so a walk over a suffix of the grid keeps the whole grid's path.

    `seed` is the source's `noise_sampler_seed`: the path is then fixed by
    the checkpoint, independent of the walk's key. A Torch seed names no JAX
    stream, so it seeds this bridge rather than reproducing the reference tree.
    """

    depth: int = MAX_BROWNIAN_DEPTH
    seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.depth) is not int or not 1 <= self.depth <= MAX_BROWNIAN_DEPTH:
            raise ValueError(f"the Brownian bridge takes 1 to {MAX_BROWNIAN_DEPTH} levels, "
                             f"the mantissa of a float32 position, not {self.depth}")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError(f"the noise sampler's seed is a nonnegative integer, not {self.seed}")

    def init(self, x, times, process, *, key):
        schedule = _sigma_integrator("DPMSolverSDE", process)
        with jax.ensure_compile_time_eval():
            sigmas = schedule.sigmas(jnp.asarray(times, jnp.float32))
            if times.shape[0] > 1 and bool(jnp.any(sigmas[:-1] <= 0)):
                raise ValueError("sigma=0 source has no finite update: DPMSolverSDE "
                                 "steps down from the interval's own sigma")
        low, high = float(schedule.sigma_min), float(schedule.sigma_max)
        if not 0 < low <= high:
            raise ValueError("the Brownian tree needs a positive sigma domain, and this "
                             f"schedule reports [{low}, {high}]")
        root = key if self.seed is None else jax.random.PRNGKey(self.seed)
        return _Brownian(root, jnp.asarray(low, jnp.float32), jnp.asarray(high, jnp.float32))

    def _noise(self, state, first, second, shape):
        """The path's standard normal draw over `[first, second]`, the one
        place a reference walk couples the source's own tree in."""
        return _brownian_noise(state, first, second, shape, self.depth)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise):
        schedule = _sigma_integrator("DPMSolverSDE", process)
        _, sigma_t = process.rates(t, like=x)
        _, sigma_s = process.rates(t_next, like=x)

        def ancestral(target):
            """The source's `sigma_up`, capped at the target level, and the
            `sigma_down` a step actually reaches."""
            up = jnp.minimum(target, (target ** 2 * (sigma_t ** 2 - target ** 2) / sigma_t ** 2) ** 0.5)
            return (target ** 2 - up ** 2) ** 0.5, up

        def midpoint(_):
            sigma_mid = jnp.exp(0.5 * (jnp.log(sigma_t) + jnp.log(sigma_s)))
            first_down, first_up = ancestral(sigma_mid)
            noise = self._noise(state, jnp.min(sigma_t), jnp.min(sigma_mid), x.shape)
            x_mid = _sde_step(x, denoised, sigma_t, first_down) + noise * first_up
            denoised_mid, _ = denoise(x_mid, schedule.t_of_sigma(sigma_mid.reshape(-1)))
            down, up = ancestral(sigma_s)
            noise = self._noise(state, jnp.min(sigma_t), jnp.min(sigma_s), x.shape)
            return _sde_step(x, denoised_mid, sigma_t, down) + noise * up

        stepped = lax.cond(jnp.all(sigma_s == 0), lambda _: denoised, midpoint, None)
        return jnp.where(sigma_s > 0, stepped, denoised), state
