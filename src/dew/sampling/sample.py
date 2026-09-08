"""The reverse process as one scan over a time grid."""

from collections.abc import Sequence
from typing import overload

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

from dew.diffusion.discrete import DiscreteDenoiser
from dew.diffusion.process import Denoiser
from dew.sampling.guidance import CFG
from dew.sampling.solvers import Solver


@overload
def sample[StateT](denoise: Denoiser | DiscreteDenoiser, x_T: jax.Array, steps: int, *, solver: Solver[StateT],
                   guidance: CFG | None = None, key: jax.Array, times: None = None,
                   final_denoise: bool = True) -> jax.Array: ...


@overload
def sample[StateT](denoise: Denoiser | DiscreteDenoiser, x_T: jax.Array, steps: None = None, *, solver: Solver[StateT],
                   guidance: CFG | None = None, key: jax.Array, times: ArrayLike | Sequence[float],
                   final_denoise: bool = True) -> jax.Array: ...


def sample[StateT](denoise: Denoiser | DiscreteDenoiser, x_T: jax.Array, steps: int | None = None, *, solver: Solver[StateT],
                   guidance: CFG | None = None, key: jax.Array, times: ArrayLike | Sequence[float] | None = None,
                   final_denoise: bool = True) -> jax.Array:
    """`steps` points from T to 0: a solver step across each interval, then the
    model's clean prediction at the last point.

    `denoise` is `process.denoiser(...)`, which carries the process the solver
    reads; `guidance` wraps it. Every step's noise comes from `key` folded
    with the step index, so a trajectory is reproducible from one key.
    An explicit `times` grid is the trajectory when given, descending and
    concrete, for a source whose sampler pairs its own sigma and model-time
    tables; it decides the length, so a grid of `steps + 1` points ending
    on the terminal is legal and a single point walks nothing. Exactly one
    of `steps` and `times` is passed. `final_denoise=False` returns the last
    point's state without the closing clean prediction, the way those
    samplers end.
    """
    if (steps is None) == (times is None):
        raise ValueError("pass exactly one of steps and times")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps must be a positive integer")
    process = denoise.process
    if guidance is None:
        predict = denoise
    elif isinstance(denoise, Denoiser):
        predict = guidance(denoise)
    else:
        raise TypeError("guidance needs a continuous Denoiser; the masked diffusion LM takes none")
    with jax.ensure_compile_time_eval():
        if times is None:
            assert steps is not None
            times = process.times(steps)
        else:
            times = jnp.asarray(times)
            if times.ndim != 1 or times.shape[0] < 1:
                raise ValueError(f"times must be a descending grid of at least one point, got {times.shape}")
        if bool(jnp.any(~jnp.isfinite(times))) or bool(jnp.any(jnp.diff(times) > 0)):
            raise ValueError("times must be a finite descending grid")
    batch = x_T.shape[0]
    if times.shape[0] == 1:
        return predict(x_T, jnp.full((batch,), times[0]))[0] if final_denoise else x_T

    with jax.ensure_compile_time_eval():
        initial = solver.init(x_T, times, process, key=key)

    def body(carry, inputs):
        x, state = carry
        t, t_next, index = inputs
        t = jnp.full((batch,), t)
        t_next = jnp.full((batch,), t_next)
        denoised, eps = predict(x, t)
        x, state = solver.step(x, t, t_next, denoised, eps, state, jax.random.fold_in(key, index), process, predict)
        return (x, state), None

    (x, _), _ = lax.scan(
        body, (x_T, initial),
        (times[:-1], times[1:], jnp.arange(times.shape[0] - 1)))
    if not final_denoise:
        return x
    return predict(x, jnp.full((batch,), times[-1]))[0]
