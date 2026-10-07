"""The reverse process as one scan over a time grid."""

from collections.abc import Sequence
from typing import overload

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

from dew.diffusion.discrete import DiscreteDenoiser
from dew.diffusion.process import Denoiser
from dew.sampling.guidance import Guidance, Walk
from dew.sampling.solvers import Solver


@overload
def sample[StateT](
    denoise: Denoiser | DiscreteDenoiser,
    x_T: jax.Array,
    steps: int,
    *,
    solver: Solver[StateT],
    guidance: Guidance | None = None,
    key: int | jax.Array,
    times: None = None,
    final_denoise: bool = True,
) -> jax.Array: ...


@overload
def sample[StateT](
    denoise: Denoiser | DiscreteDenoiser,
    x_T: jax.Array,
    steps: None = None,
    *,
    solver: Solver[StateT],
    guidance: Guidance | None = None,
    key: int | jax.Array,
    times: ArrayLike | Sequence[float],
    final_denoise: bool = True,
) -> jax.Array: ...


def sample[StateT](
    denoise: Denoiser | DiscreteDenoiser,
    x_T: jax.Array,
    steps: int | None = None,
    *,
    solver: Solver[StateT],
    guidance: Guidance | None = None,
    key: int | jax.Array,
    times: ArrayLike | Sequence[float] | None = None,
    final_denoise: bool = True,
) -> jax.Array:
    """Run the reverse process from `x_T` with `solver` and return the final sample.

    The time grid runs from T to 0. The solver takes one step across each
    interval, and the result is the model's clean prediction at the last
    point. Pass exactly one of `steps` and `times`. With `steps`, the
    process supplies its grid (`times(steps)`): that many points for a
    Gaussian process, and MDLM's grid of that many reveal steps for a masked
    one (`DiscreteProcess.times`). An explicit `times` grid is
    used as the trajectory, descending and concrete, for a source whose
    solver pairs its own sigma and model-time tables. Its length sets the
    number of steps, so a grid of `steps + 1` points ending on the terminal
    is allowed, and a single point takes no solver step.
    `final_denoise=False` returns the last point's state without the closing
    clean prediction, which is how those solvers end.

    `denoise` is `process.denoiser(...)`, which holds the process the solver
    reads, and `guidance` wraps it. Every step's noise comes from `key`
    folded with the step index, so a trajectory is reproducible from one
    key.

    The trajectory is one `lax.scan`, traced at each call. Under a caller's
    `jax.jit` it compiles once, which is how the pipelines and objectives
    call it. Called eagerly, it traces and compiles the scan again every
    time.
    """
    if (steps is None) == (times is None):
        raise ValueError("pass exactly one of steps and times")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps must be a positive integer")
    from dew.nn.inputs import request_key
    key = request_key(key)
    process = denoise.process

    # A process whose model predicts over an interval reads, at each step,
    # the interval to the next grid point.
    spanned = denoise if isinstance(denoise, Denoiser) and denoise.process.interval else None
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
    count = times.shape[0] - 1
    walk = Walk.over(denoise, guidance, count)
    batch = x_T.shape[0]
    guided = walk.init(x_T)
    if times.shape[0] == 1:
        return walk.at(guided, count)(x_T, jnp.full((batch,), times[0]))[0] if final_denoise else x_T

    with jax.ensure_compile_time_eval():
        initial = solver.init(x_T, times, process, key=key)

    def body(carry, inputs):
        x, state, guided = carry
        t, t_next, index = inputs
        t = jnp.full((batch,), t)
        t_next = jnp.full((batch,), t_next)
        stepping = walk if spanned is None else Walk.over(spanned.spanning(t, t_next), guidance, count)
        (denoised, eps), guided = stepping.step(x, t, index, guided)
        x, state = solver.step(x, t, t_next, denoised, eps, state, jax.random.fold_in(key, index),
                               process, stepping.at(guided, index))
        return (x, state, guided), None

    (x, _, guided), _ = lax.scan(
        body, (x_T, initial, guided),
        (times[:-1], times[1:], jnp.arange(times.shape[0] - 1)))
    if not final_denoise:
        return x
    return walk.at(guided, count)(x, jnp.full((batch,), times[-1]))[0]
