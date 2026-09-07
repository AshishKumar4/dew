"""The reverse process as one scan over a time grid."""

import jax
import jax.numpy as jnp
from jax import lax


def sample(denoise, x_T, steps=None, *, solver, guidance=None, key, times=None, final_denoise=True):
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
    process = denoise.process
    if guidance is not None:
        denoise = guidance(denoise)
    with jax.ensure_compile_time_eval():
        if times is None:
            times = process.times(steps)
        else:
            times = jnp.asarray(times, jnp.float32)
            if times.ndim != 1 or times.shape[0] < 1:
                raise ValueError(f"times must be a descending grid of at least one point, got {times.shape}")
    batch = x_T.shape[0]
    if times.shape[0] == 1:
        return denoise(x_T, jnp.full((batch,), times[0]))[0] if final_denoise else x_T

    def body(carry, inputs):
        x, state = carry
        t, t_next, index = inputs
        t = jnp.full((batch,), t)
        t_next = jnp.full((batch,), t_next)
        denoised, eps = denoise(x, t)
        x, state = solver.step(x, t, t_next, denoised, eps, state,
                               jax.random.fold_in(key, index), process, denoise)
        return (x, state), None

    (x, _), _ = lax.scan(
        body, (x_T, solver.init(x_T, times, process)),
        (times[:-1], times[1:], jnp.arange(times.shape[0] - 1)))
    if not final_denoise:
        return x
    return denoise(x, jnp.full((batch,), times[-1]))[0]
