"""The reverse process as one scan over a time grid."""

import jax
import jax.numpy as jnp
from jax import lax


def sample(denoise, x_T, steps: int, *, solver, guidance=None, key):
    """`steps` points from T to 0: a solver step across each interval, then the
    model's clean prediction at the last point.

    `denoise` is `process.denoiser(...)`, which carries the process the solver
    reads; `guidance` wraps it. Every step's noise comes from `key` folded
    with the step index, so a trajectory is reproducible from one key.
    """
    process = denoise.process
    if guidance is not None:
        denoise = guidance(denoise)
    times = process.times(steps)
    batch = x_T.shape[0]

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
        body, (x_T, solver.init(x_T)),
        (times[:-1], times[1:], jnp.arange(times.shape[0] - 1)))
    return denoise(x, jnp.full((batch,), times[-1]))[0]
