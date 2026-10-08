"""Step-by-step sampler trajectories used by numerical reference tests."""

import jax
import jax.numpy as jnp


def walk(solver, process, model, x_T, times, key=0):
    """Record each solver interval, with the sampler's folded per-step keys."""
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, *x_T.shape[1:])), jnp.ones((1,)))
    denoise = process.denoiser(model, params, {})
    with jax.ensure_compile_time_eval():
        times = jnp.asarray(times, jnp.float32)
    return sample_walk(denoise, solver, x_T, times, key)


def sample_walk(denoise, solver, x_T, times, key=0):
    """Walk without the closing denoise and retain every intermediate state."""
    key = jax.random.PRNGKey(key) if isinstance(key, int) else key
    process = denoise.process

    def body(carry, inputs):
        x, state = carry
        t, t_next, index = inputs
        t = jnp.full((x.shape[0],), t)
        t_next = jnp.full((x.shape[0],), t_next)
        denoised, eps = denoise(x, t)
        x, state = solver.step(x, t, t_next, denoised, eps, state, jax.random.fold_in(key, index),
                               process, denoise)
        return (x, state), x

    _, latents = jax.lax.scan(body, (x_T, solver.init(x_T, times, process, key=key)),
                              (times[:-1], times[1:], jnp.arange(times.shape[0] - 1)))
    return latents
