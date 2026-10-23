"""Diagnostic derivative regrouping, separating a WY primal from chronological tangents."""
import jax
import jax.numpy as jnp

from dew.nn.kernels import recurrence, recurrence_chunk as c

ORIGINAL = c.chunk_delta_rule


def chronological(query, key, value, g, beta, state, *, starts, delta, scale, dtype):
    b, s, h, dk = key.shape
    query = query * (dk**-.5 if scale is None else scale)
    resets = jnp.zeros((b, s), bool) if starts is None else starts

    def one(state, inputs):
        q, k, v, log_decay, write, reset = inputs
        state = jnp.where(reset[:, None, None, None], 0.0, state)
        out, state = recurrence.update(state, q, k, v, jnp.exp(log_decay), write[..., None], delta=delta)
        return state, out

    state, out = jax.lax.scan(one, state, tuple(jnp.moveaxis(x, 1, 0)
                                                for x in (query, key, value, g, beta, resets)))
    return jnp.moveaxis(out, 0, 1).astype(dtype), state


def stable_jvp(query, key, value, g, beta, state=None, chunk_size=c.CHUNK_SIZE, *, starts=None,
               delta=True, scale=None, dtype=None):
    dtype = jnp.result_type(query, key, value) if dtype is None else dtype
    if state is None:
        state = jnp.zeros((*key.shape[:1], key.shape[2], key.shape[3], value.shape[3]), value.dtype)

    @jax.custom_jvp
    def run(query, key, value, g, beta, state):
        return ORIGINAL(query, key, value, g, beta, state, chunk_size, starts=starts,
                        delta=delta, scale=scale, dtype=dtype)

    @run.defjvp
    def tangent(primals, tangents):
        primal = run(*primals)
        _, tangent = jax.jvp(lambda *xs: chronological(*xs, starts=starts, delta=delta, scale=scale,
                                                       dtype=dtype), primals, tangents)
        return primal, tangent

    return run(query, key, value, g, beta, state)
