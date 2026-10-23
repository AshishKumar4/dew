"""A diagnostic regrouping: contract the scalar-decay-free WY coefficients first."""
import jax
import jax.numpy as jnp

from dew.nn.kernels import recurrence_algebra as a
from dew.nn.kernels.recurrence_chunk import CHUNK_SIZE


def scalar_regroup(query, key, value, g, beta, state=None, chunk_size=CHUNK_SIZE, *, starts=None,
                   delta=True, scale=None, dtype=None):
    assert starts is None and delta and g.shape[-1] == 1
    b, s, h, dk = key.shape
    dv = value.shape[-1]
    pad = (-s) % chunk_size
    t = s + pad

    def block(x):
        x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        return jnp.moveaxis(x, 2, 1).reshape(b, h, -1, chunk_size, x.shape[-1])

    query, key, keyed, valued, g = (block(x) for x in
                                    (query * (dk**-.5 if scale is None else scale), key,
                                     key * beta[..., None], value * beta[..., None], g))
    grown, pairs = a.decays(g)
    unit = jnp.tril(jnp.ones(pairs.shape[-3:-1], dtype=bool))[..., None].astype(g.dtype)
    inverse = a.factor(a.intra(key, keyed, unit))
    coefficient = a.portable("local", a.paired("scores", query, key, unit), inverse)
    projected = query - a.portable("corrected", coefficient, keyed)
    end_coefficient = a.portable("write", jnp.swapaxes(key, -1, -2), inverse)
    transition = (jnp.eye(dk, dtype=g.dtype)
                  - a.portable("corrected", end_coefficient, keyed)) * grown[..., -1:, :]
    write = a.portable("write", end_coefficient * pairs[..., -1, :, 0][..., None, :], valued)
    if state is None:
        state = jnp.zeros((b, h, dk, dv), g.dtype)

    def step(state, tw):
        transition, write = tw
        return a.portable("compose", transition, state) + write, state

    final, entered = jax.lax.scan(step, state, (jnp.moveaxis(transition, 2, 0),
                                               jnp.moveaxis(write, 2, 0)))
    entered = jnp.moveaxis(entered, 0, 2)
    out = (a.remembered(projected * grown, entered)
           + a.remembered(coefficient * pairs[..., 0], valued, name="local"))
    out = jnp.moveaxis(out.reshape(b, h, t, dv), 1, 2)[:, :s]
    return out.astype(dtype or value.dtype), final
