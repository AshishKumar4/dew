"""Recurrences composed of the chunked op (`recurrence_chunk.chunk`) and an auxiliary scan, no kernel of their
own (kernel-templates.md §B, reconciliation 1): LightNet's (`lightnet`), whose keys a running log-normalizer
over time scales, and the gated slot attentions, GSA's and ABC's (`slot_attention`), a recurrence into
slots read through a softmax by a second. Each takes a row's tokens as one document.

Their auxiliary scan is fla's additive gate (`additive_gate`, fla/ops/gsa/gate.py at 413ad2c): a running
`z_t = log(exp(z_{t-1}) + exp(x_t))` over time, each token's weight `exp(x_t - z_t)` and its decay
`exp(z_{t-1} - z_t)`, so a state of such weights is the softmax over time of every write so far.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import recurrence
from .recurrence_chunk import Chunk, chunk


def additive_gate(x: jax.Array, entering: jax.Array | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`(weights, log_decay, normalizer)` of `x` `[B, S, ...]` over its time axis (1) in fp32: the running
    log-normalizer `z_t = logaddexp(z_{t-1}, x_t)` from `entering` (`[B, ...]`, none where None: `-inf`),
    each token's weight `exp(x_t - z_t)`, its log decay `z_{t-1} - z_t` (zero where nothing came before it,
    fla's convention), and the last `z`, the normalizer a next call enters with (`entering`'s, or `-inf`,
    for no tokens). An absent write (`-inf`) weighs zero and takes a zero gradient, as fla's gate's backward
    gives it, however many absent writes a scan combines (`_logaddexp`)."""
    x = x.astype(jnp.float32)
    seed = (jnp.full(x.shape[:1] + x.shape[2:], -jnp.inf, jnp.float32) if entering is None
            else entering.astype(jnp.float32))
    if x.shape[1] == 0:
        return x, x, seed
    z = _logaddexp(jax.lax.associative_scan(_logaddexp, x, axis=1), seed[:, None])
    before = jnp.concatenate([seed[:, None], z[:, :-1]], axis=1)
    started = jnp.isfinite(before)
    log_decay = jnp.where(started, jnp.where(started, before, 0.0) - z, 0.0)
    present = jnp.isfinite(z) & (x > -jnp.inf)
    weights = jnp.where(present, jnp.exp(jnp.where(present, x - jnp.where(present, z, 0.0), 0.0)), 0.0)
    return weights, log_decay, z[:, -1]


def _logaddexp(a: jax.Array, b: jax.Array) -> jax.Array:
    """`jnp.logaddexp(a, b)` (its `log1p`: a faint increment kept), `-inf` where both are absent (`-inf`), a
    constant there with no derivative into either: `jnp.logaddexp`'s forms `exp(-inf - -inf)`, a NaN a zero
    cotangent keeps, so both absent it takes finite stand-ins and selects the constant."""
    absent = (a == -jnp.inf) & (b == -jnp.inf)
    found = jnp.logaddexp(jnp.where(absent, 0.0, a), jnp.where(absent, 0.0, b))
    return jnp.where(absent, -jnp.inf, found)


def lightnet(query: jax.Array, key: jax.Array, value: jax.Array, *, scale: float | None = None,
             state: jax.Array | None = None, normalizer: jax.Array | None = None):
    """`(output [B, S, H, Dv], (memory [B, H, Dk, Dv], normalizer [B, H, Dk]))` of LightNet's recurrence
    (fla's `chunk_lightnet`): a key dimension's running log-normalizer over time (`additive_gate`, entering
    `normalizer`), the keys its weights and the memory decayed a key dimension by it, `S_t = diag(exp(z_{t-1}
    - z_t)) S_{t-1} + exp(k_t - z_t) v_t^T`, read `scale q_t^T S_t` (`Dk^-1/2` where None) from `state`. Its
    queries as given (the layer's SiLU its own), its heads each its keys' and values'."""
    weights, log_decay, last = additive_gate(key, normalizer)
    if query.shape[1] == 0:
        return value, (_entered(state, key, value), last)
    hooks = recurrence.Hooks(decay='vector', delta=False, scale=_scaled(query, scale))
    given = Chunk(query, weights.astype(key.dtype), value, log_decay=log_decay, state=state)
    out, final = chunk(hooks, given)
    return out, (final, last)


def slot_attention(query: jax.Array, key: jax.Array, value: jax.Array, slots: jax.Array,
                   log_decay: jax.Array | None = None, *, scale: float | None = None,
                   state: tuple[jax.Array, jax.Array] | None = None):
    """`(output [B, S, H, Dv], (keys' memory [B, H, Dk, M], values' memory [B, H, M, Dv]))` of gated slot
    attention (fla's `chunk_gsa`; ABC's, `chunk_abc`, where `log_decay` is None): `M` slots, each token's
    slot writes `slots` `[B, S, H, M]` decaying a slot by `log_decay` `[B, S, H, M]` (fp32), or where None
    the slots' own running normalizer over time (`additive_gate`: the writes its weights, ABC's
    softmax over time). Two recurrences, each the chunked op's: the keys into the slots, decayed along the
    slots (`Hooks.value_decay`), read `scale q_t^T` (`Dk^-1/2` where None); a softmax over the slots of that
    read in fp32; then the slots into the values, decayed a slot, read by the softmax. Its heads each its
    queries', keys' and values', `state` the two memories entering (zero where None)."""
    keys_memory, values_memory = (None, None) if state is None else state
    if query.shape[1] == 0:
        return value, (_entered(keys_memory, key, slots), _entered(values_memory, slots, value))
    if log_decay is None:
        weights, log_decay, _ = additive_gate(slots)
        slots = weights.astype(slots.dtype)
    log_decay = log_decay.astype(jnp.float32)
    into = recurrence.Hooks(decay='none', delta=False, value_decay=True, scale=_scaled(query, scale))
    read, keys_final = chunk(into, Chunk(query, key, slots, value_decay=log_decay, state=keys_memory))
    weights = jax.nn.softmax(read.astype(jnp.float32), axis=-1).astype(value.dtype)
    out_of = recurrence.Hooks(decay='vector', delta=False, scale=1.0)
    out, values_final = chunk(out_of, Chunk(weights, slots, value, log_decay=log_decay, state=values_memory))
    return out, (keys_final, values_final)


def _scaled(query: jax.Array, scale: float | None) -> float:
    """A readout's scale: `scale`, or the queries' `Dk^-1/2` where None, as fla's defaults are."""
    return query.shape[-1] ** -0.5 if scale is None else scale


def _entered(memory: jax.Array | None, key: jax.Array, value: jax.Array) -> jax.Array:
    """A call of no tokens' final memory: the one entering, or zero `[B, H, Dk, Dv]` (fp32) where None."""
    if memory is not None:
        return memory
    return jnp.zeros((key.shape[0], key.shape[2], key.shape[3], value.shape[3]), jnp.float32)
