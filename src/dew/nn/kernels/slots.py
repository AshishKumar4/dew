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


def additive_gate(x: jax.Array, entering: tuple[jax.Array, jax.Array] | None = None
                  ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array]]:
    """`(weights, log_decay, normalizer)` of `x` `[B, S, ...]` over its time axis (1) in fp32: the running
    log-normalizer `z_t = logaddexp(z_{t-1}, x_t)` from `entering` (none where None), each token's weight
    `exp(x_t - z_t)`, its log decay `z_{t-1} - z_t` (zero where nothing came before it, fla's convention),
    and the last normalizer, the state a next call enters with (`entering`, or none, for no tokens). A
    normalizer is `(largest, mass)` `[B, ...]` each: its largest write and its log mass relative to it, `z`
    their sum (`-inf, 0` none). The scan carries that pair (`_mass`) and forms each weight and decay in its
    relative frame before rounding, and a call returns and enters it so: in absolute scale a normalizer
    rounds away a mass's increment where the writes are large (two equal writes at 2^24 took `log 2` off
    it, each weight one where it is a half; carried between calls, a third token weighed a half where it
    is a third: a review's P0s of 1a60ffc17 and 969895771). An absent write (`-inf`) weighs zero and takes a
    zero gradient, as fla's gate's backward gives it."""
    x = x.astype(jnp.float32)
    shape = x.shape[:1] + x.shape[2:]
    seed_top, seed_mass = ((jnp.full(shape, -jnp.inf, jnp.float32), jnp.zeros(shape, jnp.float32))
                           if entering is None else tuple(part.astype(jnp.float32) for part in entering))
    if x.shape[1] == 0:
        return x, x, (seed_top, seed_mass)
    written = jnp.concatenate([seed_top[:, None], x], axis=1)
    masses = jnp.concatenate([seed_mass[:, None], jnp.zeros_like(x)], axis=1)
    largest, mass = jax.lax.associative_scan(_mass, (written, masses), axis=1)
    started = largest > -jnp.inf
    # Each token's normalizer's largest write and relative mass, and the prefix's before it.
    top, log_mass = largest[:, 1:], mass[:, 1:]
    before, before_mass = largest[:, :-1], mass[:, :-1]
    shift = jnp.where(started[:, :-1], jnp.where(started[:, :-1], before, 0.0) - top, 0.0)
    log_decay = jnp.where(started[:, :-1], shift + (before_mass - log_mass), 0.0)
    present = started[:, 1:] & (x > -jnp.inf)
    relative = jnp.where(present, x - jnp.where(present, top, 0.0), 0.0) - log_mass
    weights = jnp.where(present, jnp.exp(jnp.where(present, relative, 0.0)), 0.0)
    return weights, log_decay, (largest[:, -1], mass[:, -1])


def log_normalizer(normalizer: tuple[jax.Array, jax.Array]) -> jax.Array:
    """A normalizer's (`additive_gate`'s) log in absolute scale, `largest + mass`: a diagnostic, not the
    state a call continues from (rounded where the writes are large)."""
    largest, mass = normalizer
    return jnp.where(largest > -jnp.inf, largest + mass, -jnp.inf)


def _mass(a: tuple, b: tuple) -> tuple:
    """Two prefixes' writes combined, each `(largest, log mass relative to it)`: the larger largest and the
    log of both masses relative to it (`jnp.logaddexp`'s `log1p`, its increment kept); both absent (`-inf`),
    `(-inf, 0)`, each leaf a constant with no derivative into either."""
    (top_a, mass_a), (top_b, mass_b) = a, b
    top = jnp.maximum(top_a, top_b)
    started = top > -jnp.inf
    frame = jnp.where(started, top, 0.0)

    def relative(largest, mass):
        here = largest > -jnp.inf
        return jnp.where(here, jnp.where(here, largest, 0.0) - frame + mass, -jnp.inf)

    ends = (relative(top_a, mass_a), relative(top_b, mass_b))
    summed = jnp.logaddexp(*(jnp.where(started, end, 0.0) for end in ends))
    # Both absent, each leaf a constant: `maximum`'s own derivative splits between two absent writes.
    return jnp.where(started, top, -jnp.inf), jnp.where(started, summed, 0.0)


def lightnet(query: jax.Array, key: jax.Array, value: jax.Array, *, scale: float | None = None,
             state: jax.Array | None = None, normalizer: tuple[jax.Array, jax.Array] | None = None):
    """`(output [B, S, H, Dv], (memory [B, H, Dk, Dv], normalizer))` of LightNet's recurrence
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
                   state: tuple | None = None):
    """`(output [B, S, H, Dv], (keys' memory [B, H, Dk, M], values' memory [B, H, M, Dv][, normalizer]))` of
    gated slot attention (fla's `chunk_gsa`; ABC's, `chunk_abc`, where `log_decay` is None):
    `M` slots, each token's slot writes `slots` `[B, S, H, M]` decaying a slot by `log_decay` `[B, S, H, M]`
    (fp32), or where None the slots' own running normalizer over time (`additive_gate`: the writes its
    weights, ABC's softmax over time). Two recurrences, each the chunked op's: the keys into the slots,
    decayed along the slots (`Hooks.value_decay`), read `scale q_t^T` (`Dk^-1/2` where None); a softmax over
    the slots of that read in fp32; then the slots into the values, decayed a slot, read by the softmax. Its
    heads each its queries', keys' and values', `state` the memories entering (zero where None) and, ABC's,
    the slots' normalizer entering (`additive_gate`'s pair, `[B, H, M]` each; none where absent or None):
    ABC's state carries its normalizer, so a
    call split in two is the call whole (each part's gate started afresh read a second token at twice its
    weight: a review's P0 of 1a60ffc17); GSA's, its decays given, is its two memories."""
    keys_memory, values_memory, *rest = (None, None) if state is None else state
    carried: tuple = ()
    if log_decay is None:
        entering: tuple[jax.Array, jax.Array] | None = rest[0] if rest else None
        weights, log_decay, normalizer = additive_gate(slots, entering)
        slots, carried = weights.astype(slots.dtype), (normalizer,)
    if query.shape[1] == 0:
        return value, (_entered(keys_memory, key, slots), _entered(values_memory, slots, value), *carried)
    log_decay = log_decay.astype(jnp.float32)
    into = recurrence.Hooks(decay='none', delta=False, value_decay=True, scale=_scaled(query, scale))
    read, keys_final = chunk(into, Chunk(query, key, slots, value_decay=log_decay, state=keys_memory))
    weights = jax.nn.softmax(read.astype(jnp.float32), axis=-1).astype(value.dtype)
    out_of = recurrence.Hooks(decay='vector', delta=False, scale=1.0)
    out, values_final = chunk(out_of, Chunk(weights, slots, value, log_decay=log_decay, state=values_memory))
    return out, (keys_final, values_final, *carried)


def _scaled(query: jax.Array, scale: float | None) -> float:
    """A readout's scale: `scale`, or the queries' `Dk^-1/2` where None, as fla's defaults are."""
    return query.shape[-1] ** -0.5 if scale is None else scale


def _entered(memory: jax.Array | None, key: jax.Array, value: jax.Array) -> jax.Array:
    """A call of no tokens' final memory: the one entering, or zero `[B, H, Dk, Dv]` (fp32) where None."""
    if memory is not None:
        return memory
    return jnp.zeros((key.shape[0], key.shape[2], key.shape[3], value.shape[3]), jnp.float32)
