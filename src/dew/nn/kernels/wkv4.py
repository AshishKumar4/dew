"""RWKV4's WKV (row 1455; fla's `fused_recurrent_rwkv4`, fla/ops/rwkv4): a recurrence a channel, each
channel's past held as a numerator and a denominator scaled by a running maximum,

    out_t  = (e^{eps - tau} alpha + e^{u + k_t - tau} v_t) / (e^{eps - tau} beta + e^{u + k_t - tau}),
             tau = max(u + k_t, eps)
    eps'   = max(w + eps, k_t),  alpha' = e^{w + eps - eps'} alpha + e^{k_t - eps'} v_t,
             beta' = e^{w + eps - eps'} beta + e^{k_t - eps'}

with `w = -exp(time_decay)` a channel's decay and `u` (time_first) the current token's bonus. Width one a
channel holds no product, so it runs as a scan rather than on the recurrence template's chunked mainloop: a
span's carry composes with the next's associatively (`_joined`), one `associative_scan` of every token's
`(e^{k} v, e^{k}, max k, 1)` over the sequence, segmented at packed documents' first tokens."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from dew.nn.precision import at_least_fp32

EMPTY = -1e38
"""An empty past's running maximum: a span that holds no token (`exp` of it against any token's is zero)."""


class SpanCarry(NamedTuple):
    """A span of tokens' carry: its numerator and denominator scaled by `e^{-top}`, `top` their running
    maximum (fla's alpha, beta and eps), `count` its tokens (each decaying what precedes it once) and
    `fresh` whether it begins a document (what precedes it is dropped)."""

    numerator: jax.Array
    denominator: jax.Array
    top: jax.Array
    count: jax.Array
    fresh: jax.Array


def _joined(decay):
    """The composition of a span with the span after it (`associative_scan`'s operator): the earlier one's
    past decayed by `decay` once a token of the later one, both rescaled to their larger maximum, where the
    later one begins no document."""

    def join(before: SpanCarry, after: SpanCarry) -> SpanCarry:
        aged = before.top + after.count * decay
        top = jnp.maximum(aged, after.top)
        kept, added = jnp.exp(aged - top), jnp.exp(after.top - top)
        joined = SpanCarry(kept * before.numerator + added * after.numerator,
                      kept * before.denominator + added * after.denominator, top,
                      before.count + after.count, before.fresh | after.fresh)
        dropped = after._replace(count=joined.count, fresh=joined.fresh)
        return jax.tree.map(lambda one, other: jnp.where(after.fresh, other, one), joined, dropped)
    return join


def _empty(batch: int, channels: int, work) -> jax.Array:
    """An empty past's carry `[B, 3, C]`: no numerator or denominator, the running maximum `EMPTY`."""
    return jnp.stack([jnp.zeros((batch, channels), work), jnp.zeros((batch, channels), work),
                      jnp.full((batch, channels), EMPTY, work)], axis=1)


def wkv4(time_decay, time_first, key, value, state=None, starts=None):
    """RWKV4's WKV over a sequence. `time_decay` and `time_first` `[C]` (the decay `w = -exp(time_decay)`,
    the bonus `u`), `key` and `value` `[B, S, C]`; `state` `[B, 3, C]` the carry entering (fla's alpha,
    beta and eps; an empty past where None); `starts` `[B, S]` each packed document's first token, which
    enters an empty past. Returns `(out [B, S, C] in the inputs' dtype, state [B, 3, C] in at least fp32)`,
    as fla's `fused_recurrent_rwkv4` (its state's last token)."""
    dtype = jnp.result_type(key, value)
    work = at_least_fp32(dtype)
    decay = -jnp.exp(jnp.asarray(time_decay, work))
    bonus = jnp.asarray(time_first, work)
    key, value = jnp.asarray(key, work), jnp.asarray(value, work)
    batch, length, channels = key.shape
    state = _empty(batch, channels, work) if state is None else state
    fresh = (jnp.zeros((batch, length), bool) if starts is None else jnp.asarray(starts, bool))[..., None]
    fresh = jnp.broadcast_to(fresh, key.shape)
    entering = SpanCarry(*(jnp.asarray(state, work)[:, i][:, None] for i in range(3)),
                    jnp.zeros((batch, 1, channels), work), jnp.zeros((batch, 1, channels), bool))
    tokens = SpanCarry(value, jnp.ones_like(key), key, jnp.ones_like(key), fresh)
    spans = jax.tree.map(lambda first, rest: jnp.concatenate([first, rest], axis=1), entering, tokens)
    carried = jax.lax.associative_scan(_joined(decay), spans, axis=1)
    # Each token reads the carry before it (an empty one at a document's first token) and itself at its
    # bonus.
    before = jax.tree.map(lambda x: x[:, :-1], carried)
    numerator, denominator = (jnp.where(fresh, 0.0, x) for x in (before.numerator, before.denominator))
    top = jnp.where(fresh, EMPTY, before.top)
    own = bonus + key
    tau = jnp.maximum(own, top)
    past, present = jnp.exp(top - tau), jnp.exp(own - tau)
    out = (past * numerator + present * value) / (past * denominator + present)
    final = jnp.stack([carried.numerator[:, -1], carried.denominator[:, -1], carried.top[:, -1]], axis=1)
    return out.astype(dtype), final


def recurrent_wkv4(time_decay, time_first, key, value, state=None, starts=None):
    """`wkv4`'s recurrence a token at a time as fla's kernel steps it (fused_recurrent.py:94-114), in at
    least fp32 (float64 where the inputs are); operands and results as `wkv4`'s."""
    dtype = jnp.result_type(key, value)
    work = at_least_fp32(dtype)
    decay = -jnp.exp(jnp.asarray(time_decay, work))
    bonus = jnp.asarray(time_first, work)
    key, value = jnp.asarray(key, work), jnp.asarray(value, work)
    batch, length, channels = key.shape
    state = _empty(batch, channels, work) if state is None else state
    starts = jnp.zeros((batch, length), bool) if starts is None else jnp.asarray(starts, bool)

    def one_token(carry, step):
        alpha, beta, eps = carry
        k, v, begun = step
        alpha, beta = (jnp.where(begun[:, None], 0.0, x) for x in (alpha, beta))
        eps = jnp.where(begun[:, None], EMPTY, eps)
        own = bonus + k
        tau = jnp.maximum(own, eps)
        past, present = jnp.exp(eps - tau), jnp.exp(own - tau)
        out = (past * alpha + present * v) / (past * beta + present)
        aged = decay + eps
        eps = jnp.maximum(aged, k)
        kept, added = jnp.exp(aged - eps), jnp.exp(k - eps)
        return (kept * alpha + added * v, kept * beta + added, eps), out

    entering = tuple(jnp.asarray(state, work)[:, i] for i in range(3))
    steps = tuple(jnp.moveaxis(x, 1, 0) for x in (key, value, starts))
    final, out = jax.lax.scan(one_token, entering, steps)
    return jnp.moveaxis(out, 0, 1).astype(dtype), jnp.stack(final, axis=1)
