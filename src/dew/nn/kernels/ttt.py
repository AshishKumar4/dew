"""TTT-Linear (row 1448; fla's `chunk_ttt_linear_ref`, fla/ops/ttt/naive.py): a memory `W` `[D, D]` and its
bias `b` `[D]` learned at test time, a mini-batch of `chunk` tokens at a time, each mini-batch's inner
gradients `u_s` taken at the memory entering it (`titans.inner_gradient`, the LayerNorm reconstruction loss's,
`|LN(k W + b) w + c - (v - k)|^2`), each token's output reading that memory less its mini-batch's updates so
far at its own learning rate,

    o_t = q_t (W_0 - eta_t sum_{s <= t} k_s^T u_s) + b_0 - eta_t sum_{s <= t} u_s,

the memory leaving the mini-batch at its last token's rate, `W_0 - eta_last sum_s k_s^T u_s`, and each output
read out through its LayerNorm residual (`titans.read_out`), `q` scaled by `scale`. A C-class orchestrator
(docs/design/kernel-templates.md: its orchestration outside the recurrence tile), as Titans'
(`titans.titans`): a mini-batch's gradients at once, then each token's rated prefix of them, each token
reading the memory its updates so far leave at its rate (`_rated_prefixes`), so no query meets a key (a live
score of 2^70 by 2^70 overflowed beside a zero update). A partial last mini-batch's state leaves at its last
token's rate and its absent tokens write nothing, their readouts zero before the LayerNorm sees them, where
fla's reference pads them and their gradients at the bias (a zero key's `k W + b`) write the bias."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dew.nn.aligned import aligned_sum
from dew.nn.kernels.titans import absent_zeroed, mini_batches, prepared_operands, read_out, real_gradients


def _rated_prefixes(k, update, eta, real):
    """Each token's rated prefix of its mini-batch's updates: `eta_t sum_{s <= t} k_s^T u_s` `[B, H, C, D,
    Dv]` and `eta_t sum_{s <= t} u_s` `[B, H, C, 1, Dv]` over the real tokens `s` (`real` `[B, H, C]`), `k`
    `[B, H, C, D]`, `update` `[B, H, C, Dv]` and `eta` `[B, H, C]`: each one exact product rounded once
    (`aligned_sum`), the rate meeting each key and update before any sum does, as fla's chunk kernel rates
    its keys first (fla/ops/ttt/chunk.py:119-125): keys of `2^70` and updates of `2^69` under a rate of
    `2^-20` write `2^119`, where their unrated sum overflowed (train 1470's review). A later or absent
    token's key and update selected out before any product, so its own non-finite update reaches no earlier
    prefix (a zero factor times it was NaN; a head review). One prefix at a time (`lax.map`, each exact sum
    over the mini-batch's `C` tokens: `C^2 D Dv` exact products a mini-batch, where the plain prefix carries
    `C D Dv`), its working set one prefix's."""
    index = jnp.arange(k.shape[2])

    def prefix(at):
        t, rate = at
        upto = ((index <= t) & real)[..., None]
        keys, updates = (jnp.where(upto, x, jnp.zeros((), x.dtype)) for x in (k, update))
        rated_keys = aligned_sum([[rate[..., None, None, None], keys[..., :, None], updates[..., None, :]]],
                                 axis=2)
        return rated_keys[:, :, 0], aligned_sum([[rate[..., None, None], updates]], axis=2)

    keys, values = jax.lax.map(prefix, (index, jnp.moveaxis(eta, 2, 0)))
    return jnp.moveaxis(keys, 0, 2), jnp.moveaxis(values, 0, 2)


def ttt_linear(query, key, value, weight, bias, eta, state=None, *, chunk: int = 16, scale=None,
               eps: float = 1e-6, valid=None):
    """TTT-Linear a mini-batch of `chunk` tokens at a time: `query`, `key` and `value` `[B, T, H, D]`, the
    inner loss's LayerNorm affine `weight` and `bias` `[H, D]`, the learning rate `eta` `[B, T, H]` and
    `state` the entering `(W, b)` (`[B, H, D, D]`, `[B, H, 1, D]`; zeros where None); the output
    `[B, T, H, D]` in the operands' dtype and the final `(W, b)` in `at_least_fp32` of it, which every
    step computes in. A token `valid` (`[B, T]`) marks absent is none: its operands zero
    (`titans.absent_zeroed`), the state as it was."""
    dtype = jnp.result_type(query, key, value)
    # TTT's memory `[D, Dv]` and bias `[1, Dv]`; its queries scaled.
    shapes = ((key.shape[-1], value.shape[-1]), (1, value.shape[-1]))
    scale = key.shape[-1] ** -0.5 if scale is None else scale
    q, k, v, (weight, bias), (eta,), states = prepared_operands(
        query, key, value, weight, bias, (eta,), state, shapes, scale)
    q, k, v, eta = absent_zeroed(valid, q, k, v, eta)

    def mini_batch(entering, at):
        memory, offset = entering
        q, k, v, eta, real = at
        # Each real token's gradient at the memory entering the mini-batch.
        update = real_gradients(k, v, k @ memory + offset, real, weight, bias, eps)
        keys, values = _rated_prefixes(k, update, eta, real)
        out = (jnp.einsum('bhtk,bhtkv->bhtv', q, memory[:, :, None] - keys)
               + (offset[:, :, None] - values)[:, :, :, 0])
        # The state leaves as its last real token reads it: at its rate, every real token's update.
        last = jnp.max(jnp.where(real, jnp.arange(real.shape[-1]), -1), axis=-1)
        at_last = jnp.maximum(last, 0)[:, :, None, None, None]
        leaving = (jnp.take_along_axis(rated, at_last, axis=2)[:, :, 0] for rated in (keys, values))
        return tuple(state - jnp.where(last[:, :, None, None] >= 0, rated, 0.0)
                     for state, rated in zip(entering, leaving, strict=True)), jnp.moveaxis(out, 2, 0)

    keys, *rest = mini_batches(chunk, k, q, v, eta, valid=valid)
    final, out = jax.lax.scan(mini_batch, states, (rest[0], keys, *rest[1:]))
    return read_out(out, k.shape[2], weight, bias, eps, dtype), final
